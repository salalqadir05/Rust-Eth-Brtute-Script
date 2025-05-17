use anyhow::{Context, Result};
use cust::context::Context as CudaContext;
use cust::device::Device;
use cust::function::Function;
use cust::module::{Module, Symbol};
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::ffi::CString;
use std::sync::Arc;
use std::fs;
use std::time::Instant;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

// Global CUDA context and module
lazy_static::lazy_static! {
    static ref CUDA_CONTEXT: Mutex<Option<(Arc<CudaContext>, Arc<Module>)>> = Mutex::new(None);
}

pub struct GpuWorker<'m> {
    function: Function<'m>,
    stream: Stream,
    wordlist_len: i32,
    known_indices: Vec<i32>,
    known_count: i32,
    target_address: Vec<u8>,
    match_mode: i32,
    match_prefix_len: i32,
    worker_id: u32,
    total_workers: u32,
    resume_from: u64,
    _module: Arc<Module>, // Keep module alive
}

fn get_bip39_wordlist() -> Vec<String> {
    let content = include_str!("../words.txt");
    content.lines().map(|l| l.to_string()).collect()
}

pub fn init_gpu_context(device_id: u32) -> Result<(Arc<CudaContext>, Arc<Module>)> {
    let mut context_guard = CUDA_CONTEXT.lock().unwrap();
    
    // If context already exists, return it
    if let Some((ctx, module)) = context_guard.as_ref() {
        return Ok((Arc::clone(ctx), Arc::clone(module)));
    }

    // Otherwise create new context
    let device_count = Device::num_devices()
        .with_context(|| "Failed to get number of CUDA devices")?;
    if device_id as usize >= device_count.try_into().unwrap() {
        anyhow::bail!("Requested GPU {} but only {} available", device_id, device_count);
    }
    let device = Device::get_device(device_id)
        .with_context(|| format!("Failed to get device {}", device_id))?;
    let context = Arc::new(CudaContext::new(device)
        .with_context(|| format!("Failed to create CUDA context for device {}", device_id))?);

    let ptx = include_str!("gpu_kernel.ptx");
    let ptx_cstr = CString::new(ptx)
        .with_context(|| "PTX contains null byte")?;
    let module = Arc::new(Module::load_from_string(ptx_cstr.as_c_str())
        .with_context(|| "Failed to load PTX module")?);

    let wordlist = get_bip39_wordlist();
    let mut host_words = [0u8; 20480]; // 2048 words * 10 bytes
    for (i, word) in wordlist.iter().enumerate().take(2048) {
        let bytes = word.as_bytes();
        let len = bytes.len().min(9);
        host_words[i * 10..i * 10 + len].copy_from_slice(&bytes[..len]);
        host_words[i * 10 + len] = 0;
    }

    let symbol_name = CString::new("wordlist").unwrap();
    let mut symbol: Symbol<[u8; 20480]> = module.get_global(&symbol_name)
        .with_context(|| "Failed to get symbol 'wordlist' from module")?;
    unsafe {
        symbol.copy_from(&host_words)
            .with_context(|| "Failed to copy wordlist to device")?;
    }

    // Store the context and module
    *context_guard = Some((Arc::clone(&context), Arc::clone(&module)));
    Ok((context, module))
}

impl<'m> GpuWorker<'m> {
    pub fn new(
        _ctx: &CudaContext,
        module: Arc<Module>,
        wordlist: Arc<Vec<String>>,
        known_words: Arc<Vec<String>>,
        address: &String,
        match_mode: i32,
        match_prefix_len: i32,
        worker_id: u32,
        total_workers: u32,
        resume_from: u64,
    ) -> Result<Self> {
        let offset_file = format!("worker{}.offset", worker_id);
        let resume_offset = if Path::new(&offset_file).exists() {
            let content = fs::read_to_string(&offset_file)?;
            content.trim().parse::<u64>().unwrap_or(resume_from)
        } else {
            resume_from
        };

        let known_indices: Vec<i32> = known_words.iter().map(|w| {
            wordlist.iter().position(|x| x == w)
                .ok_or_else(|| anyhow::anyhow!("Unknown seed word '{}'", w))
                .map(|idx| idx as i32)
        }).collect::<Result<Vec<i32>>>()?;
        let known_count = known_indices.len() as i32;

        let addr_str = address.strip_prefix("0x").unwrap_or(address);
        let mut target_address = Vec::with_capacity(addr_str.len() / 2);
        for i in 0..(addr_str.len() / 2) {
            let byte = u8::from_str_radix(&addr_str[2 * i..2 * i + 2], 16)
                .with_context(|| format!("Invalid hex in address at pos {}", i))?;
            target_address.push(byte);
        }

        let function = module.get_function("search_seeds")
            .with_context(|| "Failed to get 'search_seeds' function")?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .with_context(|| "Failed to create CUDA stream")?;

        Ok(Self {
            function,
            stream,
            wordlist_len: wordlist.len() as i32,
            known_indices,
            known_count,
            target_address,
            match_mode,
            match_prefix_len,
            worker_id,
            total_workers,
            resume_from: resume_offset,
            _module: module,
        })
    }

    pub fn run_loop(&mut self, found_flag: &AtomicBool) -> Result<Option<String>> {
        let d_seeds_tested = DeviceBuffer::<u64>::zeroed(1)?;
        let d_seeds_found = DeviceBuffer::<u64>::zeroed(1)?;
        let d_known = DeviceBuffer::from_slice(&self.known_indices)?;
        let d_address = DeviceBuffer::from_slice(&self.target_address)?;
        let d_found_mnemonic = DeviceBuffer::<i32>::zeroed(12)?;

        let unknown_count = 12 - self.known_count as u32;
        let total_candidates = 2048u64.pow(unknown_count);
        let batch_size = total_candidates / self.total_workers as u64;
        let threads = 256;
        let blocks = ((batch_size + threads as u64 - 1) / threads as u64) as u32;

        let mut current_offset = self.resume_from;
        let start_time = Instant::now();
        let mut total_tested = 0u64;

        while current_offset < batch_size * (self.worker_id as u64 + 1) {
            if found_flag.load(Ordering::SeqCst) {
                println!("Worker {} stopping due to found flag", self.worker_id);
                break;
            }

            let batch_remaining = batch_size * (self.worker_id as u64 + 1) - current_offset;
            let current_batch_size = std::cmp::min(batch_remaining, batch_size);
            let func = &self.function;
            let stream = &self.stream;
            unsafe {
                launch!(func<<<blocks, threads, 0, stream>>>(
                    d_seeds_tested.as_device_ptr(),
                    d_seeds_found.as_device_ptr(),
                    current_offset,
                    current_batch_size,
                    self.wordlist_len,
                    self.known_count,
                    d_known.as_device_ptr(),
                    d_address.as_device_ptr(),
                    self.match_mode,
                    self.match_prefix_len,
                    d_found_mnemonic.as_device_ptr(),
                ))?;
            }

            self.stream.synchronize()?;

            let mut tested = [0u64];
            let mut found = [0u64];
            d_seeds_tested.copy_to(&mut tested)?;
            d_seeds_found.copy_to(&mut found)?;

            total_tested += tested[0];
            let elapsed = start_time.elapsed();
            let rate = total_tested as f64 / elapsed.as_secs_f64();

            println!(
                "Worker {} tested {} (total: {}); found {}; rate {:.2} seeds/sec",
                self.worker_id, tested[0], total_tested, found[0], rate
            );

            if found[0] > 0 {
                let mut mnemonic_indices = [0i32; 12];
                d_found_mnemonic.copy_to(&mut mnemonic_indices)?;

                let wordlist = get_bip39_wordlist();
                let mut mnemonic_words = Vec::new();

                let mut ki = 0;
                let mut ui = 0;
                for i in 0..12 {
                    if ki < self.known_count as usize {
                        mnemonic_words.push(wordlist[self.known_indices[ki] as usize].clone());
                        ki += 1;
                    } else {
                        mnemonic_words.push(wordlist[mnemonic_indices[ui] as usize].clone());
                        ui += 1;
                    }
                }

                let mnemonic = mnemonic_words.join(" ");
                println!("Worker {} found mnemonic: {}", self.worker_id, mnemonic);
                found_flag.store(true, Ordering::SeqCst);
                return Ok(Some(mnemonic));
            }

            current_offset += current_batch_size;
            fs::write(format!("worker{}.offset", self.worker_id), current_offset.to_string())?;
        }

        Ok(None)
    }
}
