use anyhow::{Context, Result};
use cust::context::Context as CudaContext;
use cust::device::Device;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use cust::memory::{AsyncCopyDestination, DeviceBuffer};
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use cust::DeviceCopy;
use cust::launch;
use cust::memory::CopyDestination;
#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
struct GpuWord {
    bytes: [u8; 10], // 9 chars + null terminator
    len: u8,
}

pub fn init_gpu_context(device_id: u32) -> Result<(CudaContext, Module)> {
    let device = Device::get_device(device_id)
        .with_context(|| format!("Failed to get device {}", device_id))?;
    println!("Device name: {}", device.name()?);
    let ctx = CudaContext::new(device)
        .with_context(|| format!("Failed to create CUDA context for device {}", device_id))?;
    ctx.set_flags(cust::context::ContextFlags::SCHED_AUTO)
        .with_context(|| "Failed to set context flags")?;

    let ptx = include_str!("gpu_kernel.ptx");
    let module = Module::from_ptx(ptx, &[])
        .context("Failed to load PTX module")?;

    let wordlist = get_bip39_wordlist();
    let gpu_words = convert_wordlist_to_gpu_format(&wordlist);
    let mut wordlist_array: [GpuWord; 2048] = [GpuWord { bytes: [0; 10], len: 0 }; 2048];
    for (i, word) in gpu_words.into_iter().enumerate() {
        if i >= 2048 { break; }
        wordlist_array[i] = word;
    }
    let symbol_name = std::ffi::CString::new("wordlist").unwrap();
    let mut symbol = module.get_global::<[GpuWord; 2048]>(&symbol_name)
        .with_context(|| "Failed to get symbol 'wordlist' from module")?;
    symbol.copy_to(&mut wordlist_array)
        .with_context(|| "Failed to copy wordlist to device")?;

    Ok((ctx, module))
}

fn get_bip39_wordlist() -> Vec<String> {
    let content = include_str!("../words.txt");
    content.lines().map(|l| l.trim().to_string()).collect()
}

fn convert_wordlist_to_gpu_format(wordlist: &[String]) -> Vec<GpuWord> {
    wordlist.iter().map(|word| {
        let mut gpu_word = GpuWord { bytes: [0; 10], len: word.len() as u8 };
        let bytes = word.as_bytes();
        let len = bytes.len().min(9);
        gpu_word.bytes[..len].copy_from_slice(&bytes[..len]);
        gpu_word
    }).collect()
}

pub struct GpuWorker {
    module: Module,
    stream: Stream,
    known_indices: Vec<u16>,
    known_count: i32,
    target_address: Vec<u8>,
    match_mode: i32,
    match_prefix_len: i32,
    worker_id: u32,
    total_workers: u32,
    resume_from: u64,
}

impl GpuWorker {
    pub fn new(
        ctx: &CudaContext,
        module: Module,
        wordlist: Arc<Vec<String>>,
        known_words: Arc<Vec<String>>,
        address: &str,
        match_mode: i32,
        match_prefix_len: i32,
        worker_id: u32,
        total_workers: u32,
        resume_from: u64,
    ) -> Result<Self> {
        if match_mode < 0 || match_mode > 2 {
            anyhow::bail!("Invalid match_mode: must be 0, 1, or 2");
        }
        let match_prefix_len = match_prefix_len.clamp(0, 20);

        let offset_file = format!("worker{}.offset", worker_id);
        let resume_offset = if Path::new(&offset_file).exists() {
            fs::read_to_string(&offset_file)?
                .trim()
                .parse::<u64>()
                .unwrap_or(resume_from)
        } else {
            resume_from
        };

        let known_indices: Vec<u16> = known_words.iter().map(|w| {
            wordlist.iter().position(|x| x == w)
                .ok_or_else(|| anyhow::anyhow!("Unknown seed word '{}'", w))
                .map(|idx| idx as u16)
        }).collect::<Result<Vec<u16>>>()?;

        if known_indices.len() > 12 {
            anyhow::bail!("Too many known words: maximum is 12");
        }

        let raw = address.trim();
        let addr_str = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")).unwrap_or(raw);
        let mut target_address = Vec::with_capacity(20);
        match addr_str.len() {
            0 => {
                target_address.resize(20, 0);
                println!("Address mode: zero address (wildcard)");
            }
            1..=40 => {
                if addr_str.len() % 2 != 0 {
                    anyhow::bail!("Address must have even number of hex digits, got {}", addr_str.len());
                }
                if !addr_str.chars().all(|c| c.is_ascii_hexdigit()) {
                    anyhow::bail!("Address contains non-hex characters: '{}'", addr_str);
                }
                for i in (0..addr_str.len()).step_by(2) {
                    let byte = u8::from_str_radix(&addr_str[i..i+2], 16)
                        .with_context(|| format!("Invalid hex at byte {}", i/2))?;
                    target_address.push(byte);
                }
                if target_address.len() < 20 {
                    println!("Address mode: partial prefix ({} bytes)", target_address.len());
                    target_address.resize(20, 0);
                } else {
                    println!("Address mode: full address (20 bytes)");
                }
            }
            _ => anyhow::bail!("Address too long: expected up to 40 hex digits, got {}", addr_str.len()),
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            known_indices: known_indices.clone(),
            known_count: known_indices.len() as i32,
            target_address,
            match_mode,
            match_prefix_len,
            worker_id,
            total_workers,
            resume_from: resume_offset,
        })
    }

    pub fn run_loop(&mut self, found_flag: &AtomicBool) -> Result<Option<String>> {
        let unknown_count = 12 - self.known_count as u32;
        let total_candidates = 2048u64.pow(unknown_count);
        let per_worker = total_candidates / self.total_workers as u64;
        let start_offset = self.worker_id as u64 * per_worker;
        let end_offset = start_offset + per_worker;

        const MAX_CANDIDATES_PER_LAUNCH: u64 = 1;
        const THREADS: u32 = 1;
        let mut current_offset = self.resume_from.max(start_offset);
        let start_time = Instant::now();
        let mut total_tested = 0u64;

        let mut  d_seeds_tested = DeviceBuffer::<u64>::zeroed(1)?;
        let  mut d_seeds_found = DeviceBuffer::<u64>::zeroed(1)?;
        let d_known = DeviceBuffer::from_slice(&self.known_indices)?;
        let d_address = DeviceBuffer::from_slice(&self.target_address)?;
        let d_found_mnemonic = DeviceBuffer::<u16>::zeroed(MAX_CANDIDATES_PER_LAUNCH as usize * 12)?;
        let wordlist = get_bip39_wordlist();

        let function = self.module.get_function("search_seeds")?;
        while current_offset < end_offset && !found_flag.load(Ordering::SeqCst) {
            let remaining = end_offset - current_offset;
            let this_launch = remaining.min(MAX_CANDIDATES_PER_LAUNCH);
            let blocks = ((this_launch + THREADS as u64 - 1) / THREADS as u64) as u32;

            println!("Worker {}: Launching kernel at offset {}, batch size {}", self.worker_id, current_offset, this_launch);
            let t0 = Instant::now();
            let stream = &self.stream;
            unsafe {
                let launch_result = launch!(function<<<blocks, THREADS, 0, stream>>>(
                    d_seeds_tested.as_device_ptr(),
                    d_seeds_found.as_device_ptr(),
                    current_offset,
                    this_launch,
                    2048,
                    self.known_count,
                    d_known.as_device_ptr(),
                    d_address.as_device_ptr(),
                    self.match_mode,
                    self.match_prefix_len,
                    d_found_mnemonic.as_device_ptr()
                ));
                if let Err(e) = launch_result {
                    eprintln!("Worker {}: Kernel launch failed: {:?}", self.worker_id, e);
                    return Err(e.into());
                }
            }
            if let Err(e) = self.stream.synchronize() {
                eprintln!("Worker {}: Stream synchronization failed: {:?}", self.worker_id, e);
                return Err(e.into());
            }
            let dt = t0.elapsed();

            let tested = d_seeds_tested.as_host_vec()?.get(0).copied().unwrap_or(0);
            let found = d_seeds_found.as_host_vec()?.get(0).copied().unwrap_or(0);
            total_tested += tested;
            let rate = if dt.as_secs_f64() > 0.0 { tested as f64 / dt.as_secs_f64() } else { 0.0 };
            println!(
                "Worker {}: [kernel time: {:.1} ms] tested {}/{}, {:.1} seeds/sec",
                self.worker_id, dt.as_secs_f64() * 1e3, tested, this_launch, rate
            );

            if found > 0 {
                let mnemonic_indices = d_found_mnemonic.as_host_vec()?;
                let mut mnemonic_words = Vec::with_capacity(12);
                for i in 0..12 {
                    let idx = mnemonic_indices[i] as usize;
                    if idx >= wordlist.len() {
                        anyhow::bail!("Invalid word index: {}", idx);
                    }
                    mnemonic_words.push(wordlist[idx].clone());
                }
                let mnemonic = mnemonic_words.join(" ");
                println!("Worker {} found mnemonic: {}", self.worker_id, &mnemonic);
                found_flag.store(true, Ordering::SeqCst);
                return Ok(Some(mnemonic));
            }

            current_offset += this_launch;
            if let Err(e) = fs::write(format!("worker{}.offset", self.worker_id), current_offset.to_string()) {
                eprintln!("Worker {}: Failed to write offset: {:?}", self.worker_id, e);
            }
            unsafe {
            d_seeds_tested.async_copy_from(&[0u64],stream)?;
            d_seeds_found.async_copy_from(&[0u64], stream)?;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let total_rate = if start_time.elapsed().as_secs_f64() > 0.0 {
            total_tested as f64 / start_time.elapsed().as_secs_f64()
        } else { 0.0 };
        println!("Worker {} completed, total tested: {}, rate: {:.1} seeds/sec", self.worker_id, total_tested, total_rate);
        Ok(None)
    }
}