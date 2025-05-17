use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::process;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;

mod gpu_wrapper;
use gpu_wrapper::GpuWorker;

#[derive(Parser, Debug)]
#[command(name = "seeds", about = "Ethereum GPU brute force recovery tool")]
struct Args {
    #[clap(long, default_value = "")]
    known: String,

    #[clap(long, default_value = "")]
    address: String,

    #[clap(long, default_value = "1")]
    workers: u32,

    #[clap(long, default_value = "0")]
    match_mode: i32,

    #[clap(long, default_value = "0")]
    match_prefix_len: i32,

    #[clap(long, default_value = "words.txt")]
    wordlist: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load the BIP39 wordlist from file
    let wordlist_content = fs::read_to_string(&args.wordlist)
        .with_context(|| format!("Failed to read wordlist file '{}'", &args.wordlist))?;
    let wordlist = Arc::new(
        wordlist_content
            .lines()
            .map(str::trim)
            .map(String::from)
            .collect::<Vec<_>>()
    );

    // Parse the known seed words
    let known_words = Arc::new(
        if args.known.is_empty() {
            Vec::new()
        } else {
            args.known
                .split_whitespace()
                .map(str::trim)
                .map(String::from)
                .collect::<Vec<_>>()
        }
    );

    let found_flag = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Spawn a thread per worker
    for worker_id in 0..args.workers {
        let wordlist = Arc::clone(&wordlist);
        let known_words = Arc::clone(&known_words);
        let address = args.address.clone();
        let match_mode = args.match_mode;
        let match_prefix_len = args.match_prefix_len;
        let total_workers = args.workers;
        let found_flag = Arc::clone(&found_flag);

        let handle = thread::spawn(move || {
            let thread_res: Result<()> = (|| {
                // Early exit if another worker already found the seed
                if found_flag.load(Ordering::SeqCst) {
                    return Ok(());
                }

                // Load resume offset from workerN.offset
                let offset_path = format!("worker{}.offset", worker_id);
                let resume_from = fs::read_to_string(&offset_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0);

                // Initialize GPU
                let (ctx, module) = gpu_wrapper::init_gpu_context(0)
                    .with_context(|| format!("init_gpu_context(0) failed"))?;

                // Create worker
                let mut worker = GpuWorker::new(
                    &ctx,
                    &module,
                    wordlist,
                    known_words,
                    &address,
                    match_mode,
                    match_prefix_len,
                    worker_id,
                    total_workers,
                    resume_from,
                )?;

                // Run the brute-force loop
                if let Some(mnemonic) = worker.run_loop(&found_flag)? {
                    println!("✅ Worker {} found a valid mnemonic!", worker_id);
                    fs::write("found_seeds.txt", &mnemonic)
                        .with_context(|| "Failed to write found_seeds.txt")?;
                    found_flag.store(true, Ordering::SeqCst);
                }

                Ok(())
            })();

            if let Err(e) = thread_res {
                eprintln!("Worker {} error: {:?}", worker_id, e);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Exit depending on whether a mnemonic was found
    if found_flag.load(Ordering::SeqCst) {
        println!("✅ Mnemonic recovered and written to found_seeds.txt");
        process::exit(0);
    } else {
        println!("❌ Search completed; no matching mnemonic found.");
        process::exit(1);
    }
}
