use cust::prelude::*;
use cust::context::Context as CudaContext;
use anyhow::{Result, Context};
use std::env;
use std::process::Command;

fn check_cuda_libraries() -> Result<()> {
    println!("\nChecking CUDA libraries...");
    
    // First try ldconfig
    let output = Command::new("ldconfig")
        .arg("-p")
        .output()
        .context("Failed to run ldconfig")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let cuda_libs = [
        "libcudart.so",
        "libcublas.so",
        "libcublasLt.so",
        "libcufft.so",
        "libcurand.so",
        "libcusolver.so",
        "libcusparse.so",
        "libnvidia-ml.so"
    ];

    let mut all_found = true;
    for lib in cuda_libs.iter() {
        if stdout.contains(lib) {
            println!("Found {}", lib);
        } else {
            println!("{} not found", lib);
            all_found = false;
        }
    }

    if !all_found {
        // If some libraries are missing, try direct path
        let cuda_lib_path = "/usr/local/cuda-11.8/lib64";
        if let Ok(output) = Command::new("ls")
            .arg(cuda_lib_path)
            .output() 
        {
            if output.status.success() {
                let libs = String::from_utf8_lossy(&output.stdout);
                println!("\nChecking direct path {}:", cuda_lib_path);
                for lib in cuda_libs.iter() {
                    if libs.contains(lib) {
                        println!("Found {} in {}", lib, cuda_lib_path);
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn test_cuda_init() -> Result<()> {
    println!("Testing CUDA initialization...");
    
    // Print environment information
    println!("\nEnvironment Information:");
    println!("CUDA_PATH: {}", env::var("CUDA_PATH").unwrap_or_else(|_| "Not set".to_string()));
    println!("LD_LIBRARY_PATH: {}", env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "Not set".to_string()));
    
    // Check CUDA libraries
    if let Err(e) = check_cuda_libraries() {
        println!("Warning: Failed to check CUDA libraries: {}", e);
    }
    
    println!("\nAttempting CUDA device initialization...");
    
    // Try to get device count
    let device_count = Device::num_devices()
        .context("Failed to get number of CUDA devices")?;
    println!("Found {} CUDA devices", device_count);
    
    if device_count == 0 {
        anyhow::bail!("No CUDA devices found");
    }
    
    // Try to get the first device
    let device = Device::get_device(0)
        .context("Failed to get device 0")?;
    println!("Successfully got device 0");
    println!("Device name: {}", device.name()?);

    // Try to create a context
    let _context = CudaContext::new(device)
        .context("Failed to create CUDA context")?;
    println!("Successfully created CUDA context");

    Ok(())
} 