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
            // Try to get the actual path
            if let Ok(output) = Command::new("ldconfig")
                .arg("-p")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().find(|l| l.contains(lib)) {
                    if let Some(path) = line.split("=>").nth(1) {
                        println!("  Path: {}", path.trim());
                    }
                }
            }
        } else {
            println!("{} not found", lib);
            all_found = false;
        }
    }

    if !all_found {
        // If some libraries are missing, try direct path
        let cuda_lib_path = env::var("CUDA_LIBRARY_PATH")
            .unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());
        
        if let Ok(output) = Command::new("ls")
            .arg(&cuda_lib_path)
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
    println!("CUDA_HOME: {}", env::var("CUDA_HOME").unwrap_or_else(|_| "Not set".to_string()));
    println!("CUDA_TOOLKIT_ROOT_DIR: {}", env::var("CUDA_TOOLKIT_ROOT_DIR").unwrap_or_else(|_| "Not set".to_string()));
    println!("CUDA_TOOLKIT_VERSION: {}", env::var("CUDA_TOOLKIT_VERSION").unwrap_or_else(|_| "Not set".to_string()));
    println!("LD_LIBRARY_PATH: {}", env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "Not set".to_string()));
    
    // Check CUDA libraries
    if let Err(e) = check_cuda_libraries() {
        println!("Warning: Failed to check CUDA libraries: {}", e);
    }
    
    println!("\nAttempting CUDA device initialization...");
    
    // Try to get device count with error context
    let device_count = match Device::num_devices() {
        Ok(count) => {
            println!("Found {} CUDA devices", count);
            count
        },
        Err(e) => {
            println!("Error getting device count: {}", e);
            println!("Checking CUDA runtime version...");
            
            // Try to get CUDA runtime version
            if let Ok(output) = Command::new("nvcc")
                .arg("--version")
                .output() 
            {
                if output.status.success() {
                    println!("CUDA Compiler Version:");
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                }
            }
            
            return Err(e).context("Failed to get number of CUDA devices");
        }
    };
    
    if device_count == 0 {
        anyhow::bail!("No CUDA devices found");
    }
    
    // Try to get the first device
    let device = match Device::get_device(0) {
        Ok(d) => {
            println!("Successfully got device 0");
            d
        },
        Err(e) => {
            println!("Error getting device 0: {}", e);
            return Err(e).context("Failed to get device 0");
        }
    };

    // Print device information
    match device.name() {
        Ok(name) => println!("Device name: {}", name),
        Err(e) => println!("Warning: Could not get device name: {}", e),
    }

    // Try to create a context
    match CudaContext::new(device) {
        Ok(_) => println!("Successfully created CUDA context"),
        Err(e) => {
            println!("Error creating CUDA context: {}", e);
            return Err(e).context("Failed to create CUDA context");
        }
    }

    Ok(())
} 