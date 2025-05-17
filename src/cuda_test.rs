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

    // Get CUDA toolkit version from environment
    let cuda_version = env::var("CUDA_TOOLKIT_VERSION")
        .unwrap_or_else(|_| "11.8".to_string());
    println!("Expected CUDA version: {}", cuda_version);

    let mut all_found = true;
    for lib in cuda_libs.iter() {
        if stdout.contains(lib) {
            println!("Found {}", lib);
            // Try to get all versions of this library
            if let Ok(output) = Command::new("ldconfig")
                .arg("-p")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let mut found_versions = Vec::new();
                for line in stdout.lines() {
                    if line.contains(lib) {
                        if let Some(path) = line.split("=>").nth(1) {
                            let path = path.trim();
                            println!("  Path: {}", path);
                            // Try to extract version from path
                            if let Some(version) = path.split("so.").nth(1) {
                                found_versions.push(version.to_string());
                            }
                        }
                    }
                }
                if !found_versions.is_empty() {
                    println!("  Available versions: {}", found_versions.join(", "));
                }
            }
        } else {
            println!("{} not found", lib);
            all_found = false;
        }
    }

    // Check specific CUDA paths
    let cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-11.8/lib64",
        "/usr/lib/cuda/lib64",
        "/usr/lib/nvidia-cuda-toolkit/lib64"
    ];

    println!("\nChecking specific CUDA paths:");
    for path in cuda_paths.iter() {
        if let Ok(output) = Command::new("ls")
            .arg("-l")
            .arg(path)
            .output() 
        {
            if output.status.success() {
                println!("\nContents of {}:", path);
                println!("{}", String::from_utf8_lossy(&output.stdout));
            }
        }
    }

    // Check if CUDA libraries are in the correct order in LD_LIBRARY_PATH
    if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
        println!("\nLD_LIBRARY_PATH order:");
        for (i, path) in ld_path.split(':').enumerate() {
            println!("{}. {}", i + 1, path);
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
            
            // Try to get NVIDIA driver version
            if let Ok(output) = Command::new("nvidia-smi")
                .arg("--query-gpu=driver_version")
                .arg("--format=csv,noheader")
                .output()
            {
                if output.status.success() {
                    println!("NVIDIA Driver Version:");
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                }
            }

            // Try to get CUDA runtime error
            if let Ok(output) = Command::new("cuda-memcheck")
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    println!("CUDA Memory Checker Version:");
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                }
            }

            // Try to get CUDA device properties using nvidia-smi
            if let Ok(output) = Command::new("nvidia-smi")
                .arg("--query-gpu=name,memory.total,driver_version,cuda_version")
                .arg("--format=csv,noheader")
                .output()
            {
                if output.status.success() {
                    println!("\nGPU Information:");
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                }
            }

            // Try to get CUDA runtime error using nvprof
            if let Ok(output) = Command::new("nvprof")
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    println!("\nCUDA Profiler Version:");
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

    // Create context and set flags
    let context = match CudaContext::new(device) {
        Ok(ctx) => {
            println!("Successfully created CUDA context");
            ctx
        },
        Err(e) => {
            println!("Error creating CUDA context: {}", e);
            return Err(e).context("Failed to create CUDA context");
        }
    };
    
    match context.set_flags(cust::context::ContextFlags::SCHED_AUTO) {
        Ok(_) => println!("Successfully set context flags"),
        Err(e) => {
            println!("Warning: Failed to set context flags: {}", e);
            // Continue anyway as this is not critical
        }
    }

    Ok(())
} 