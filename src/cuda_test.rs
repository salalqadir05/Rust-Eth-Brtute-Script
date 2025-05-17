use cust::prelude::*;
use anyhow::Result;

pub fn test_cuda_init() -> Result<()> {
    println!("Testing CUDA initialization...");
    
    // Try to get device count
    match Device::num_devices() {
        Ok(count) => {
            println!("Found {} CUDA devices", count);
            
            // Try to get the first device
            if count > 0 {
                match Device::get_device(0) {
                    Ok(device) => {
                        println!("Successfully got device 0");
                        println!("Device name: {}", device.name()?);
                        println!("Compute capability: {}.{}", 
                            device.compute_capability()?.major,
                            device.compute_capability()?.minor);
                        println!("Total memory: {} MB", device.total_memory()? / 1024 / 1024);
                    },
                    Err(e) => println!("Failed to get device 0: {}", e),
                }
            }
        },
        Err(e) => println!("Failed to get device count: {}", e),
    }

    // Try to create a context
    match Device::get_device(0) {
        Ok(device) => {
            match CudaContext::new(device) {
                Ok(_) => println!("Successfully created CUDA context"),
                Err(e) => println!("Failed to create CUDA context: {}", e),
            }
        },
        Err(e) => println!("Failed to get device for context creation: {}", e),
    }

    Ok(())
} 