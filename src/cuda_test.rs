use cust::prelude::*;
use cust::context::Context as CudaContext;
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
                        // Just print basic device info that we know exists
                        println!("Device name: {}", device.name()?);
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