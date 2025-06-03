// Tenstorrent backend example for ZLUDA
// Enable with: cargo run --example tenstorrent_example --features tenstorrent

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ZLUDA with Tenstorrent backend example");

    // Initialize ZLUDA CUDA driver
    let result = unsafe { zluda::cuInit(0) };
    println!("CUDA initialization result: {:?}", result);

    // Get number of devices
    let mut device_count = 0;
    unsafe { zluda::cuDeviceGetCount(&mut device_count) };
    println!("Number of devices: {}", device_count);

    if device_count == 0 {
        println!("No Tenstorrent devices found");
        return Ok(());
    }

    // Get device
    let mut device = 0;
    let result = unsafe { zluda::cuDeviceGet(&mut device, 0) };
    println!("Get device result: {:?}", result);

    // Get device name
    let mut name = vec![0u8; 100];
    let result =
        unsafe { zluda::cuDeviceGetName(name.as_mut_ptr() as *mut i8, name.len() as i32, device) };
    println!("Get device name result: {:?}", result);

    if result == zluda::CUresult::CUDA_SUCCESS {
        let device_name = name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as char)
            .collect::<String>();
        println!("Device name: {}", device_name);
    }

    // Create context
    let mut context = std::ptr::null_mut();
    let result = unsafe { zluda::cuCtxCreate_v2(&mut context, 0, device) };
    println!("Create context result: {:?}", result);

    // Clean up
    if !context.is_null() {
        unsafe { zluda::cuCtxDestroy_v2(context) };
    }

    println!("Tenstorrent example completed successfully");
    Ok(())
}
