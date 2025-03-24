# oneAPI Level Zero Runtime Bindings for Rust

This crate provides Rust bindings to the oneAPI Level Zero runtime API, primarily for use with the ZLUDA project for CUDA compatibility on Intel GPUs.

## Features

- Complete bindings to the Level Zero API
- Dynamic library loading (no need to link against libze_loader at compile time)
- Error handling using Rust's Result type
- Thread-safe initialization

## Requirements

- oneAPI Level Zero runtime installed on the system
- `libze_loader.so` accessible in standard library paths

## Usage

```rust
use ze_runtime_sys::{ze_driver_handle_t, ze_device_handle_t, ze_init_flag_t};
use ze_runtime_sys::ze_result_t;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Level Zero driver
    let result = unsafe { ze_runtime_sys::zeInit(ze_init_flag_t::ZE_INIT_FLAG_GPU_ONLY) };
    if let Err(err) = result {
        println!("Failed to initialize Level Zero: {:?}", err);
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Level Zero init failed")));
    }
    
    // Get driver count
    let mut driver_count = 0;
    unsafe { ze_runtime_sys::zeDriverGet(&mut driver_count, std::ptr::null_mut())? };
    
    println!("Found {} Level Zero driver(s)", driver_count);
    
    // Get driver handles
    let mut drivers = vec![std::ptr::null_mut(); driver_count as usize];
    unsafe { ze_runtime_sys::zeDriverGet(&mut driver_count, drivers.as_mut_ptr() as *mut ze_driver_handle_t)? };
    
    // For each driver, get the devices
    for (i, &driver) in drivers.iter().enumerate() {
        let mut device_count = 0;
        unsafe { ze_runtime_sys::zeDeviceGet(driver, &mut device_count, std::ptr::null_mut())? };
        
        println!("Driver {} has {} device(s)", i, device_count);
        
        let mut devices = vec![std::ptr::null_mut(); device_count as usize];
        unsafe { ze_runtime_sys::zeDeviceGet(driver, &mut device_count, devices.as_mut_ptr() as *mut ze_device_handle_t)? };
        
        // Get device properties for each device
        for (j, &device) in devices.iter().enumerate() {
            let mut device_properties = Default::default();
            unsafe { ze_runtime_sys::zeDeviceGetProperties(device, &mut device_properties)? };
            
            println!("Device {}: {}", j, std::ffi::CStr::from_ptr(device_properties.name.as_ptr()).to_string_lossy());
        }
    }
    
    Ok(())
}
```

## Implementation Details

This crate uses dynamic loading of the Level Zero API functions through the `libloading` crate. 
It automatically searches for the Level Zero loader library in standard system paths and loads the 
required symbols on demand.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 