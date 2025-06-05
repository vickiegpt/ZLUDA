use std::env::VarError;

fn main() -> Result<(), VarError> {
    // Build the simple C wrapper for missing symbols
    cc::Build::new()
        .file("src/tt_metal_wrapper.cpp")
        .flag("-fPIC")
        .compile("tt_metal_wrapper");
    
    // Add the lib directory to the library search path
    println!("cargo:rustc-link-search=native=lib");
    
    // Link to our wrapper only (stub implementation for development)
    // println!("cargo:rustc-link-lib=dylib=tt_metal");  // Comment out real library
    println!("cargo:rustc-link-lib=static=tt_metal_wrapper");  // Use static library created by cc::Build
    println!("cargo:rustc-link-lib=dylib=stdc++");
    
    Ok(())
}

