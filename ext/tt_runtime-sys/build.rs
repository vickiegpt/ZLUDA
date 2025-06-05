use std::env::VarError;
use std::path::Path;

fn main() -> Result<(), VarError> {
    cc::Build::new()
        .file("src/tt_metal_wrapper.cpp")
        .cpp(true)
        .flag("-std=c++20")
        .compile("tt_metal_wrapper");
        
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib_path = Path::new(&out_dir).join("libtt_metal_wrapper.a");
    
    // If the library doesn't exist, warn but don't fail
    if !lib_path.exists() {
        println!("cargo:warning=libtt_metal_wrapper.a not found in OUT_DIR: {}", out_dir);
    } else {
        println!("cargo:warning=Found libtt_metal_wrapper.a in OUT_DIR: {}", out_dir);
    }
    
    // Add the lib directory to the library search path
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-search=native={}", out_dir);
    
    // Link to the TT Metal libraries
    println!("cargo:rustc-link-lib=static=tt_metal_wrapper");  // Our wrapper
    println!("cargo:rustc-link-lib=dylib=tt_metal");  // TT Metal library
    println!("cargo:rustc-link-lib=dylib=device");    // Device library
    println!("cargo:rustc-link-lib=dylib=stdc++");
    Ok(())
}

