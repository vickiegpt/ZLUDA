use std::env::VarError;

fn main() -> Result<(), VarError> {
    // Build the simple C wrapper for missing symbols
    cc::Build::new()
        .file("lib/libtt_metal_c_wrapper.c")
        .flag("-fPIC")
        .compile("tt_metal_c_wrapper");
    
    // Link to the TT Metal library and our wrapper
    println!("cargo:rustc-link-lib=dylib=tt_metal");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-search=native=lib");
    
    Ok(())
}

