use std::env::VarError;

fn main() -> Result<(), VarError> {
    println!("cargo:rustc-link-lib=dylib=tt_metal");
    println!("cargo:rustc-link-search=native=/opt/local/lib/");
    Ok(())
}

