use std::env::VarError;
use std::{env, path::PathBuf};

fn main() -> Result<(), VarError> {
    // Build the ze_runner C code
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src");

    cc::Build::new()
        .file(src_dir.join("runner/ze_runner.c"))
        .include(src_dir.clone()) // Include src directory for level-zero headers
        .compile("ze_runner"); // Compile to libze_runner.a

    // Link against Level Zero loader library
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=ze_loader_1");
        let env = env::var("CARGO_CFG_TARGET_ENV")?;
        if env == "msvc" {
            let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
            path.push("lib");
            println!("cargo:rustc-link-search=native={}", path.display());
        } else {
            println!("cargo:rustc-link-search=native=C:\\Windows\\System32");
        };
    } else {
        println!("cargo:rustc-link-lib=dylib=ze_loader");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");
    }

    // Tell cargo to re-run if any of these files change
    println!("cargo:rerun-if-changed=src/runner/ze_runner.c");
    println!("cargo:rerun-if-changed=src/runner/ze_runner.h");
    println!("cargo:rerun-if-changed=src/level-zero/ze_api.h");

    Ok(())
}
