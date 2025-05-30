# Intel COMGR Sys

This crate provides Rust bindings for Intel's Compiler for Graphics (COMGR) that follow AMD's COMGR API structure. It's designed to be a drop-in replacement for `amd_comgr-sys` but using Intel's compiler technology underneath.

## Features

- Provides a similar API structure to AMD's COMGR library
- Uses Intel's `icpx` compiler under the hood
- Supports various compilation actions like:
  - Source to preprocessed output
  - Source to LLVM bitcode
  - LLVM bitcode to assembly/relocatable
  - Linking and optimization
  - Generating fat binaries

## Requirements

- Intel oneAPI Base Toolkit with the Intel DPC++/C++ Compiler installed
- `icpx` command available in your PATH

## Example Usage

```rust
use intel_comgr_sys::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data objects
    let mut source_data = intel_comgr_data_t { handle: 0 };
    intel_comgr_create_data(
        intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE,
        &mut source_data
    )?;
    
    // Set source code
    let source_code = "kernel void add(global int* a, global int* b) { ... }";
    intel_comgr_data_set_bytes(
        source_data,
        source_code.as_ptr() as *const _,
        source_code.len()
    )?;
    
    // Create data sets
    let mut input_set = intel_comgr_data_set_t { handle: 0 };
    let mut output_set = intel_comgr_data_set_t { handle: 0 };
    intel_comgr_create_data_set(&mut input_set)?;
    intel_comgr_create_data_set(&mut output_set)?;
    
    // Add source to input set
    intel_comgr_data_set_add(input_set, source_data)?;
    
    // Create and configure action info
    let mut action_info = intel_comgr_action_info_t { handle: 0 };
    intel_comgr_create_action_info(&mut action_info)?;
    intel_comgr_action_info_set_language(
        action_info,
        intel_comgr_language_s::INTEL_COMGR_LANGUAGE_OPENCL_2_0
    )?;
    
    // Perform compilation
    intel_comgr_do_action(
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
        action_info,
        input_set,
        output_set
    )?;
    
    // Clean up
    intel_comgr_release_data_set(output_set)?;
    intel_comgr_release_data_set(input_set)?;
    intel_comgr_release_data(source_data)?;
    intel_comgr_release_action_info(action_info)?;
    
    Ok(())
}
```

## Implementation Details

This library wraps Intel's compiler technology through command-line interfaces like `icpx` to provide functionality similar to AMD's COMGR API. It's designed to be used with the ZLUDA project for CUDA compatibility on Intel GPUs.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 