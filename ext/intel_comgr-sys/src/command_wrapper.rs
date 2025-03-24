
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::collections::HashMap;
use tempfile::tempdir;

use crate::{
    intel_comgr_action_kind_s, intel_comgr_action_info_t, intel_comgr_data_kind_s,
    intel_comgr_data_set_t, intel_comgr_status_s, intel_comgr_status_t, intel_comgr_data_t,
    intel_comgr_language_t,
};

// Actual internal representation of data 
pub(crate) struct DataContent {
    pub(crate) kind: intel_comgr_data_kind_s,
    pub(crate) content: Vec<u8>,
    pub(crate) name: Option<String>,
}

// Store our internal data mapping keyed by handle
pub(crate) type DataMap = HashMap<u64, DataContent>;

// Global storage for data
lazy_static::lazy_static! {
    pub(crate) static ref DATA_STORE: std::sync::Mutex<DataMap> = std::sync::Mutex::new(HashMap::new());
    pub(crate) static ref DATA_SET_STORE: std::sync::Mutex<HashMap<u64, Vec<u64>>> = std::sync::Mutex::new(HashMap::new());
    pub(crate) static ref ACTION_INFO_STORE: std::sync::Mutex<HashMap<u64, ActionInfo>> = std::sync::Mutex::new(HashMap::new());
    pub(crate) static ref NEXT_HANDLE: std::sync::Mutex<u64> = std::sync::Mutex::new(1);
}

pub(crate) fn get_next_handle() -> u64 {
    let mut handle = NEXT_HANDLE.lock().unwrap();
    let current = *handle;
    *handle += 1;
    current
}

#[derive(Default, Clone)]
pub(crate) struct ActionInfo {
    pub(crate) language: Option<intel_comgr_language_t>,
    pub(crate) options: Vec<String>,
    pub(crate) working_directory: Option<String>,
    pub(crate) target: Option<String>,
}

struct ActionContext {
    temp_dir: PathBuf,
    options: Vec<String>,
    language: intel_comgr_language_t,
    input_files: Vec<(PathBuf, intel_comgr_data_kind_s)>,
    target: Option<String>,
}

pub fn perform_action(
    action_kind: intel_comgr_action_kind_s,
    action_info: intel_comgr_action_info_t,
    input_set: intel_comgr_data_set_t,
    output_set: intel_comgr_data_set_t,
) -> intel_comgr_status_t {
    // Create a temporary directory for the operation
    let dir = match tempdir() {
        Ok(d) => d,
        Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
    };
    
    // Extract info from action_info
    let action_info_lock = ACTION_INFO_STORE.lock().unwrap();
    let action_data = match action_info_lock.get(&action_info.handle) {
        Some(data) => data.clone(),
        None => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(action_info_lock);
    
    // Extract data set contents
    let data_set_lock = DATA_SET_STORE.lock().unwrap();
    let data_handles = match data_set_lock.get(&input_set.handle) {
        Some(handles) => handles.clone(),
        None => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    drop(data_set_lock);
    
    let data_store_lock = DATA_STORE.lock().unwrap();
    let mut input_files = Vec::new();
    
    // Write input files to temp directory
    for handle in &data_handles {
        if let Some(data) = data_store_lock.get(handle) {
            let file_name = match &data.name {
                Some(name) => name.clone(),
                None => {
                    // Generate name based on kind
                    match data.kind.0 {
                        1 => "input.cpp".to_string(),
                        2 => format!("include_{}.h", handle),
                        3 => "header.pch".to_string(),
                        6 => format!("input_{}.bc", handle),
                        7 => format!("input_{}.o", handle),
                        _ => format!("data_{}", handle),
                    }
                }
            };
            
            let file_path = dir.path().join(&file_name);
            if let Err(_) = fs::write(&file_path, &data.content) {
                return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
            }
            
            input_files.push((file_path, data.kind));
        }
    }
    drop(data_store_lock);
    
    let language = action_data.language.unwrap_or_default();
    
    // Build the context
    let context = ActionContext {
        temp_dir: dir.path().to_path_buf(),
        options: action_data.options,
        language,
        input_files,
        target: action_data.target,
    };
    
    // Execute the specific action
    let result = match action_kind.0 {
        0 => preprocess_source(&context),
        1 => add_precompiled_headers(&context),
        2 => compile_source_to_bc(&context),
        3 => add_device_libraries(&context),
        4 => link_bc_to_bc(&context),
        5 => optimize_bc(&context),
        6 => codegen_to_relocatable(&context),
        7 => codegen_to_assembly(&context),
        8 => compile_to_fatbin(&context),
        _ => Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT),
    };
    
    if result.is_err() {
        return result;
    }
    
    // Collect output files and add to output_set
    add_outputs_to_set(&context, output_set)?;
    
    Ok(())
}

// Helper function to add output files to the output set
fn add_outputs_to_set(ctx: &ActionContext, output_set: intel_comgr_data_set_t) -> intel_comgr_status_t {
    let output_dir = ctx.temp_dir.clone();
    
    // Get all files in the output directory
    let entries = match fs::read_dir(&output_dir) {
        Ok(entries) => entries,
        Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
    };
    
    // Look for output files based on common patterns
    for entry in entries {
        if let Ok(entry) = entry {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            
            let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
            
            // Determine the kind based on the extension
            let kind = match extension {
                "i" | "ii" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE,
                "h" | "hpp" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_INCLUDE,
                "pch" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                "bc" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC,
                "o" | "obj" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_RELOCATABLE,
                "s" | "asm" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE,
                "exe" | "out" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_EXECUTABLE,
                "fatbin" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_FATBIN,
                "log" | "txt" => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_LOG,
                _ => intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BYTES,
            };
            
            // Skip input files that were previously created
            let mut is_input = false;
            for (input_path, _) in &ctx.input_files {
                if input_path == &path {
                    is_input = true;
                    break;
                }
            }
            if is_input {
                continue;
            }
            
            // Read the file content
            let content = match fs::read(&path) {
                Ok(content) => content,
                Err(_) => continue,
            };
            
            // Add the data to our store
            let handle = get_next_handle();
            let name = path.file_name().and_then(|n| n.to_str()).map(|s| s.to_string());
            
            let data_content = DataContent {
                kind,
                content,
                name,
            };
            
            {
                let mut data_store = DATA_STORE.lock().unwrap();
                data_store.insert(handle, data_content);
                
                let mut data_set_store = DATA_SET_STORE.lock().unwrap();
                let set_handles = data_set_store.entry(output_set.handle).or_insert_with(Vec::new);
                set_handles.push(handle);
            }
        }
    }
    
    Ok(())
}

// Action implementations using icpx command

fn preprocess_source(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find source files
    let source_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if source_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each source file
    for input_file in source_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.i", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-E", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Add include paths
        for (include_path, kind) in &ctx.input_files {
            if kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_INCLUDE.0 {
                let include_dir = include_path.parent().unwrap_or(Path::new(""));
                command.arg("-I").arg(include_dir);
            }
        }
        
        // Add custom options
        command.args(&ctx.options);
        
        // Add language options based on the language
        match ctx.language.0 {
            1 => { command.args(["-cl-std=CL1.2"]); }
            2 => { command.args(["-cl-std=CL2.0"]); }
            3 => { command.args(["-fsycl"]); }
            _ => {}
        }
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn add_precompiled_headers(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find header files
    let header_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_INCLUDE.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if header_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each header file to create PCH
    for header_file in header_files {
        let file_stem = header_file.file_stem().and_then(|s| s.to_str()).unwrap_or("header");
        let pch_file = ctx.temp_dir.join(format!("{}.pch", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-x", "c++-header", header_file.to_str().unwrap(), "-o", pch_file.to_str().unwrap()]);
        
        // Add custom options
        command.args(&ctx.options);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn compile_source_to_bc(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find source files
    let source_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if source_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Find PCH files
    let pch_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_PRECOMPILED_HEADER.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    // Process each source file
    for input_file in source_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.bc", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-c", "-emit-llvm", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Add include paths
        for (include_path, kind) in &ctx.input_files {
            if kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_INCLUDE.0 {
                let include_dir = include_path.parent().unwrap_or(Path::new(""));
                command.arg("-I").arg(include_dir);
            }
        }
        
        // Add PCH if available
        if !pch_files.is_empty() {
            command.arg("-include-pch").arg(&pch_files[0]);
        }
        
        // Add language options based on the language
        match ctx.language.0 {
            1 => { command.args(["-cl-std=CL1.2"]); }
            2 => { command.args(["-cl-std=CL2.0"]); }
            3 => { command.args(["-fsycl"]); }
            _ => {}
        }
        
        // Add custom options
        command.args(&ctx.options);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn add_device_libraries(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find bitcode files
    let bc_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if bc_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Find library files (assuming they're in BC format)
    let lib_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| 
            kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0 || 
            kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_RELOCATABLE.0
        )
        .filter(|(path, _)| {
            // This is a simple heuristic - libraries might have "lib" prefix
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            file_name.starts_with("lib")
        })
        .map(|(path, _)| path.clone())
        .collect();
    
    // Process each bc file with libraries
    for input_file in bc_files {
        // Skip library files that we identified
        if lib_files.contains(&input_file) {
            continue;
        }
        
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_with_libs.bc", file_stem));
        
        let mut command = Command::new("llvm-link");
        command.arg(input_file.to_str().unwrap());
        
        // Add library files
        for lib_file in &lib_files {
            command.arg(lib_file.to_str().unwrap());
        }
        
        command.args(["-o", output_file.to_str().unwrap()]);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => {
                // If llvm-link is not available, try to use icpx with appropriate options
                let mut fallback = Command::new("icpx");
                fallback.args(["-c", "-emit-llvm", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
                
                for lib_file in &lib_files {
                    let lib_dir = lib_file.parent().unwrap_or(Path::new(""));
                    fallback.arg("-L").arg(lib_dir);
                    
                    let lib_name = lib_file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    if lib_name.starts_with("lib") {
                        // Strip "lib" prefix and add as -l
                        let lib_name = &lib_name[3..];
                        fallback.arg("-l").arg(lib_name);
                    }
                }
                
                match fallback.status() {
                    Ok(status) => status,
                    Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
                }
            }
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn link_bc_to_bc(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find all bitcode files
    let bc_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if bc_files.len() < 2 {
        // Need at least 2 files to link
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    let output_file = ctx.temp_dir.join("linked.bc");
    
    // Use llvm-link from the icpx toolchain
    let mut command = Command::new("llvm-link");
    
    for bc_file in &bc_files {
        command.arg(bc_file.to_str().unwrap());
    }
    
    command.args(["-o", output_file.to_str().unwrap()]);
    
    // Try to use llvm-link, but if not available, attempt to use icpx
    let status = match command.status() {
        Ok(status) => status,
        Err(_) => {
            // Fallback to using icpx for linking
            let mut fallback = Command::new("icpx");
            fallback.arg("-o").arg(output_file.to_str().unwrap());
            
            for bc_file in &bc_files {
                fallback.arg(bc_file.to_str().unwrap());
            }
            
            match fallback.status() {
                Ok(status) => status,
                Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
            }
        }
    };
    
    if !status.success() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
    }
    
    Ok(())
}

fn optimize_bc(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find bitcode files
    let bc_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if bc_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each file
    for input_file in bc_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}_optimized.bc", file_stem));
        
        // Determine optimization level from options or default to O2
        let opt_level = if ctx.options.iter().any(|opt| opt == "-O0") {
            "-O0"
        } else if ctx.options.iter().any(|opt| opt == "-O1") {
            "-O1"
        } else if ctx.options.iter().any(|opt| opt == "-O3") {
            "-O3"
        } else {
            "-O2"
        };
        
        // Use opt from the icpx toolchain
        let mut command = Command::new("opt");
        command.args([opt_level, input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Try to use opt, but if not available, attempt to use icpx
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => {
                // Fallback to using icpx for optimization
                let mut fallback = Command::new("icpx");
                fallback.args([opt_level, "-c", "-emit-llvm", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
                
                match fallback.status() {
                    Ok(status) => status,
                    Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
                }
            }
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn codegen_to_relocatable(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find bitcode files
    let bc_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if bc_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each file
    for input_file in bc_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.o", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-c", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Add target if specified
        if let Some(target) = &ctx.target {
            command.args(["-target", target]);
        }
        
        // Add custom options
        command.args(&ctx.options);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn codegen_to_assembly(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find bitcode files
    let bc_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if bc_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each file
    for input_file in bc_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.s", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-S", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Add target if specified
        if let Some(target) = &ctx.target {
            command.args(["-target", target]);
        }
        
        // Add custom options
        command.args(&ctx.options);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
}

fn compile_to_fatbin(ctx: &ActionContext) -> intel_comgr_status_t {
    // Find source files
    let source_files: Vec<_> = ctx.input_files.iter()
        .filter(|(_, kind)| kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_SOURCE.0)
        .map(|(path, _)| path.clone())
        .collect();
    
    if source_files.is_empty() {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR_INVALID_ARGUMENT);
    }
    
    // Process each source file
    for input_file in source_files {
        let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("input");
        let output_file = ctx.temp_dir.join(format!("{}.fatbin", file_stem));
        
        let mut command = Command::new("icpx");
        command.args(["-fsycl", input_file.to_str().unwrap(), "-o", output_file.to_str().unwrap()]);
        
        // Add include paths
        for (include_path, kind) in &ctx.input_files {
            if kind.0 == intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_INCLUDE.0 {
                let include_dir = include_path.parent().unwrap_or(Path::new(""));
                command.arg("-I").arg(include_dir);
            }
        }
        
        // For fatbin, we want to compile for multiple targets
        // Intel GPU targets (examples)
        let targets = [
            "spir64",                    // Generic SPIR-V
            "spir64_gen9",               // Intel Gen9
            "spir64_gen11",              // Intel Gen11
            "spir64_gen12",              // Intel Gen12/Xe
            "spir64_xehpg",              // Intel Arc A-Series
            "spir64_xehpc",              // Intel Data Center GPU Max Series
        ];
        
        // Add targets
        for target in &targets {
            command.arg("-Xs").arg(format!("-device={}", target));
        }
        
        // Add custom options
        command.args(&ctx.options);
        
        let status = match command.status() {
            Ok(status) => status,
            Err(_) => return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR),
        };
        
        if !status.success() {
            return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
        }
    }
    
    Ok(())
} 