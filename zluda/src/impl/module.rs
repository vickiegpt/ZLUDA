use super::ZludaObject;
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
use std::{ffi::CStr, mem, ptr};
#[cfg(feature = "intel")]
use super::{Decuda, Encuda};
#[cfg(feature = "intel")]
use std::os::raw::c_void;

pub(crate) struct SpirvModule {
    text: String,
    ast: ptx_parser::ModuleAst,
}

impl SpirvModule {
    pub fn new(text: &str) -> Result<Self, ptx_parser::ParsingError> {
        let ast = ptx_parser::parse_module_checked(text)?;
        Ok(Self {
            text: text.to_string(),
            ast,
        })
    }
}

#[cfg(feature = "amd")]
pub(crate) struct Module {
    base: hipModule_t,
}

#[cfg(feature = "intel")]
pub(crate) struct Module {
    context: ze_context_handle_t,
    device: ze_device_handle_t,
    module: ze_module_handle_t,
    functions: Vec<(String, ze_kernel_handle_t)>,
}

#[cfg(feature = "amd")]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        unsafe { hipModuleUnload(self.base) }?;
        Ok(())
    }
}

#[cfg(feature = "intel")]
impl ZludaObject for Module {
    const COOKIE: usize = 0xe9138bd040487d4a;

    type CudaHandle = CUmodule;

    fn drop_checked(&mut self) -> CUresult {
        // Clean up all kernels first
        for (_, kernel) in &self.functions {
            unsafe {
                zeKernelDestroy(*kernel);
            }
        }
        self.functions.clear();

        // Destroy the module
        let result = unsafe { zeModuleDestroy(self.module) };
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return result.into();
        }
        
        Ok(())
    }
}

#[cfg(feature = "amd")]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;
    let ast = ptx_parser::parse_module_checked(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
    let llvm_module = ptx::to_llvm_module(ast).map_err(|_| CUerror::UNKNOWN)?;
    let mut dev = 0;
    unsafe { hipCtxGetDevice(&mut dev) }?;
    let mut props = unsafe { mem::zeroed() };
    unsafe { hipGetDevicePropertiesR0600(&mut props, dev) }?;
    let elf_module = comgr::compile_bitcode(
        unsafe { CStr::from_ptr(props.gcnArchName.as_ptr()) },
        &*llvm_module.llvm_ir,
        llvm_module.linked_bitcode(),
    )
    .map_err(|_| CUerror::UNKNOWN)?;
    let mut hip_module = unsafe { mem::zeroed() };
    unsafe { hipModuleLoadData(&mut hip_module, elf_module.as_ptr().cast()) }?;
    *module = Module { base: hip_module }.wrap();
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn load_data(module: &mut CUmodule, image: *const std::ffi::c_void) -> CUresult {
    // Parse the PTX text
    let text = unsafe { CStr::from_ptr(image.cast()) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;
    
    let spirv_module = match load_data_impl(module, SpirvModule::new(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?) {
        Ok(()) => return CUresult::CUDA_SUCCESS,
        Err(e) => return e.into(),
    };
    
    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn load_data_impl(module: &mut CUmodule, spirv_module: SpirvModule) -> Result<(), CUerror> {
    // Get current context and device
    let (context, device) = get_current_context_and_device()?;
    
    // Convert PTX to SPIRV - for Intel we need to convert PTX to SPIR-V format
    let spirv_binary = ptx_to_spirv(&spirv_module)?;
    
    // Create module descriptor
    let mut module_desc = ze_module_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MODULE_DESC,
        pNext: ptr::null(),
        format: ze_module_format_t::ZE_MODULE_FORMAT_IL_SPIRV,
        inputSize: spirv_binary.len() as usize,
        pInputModule: spirv_binary.as_ptr() as *const c_void,
        pBuildFlags: ptr::null(),
        pConstants: ptr::null(),
    };
    
    // Create module
    let mut ze_module = ptr::null_mut();
    let mut build_log = ptr::null_mut();
    
    let result = unsafe {
        zeModuleCreate(
            context,
            device,
            &module_desc,
            &mut ze_module,
            &mut build_log,
        )
    };
    
    // Check if build log exists and handle it
    if !build_log.is_null() {
        // In a real implementation, you would process the build log
        unsafe { zeModuleBuildLogDestroy(build_log) };
    }
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUerror::UNKNOWN);
    }
    
    // Create an empty list to store kernels
    let functions = Vec::new();
    
    // Create and return the Module object
    *module = Module { 
        context, 
        device, 
        module: ze_module_handle_t(ze_module),
        functions 
    }.wrap();
    
    Ok(())
}

#[cfg(feature = "intel")]
fn ptx_to_spirv(spirv_module: &SpirvModule) -> Result<Vec<u8>, CUerror> {
    // Convert PTX AST to LLVM IR
    let llvm_module = ptx::to_llvm_module(spirv_module.ast.clone()).map_err(|_| CUerror::UNKNOWN)?;
    
    // For Intel, we need to use the LLVM SPIR-V target
    // This is a placeholder implementation that assumes 
    // the ptx::LlvmModule has infrastructure for SPIR-V conversion
    
    // In a real implementation, you would use LLVM's SPIR-V backend
    // Here we're using a hypothetical conversion from the llvm_module
    let spirv_binary = ptx::llvm_to_spirv(&llvm_module.llvm_ir)
        .map_err(|_| CUerror::UNKNOWN)?;
    
    Ok(spirv_binary)
}

#[cfg(feature = "intel")]
fn get_current_context_and_device() -> Result<(ze_context_handle_t, ze_device_handle_t), CUerror> {
    // Get the current thread-local context and device
    let current_ctx = super::context::CONTEXT_STACK.with(|stack| {
        let stack = stack.borrow();
        stack.last().map(|(ctx, dev)| (*ctx, *dev))
    }).ok_or(CUerror::INVALID_CONTEXT)?;
    
    // Get the ZeContext from the CUcontext
    let context = super::context::get_current_ze()?;
    
    // Return context and device handles
    Ok((context.context, context.device))
}

pub(crate) fn unload(hmod: CUmodule) -> CUresult {
    super::drop_checked::<Module>(hmod)
}

#[cfg(feature = "amd")]
pub(crate) fn get_function(
    hfunc: &mut hipFunction_t,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> hipError_t {
    unsafe { hipModuleGetFunction(hfunc, hmod.base, name) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_function(
    hfunc: &mut ze_kernel_handle_t,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> hipError_t {
    let name_str = unsafe { CStr::from_ptr(name) }.to_str()
        .map_err(|_| hipError_t::hipErrorInvalidValue)?;
    
    // Check if we already have this kernel cached
    for (func_name, kernel) in &hmod.functions {
        if func_name == name_str {
            *hfunc = *kernel;
            return hipError_t::hipSuccess;
        }
    }
    
    // Create a new kernel if not found
    let mut kernel = ptr::null_mut();
    let result = unsafe {
        zeKernelCreate(
            hmod.module,
            &ze_kernel_desc_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_KERNEL_DESC,
                pNext: ptr::null(),
                flags: 0,
                pKernelName: name,
            },
            &mut kernel,
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return match result {
            ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_NAME => hipError_t::hipErrorInvalidSymbol,
            _ => hipError_t::hipErrorInvalidValue,
        };
    }
    
    // Add to our cache
    let kernel_handle = ze_kernel_handle_t(kernel);
    if let Module { functions, .. } = hmod {
        functions.push((name_str.to_string(), kernel_handle));
    }
    
    *hfunc = kernel_handle;
    hipError_t::hipSuccess
}
