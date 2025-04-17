#[cfg(feature = "intel")]
use super::ze_module;
use super::ZludaObject;
#[cfg(feature = "intel")]
use super::{Decuda, Encuda};
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use std::os::raw::c_void;
use std::{ffi::CStr, mem, ptr};
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
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
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
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
            return ze_to_cuda_result(result);
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
    let llvm_module = ptx::to_llvm_module(&ast).map_err(|_| CUerror::UNKNOWN)?;
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

    let spirv_module = ze_module::SpirvModule::new(text).map_err(|_| CUerror::NO_BINARY_FOR_GPU)?;
    match load_data_impl(module, spirv_module) {
        Ok(()) => CUresult::SUCCESS,
        Err(e) => Err(e),
    }
}

#[cfg(feature = "intel")]
pub(crate) fn load_data_impl(
    module: &mut CUmodule,
    spirv_module: ze_module::SpirvModule,
) -> Result<(), CUerror> {
    // Get current context and device
    let (context, device) = get_current_context_and_device()?;

    // Convert PTX to SPIRV - for Intel we need to convert PTX to SPIR-V format
    let spirv_binary = ptx_to_spirv(&spirv_module)?;

    // Create module descriptor
    let module_desc = ze_module_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MODULE_DESC,
        pNext: ptr::null(),
        format: ze_module_format_t::ZE_MODULE_FORMAT_IL_SPIRV,
        inputSize: spirv_binary.len(),
        pInputModule: spirv_binary.as_ptr(),
        pBuildFlags: ptr::null(),
        pConstants: ptr::null(),
    };

    // Create module
    let mut ze_module = ptr::null_mut();
    let mut build_log = ptr::null_mut();

    let result =
        unsafe { zeModuleCreate(context, device, &module_desc, ze_module, &mut build_log) };

    // Check if build log exists and handle it
    if !build_log.is_null() {
        // In a real implementation, you would process the build log
        unsafe { zeModuleBuildLogDestroy(build_log) };
    }

    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(CUerror::UNKNOWN);
    }

    // Create and return the Module object
    // Use ze_module implementation for Intel
    super::ze_module::load_data_impl(module, spirv_module)?;
    Ok(())
}

#[cfg(feature = "intel")]
fn ptx_to_spirv(spirv_module: &ze_module::SpirvModule) -> Result<Vec<u8>, CUerror> {
    // Convert PTX AST to LLVM IR
    let llvm_module = ptx::to_llvm_module(spirv_module.ast.clone()).map_err(|_| CUerror::UNKNOWN)?;

    // For Intel, we need to use the LLVM SPIR-V target
    // This is a placeholder implementation that assumes
    // the ptx::LlvmModule has infrastructure for SPIR-V conversion

    // In a real implementation, you would use LLVM's SPIR-V backend
    // Here we're using the robust conversion from the llvm_module
    let spirv_binary = ptx::llvm_to_spirv_robust(unsafe {
        std::str::from_raw_parts(llvm_module.llvm_ir.as_ptr(), llvm_module.llvm_ir.len())
    })
    .map_err(|_| CUerror::UNKNOWN)?;

    Ok(spirv_binary)
}

#[cfg(feature = "intel")]
fn get_current_context_and_device() -> Result<(ze_context_handle_t, ze_device_handle_t), CUerror> {
    // Get the current thread-local context and device
    let current_ctx = super::context::CONTEXT_STACK
        .with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|(ctx, dev)| (*ctx, *dev))
        })
        .ok_or(CUerror::INVALID_CONTEXT)?;

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
    hfunc: &mut CUfunction,
    hmod: &Module,
    name: *const ::core::ffi::c_char,
) -> CUresult {
    let name_str = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| CUerror::INVALID_VALUE)?;

    // Check if kernel already exists
    if let Some((_, kernel)) = hmod.functions.iter().find(|(n, _)| n == name_str) {
        *hfunc = ZeKernel {
            context: hmod.context,
            device: hmod.device,
            module: hmod.module,
            kernel: *kernel,
        }
        .wrap();
        return CUresult::SUCCESS;
    }

    // Create new kernel
    let mut kernel = ptr::null_mut();
    let kernel_desc = ze_kernel_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_KERNEL_DESC,
        pNext: ptr::null(),
        flags: 0,
        pKernelName: name,
    };

    let result = unsafe { zeKernelCreate(hmod.module, &kernel_desc, kernel) };

    match result {
        ze_result_t::ZE_RESULT_SUCCESS => {
            let kernel_wrapper = ZeKernel {
                context: hmod.context,
                device: hmod.device,
                module: hmod.module,
                kernel: unsafe { *kernel },
            };

            // Store the kernel in the module's function list
            let mut module_mut = hmod as *const Module as *mut Module;
            unsafe {
                (*module_mut)
                    .functions
                    .push((name_str.to_string(), *kernel));
            }

            *hfunc = kernel_wrapper.wrap();
            CUresult::SUCCESS
        }
        ze_result_t::ZE_RESULT_ERROR_INVALID_KERNEL_NAME => CUresult::ERROR_INVALID_IMAGE,
        _ => CUresult::ERROR_INVALID_VALUE,
    }
}

#[cfg(feature = "intel")]
pub(crate) struct ZeKernel {
    pub context: ze_context_handle_t,
    pub device: ze_device_handle_t,
    pub module: ze_module_handle_t,
    pub kernel: ze_kernel_handle_t,
}
#[cfg(feature = "intel")]
unsafe impl Send for ZeKernel {}
#[cfg(feature = "intel")]
unsafe impl Sync for ZeKernel {}
#[cfg(feature = "intel")]
impl ZludaObject for ZeKernel {
    const COOKIE: usize = 0xad74ceadb9b2d51c;

    type CudaHandle = CUfunction;

    fn drop_checked(&mut self) -> CUresult {
        let result = unsafe { zeKernelDestroy(self.kernel) };
        if result != ze_result_t::ZE_RESULT_SUCCESS {
            return ze_to_cuda_result(result);
        }
        Ok(())
    }
}

#[cfg(feature = "intel")]
fn ze_to_cuda_result(result: ze_result_t) -> CUresult {
    match result {
        ze_result_t::ZE_RESULT_SUCCESS => CUresult::SUCCESS,
        ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
        | ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => CUresult::ERROR_OUT_OF_MEMORY,
        ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST => CUresult::ERROR_NO_DEVICE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE => CUresult::ERROR_INVALID_HANDLE,
        ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER => CUresult::ERROR_INVALID_VALUE,
        ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => CUresult::ERROR_NOT_INITIALIZED,
        _ => CUresult::ERROR_UNKNOWN,
    }
}
