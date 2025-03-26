#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
#[cfg(feature = "intel")]
use cuda_types::cuda::CUerror;
use std::mem;
use std::ptr;
use super::context;

#[cfg(feature = "amd")]
pub(crate) fn alloc_v2(dptr: *mut hipDeviceptr_t, bytesize: usize) -> hipError_t {
    unsafe { hipMalloc(dptr.cast(), bytesize) }?;
    // TODO: parametrize for non-Geekbench
    unsafe { hipMemsetD8(*dptr, 0, bytesize) }
}

#[cfg(feature = "intel")]
pub(crate) fn alloc_v2(dptr: *mut hipDeviceptr_t, bytesize: usize) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Create device memory allocation descriptor
    let mut device_desc = ze_device_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        pNext: ptr::null(),
        flags: 0,
        ordinal: 0,
    };
    
    // Allocate memory
    let mut device_ptr = ptr::null_mut();
    let result = unsafe {
        zeMemAllocDevice(
            ctx.context,
            &device_desc,
            bytesize,
            64, // alignment
            ctx.device,
            &mut device_ptr
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorOutOfMemory;
    }
    
    // Store the allocated pointer
    unsafe { *dptr = hipDeviceptr_t(device_ptr); }
    
    // Zero initialize the memory
    if bytesize > 0 {
        set_d8_v2(hipDeviceptr_t(device_ptr), 0, bytesize)?;
    }
    
    // Track allocation in context
    ctx.add_allocation(device_ptr);
    
    hipError_t::hipSuccess
}

#[cfg(feature = "amd")]
pub(crate) fn free_v2(dptr: hipDeviceptr_t) -> hipError_t {
    unsafe { hipFree(dptr.0) }
}

#[cfg(feature = "intel")]
pub(crate) fn free_v2(dptr: hipDeviceptr_t) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Free memory
    let result = unsafe {
        zeMemFree(ctx.context, dptr.0)
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Remove allocation from tracking
    ctx.remove_allocation(dptr.0);
    
    hipError_t::hipSuccess
}

#[cfg(feature = "amd")]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) }
}

#[cfg(feature = "intel")]
pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };
    
    // Append copy command
    let result = unsafe {
        zeCommandListAppendMemoryCopy(
            command_list,
            dst_host,
            src_device.0,
            byte_count,
            ptr::null_mut(), // No wait event
            0,               // Number of wait events
            ptr::null_mut(), // No signal event
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) }
}

#[cfg(feature = "intel")]
pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };
    
    // Append copy command
    let result = unsafe {
        zeCommandListAppendMemoryCopy(
            command_list,
            dst_device.0,
            src_host as *mut ::core::ffi::c_void,
            byte_count,
            ptr::null_mut(), // No wait event
            0,               // Number of wait events
            ptr::null_mut(), // No signal event
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

#[cfg(feature = "intel")]
pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    // Intel Level Zero doesn't have a direct equivalent to hipMemGetAddressRange
    // In a production implementation, you would need to track allocations and their sizes
    // For now, return the same pointer as the base and assume we don't know the size
    
    if !pbase.is_null() {
        unsafe { *pbase = dptr; }
    }
    
    if !psize.is_null() {
        // We don't know the size, so use 0 or query it from allocation tracking in a real implementation
        unsafe { *psize = 0; }
    }
    
    hipError_t::hipSuccess
}

#[cfg(feature = "amd")]
pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, ui: ::core::ffi::c_uint, n: usize) -> hipError_t {
    unsafe { hipMemsetD32(dst, ui, n) }
}

#[cfg(feature = "intel")]
pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, ui: ::core::ffi::c_uint, n: usize) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };
    
    // Append fill command
    let result = unsafe {
        zeCommandListAppendMemoryFill(
            command_list,
            dst.0,
            &ui as *const _ as *const ::core::ffi::c_void,
            std::mem::size_of::<::core::ffi::c_uint>(),
            n * std::mem::size_of::<::core::ffi::c_uint>(),
            ptr::null_mut(), // No wait event
            0,               // Number of wait events
            ptr::null_mut(), // No signal event
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

#[cfg(feature = "amd")]
pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    unsafe { hipMemsetD8(dst, value, n) }
}

#[cfg(feature = "intel")]
pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    // Get current context
    let ctx = match context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(_) => return hipError_t::hipErrorInvalidContext,
    };
    
    // Get a command list
    let command_list = match get_immediate_command_list(&ctx) {
        Ok(cl) => cl,
        Err(e) => return e,
    };
    
    // Append fill command
    let result = unsafe {
        zeCommandListAppendMemoryFill(
            command_list,
            dst.0,
            &value as *const _ as *const ::core::ffi::c_void,
            std::mem::size_of::<::core::ffi::c_uchar>(),
            n,
            ptr::null_mut(), // No wait event
            0,               // Number of wait events
            ptr::null_mut(), // No signal event
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Close and execute the command list
    execute_immediate_command_list(&ctx, command_list)
}

// Helper functions for Intel Level Zero implementation

#[cfg(feature = "intel")]
fn get_immediate_command_list(ctx: &context::Context) -> Result<ze_command_list_handle_t, hipError_t> {
    // Create a new immediate command list
    let mut desc = ze_command_list_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        pNext: ptr::null(),
        commandQueueGroupOrdinal: 0,
        flags: 0,
    };
    
    let mut command_list = ptr::null_mut();
    let result = unsafe {
        zeCommandListCreate(
            ctx.context,
            ctx.device,
            &mut desc,
            &mut command_list,
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return Err(hipError_t::hipErrorInvalidValue);
    }
    
    let handle = ze_command_list_handle_t(command_list);
    
    // Track the command list in the context
    ctx.add_command_list(handle);
    
    Ok(handle)
}

#[cfg(feature = "intel")]
fn execute_immediate_command_list(ctx: &context::Context, command_list: ze_command_list_handle_t) -> hipError_t {
    // Create a command queue
    let mut queue_desc = ze_command_queue_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        pNext: ptr::null(),
        ordinal: 0,
        index: 0,
        flags: 0,
        mode: ze_command_queue_mode_t::ZE_COMMAND_QUEUE_MODE_DEFAULT,
        priority: ze_command_queue_priority_t::ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    
    let mut command_queue = ptr::null_mut();
    let result = unsafe {
        zeCommandQueueCreate(
            ctx.context,
            ctx.device,
            &mut queue_desc,
            &mut command_queue,
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up command list
        unsafe { zeCommandListDestroy(command_list) };
        return hipError_t::hipErrorInvalidValue;
    }
    
    let queue_handle = ze_command_queue_handle_t(command_queue);
    
    // Track the command queue in the context
    ctx.add_command_queue(queue_handle);
    
    // Close the command list
    let result = unsafe {
        zeCommandListClose(command_list)
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up resources
        unsafe {
            zeCommandListDestroy(command_list);
            zeCommandQueueDestroy(command_queue);
        }
        ctx.remove_command_list(command_list);
        ctx.remove_command_queue(queue_handle);
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Execute the command list
    let result = unsafe {
        zeCommandQueueExecuteCommandLists(
            queue_handle,
            1,
            &command_list,
            ptr::null_mut(),
        )
    };
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        // Clean up resources
        unsafe {
            zeCommandListDestroy(command_list);
            zeCommandQueueDestroy(command_queue);
        }
        ctx.remove_command_list(command_list);
        ctx.remove_command_queue(queue_handle);
        return hipError_t::hipErrorInvalidValue;
    }
    
    // Synchronize the queue
    let result = unsafe {
        zeCommandQueueSynchronize(queue_handle, u64::MAX)
    };
    
    // Clean up resources
    unsafe {
        zeCommandListDestroy(command_list);
        zeCommandQueueDestroy(command_queue);
    }
    ctx.remove_command_list(command_list);
    ctx.remove_command_queue(queue_handle);
    
    if result != ze_result_t::ZE_RESULT_SUCCESS {
        return hipError_t::hipErrorInvalidValue;
    }
    
    hipError_t::hipSuccess
}
