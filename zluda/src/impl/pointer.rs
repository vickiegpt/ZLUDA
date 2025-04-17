use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
use std::{ffi::c_void, ptr};
#[cfg(feature = "intel")]
use ze_runtime_sys::*;

#[cfg(feature = "amd")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    if data == ptr::null_mut() {
        return CUresult::ErrorInvalidValue;
    }
    match attribute {
        // TODO: implement by getting device ordinal & allocation start,
        // then go through every context for that device
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => CUresult::ErrorNotSupported,
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            let mut cuda_result = CUmemorytype(0);
            CUpointerGetAttribute(
                (&mut cuda_result as *mut CUmemorytype).cast::<c_void>(),
                attribute,
                ptr,
            )?;
            let hip_result = memory_type_amd(cuda_result)?;
            unsafe { *(data.cast()) = hip_result };
            Ok(())
        }
        _ => unsafe { CUpointerGetAttribute(data, attribute, ptr) },
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    if data == ptr::null_mut() {
        return CUresult::ERROR_INVALID_VALUE;
    }

    // Get the current context for querying memory properties
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    match attribute {
        // TODO: implement context attribute for Intel devices
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // For Intel, we can get the context from the memory allocation
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // Store the context handle
            *(data.cast::<ze_context_handle_t>()) = ze_context.context;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Query memory attributes using Level Zero memory APIs
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // Convert Level Zero memory type to CUDA memory type
            let cuda_type = match alloc_props.type_ {
                ze_memory_type_t::ZE_MEMORY_TYPE_HOST => CUmemorytype::CU_MEMORYTYPE_HOST,
                ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => CUmemorytype::CU_MEMORYTYPE_DEVICE,
                ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => CUmemorytype::CU_MEMORYTYPE_UNIFIED,
                _ => return CUresult::ERROR_INVALID_VALUE,
            };

            *(data.cast()) = cuda_type;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // In Level Zero, device pointers are represented the same way
            *(data.cast::<CUdeviceptr>()) = ptr;
            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            // For host-mapped memory, need to query base address
            let mut base_ptr = ptr::null_mut();
            let mut size = 0;

            let result = unsafe {
                zeMemGetAddressRange(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut base_ptr,
                    &mut size,
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                *(data.cast::<*mut c_void>()) = ptr::null_mut();
            } else {
                *(data.cast::<*mut c_void>()) = base_ptr;
            }

            CUresult::SUCCESS
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            // Check if memory is managed (shared memory in Level Zero)
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = unsafe {
                zeMemGetAllocProperties(
                    ze_context.context,
                    ptr.0 as *const c_void,
                    &mut alloc_props,
                    ptr::null_mut(),
                )
            };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return super::ze_to_cuda_result(result);
            }

            // In Level Zero, shared memory is considered managed
            *(data.cast::<i32>()) = if alloc_props.type_ == ze_memory_type_t::ZE_MEMORY_TYPE_SHARED
            {
                1
            } else {
                0
            };

            CUresult::SUCCESS
        }

        // For other attributes, handle based on Intel capabilities or return unsupported
        _ => CUresult::ERROR_NOT_SUPPORTED,
    }
}

#[cfg(feature = "amd")]
fn memory_type_amd(cu: CUmemorytype) -> Result<CUmemorytype, CUresult> {
    match cu {
        CUmemorytype::CU_MEMORYTYPE_HOST => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        CUmemorytype::CU_MEMORYTYPE_DEVICE => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        CUmemorytype::CU_MEMORYTYPE_ARRAY => Ok(CUmemorytype::CU_MEMORYTYPE_ARRAY),
        CUmemorytype::CU_MEMORYTYPE_UNIFIED => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(CUresult::ErrorInvalidValue),
    }
}

#[cfg(feature = "intel")]
fn memory_type_intel(ze_type: ze_memory_type_t) -> Result<CUmemorytype, ze_result_t> {
    match ze_type {
        ze_memory_type_t::ZE_MEMORY_TYPE_HOST => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT),
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn get_pointer_attribute(
    attribute: CUpointer_attribute,
    ptr: CUdeviceptr,
) -> CUresult {
    // Get the current context
    let ze_context = match super::context::get_current_ze() {
        Ok(ctx) => ctx,
        Err(e) => return e.into(),
    };

    match attribute {
        CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT => {
            // Return the current context
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Query memory type
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = zeMemGetAllocProperties(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut alloc_props,
                ptr::null_mut(),
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return result.into();
            }

            // Convert Level Zero memory type to CUDA memory type
            let cuda_type = match alloc_props.type_ {
                ze_memory_type_t::ZE_MEMORY_TYPE_HOST => CUmemorytype::CU_MEMORYTYPE_HOST,
                ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => CUmemorytype::CU_MEMORYTYPE_DEVICE,
                ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => CUmemorytype::CU_MEMORYTYPE_UNIFIED,
                _ => return Err(ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT).into(),
            };

            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // Device pointer is the same in Level Zero
            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_HOST_POINTER => {
            // Query host pointer
            let mut base_ptr = ptr::null_mut();
            let mut size = 0;

            let result = zeMemGetAddressRange(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut base_ptr,
                &mut size,
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return result.into();
            }

            Ok(())
        }

        CUpointer_attribute::CU_POINTER_ATTRIBUTE_IS_MANAGED => {
            // Check if memory is managed (shared memory in Level Zero)
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };

            let result = zeMemGetAllocProperties(
                ze_context.context,
                ptr.0 as *const c_void,
                &mut alloc_props,
                ptr::null_mut(),
            );

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return result.into();
            }

            Ok(())
        }

        // For other attributes, handle based on Intel capabilities or return unsupported
        _ => ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE.into(),
    }
}
