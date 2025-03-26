use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
use std::{ffi::c_void, ptr};

#[cfg(feature = "amd")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: hipPointer_attribute,
    ptr: hipDeviceptr_t,
) -> hipError_t {
    if data == ptr::null_mut() {
        return hipError_t::ErrorInvalidValue;
    }
    match attribute {
        // TODO: implement by getting device ordinal & allocation start,
        // then go through every context for that device
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_CONTEXT => hipError_t::ErrorNotSupported,
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            let mut hip_result = hipMemoryType(0);
            hipPointerGetAttribute(
                (&mut hip_result as *mut hipMemoryType).cast::<c_void>(),
                attribute,
                ptr,
            )?;
            let cuda_result = memory_type_amd(hip_result)?;
            unsafe { *(data.cast()) = cuda_result };
            Ok(())
        }
        _ => unsafe { hipPointerGetAttribute(data, attribute, ptr) },
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn get_attribute(
    data: *mut c_void,
    attribute: hipPointer_attribute,
    ptr: hipDeviceptr_t,
) -> hipError_t {
    if data == ptr::null_mut() {
        return ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT.into();
    }
    
    match attribute {
        // TODO: implement context attribute for Intel devices
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_CONTEXT => ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE.into(),
        
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_MEMORY_TYPE => {
            // Get the current context for querying memory properties
            let ze_context = match super::context::get_current_ze() {
                Ok(ctx) => ctx,
                Err(e) => return e.into(),
            };
            
            // Query memory attributes using Level Zero memory APIs
            let mut alloc_props = ze_memory_allocation_properties_t {
                stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                pNext: ptr::null_mut(),
                type_: ze_memory_type_t::ZE_MEMORY_TYPE_UNKNOWN,
                id: 0,
                pageSize: 0,
            };
            
            let result = zeMemGetAllocProperties(
                ze_context.context,
                ptr as *const c_void,
                &mut alloc_props,
                ptr::null_mut(),
            );
            
            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return result.into();
            }
            
            let cuda_result = memory_type_intel(alloc_props.type_)?;
            *(data.cast()) = cuda_result;
            
            Ok(())
        },
        
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_DEVICE_POINTER => {
            // In Level Zero, device pointers are represented the same way
            *(data.cast::<CUdeviceptr>()) = ptr;
            Ok(())
        },
        
        hipPointer_attribute::HIP_POINTER_ATTRIBUTE_HOST_POINTER => {
            // For host-mapped memory, need to query base address
            let ze_context = match super::context::get_current_ze() {
                Ok(ctx) => ctx,
                Err(e) => return e.into(),
            };
            
            // Get the base address - if this is host memory it will be accessible
            let mut base_ptr = ptr::null_mut();
            let result = zeMemGetAddressRange(
                ze_context.context,
                ptr as *const c_void,
                &mut base_ptr,
                ptr::null_mut(),
            );
            
            if result != ze_result_t::ZE_RESULT_SUCCESS {
                *(data.cast::<*mut c_void>()) = ptr::null_mut();
            } else {
                *(data.cast::<*mut c_void>()) = base_ptr;
            }
            
            Ok(())
        },
        
        // For other attributes, handle based on Intel capabilities or return unsupported
        _ => ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE.into(),
    }
}

#[cfg(feature = "amd")]
fn memory_type_amd(cu: hipMemoryType) -> Result<CUmemorytype, hipErrorCode_t> {
    match cu {
        hipMemoryType::hipMemoryTypeHost => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        hipMemoryType::hipMemoryTypeDevice => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        hipMemoryType::hipMemoryTypeArray => Ok(CUmemorytype::CU_MEMORYTYPE_ARRAY),
        hipMemoryType::hipMemoryTypeUnified => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(hipErrorCode_t::InvalidValue),
    }
}

#[cfg(feature = "intel")]
fn memory_type_intel(ze_type: ze_memory_type_t) -> Result<CUmemorytype, hipErrorCode_t> {
    match ze_type {
        ze_memory_type_t::ZE_MEMORY_TYPE_HOST => Ok(CUmemorytype::CU_MEMORYTYPE_HOST),
        ze_memory_type_t::ZE_MEMORY_TYPE_DEVICE => Ok(CUmemorytype::CU_MEMORYTYPE_DEVICE),
        ze_memory_type_t::ZE_MEMORY_TYPE_SHARED => Ok(CUmemorytype::CU_MEMORYTYPE_UNIFIED),
        _ => Err(hipErrorCode_t::InvalidValue),
    }
}
