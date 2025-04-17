pub(crate) mod r#impl;

// Import necessary for FromCuda
use crate::r#impl::FromCuda;
// Import std::ptr for null_mut
use std::ptr;
// Import Ze types
use ze_runtime_sys::ze_device_handle_t;
// Import CUerror for Result
use cuda_types::cuda::CUerror;

// Define Result type to match FromCuda error return type
type Result<T> = std::result::Result<T, CUerror>;

// Fix implementation of FromCuda for ze_device_handle_t
#[cfg(feature = "intel")]
impl FromCuda<'_, *mut i32> for *mut ze_device_handle_t {
    fn from_cuda(_: &*mut i32) -> Result<Self> {
        // Simplified implementation - just a placeholder
        Ok(ptr::null_mut())
    }
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::unimplemented()
            }
        )*
    };
}

#[cfg(feature = "amd")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}
#[cfg(feature = "intel")]
macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}
#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, i32> for ze_device_handle_t {
    // type Error = CUresult; // 或者您使用的任何错误类型

    fn from_cuda(cuda_value: &'a i32) -> Result<Self> {
        // 这里实现从i32到ze_device_handle_t的转换逻辑
        // 例如：
        if *cuda_value < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE); // 使用适当的错误
        }

        // 根据索引获取设备句柄
        let device_handle = unsafe {
            // 假设有一个函数可以根据索引获取设备句柄
            get_device_handle_by_index(*cuda_value as usize)?
        };

        Ok(device_handle)
    }
}

#[cfg(feature = "intel")]
impl<'a> FromCuda<'a, cuda_types::cuda::CUdeviceptr_v2> for cuda_types::cuda::CUdeviceptr_v2 {
    // type Error = CUresult; // 或者您使用的任何错误类型

    fn from_cuda(cuda_value: cuda_types::cuda::CUdeviceptr_v2) -> Result<Self, Self::Error> {
        // 这里实现从i32到ze_device_handle_t的转换逻辑
        // 例如：
        if unsafe { *cuda_value.0 as i32 } < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE); // 使用适当的错误
        };

        Ok(cuda_value)
    }
}
#[cfg(feature = "amd")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

#[cfg(feature = "intel")]
macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*);
                Ok(())
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxGetLimit,
            cuCtxSetCurrent,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRetain,
            cuDeviceGetProperties,
            cuDeviceGetUuid,
            cuDeviceGetUuid_v2,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuFuncGetAttribute,
            cuInit,
            cuMemAlloc_v2,
            cuMemFree_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyHtoD_v2,
            cuModuleGetFunction,
            cuModuleLoadData,
            cuModuleUnload,
            cuPointerGetAttribute,
            cuMemGetAddressRange_v2,
            cuMemsetD32_v2,
            cuMemsetD8_v2
        ],
    implemented_in_function <= [cuLaunchKernel,]
);
