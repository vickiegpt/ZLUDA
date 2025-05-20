// 导出ze_runner接口供ptx crate使用

use std::os::raw::{c_char, c_int, c_void};

/// 与C库中的ze_runner_result_t对应
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ZeRunnerResult {
    Success = 0,
    ErrorInit = 1,
    ErrorNoDriver = 2,
    ErrorNoDevice = 3,
    ErrorContext = 4,
    ErrorCommandList = 5,
    ErrorModule = 6,
    ErrorKernel = 7,
    ErrorMemory = 8,
    ErrorKernelExecution = 9,
    ErrorSync = 10,
    ErrorInvalidArgument = 11,
}

impl ZeRunnerResult {
    /// 将数字状态码转换为枚举值
    pub fn from_status(status: c_int) -> Self {
        match status {
            0 => ZeRunnerResult::Success,
            1 => ZeRunnerResult::ErrorInit,
            2 => ZeRunnerResult::ErrorNoDriver,
            3 => ZeRunnerResult::ErrorNoDevice,
            4 => ZeRunnerResult::ErrorContext,
            5 => ZeRunnerResult::ErrorCommandList,
            6 => ZeRunnerResult::ErrorModule,
            7 => ZeRunnerResult::ErrorKernel,
            8 => ZeRunnerResult::ErrorMemory,
            9 => ZeRunnerResult::ErrorKernelExecution,
            10 => ZeRunnerResult::ErrorSync,
            11 => ZeRunnerResult::ErrorInvalidArgument,
            _ => ZeRunnerResult::ErrorInit, // 默认为初始化错误
        }
    }
}

/// 外部C函数声明
unsafe extern "C" {
    pub fn run_spirv_kernel(
        name: *const c_char,
        spirv_data: *const c_void,
        spirv_size: usize,
        input: *const c_void,
        input_size: usize,
        output: *mut c_void,
        output_size: usize,
    ) -> c_int;
}
