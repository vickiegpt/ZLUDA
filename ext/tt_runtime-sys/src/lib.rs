// Generated automatically by zluda_bindgen
// DO NOT EDIT MANUALLY
#![allow(warnings)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CoreCoord {
    pub x: u32,
    pub y: u32,
}
pub const tt_DataFormat_tt_DataFormat_Invalid: tt_DataFormat = 0;
pub const tt_DataFormat_tt_DataFormat_Bfp8_b: tt_DataFormat = 1;
pub const tt_DataFormat_tt_DataFormat_Bfp4_b: tt_DataFormat = 2;
pub const tt_DataFormat_tt_DataFormat_Float32: tt_DataFormat = 3;
pub const tt_DataFormat_tt_DataFormat_Bfp2_b: tt_DataFormat = 4;
pub const tt_DataFormat_tt_DataFormat_Float16_b: tt_DataFormat = 5;
pub const tt_DataFormat_tt_DataFormat_Bfp8: tt_DataFormat = 6;
pub const tt_DataFormat_tt_DataFormat_Bfp4: tt_DataFormat = 7;
pub const tt_DataFormat_tt_DataFormat_Bfp2: tt_DataFormat = 8;
pub const tt_DataFormat_tt_DataFormat_Float16: tt_DataFormat = 9;
pub const tt_DataFormat_tt_DataFormat_RawUInt8: tt_DataFormat = 10;
pub const tt_DataFormat_tt_DataFormat_RawUInt16: tt_DataFormat = 11;
pub const tt_DataFormat_tt_DataFormat_RawUInt32: tt_DataFormat = 12;
pub const tt_DataFormat_tt_DataFormat_Int8: tt_DataFormat = 13;
pub const tt_DataFormat_tt_DataFormat_UInt8: tt_DataFormat = 14;
pub const tt_DataFormat_tt_DataFormat_UInt16: tt_DataFormat = 15;
pub const tt_DataFormat_tt_DataFormat_Int32: tt_DataFormat = 16;
pub const tt_DataFormat_tt_DataFormat_UInt32: tt_DataFormat = 17;
pub type tt_DataFormat = ::core::ffi::c_uint;
pub const tt_BufferType_tt_BufferType_DRAM: tt_BufferType = 0;
pub const tt_BufferType_tt_BufferType_L1: tt_BufferType = 1;
pub const tt_BufferType_tt_BufferType_L1_SMALL: tt_BufferType = 2;
pub const tt_BufferType_tt_BufferType_TRACE: tt_BufferType = 3;
pub const tt_BufferType_tt_BufferType_SYSTEM_MEMORY: tt_BufferType = 4;
pub type tt_BufferType = ::core::ffi::c_uint;
pub const tt_DataMovementProcessor_tt_DataMovementProcessor_RISCV_0: tt_DataMovementProcessor = 0;
pub const tt_DataMovementProcessor_tt_DataMovementProcessor_RISCV_1: tt_DataMovementProcessor = 1;
pub type tt_DataMovementProcessor = ::core::ffi::c_uint;
pub const tt_NOC_tt_NOC_RISCV_0_default: tt_NOC = 0;
pub const tt_NOC_tt_NOC_RISCV_1_default: tt_NOC = 1;
pub type tt_NOC = ::core::ffi::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Device {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Program {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Buffer {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_CircularBuffer {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tt_Kernel {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_InterleavedBufferConfig {
    pub device: *mut tt_Device,
    pub size: u32,
    pub page_size: u32,
    pub buffer_type: tt_BufferType,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_DataMovementConfig {
    pub processor: tt_DataMovementProcessor,
    pub noc: tt_NOC,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_ComputeConfig {
    pub compile_args: *mut u32,
    pub compile_args_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct tt_CircularBufferConfig {
    pub size: u32,
    pub data_format: tt_DataFormat,
    pub page_size: u32,
}
extern "C" {
    pub fn tt_metal_CreateDevice(device_id: ::core::ffi::c_int) -> *mut tt_Device;
}
extern "C" {
    pub fn tt_metal_CloseDevice(device: *mut tt_Device) -> ::core::ffi::c_int;
}
extern "C" {
    pub fn tt_metal_CreateProgram() -> *mut tt_Program;
}
extern "C" {
    pub fn tt_metal_CreateBuffer(
        config: *const tt_InterleavedBufferConfig,
    ) -> *mut tt_Buffer;
}
extern "C" {
    pub fn tt_metal_CreateCircularBuffer(
        program: *mut tt_Program,
        core: CoreCoord,
        config: *const tt_CircularBufferConfig,
    ) -> *mut tt_CircularBuffer;
}
extern "C" {
    pub fn tt_metal_CreateKernel(
        program: *mut tt_Program,
        kernel_file: *const ::core::ffi::c_char,
        core: CoreCoord,
        config: *const tt_DataMovementConfig,
    ) -> *mut tt_Kernel;
}
extern "C" {
    pub fn tt_metal_CreateComputeKernel(
        program: *mut tt_Program,
        kernel_file: *const ::core::ffi::c_char,
        core: CoreCoord,
        config: *const tt_ComputeConfig,
    ) -> *mut tt_Kernel;
}
extern "C" {
    pub fn tt_metal_SetRuntimeArgs(
        program: *mut tt_Program,
        kernel: *mut tt_Kernel,
        core: CoreCoord,
        args: *const u32,
        args_count: usize,
    );
}
extern "C" {
    pub fn tt_metal_LaunchProgram(
        device: *mut tt_Device,
        program: *mut tt_Program,
    ) -> ::core::ffi::c_int;
}
extern "C" {
    pub fn tt_metal_TileSize(data_format: tt_DataFormat) -> u32;
}
extern "C" {
    pub fn tt_metal_WriteToBuffer(buffer: *mut tt_Buffer, data: *const u32, size: usize);
}
extern "C" {
    pub fn tt_metal_ReadFromBuffer(buffer: *mut tt_Buffer, data: *mut u32, size: usize);
}
extern "C" {
    pub fn tt_metal_buffer_address(buffer: *mut tt_Buffer) -> u64;
}
extern "C" {
    pub fn tt_metal_create_random_vector_of_bfp8(
        size: u32,
        is_exp_a: ::core::ffi::c_int,
        seed: u32,
        timestamp: u64,
    ) -> *mut u32;
}
unsafe impl Send for tt_Device {}
unsafe impl Sync for tt_Device {}
unsafe impl Send for tt_Program {}
unsafe impl Sync for tt_Program {}
unsafe impl Send for tt_Buffer {}
unsafe impl Sync for tt_Buffer {}
unsafe impl Send for tt_CircularBuffer {}
unsafe impl Sync for tt_CircularBuffer {}
unsafe impl Send for tt_Kernel {}
unsafe impl Sync for tt_Kernel {}
