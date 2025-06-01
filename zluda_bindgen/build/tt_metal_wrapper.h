// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_METAL_WRAPPER_H
#define TT_METAL_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Core coordinate structure
typedef struct {
    uint32_t x;
    uint32_t y;
} CoreCoord;

// Data format enumeration
typedef enum {
    tt_DataFormat_Invalid = 0,
    tt_DataFormat_Bfp8_b = 1,
    tt_DataFormat_Bfp4_b = 2,
    tt_DataFormat_Float32 = 3,
    tt_DataFormat_Bfp2_b = 4,
    tt_DataFormat_Float16_b = 5,
    tt_DataFormat_Bfp8 = 6,
    tt_DataFormat_Bfp4 = 7,
    tt_DataFormat_Bfp2 = 8,
    tt_DataFormat_Float16 = 9,
    tt_DataFormat_RawUInt8 = 10,
    tt_DataFormat_RawUInt16 = 11,
    tt_DataFormat_RawUInt32 = 12,
    tt_DataFormat_Int8 = 13,
    tt_DataFormat_UInt8 = 14,
    tt_DataFormat_UInt16 = 15,
    tt_DataFormat_Int32 = 16,
    tt_DataFormat_UInt32 = 17
} tt_DataFormat;

// Buffer type enumeration  
typedef enum {
    tt_BufferType_DRAM = 0,
    tt_BufferType_L1 = 1,
    tt_BufferType_L1_SMALL = 2,
    tt_BufferType_TRACE = 3,
    tt_BufferType_SYSTEM_MEMORY = 4
} tt_BufferType;

// Processor type enumeration
typedef enum {
    tt_DataMovementProcessor_RISCV_0 = 0,
    tt_DataMovementProcessor_RISCV_1 = 1
} tt_DataMovementProcessor;

// NOC enumeration
typedef enum {
    tt_NOC_RISCV_0_default = 0,
    tt_NOC_RISCV_1_default = 1
} tt_NOC;

// Opaque pointer types
typedef struct tt_Device tt_Device;
typedef struct tt_Program tt_Program;
typedef struct tt_Buffer tt_Buffer;
typedef struct tt_CircularBuffer tt_CircularBuffer;
typedef struct tt_Kernel tt_Kernel;

// Configuration structures
typedef struct {
    tt_Device* device;
    uint32_t size;
    uint32_t page_size;
    tt_BufferType buffer_type;
} tt_InterleavedBufferConfig;

typedef struct {
    tt_DataMovementProcessor processor;
    tt_NOC noc;
} tt_DataMovementConfig;

typedef struct {
    uint32_t* compile_args;
    size_t compile_args_count;
} tt_ComputeConfig;

typedef struct {
    uint32_t size;
    tt_DataFormat data_format;
    uint32_t page_size;
} tt_CircularBufferConfig;

// Core TT Metal API functions
tt_Device* tt_metal_CreateDevice(int device_id);
int tt_metal_CloseDevice(tt_Device* device);

tt_Program* tt_metal_CreateProgram(void);

tt_Buffer* tt_metal_CreateBuffer(const tt_InterleavedBufferConfig* config);

tt_CircularBuffer* tt_metal_CreateCircularBuffer(
    tt_Program* program, 
    CoreCoord core, 
    const tt_CircularBufferConfig* config
);

tt_Kernel* tt_metal_CreateKernel(
    tt_Program* program,
    const char* kernel_file,
    CoreCoord core,
    const tt_DataMovementConfig* config
);

tt_Kernel* tt_metal_CreateComputeKernel(
    tt_Program* program,
    const char* kernel_file,
    CoreCoord core,
    const tt_ComputeConfig* config
);

void tt_metal_SetRuntimeArgs(
    tt_Program* program,
    tt_Kernel* kernel,
    CoreCoord core,
    const uint32_t* args,
    size_t args_count
);

int tt_metal_LaunchProgram(tt_Device* device, tt_Program* program);

uint32_t tt_metal_TileSize(tt_DataFormat data_format);

void tt_metal_WriteToBuffer(tt_Buffer* buffer, const uint32_t* data, size_t size);
void tt_metal_ReadFromBuffer(tt_Buffer* buffer, uint32_t* data, size_t size);

uint64_t tt_metal_buffer_address(tt_Buffer* buffer);

// Utility functions
uint32_t* tt_metal_create_random_vector_of_bfp8(uint32_t size, int is_exp_a, uint32_t seed, uint64_t timestamp);

#ifdef __cplusplus
}
#endif

#endif // TT_METAL_WRAPPER_H