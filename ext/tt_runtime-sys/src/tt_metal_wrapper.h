// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_METAL_WRAPPER_H
#define TT_METAL_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types
typedef struct tt_Device tt_Device;
typedef struct tt_Program tt_Program;
typedef struct tt_Buffer tt_Buffer;
typedef struct tt_Kernel tt_Kernel;
typedef struct tt_CommandQueue tt_CommandQueue;
typedef struct tt_Event tt_Event;
typedef struct tt_CircularBuffer tt_CircularBuffer;
typedef struct tt_Semaphore tt_Semaphore;
typedef struct tt_Trace tt_Trace;

// Result codes
typedef enum {
    tt_Result_Success = 0,
    tt_Result_Error_InvalidArgument,
    tt_Result_Error_DeviceError,
    tt_Result_Error_NotImplemented,
    tt_Result_Error_CompilationFailed,
    tt_Result_Error_RuntimeError,
    tt_Result_Error_Unknown
} tt_Result;

// Data formats
typedef enum {
    tt_DataFormat_Float32,
    tt_DataFormat_Float16,
    tt_DataFormat_Float16_b,
    tt_DataFormat_Bfp8,
    tt_DataFormat_Bfp8_b
} tt_DataFormat;

// Event status
typedef enum {
    tt_EventStatus_Complete,
    tt_EventStatus_Running,
    tt_EventStatus_Error
} tt_EventStatus;

// Core coordinate
typedef struct {
    uint32_t x;
    uint32_t y;
} CoreCoord;

// Circular buffer config
typedef struct {
    uint32_t page_size;
    uint32_t buffer_type;
} tt_CircularBufferConfig;

// Data movement config
typedef struct {
    uint32_t processor;
    uint32_t noc;
} tt_DataMovementConfig;

// Device Management APIs
tt_Result tt_metal_QueryDevices(uint32_t *device_count, uint32_t *device_ids);
tt_Device *tt_metal_CreateDevice(int device_id);
tt_Result tt_metal_CloseDevice(tt_Device *device);

// Program APIs
tt_Program *tt_metal_CreateProgram(tt_Device *device);
tt_Result tt_metal_DestroyProgram(tt_Program *program);
tt_Result tt_metal_LoadFromLLVM(tt_Program *program, const char *llvm_ir);
tt_Result tt_metal_CompileProgram(tt_Program *program, const char *options);

// Buffer APIs
tt_Buffer *tt_metal_CreateBuffer(tt_Device *device, uint64_t size);
tt_Result tt_metal_DestroyBuffer(tt_Buffer *buffer);
tt_Result tt_metal_AssignGlobalBufferToProgram(tt_Program *program, tt_Buffer *buffer, const char *name);
uint64_t tt_metal_GetBufferAddress(tt_Buffer *buffer);

// Buffer Data Transfer APIs
tt_Result tt_metal_WriteToBuffer(tt_Buffer *buffer, const void *data, uint64_t size);
tt_Result tt_metal_ReadFromBuffer(tt_Buffer *buffer, void *data, uint64_t size);
tt_Result tt_metal_WriteToBufferOffset(tt_Buffer *buffer, const void *data, uint64_t offset, uint64_t size);
tt_Result tt_metal_ReadFromBufferOffset(tt_Buffer *buffer, void *data, uint64_t offset, uint64_t size);

// Kernel APIs
tt_Kernel *tt_metal_CreateKernel(tt_Program *program, const char *kernel_file, CoreCoord core, const tt_DataMovementConfig *config);
tt_Kernel *tt_metal_CreateKernelFromString(tt_Program *program, const char *kernel_source, const char *kernel_name);
tt_Result tt_metal_DestroyKernel(tt_Kernel *kernel);
tt_Result tt_metal_SetRuntimeArgs(tt_Program *program, const char *kernel_name, const tt_Buffer **args, int32_t num_args);

// Command Queue APIs
tt_CommandQueue *tt_metal_CreateCommandQueue(tt_Device *device);
tt_Result tt_metal_DestroyCommandQueue(tt_CommandQueue *command_queue);

// Execution APIs
tt_Result tt_metal_LaunchProgram(tt_Program *program);
tt_Result tt_metal_WaitForCompletion(tt_Program *program);
tt_Result tt_metal_Synchronize(tt_Device *device);

// Utility functions
uint32_t tt_metal_TileSize(tt_DataFormat data_format);

// Additional APIs
tt_Result tt_metal_EnqueueWriteBuffer(tt_CommandQueue *command_queue, tt_Buffer *buffer, bool blocking, uint64_t offset, uint64_t size, const void *data, tt_Event **event);
tt_Result tt_metal_EnqueueReadBuffer(tt_CommandQueue *command_queue, tt_Buffer *buffer, bool blocking, uint64_t offset, uint64_t size, void *data, tt_Event **event);
tt_Result tt_metal_EnqueueProgram(tt_CommandQueue *command_queue, tt_Program *program, tt_Event **event);
tt_Result tt_metal_EnqueueRecordEvent(tt_CommandQueue *command_queue, tt_Event **event);
tt_Result tt_metal_EnqueueWaitForEvent(tt_CommandQueue *command_queue, tt_Event *event);
tt_Result tt_metal_Finish(tt_CommandQueue *command_queue);
tt_Result tt_metal_DestroyEvent(tt_Event *event);
tt_EventStatus tt_metal_EventQuery(tt_Event *event);
tt_Result tt_metal_EventSynchronize(tt_Event *event);

// Circular Buffer APIs
tt_CircularBuffer *tt_metal_CreateCircularBuffer(tt_Program *program, CoreCoord core, const tt_CircularBufferConfig *config);
tt_Result tt_metal_DestroyCircularBuffer(tt_CircularBuffer *circular_buffer);
uint32_t tt_metal_CircularBufferPagesAvailableAtFront(tt_CircularBuffer *circular_buffer);
tt_Result tt_metal_CircularBufferWaitFront(tt_CircularBuffer *circular_buffer, uint32_t min_pages);
uint32_t tt_metal_CircularBufferPagesReservableAtBack(tt_CircularBuffer *circular_buffer);
tt_Result tt_metal_CircularBufferReserveBack(tt_CircularBuffer *circular_buffer, uint32_t pages);
tt_Result tt_metal_CircularBufferPushBack(tt_CircularBuffer *circular_buffer, const void *data, uint32_t pages);
tt_Result tt_metal_CircularBufferPopFront(tt_CircularBuffer *circular_buffer, void *data, uint32_t pages);

// Semaphore APIs
tt_Semaphore *tt_metal_CreateSemaphore(tt_Device *device, uint32_t initial_value);
tt_Result tt_metal_DestroySemaphore(tt_Semaphore *semaphore);
tt_Result tt_metal_SemaphoreSet(tt_Semaphore *semaphore, uint32_t value);
tt_Result tt_metal_SemaphoreWait(tt_Semaphore *semaphore, uint32_t value);
tt_Result tt_metal_SemaphoreIncrement(tt_Semaphore *semaphore);

// Trace APIs
tt_Result tt_metal_BeginTraceCapture(tt_Device *device, const char *trace_name);
tt_Result tt_metal_EndTraceCapture(tt_Device *device, tt_Trace **trace);
tt_Result tt_metal_ReplayTrace(tt_Device *device, tt_Trace *trace);
tt_Result tt_metal_ReleaseTrace(tt_Trace *trace);
tt_Result tt_metal_EnqueueTrace(tt_CommandQueue *command_queue, tt_Trace *trace, tt_Event **event);
tt_Result tt_metal_LoadTrace(tt_Device *device, const char *trace_file, tt_Trace **trace);
tt_Result tt_metal_LightMetalBeginCapture(tt_Device *device, const char *capture_name);
tt_Result tt_metal_LightMetalEndCapture(tt_Device *device);
tt_Result tt_metal_DumpDeviceProfileResults(tt_Device *device, const char *output_file);
tt_Result tt_metal_GetCompileTimeArgValue(tt_Program *program, const char *kernel_name, const char *arg_name, void *value, size_t value_size);

// Additional debugging function
tt_Result tt_metal_GetKernelInfo(tt_Kernel *kernel, char *info_buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // TT_METAL_WRAPPER_H