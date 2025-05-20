/**
 * Level Zero API 头文件
 * 包含最基本的定义以满足ze_runner的编译需求
 *
 * 注意：这是一个简化版本，仅用于示例目的
 * 实际使用中应替换为完整的Level Zero头文件
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 结构类型定义
typedef enum _ze_structure_type_t {
  ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0x1,
  ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x2,
  ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x3,
  ZE_STRUCTURE_TYPE_MODULE_DESC = 0x4,
  ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x5,
  // 其他结构类型...
} ze_structure_type_t;

// 结果码定义
typedef enum _ze_result_t {
  ZE_RESULT_SUCCESS = 0,
  ZE_RESULT_ERROR_UNINITIALIZED = 1,
  ZE_RESULT_ERROR_DEVICE_LOST = 2,
  ZE_RESULT_ERROR_INVALID_ARGUMENT = 3,
  ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY = 4,
  ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY = 5,
  ZE_RESULT_ERROR_MODULE_BUILD_FAILURE = 6,
  ZE_RESULT_ERROR_INVALID_KERNEL_NAME = 7,
  // 其他错误码...
} ze_result_t;

// 命令队列优先级
typedef enum _ze_command_queue_priority_t {
  ZE_COMMAND_QUEUE_PRIORITY_NORMAL = 0,
  ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW = 1,
  ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH = 2,
} ze_command_queue_priority_t;

// 命令队列模式
typedef enum _ze_command_queue_mode_t {
  ZE_COMMAND_QUEUE_MODE_DEFAULT = 0,
  ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1,
  ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2,
} ze_command_queue_mode_t;

// 模块格式
typedef enum _ze_module_format_t {
  ZE_MODULE_FORMAT_IL_SPIRV = 0,
} ze_module_format_t;

// 句柄定义
typedef struct _ze_driver_handle_t *ze_driver_handle_t;
typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;
typedef struct _ze_command_list_handle_t *ze_command_list_handle_t;
typedef struct _ze_module_handle_t *ze_module_handle_t;
typedef struct _ze_module_build_log_handle_t *ze_module_build_log_handle_t;
typedef struct _ze_kernel_handle_t *ze_kernel_handle_t;
typedef struct _ze_event_handle_t *ze_event_handle_t;

// 描述符结构体
typedef struct _ze_context_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
} ze_context_desc_t;

typedef struct _ze_command_queue_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t ordinal;
  uint32_t index;
  uint32_t flags;
  ze_command_queue_mode_t mode;
  ze_command_queue_priority_t priority;
} ze_command_queue_desc_t;

typedef struct _ze_device_mem_alloc_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
  uint32_t ordinal;
} ze_device_mem_alloc_desc_t;

typedef struct _ze_module_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  ze_module_format_t format;
  size_t inputSize;
  const uint8_t *pInputModule;
  const char *pBuildFlags;
  const void *pConstants;
} ze_module_desc_t;

typedef struct _ze_kernel_desc_t {
  ze_structure_type_t stype;
  const void *pNext;
  uint32_t flags;
  const char *pKernelName;
} ze_kernel_desc_t;

typedef struct _ze_group_count_t {
  uint32_t groupCountX;
  uint32_t groupCountY;
  uint32_t groupCountZ;
} ze_group_count_t;

// API 函数原型
ze_result_t zeInit(uint32_t flags);
ze_result_t zeDriverGet(uint32_t *pCount, ze_driver_handle_t *phDrivers);
ze_result_t zeDeviceGet(ze_driver_handle_t hDriver, uint32_t *pCount,
                        ze_device_handle_t *phDevices);
ze_result_t zeContextCreate(ze_driver_handle_t hDriver,
                            const ze_context_desc_t *desc,
                            ze_context_handle_t *phContext);
ze_result_t zeContextDestroy(ze_context_handle_t hContext);
ze_result_t
zeCommandListCreateImmediate(ze_context_handle_t hContext,
                             ze_device_handle_t hDevice,
                             const ze_command_queue_desc_t *altdesc,
                             ze_command_list_handle_t *phCommandList);
ze_result_t zeCommandListDestroy(ze_command_list_handle_t hCommandList);
ze_result_t zeModuleCreate(ze_context_handle_t hContext,
                           ze_device_handle_t hDevice,
                           const ze_module_desc_t *desc,
                           ze_module_handle_t *phModule,
                           ze_module_build_log_handle_t *phBuildLog);
ze_result_t zeModuleDestroy(ze_module_handle_t hModule);
ze_result_t
zeModuleBuildLogGetString(ze_module_build_log_handle_t hModuleBuildLog,
                          size_t *pSize, char *pBuildLog);
ze_result_t
zeModuleBuildLogDestroy(ze_module_build_log_handle_t hModuleBuildLog);
ze_result_t zeKernelCreate(ze_module_handle_t hModule,
                           const ze_kernel_desc_t *desc,
                           ze_kernel_handle_t *phKernel);
ze_result_t zeKernelDestroy(ze_kernel_handle_t hKernel);
ze_result_t zeKernelSetGroupSize(ze_kernel_handle_t hKernel,
                                 uint32_t groupSizeX, uint32_t groupSizeY,
                                 uint32_t groupSizeZ);
ze_result_t zeKernelSetArgumentValue(ze_kernel_handle_t hKernel,
                                     uint32_t argIndex, size_t argSize,
                                     const void *pArgValue);
ze_result_t zeMemAllocDevice(ze_context_handle_t hContext,
                             const ze_device_mem_alloc_desc_t *device_desc,
                             size_t size, size_t alignment,
                             ze_device_handle_t hDevice, void **pptr);
ze_result_t zeMemFree(ze_context_handle_t hContext, void *ptr);
ze_result_t zeCommandListAppendMemoryCopy(ze_command_list_handle_t hCommandList,
                                          void *dstptr, const void *srcptr,
                                          size_t size,
                                          ze_event_handle_t hSignalEvent,
                                          uint32_t numWaitEvents,
                                          ze_event_handle_t *phWaitEvents);
ze_result_t zeCommandListAppendBarrier(ze_command_list_handle_t hCommandList,
                                       ze_event_handle_t hSignalEvent,
                                       uint32_t numWaitEvents,
                                       ze_event_handle_t *phWaitEvents);
ze_result_t zeCommandListAppendLaunchKernel(
    ze_command_list_handle_t hCommandList, ze_kernel_handle_t hKernel,
    const ze_group_count_t *pLaunchFuncArgs, ze_event_handle_t hSignalEvent,
    uint32_t numWaitEvents, ze_event_handle_t *phWaitEvents);
ze_result_t zeCommandListHostSynchronize(ze_command_list_handle_t hCommandList,
                                         uint64_t timeout);

#ifdef __cplusplus
}
#endif