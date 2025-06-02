#include "ze_runner.h"
#include "../level-zero/ze_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义缺少的常量 - 如果您的Level Zero版本中没有显式定义这些常量，我们自己定义它们
// 注意：确保这些值与实际的Level Zero API兼容
#ifndef ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY
#define ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY 0x1
#endif

// 调试信息打印
#define DEBUG_PRINT(fmt, ...) printf("ZE_RUNNER: " fmt "\n", ##__VA_ARGS__)

// 检查Level Zero API调用
#define CHECK_ZE(expr)                                                   \
    do {                                                                \
        ze_result_t _result = (expr);                                   \
        if (_result != ZE_RESULT_SUCCESS) {                             \
            DEBUG_PRINT("Level Zero API error: %d at %s:%d", _result, __FILE__, __LINE__); \
            return _result == ZE_RESULT_SUCCESS ? ZE_RUNNER_SUCCESS : (ze_runner_result_t)(_result + 100); \
        }                                                               \
    } while (0)

// 主函数：运行SPIR-V内核
ze_runner_result_t run_spirv_kernel(
    const char* name,
    const void* spirv_data,
    size_t spirv_size,
    const void* input,
    size_t input_size,
    void* output,
    size_t output_size
) {
    DEBUG_PRINT("Starting with kernel: %s", name);
    DEBUG_PRINT("SPIR-V size: %zu bytes", spirv_size);
    DEBUG_PRINT("Input size: %zu bytes, Output size: %zu bytes", input_size, output_size);

    // 参数检查
    if (!name || !spirv_data || spirv_size == 0 || 
        !input || input_size == 0 || 
        !output || output_size == 0) {
        DEBUG_PRINT("Invalid arguments provided");
        return ZE_RUNNER_ERROR_INVALID_ARGUMENT;
    }

    // 初始化Level Zero
    CHECK_ZE(zeInit(0));
    DEBUG_PRINT("Level Zero initialized successfully");

    // 获取驱动程序
    uint32_t driver_count = 0;
    CHECK_ZE(zeDriverGet(&driver_count, NULL));
    
    if (driver_count == 0) {
        DEBUG_PRINT("No drivers found");
        return ZE_RUNNER_ERROR_NO_DRIVER;
    }
    
    ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(driver_count * sizeof(ze_driver_handle_t));
    if (!drivers) {
        DEBUG_PRINT("Failed to allocate memory for drivers");
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    CHECK_ZE(zeDriverGet(&driver_count, drivers));
    DEBUG_PRINT("Found %u driver(s)", driver_count);
    
    // 获取设备
    uint32_t device_count = 0;
    CHECK_ZE(zeDeviceGet(drivers[0], &device_count, NULL));
    
    if (device_count == 0) {
        DEBUG_PRINT("No devices found");
        free(drivers);
        return ZE_RUNNER_ERROR_NO_DEVICE;
    }
    
    ze_device_handle_t* devices = (ze_device_handle_t*)malloc(device_count * sizeof(ze_device_handle_t));
    if (!devices) {
        DEBUG_PRINT("Failed to allocate memory for devices");
        free(drivers);
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    CHECK_ZE(zeDeviceGet(drivers[0], &device_count, devices));
    DEBUG_PRINT("Found %u device(s)", device_count);
    
    // 创建上下文
    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0
    };
    
    ze_context_handle_t context = NULL;
    CHECK_ZE(zeContextCreate(drivers[0], &context_desc, &context));
    DEBUG_PRINT("Context created successfully");
    
    // 创建命令队列描述符
    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = NULL,
        .ordinal = 0,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    
    // 创建即时命令列表
    ze_command_list_handle_t command_list = NULL;
    ze_result_t cmd_result = zeCommandListCreateImmediate(
        context, 
        devices[0], 
        &queue_desc, 
        &command_list
    );
    
    if (cmd_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to create command list: %d", cmd_result);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_COMMAND_LIST;
    }
    
    DEBUG_PRINT("Command list created successfully");
    
    // 设置编译标志，启用原子操作支持并禁用可能干扰原子操作的优化
    const char* build_flags = "-ze-intel-enable-atomics -ze-opt-disable -ze-intel-enable-spirv-linkage -ze-intel-has-buffer-offset-impl -ze-intel-enable-persistent-usc-cache -ze-ext=intel-subgroups -ze-ext=ze_ext_subgroups";
    
    // 创建模块
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .pNext = NULL,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spirv_size,
        .pInputModule = spirv_data,
        .pBuildFlags = build_flags,
        .pConstants = NULL
    };
    
    ze_module_handle_t module = NULL;
    ze_module_build_log_handle_t build_log = NULL;
    ze_result_t mod_result = zeModuleCreate(
        context, 
        devices[0], 
        &module_desc, 
        &module, 
        &build_log
    );
    
    // 检查模块创建结果
    if (mod_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to create module: %d", mod_result);
        
        // 获取构建日志
        if (build_log) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(build_log, &log_size, NULL);
            
            if (log_size > 0) {
                char* log_text = (char*)malloc(log_size);
                if (log_text) {
                    zeModuleBuildLogGetString(build_log, &log_size, log_text);
                    DEBUG_PRINT("Module build log: %s", log_text);
                    free(log_text);
                }
            }
            
            zeModuleBuildLogDestroy(build_log);
        }
        
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_MODULE;
    }
    
    DEBUG_PRINT("Module created successfully");
    
    // 创建内核
    ze_kernel_desc_t kernel_desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pNext = NULL,
        .flags = ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY,  // 尝试使用显式的内存可见性
        .pKernelName = name
    };
    
    ze_kernel_handle_t kernel = NULL;
    ze_result_t kern_result = zeKernelCreate(module, &kernel_desc, &kernel);
    
    if (kern_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to create kernel: %d", kern_result);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_KERNEL_EXECUTION;
    }
    
    DEBUG_PRINT("Kernel created successfully");
    
    // 设置内核组大小
    CHECK_ZE(zeKernelSetGroupSize(kernel, 1, 1, 1));
    
    // 分配设备内存
    ze_device_mem_alloc_desc_t mem_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
        .ordinal = 0
    };
    
    // 输入缓冲区
    void* d_input = NULL;
    ze_result_t alloc_result = zeMemAllocDevice(
        context,
        &mem_desc,
        input_size,
        1,  // alignment
        devices[0],
        &d_input
    );
    
    if (alloc_result != ZE_RESULT_SUCCESS || d_input == NULL) {
        DEBUG_PRINT("Failed to allocate input memory: %d", alloc_result);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    // 输出缓冲区
    void* d_output = NULL;
    alloc_result = zeMemAllocDevice(
        context,
        &mem_desc,
        output_size,
        1,  // alignment
        devices[0],
        &d_output
    );
    
    if (alloc_result != ZE_RESULT_SUCCESS || d_output == NULL) {
        DEBUG_PRINT("Failed to allocate output memory: %d", alloc_result);
        zeMemFree(context, d_input);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    DEBUG_PRINT("Device memory allocated: input=%p (%zu bytes), output=%p (%zu bytes)",
               d_input, input_size, d_output, output_size);
    
    // 复制输入数据到设备
    ze_result_t copy_result = zeCommandListAppendMemoryCopy(
        command_list,
        d_input,
        input,
        input_size,
        NULL,  // no event
        0,     // no wait events
        NULL   // no signal event
    );
    
    if (copy_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to copy input data to device: %d", copy_result);
        zeMemFree(context, d_output);
        zeMemFree(context, d_input);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    // 设置内核参数
    CHECK_ZE(zeKernelSetArgumentValue(kernel, 0, sizeof(d_input), &d_input));
    CHECK_ZE(zeKernelSetArgumentValue(kernel, 1, sizeof(d_output), &d_output));
    
    // 启动内核
    ze_group_count_t launch_args = {
        .groupCountX = 1,
        .groupCountY = 1,
        .groupCountZ = 1
    };
    
    DEBUG_PRINT("Launching kernel");
    ze_result_t launch_result = zeCommandListAppendLaunchKernel(
        command_list,
        kernel,
        &launch_args,
        NULL,  // no event
        0,     // no wait events
        NULL   // no signal event
    );
    
    if (launch_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to launch kernel: %d", launch_result);
        zeMemFree(context, d_output);
        zeMemFree(context, d_input);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_KERNEL_EXECUTION;
    }
    
    // 添加内存屏障
    CHECK_ZE(zeCommandListAppendBarrier(command_list, NULL, 0, NULL));
    
    // 复制输出数据回主机
    DEBUG_PRINT("Copying results back to host");
    copy_result = zeCommandListAppendMemoryCopy(
        command_list,
        output,
        d_output,
        output_size,
        NULL,  // no event
        0,     // no wait events
        NULL   // no signal event
    );
    
    if (copy_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Failed to copy output data to host: %d", copy_result);
        zeMemFree(context, d_output);
        zeMemFree(context, d_input);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_MEMORY;
    }
    
    // 同步命令列表
    DEBUG_PRINT("Synchronizing command list");
    ze_result_t sync_result = zeCommandListHostSynchronize(command_list, UINT64_MAX);
    
    if (sync_result != ZE_RESULT_SUCCESS) {
        DEBUG_PRINT("Command list synchronization failed: %d", sync_result);
        zeMemFree(context, d_output);
        zeMemFree(context, d_input);
        zeKernelDestroy(kernel);
        zeModuleDestroy(module);
        zeCommandListDestroy(command_list);
        zeContextDestroy(context);
        free(devices);
        free(drivers);
        return ZE_RUNNER_ERROR_SYNC;
    }
    
    // 清理资源
    DEBUG_PRINT("Cleaning up resources");
    zeMemFree(context, d_output);
    zeMemFree(context, d_input);
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    zeCommandListDestroy(command_list);
    zeContextDestroy(context);
    free(devices);
    free(drivers);
    
    DEBUG_PRINT("Kernel execution completed successfully");
    return ZE_RUNNER_SUCCESS;
} 