/**
 * Level Zero Runner for ZLUDA
 * C interface for running SPIR-V kernels using Level Zero API
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Status codes returned by the ze_runner
 */
typedef enum {
  ZE_RUNNER_SUCCESS = 0,            // Operation was successful
  ZE_RUNNER_ERROR_INIT = 1,         // Failed to initialize Level Zero
  ZE_RUNNER_ERROR_NO_DRIVER = 2,    // No Level Zero driver found
  ZE_RUNNER_ERROR_NO_DEVICE = 3,    // No Level Zero device found
  ZE_RUNNER_ERROR_CONTEXT = 4,      // Failed to create context
  ZE_RUNNER_ERROR_COMMAND_LIST = 5, // Failed to create command list
  ZE_RUNNER_ERROR_MODULE = 6,       // Failed to create module
  ZE_RUNNER_ERROR_KERNEL = 7,       // Failed to create kernel
  ZE_RUNNER_ERROR_MEMORY = 8,       // Memory allocation or memory copy failed
  ZE_RUNNER_ERROR_KERNEL_EXECUTION = 9, // Kernel execution failed
  ZE_RUNNER_ERROR_SYNC = 10,            // Synchronization failed
  ZE_RUNNER_ERROR_INVALID_ARGUMENT = 11 // Invalid argument provided
} ze_runner_result_t;

/**
 * Run a SPIR-V kernel using Level Zero
 *
 * @param name          Kernel function name
 * @param spirv_data    SPIR-V binary data
 * @param spirv_size    Size of SPIR-V binary in bytes
 * @param input         Pointer to input data
 * @param input_size    Size of input data in bytes
 * @param output        Pointer to output buffer
 * @param output_size   Size of output buffer in bytes
 *
 * @return Status code indicating success or failure
 */
ze_runner_result_t run_spirv_kernel(const char *name, const void *spirv_data,
                                    size_t spirv_size, const void *input,
                                    size_t input_size, void *output,
                                    size_t output_size);

#ifdef __cplusplus
}
#endif