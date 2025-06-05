// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal_wrapper.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <thread>

// Include actual TT Metal headers
#include <hostdevcommon/common_values.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/trace.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

// Additional using declarations for convenience
using namespace tt::tt_metal;

extern "C" {

// Device Management APIs
tt_Result tt_metal_QueryDevices(uint32_t *device_count, uint32_t *device_ids) {
  try {
    size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
    
    if (device_count) {
      *device_count = static_cast<uint32_t>(num_devices);
    }
    
    if (device_ids && num_devices > 0) {
      for (size_t i = 0; i < num_devices; i++) {
        device_ids[i] = static_cast<uint32_t>(i);
      }
    }
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_DeviceError;
  }
}

tt_Device *tt_metal_CreateDevice(int device_id) {
  try {
    // Create a TT Metal device using the host API
    tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(
      static_cast<chip_id_t>(device_id),
      1,  // num_hw_cqs
      DEFAULT_L1_SMALL_SIZE,
      DEFAULT_TRACE_REGION_SIZE
    );
    
    if (!device) {
      return nullptr;
    }
    
    return reinterpret_cast<tt_Device *>(device);
  } catch (...) {
    return nullptr;
  }
}

tt_Result tt_metal_CloseDevice(tt_Device *device) {
  try {
    if (!device)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    bool success = tt::tt_metal::CloseDevice(tt_device);
    
    return success ? tt_Result_Success : tt_Result_Error_DeviceError;
  } catch (...) {
    return tt_Result_Error_DeviceError;
  }
}

// Program APIs
tt_Program *tt_metal_CreateProgram(tt_Device *device) {
  try {
    if (!device)
      return nullptr;
    
    // Create a new program using TT Metal API
    tt::tt_metal::Program* program = new tt::tt_metal::Program(tt::tt_metal::CreateProgram());
    
    return reinterpret_cast<tt_Program *>(program);
  } catch (...) {
    return nullptr;
  }
}

tt_Result tt_metal_DestroyProgram(tt_Program *program) {
  try {
    if (!program)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    delete tt_program;
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_LoadFromLLVM(tt_Program *program, const char *llvm_ir) {
  try {
    if (!program || !llvm_ir)
      return tt_Result_Error_InvalidArgument;
    // This would need to be implemented based on TT Metal's LLVM loading
    // capabilities
    return tt_Result_Error_NotImplemented;
  } catch (...) {
    return tt_Result_Error_CompilationFailed;
  }
}

tt_Result tt_metal_CompileProgram(tt_Program *program, const char *options) {
  try {
    if (!program)
      return tt_Result_Error_InvalidArgument;
    // Compilation is typically handled automatically in TT Metal
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_CompilationFailed;
  }
}

// Buffer APIs
tt_Buffer *tt_metal_CreateBuffer(tt_Device *device, uint64_t size) {
  try {
    if (!device)
      return nullptr;
    
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    // Create an interleaved buffer configuration
    tt::tt_metal::InterleavedBufferConfig config = {
      .device = tt_device,
      .size = size,
      .page_size = size,  // For simplicity, one page
      .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    
    // Create the buffer
    std::shared_ptr<tt::tt_metal::Buffer> buffer = tt::tt_metal::CreateBuffer(config);
    
    if (!buffer) {
      return nullptr;
    }
    
    // Store the shared_ptr in a wrapper to prevent it from being deleted
    auto* buffer_wrapper = new std::shared_ptr<tt::tt_metal::Buffer>(buffer);
    
    return reinterpret_cast<tt_Buffer *>(buffer_wrapper);
  } catch (...) {
    return nullptr;
  }
}

tt_Result tt_metal_DestroyBuffer(tt_Buffer *buffer) {
  try {
    if (!buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Delete the shared_ptr wrapper
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    delete buffer_wrapper;
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_AssignGlobalBufferToProgram(tt_Program *program,
                                               tt_Buffer *buffer,
                                               const char *name) {
  try {
    if (!program || !buffer || !name)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    
    // Assign the buffer to the program
    tt::tt_metal::AssignGlobalBufferToProgram(*buffer_wrapper, *tt_program);
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

uint64_t tt_metal_GetBufferAddress(tt_Buffer *buffer) {
  try {
    if (!buffer)
      return 0;
    
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    tt::tt_metal::Buffer* tt_buffer = buffer_wrapper->get();
    
    if (!tt_buffer)
      return 0;
    
    return static_cast<uint64_t>(tt_buffer->address());
  } catch (...) {
    return 0;
  }
}

// Buffer Data Transfer APIs
tt_Result tt_metal_WriteToBuffer(tt_Buffer *buffer, const void *data,
                                 uint64_t size) {
  try {
    if (!buffer || !data)
      return tt_Result_Error_InvalidArgument;
    
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    tt::tt_metal::Buffer* tt_buffer = buffer_wrapper->get();
    
    if (!tt_buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Get the device and its command queue
    tt::tt_metal::IDevice* device = tt_buffer->device();
    tt::tt_metal::CommandQueue& cq = device->command_queue(0);
    
    // Create a buffer region for the entire buffer
    tt::tt_metal::BufferRegion region(0, size);
    
    // Enqueue the write operation (blocking)
    cq.enqueue_write_buffer(*buffer_wrapper, const_cast<void*>(data), region, true);
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_ReadFromBuffer(tt_Buffer *buffer, void *data,
                                  uint64_t size) {
  try {
    if (!buffer || !data)
      return tt_Result_Error_InvalidArgument;
    
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    tt::tt_metal::Buffer* tt_buffer = buffer_wrapper->get();
    
    if (!tt_buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Get the device and its command queue
    tt::tt_metal::IDevice* device = tt_buffer->device();
    tt::tt_metal::CommandQueue& cq = device->command_queue(0);
    
    // Create a buffer region for the entire buffer
    tt::tt_metal::BufferRegion region(0, size);
    
    // Enqueue the read operation (blocking)
    cq.enqueue_read_buffer(*buffer_wrapper, data, region, true);
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_WriteToBufferOffset(tt_Buffer *buffer, const void *data,
                                       uint64_t offset, uint64_t size) {
  try {
    if (!buffer || !data)
      return tt_Result_Error_InvalidArgument;
    
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    tt::tt_metal::Buffer* tt_buffer = buffer_wrapper->get();
    
    if (!tt_buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Get the device and its command queue
    tt::tt_metal::IDevice* device = tt_buffer->device();
    tt::tt_metal::CommandQueue& cq = device->command_queue(0);
    
    // Create a buffer region with offset
    tt::tt_metal::BufferRegion region(offset, size);
    
    // Enqueue the write operation (blocking)
    cq.enqueue_write_buffer(*buffer_wrapper, const_cast<void*>(data), region, true);
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_ReadFromBufferOffset(tt_Buffer *buffer, void *data,
                                        uint64_t offset, uint64_t size) {
  try {
    if (!buffer || !data)
      return tt_Result_Error_InvalidArgument;
    
    auto* buffer_wrapper = reinterpret_cast<std::shared_ptr<tt::tt_metal::Buffer>*>(buffer);
    tt::tt_metal::Buffer* tt_buffer = buffer_wrapper->get();
    
    if (!tt_buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Get the device and its command queue
    tt::tt_metal::IDevice* device = tt_buffer->device();
    tt::tt_metal::CommandQueue& cq = device->command_queue(0);
    
    // Create a buffer region with offset
    tt::tt_metal::BufferRegion region(offset, size);
    
    // Enqueue the read operation (blocking)
    cq.enqueue_read_buffer(*buffer_wrapper, data, region, true);
    
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

// Kernel wrapper to store kernel handle
struct KernelWrapper {
  tt::tt_metal::KernelHandle handle;
  tt::tt_metal::Program* program;
  std::string name;
  
  KernelWrapper(tt::tt_metal::KernelHandle h, tt::tt_metal::Program* p, const std::string& n)
    : handle(h), program(p), name(n) {}
};

// Kernel APIs
tt_Kernel *tt_metal_CreateKernel(tt_Program *program, const char *kernel_file,
                                 CoreCoord core,
                                 const tt_DataMovementConfig *config) {
  try {
    printf("ZLUDA DEBUG: tt_metal_CreateKernel called with program=%p, "
           "kernel_file=%s, core=(%u,%u), config=%p\n",
           program, kernel_file ? kernel_file : "NULL", core.x, core.y, config);

    if (!program) {
      printf("ZLUDA ERROR: program is NULL\n");
      return nullptr;
    }
    if (!kernel_file) {
      printf("ZLUDA ERROR: kernel_file is NULL\n");
      return nullptr;
    }
    if (!config) {
      printf("ZLUDA ERROR: config is NULL\n");
      return nullptr;
    }

    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    
    // Convert CoreCoord to TT Metal CoreCoord
    tt::tt_metal::CoreCoord tt_core(static_cast<std::size_t>(core.x), static_cast<std::size_t>(core.y));
    
    // Create TT Metal DataMovementConfig
    tt::tt_metal::DataMovementConfig dm_config;
    dm_config.processor = static_cast<tt::tt_metal::DataMovementProcessor>(config->processor);
    dm_config.noc = static_cast<tt::tt_metal::NOC>(config->noc);
    
    // Create the kernel using TT Metal API
    tt::tt_metal::KernelHandle kernel_handle = tt::tt_metal::CreateKernel(
        *tt_program,
        std::string(kernel_file),
        tt_core,
        dm_config
    );
    
    // Create wrapper to store kernel info
    std::string kernel_name = kernel_file;
    size_t last_slash = kernel_name.rfind('/');
    if (last_slash != std::string::npos) {
      kernel_name = kernel_name.substr(last_slash + 1);
    }
    
    KernelWrapper* wrapper = new KernelWrapper(kernel_handle, tt_program, kernel_name);
    
    printf("ZLUDA DEBUG: Created kernel with handle %u for file %s at core (%u,%u)\n",
           kernel_handle, kernel_file, core.x, core.y);
    
    return reinterpret_cast<tt_Kernel *>(wrapper);
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_CreateKernel: %s\n", e.what());
    return nullptr;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_CreateKernel\n");
    return nullptr;
  }
}

tt_Kernel *tt_metal_CreateKernelFromString(tt_Program *program,
                                           const char *kernel_source,
                                           const char *kernel_name) {
  try {
    printf("ZLUDA DEBUG: tt_metal_CreateKernelFromString called with "
           "program=%p, kernel_name=%s\n",
           program, kernel_name ? kernel_name : "NULL");

    if (!program) {
      printf("ZLUDA ERROR: program is NULL\n");
      return nullptr;
    }
    if (!kernel_source) {
      printf("ZLUDA ERROR: kernel_source is NULL\n");
      return nullptr;
    }
    if (!kernel_name) {
      printf("ZLUDA ERROR: kernel_name is NULL\n");
      return nullptr;
    }

    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    
    // Default to core (0,0) for now - in a real implementation this should be configurable
    tt::tt_metal::CoreCoord tt_core(static_cast<std::size_t>(0), static_cast<std::size_t>(0));
    
    // Default data movement config
    tt::tt_metal::DataMovementConfig dm_config;
    
    // Create the kernel from string using TT Metal API
    tt::tt_metal::KernelHandle kernel_handle = tt::tt_metal::CreateKernelFromString(
        *tt_program,
        std::string(kernel_source),
        tt_core,
        dm_config
    );
    
    // Create wrapper to store kernel info
    KernelWrapper* wrapper = new KernelWrapper(kernel_handle, tt_program, std::string(kernel_name));
    
    printf("ZLUDA DEBUG: Created kernel '%s' with handle %u from source\n", 
           kernel_name, kernel_handle);

    return reinterpret_cast<tt_Kernel *>(wrapper);
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_CreateKernelFromString: %s\n",
           e.what());
    return nullptr;
  } catch (...) {
    printf(
        "ZLUDA ERROR: Unknown exception in tt_metal_CreateKernelFromString\n");
    return nullptr;
  }
}

tt_Result tt_metal_DestroyKernel(tt_Kernel *kernel) {
  try {
    if (!kernel) {
      printf("ZLUDA ERROR: kernel is NULL in DestroyKernel\n");
      return tt_Result_Error_InvalidArgument;
    }

    // Cast back to kernel wrapper
    KernelWrapper *wrapper = reinterpret_cast<KernelWrapper *>(kernel);

    printf("ZLUDA DEBUG: Destroying kernel '%s' with handle %u\n", 
           wrapper->name.c_str(), wrapper->handle);

    // Delete the wrapper (the kernel itself is managed by the program)
    delete wrapper;

    printf("ZLUDA DEBUG: Kernel destroyed successfully\n");
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_DestroyKernel: %s\n", e.what());
    return tt_Result_Error_Unknown;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_DestroyKernel\n");
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_SetRuntimeArgs(tt_Program *program, const char *kernel_name,
                                  const tt_Buffer **args, int32_t num_args) {
  try {
    printf("ZLUDA DEBUG: tt_metal_SetRuntimeArgs called with program=%p, "
           "kernel_name=%s, num_args=%d\n",
           program, kernel_name ? kernel_name : "NULL", num_args);

    if (!program) {
      printf("ZLUDA ERROR: program is NULL in SetRuntimeArgs\n");
      return tt_Result_Error_InvalidArgument;
    }
    if (!kernel_name) {
      printf("ZLUDA ERROR: kernel_name is NULL in SetRuntimeArgs\n");
      return tt_Result_Error_InvalidArgument;
    }
    if (num_args > 0 && !args) {
      printf("ZLUDA ERROR: args is NULL but num_args > 0 in SetRuntimeArgs\n");
      return tt_Result_Error_InvalidArgument;
    }

    // NOTE: This is a simplified implementation. In TT Metal, runtime args are set
    // per kernel handle and core, not by kernel name. We would need to:
    // 1. Find all kernels with the given name in the program
    // 2. Get their handles and cores
    // 3. Convert buffer addresses to uint32_t runtime args
    // 4. Call SetRuntimeArgs for each kernel/core combination
    
    // For now, just log what we would do
    printf("ZLUDA DEBUG: Would set %d runtime arguments for kernel '%s'\n",
           num_args, kernel_name);
    for (int32_t i = 0; i < num_args; i++) {
      if (args[i]) {
        uint64_t addr = tt_metal_GetBufferAddress(const_cast<tt_Buffer *>(args[i]));
        printf("ZLUDA DEBUG: Arg[%d]: buffer=%p, address=0x%lx\n", i, args[i], addr);
      } else {
        printf("ZLUDA DEBUG: Arg[%d]: NULL buffer\n", i);
      }
    }

    // This is a stub - proper implementation would require kernel handle tracking
    printf("ZLUDA WARNING: SetRuntimeArgs not fully implemented - kernel name lookup needed\n");
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_SetRuntimeArgs: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_SetRuntimeArgs\n");
    return tt_Result_Error_RuntimeError;
  }
}

// Command Queue APIs
tt_CommandQueue *tt_metal_CreateCommandQueue(tt_Device *device) {
  try {
    if (!device) {
      printf("ZLUDA ERROR: device is NULL in CreateCommandQueue\n");
      return nullptr;
    }

    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    // Get the device's default command queue (index 0)
    tt::tt_metal::CommandQueue* cq = &(tt_device->command_queue(0));
    
    printf("ZLUDA DEBUG: Returning command queue %p for device %p\n", cq, device);
    return reinterpret_cast<tt_CommandQueue *>(cq);
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_CreateCommandQueue: %s\n",
           e.what());
    return nullptr;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_CreateCommandQueue\n");
    return nullptr;
  }
}

tt_Result tt_metal_DestroyCommandQueue(tt_CommandQueue *command_queue) {
  try {
    if (!command_queue) {
      printf("ZLUDA ERROR: command_queue is NULL in DestroyCommandQueue\n");
      return tt_Result_Error_InvalidArgument;
    }

    // Command queues are owned by the device, so we don't delete them
    printf("ZLUDA DEBUG: Command queue %p reference released (owned by device)\n", command_queue);
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_DestroyCommandQueue: %s\n",
           e.what());
    return tt_Result_Error_Unknown;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_DestroyCommandQueue\n");
    return tt_Result_Error_Unknown;
  }
}

// Execution APIs
tt_Result tt_metal_LaunchProgram(tt_Program *program) {
  try {
    if (!program)
      return tt_Result_Error_InvalidArgument;
      
    // For now, this is just a placeholder since program execution
    // is typically done through EnqueueProgram in TT Metal
    printf("ZLUDA DEBUG: LaunchProgram called - use EnqueueProgram instead\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_WaitForCompletion(tt_Program *program) {
  try {
    if (!program)
      return tt_Result_Error_InvalidArgument;
      
    printf("ZLUDA DEBUG: WaitForCompletion called - programs complete through command queue\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_Synchronize(tt_Device *device) {
  try {
    if (!device)
      return tt_Result_Error_InvalidArgument;
      
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    // Synchronize all command queues on the device
    tt::tt_metal::CommandQueue& cq = tt_device->command_queue(0);
    cq.finish({});  // Finish with empty sub-device list means all sub-devices
    
    printf("ZLUDA DEBUG: Device %p synchronized\n", device);
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_DeviceError;
  }
}

// Utility functions
uint32_t tt_metal_TileSize(tt_DataFormat data_format) {
  switch (data_format) {
  case tt_DataFormat_Float32:
    return 32 * 32 * 4;
  case tt_DataFormat_Float16:
  case tt_DataFormat_Float16_b:
    return 32 * 32 * 2;
  case tt_DataFormat_Bfp8:
  case tt_DataFormat_Bfp8_b:
    return 32 * 32 * 1;
  default:
    return 0;
  }
}

// Execution APIs with proper handling
tt_Result tt_metal_EnqueueProgram(tt_CommandQueue *command_queue,
                                  tt_Program *program, tt_Event **event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!program)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing program %p on command queue %p\n", program,
           command_queue);

    tt::tt_metal::CommandQueue* cq = reinterpret_cast<tt::tt_metal::CommandQueue*>(command_queue);
    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    
    // Enqueue the program (non-blocking)
    cq->enqueue_program(*tt_program, false);

    // If the caller requested an event, create one
    if (event) {
      // For now, create a dummy event - in real implementation would use actual TT Metal events
      *event = reinterpret_cast<tt_Event *>(new char[1]);
      printf("ZLUDA DEBUG: Created event %p\n", *event);
    }

    printf("ZLUDA DEBUG: Program enqueued successfully\n");
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueProgram: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueProgram\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_Finish(tt_CommandQueue *command_queue) {
  try {
    if (!command_queue) {
      printf("ZLUDA ERROR: command_queue is NULL in Finish\n");
      return tt_Result_Error_InvalidArgument;
    }

    printf("ZLUDA DEBUG: Finishing command queue %p\n", command_queue);

    tt::tt_metal::CommandQueue* cq = reinterpret_cast<tt::tt_metal::CommandQueue*>(command_queue);
    cq->finish({});  // Finish with empty sub-device list

    printf("ZLUDA DEBUG: Command queue finished successfully\n");
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_Finish: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_Finish\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EnqueueRecordEvent(tt_CommandQueue *command_queue,
                                      tt_Event **event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!event)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing record event on command queue %p\n",
           command_queue);

    // Create a dummy event for now
    *event = reinterpret_cast<tt_Event *>(new char[1]);
    printf("ZLUDA DEBUG: Created event %p\n", *event);

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueRecordEvent: %s\n",
           e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueRecordEvent\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EnqueueWaitForEvent(tt_CommandQueue *command_queue,
                                       tt_Event *event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!event)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing wait for event %p on command queue %p\n",
           event, command_queue);

    // Stub implementation for now

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueWaitForEvent: %s\n",
           e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueWaitForEvent\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EnqueueReadBuffer(tt_CommandQueue *command_queue,
                                     tt_Buffer *buffer, bool blocking,
                                     uint64_t offset, uint64_t size, void *data,
                                     tt_Event **event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!buffer)
      return tt_Result_Error_InvalidArgument;
    if (!data)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing read buffer %p on command queue %p "
           "(offset=%lu, size=%lu, blocking=%s)\n",
           buffer, command_queue, offset, size, blocking ? "yes" : "no");

    // Stub implementation for now

    // If the caller requested an event
    if (event) {
      // Create a dummy event and set it as output parameter
      *event = reinterpret_cast<tt_Event *>(new char[1]);
      printf("ZLUDA DEBUG: Created event %p\n", *event);
    }

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueReadBuffer: %s\n",
           e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueReadBuffer\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EnqueueWriteBuffer(tt_CommandQueue *command_queue,
                                      tt_Buffer *buffer, bool blocking,
                                      uint64_t offset, uint64_t size,
                                      const void *data, tt_Event **event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!buffer)
      return tt_Result_Error_InvalidArgument;
    if (!data)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing write buffer %p on command queue %p "
           "(offset=%lu, size=%lu, blocking=%s)\n",
           buffer, command_queue, offset, size, blocking ? "yes" : "no");

    // Stub implementation for now

    // If the caller requested an event
    if (event) {
      // Create a dummy event and set it as output parameter
      *event = reinterpret_cast<tt_Event *>(new char[1]);
      printf("ZLUDA DEBUG: Created event %p\n", *event);
    }

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueWriteBuffer: %s\n",
           e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueWriteBuffer\n");
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EnqueueTrace(tt_CommandQueue *command_queue, tt_Trace *trace,
                                tt_Event **event) {
  try {
    if (!command_queue)
      return tt_Result_Error_InvalidArgument;
    if (!trace)
      return tt_Result_Error_InvalidArgument;

    printf("ZLUDA DEBUG: Enqueueing trace %p on command queue %p\n", trace,
           command_queue);

    // Stub implementation for now

    // If the caller requested an event
    if (event) {
      // Create a dummy event and set it as output parameter
      *event = reinterpret_cast<tt_Event *>(new char[1]);
      printf("ZLUDA DEBUG: Created event %p\n", *event);
    }

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EnqueueTrace: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EnqueueTrace\n");
    return tt_Result_Error_RuntimeError;
  }
}

// Circular Buffer APIs
tt_CircularBuffer *
tt_metal_CreateCircularBuffer(tt_Program *program, CoreCoord core,
                              const tt_CircularBufferConfig *config) {
  try {
    if (!program || !config)
      return nullptr;

    printf("ZLUDA DEBUG: Creating circular buffer for program %p at core "
           "(%u,%u)\n",
           program, core.x, core.y);

    tt::tt_metal::Program* tt_program = reinterpret_cast<tt::tt_metal::Program*>(program);
    tt::tt_metal::CoreCoord tt_core(static_cast<std::size_t>(core.x), static_cast<std::size_t>(core.y));
    
    // Create a CircularBufferConfig for TT Metal
    tt::tt_metal::CircularBufferConfig cb_config;
    cb_config.set_page_size(config->page_size);
    cb_config.set_buffer_type(static_cast<tt::tt_metal::BufferType>(config->buffer_type));
    
    // Create the circular buffer using TT Metal API
    tt::tt_metal::CBHandle cb_handle = tt::tt_metal::CreateCircularBuffer(
        *tt_program,
        tt_core,
        cb_config
    );
    
    // Store the handle in a wrapper
    auto* cb_wrapper = new tt::tt_metal::CBHandle(cb_handle);
    
    printf("ZLUDA DEBUG: Created circular buffer with handle %u\n", cb_handle);
    return reinterpret_cast<tt_CircularBuffer *>(cb_wrapper);
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in CreateCircularBuffer: %s\n", e.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

tt_Result tt_metal_DestroyCircularBuffer(tt_CircularBuffer *circular_buffer) {
  try {
    if (!circular_buffer)
      return tt_Result_Error_InvalidArgument;
    
    // Delete the handle wrapper (the actual circular buffer is managed by the program)
    auto* cb_handle = reinterpret_cast<tt::tt_metal::CBHandle*>(circular_buffer);
    delete cb_handle;
    
    printf("ZLUDA DEBUG: Circular buffer handle released\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

uint32_t tt_metal_CircularBufferPagesAvailableAtFront(
    tt_CircularBuffer *circular_buffer) {
  return 0;
}

tt_Result tt_metal_CircularBufferWaitFront(tt_CircularBuffer *circular_buffer,
                                           uint32_t min_pages) {
  return tt_Result_Error_NotImplemented;
}

uint32_t tt_metal_CircularBufferPagesReservableAtBack(
    tt_CircularBuffer *circular_buffer) {
  return 0;
}

tt_Result tt_metal_CircularBufferReserveBack(tt_CircularBuffer *circular_buffer,
                                             uint32_t pages) {
  return tt_Result_Error_NotImplemented;
}

tt_Result tt_metal_CircularBufferPushBack(tt_CircularBuffer *circular_buffer,
                                          const void *data, uint32_t pages) {
  return tt_Result_Error_NotImplemented;
}

tt_Result tt_metal_CircularBufferPopFront(tt_CircularBuffer *circular_buffer,
                                          void *data, uint32_t pages) {
  return tt_Result_Error_NotImplemented;
}

// Semaphore APIs
tt_Semaphore *tt_metal_CreateSemaphore(tt_Device *device,
                                       uint32_t initial_value) {
  try {
    if (!device)
      return nullptr;
    
    printf("ZLUDA DEBUG: Creating semaphore with initial value %u\n", initial_value);
    
    // In TT Metal, semaphores are created per-program, not per-device
    // For now, we'll create a placeholder that stores the initial value
    uint32_t* semaphore_value = new uint32_t(initial_value);
    
    printf("ZLUDA DEBUG: Created semaphore placeholder %p\n", semaphore_value);
    return reinterpret_cast<tt_Semaphore *>(semaphore_value);
  } catch (...) {
    return nullptr;
  }
}

tt_Result tt_metal_DestroySemaphore(tt_Semaphore *semaphore) {
  try {
    if (!semaphore)
      return tt_Result_Error_InvalidArgument;
    
    uint32_t* semaphore_value = reinterpret_cast<uint32_t*>(semaphore);
    delete semaphore_value;
    
    printf("ZLUDA DEBUG: Semaphore destroyed\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_SemaphoreSet(tt_Semaphore *semaphore, uint32_t value) {
  try {
    if (!semaphore)
      return tt_Result_Error_InvalidArgument;
    
    uint32_t* semaphore_value = reinterpret_cast<uint32_t*>(semaphore);
    *semaphore_value = value;
    
    printf("ZLUDA DEBUG: Semaphore set to %u\n", value);
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_SemaphoreWait(tt_Semaphore *semaphore, uint32_t value) {
  try {
    if (!semaphore)
      return tt_Result_Error_InvalidArgument;
    
    uint32_t* semaphore_value = reinterpret_cast<uint32_t*>(semaphore);
    
    // Simple wait implementation - in real TT Metal this would be hardware-level
    while (*semaphore_value < value) {
      // Busy wait - not efficient but matches the API
      std::this_thread::yield();
    }
    
    printf("ZLUDA DEBUG: Semaphore wait completed for value %u\n", value);
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_SemaphoreIncrement(tt_Semaphore *semaphore) {
  try {
    if (!semaphore)
      return tt_Result_Error_InvalidArgument;
    
    uint32_t* semaphore_value = reinterpret_cast<uint32_t*>(semaphore);
    (*semaphore_value)++;
    
    printf("ZLUDA DEBUG: Semaphore incremented to %u\n", *semaphore_value);
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

// Trace APIs
tt_Result tt_metal_BeginTraceCapture(tt_Device *device,
                                     const char *trace_name) {
  try {
    if (!device || !trace_name)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    printf("ZLUDA DEBUG: Beginning trace capture '%s' on device %p\n", trace_name, device);
    
    // Use TT Metal trace API - assuming trace ID 1 for simplicity
    tt_device->begin_trace(0, 1);  // cq_id=0, tid=1
    
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in BeginTraceCapture: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_EndTraceCapture(tt_Device *device, tt_Trace **trace) {
  try {
    if (!device)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    printf("ZLUDA DEBUG: Ending trace capture on device %p\n", device);
    
    // End the trace capture
    tt_device->end_trace(0, 1);  // cq_id=0, tid=1
    
    if (trace) {
      // Get the trace buffer and store it
      auto trace_buffer = tt_device->get_trace(1);  // tid=1
      *trace = reinterpret_cast<tt_Trace*>(new std::shared_ptr<tt::tt_metal::TraceBuffer>(trace_buffer));
      printf("ZLUDA DEBUG: Created trace handle %p\n", *trace);
    }
    
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in EndTraceCapture: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_ReplayTrace(tt_Device *device, tt_Trace *trace) {
  try {
    if (!device || !trace)
      return tt_Result_Error_InvalidArgument;
    
    tt::tt_metal::IDevice* tt_device = reinterpret_cast<tt::tt_metal::IDevice*>(device);
    
    printf("ZLUDA DEBUG: Replaying trace %p on device %p\n", trace, device);
    
    // Replay the trace - blocking on device and worker thread
    tt_device->replay_trace(0, 1, true, true);  // cq_id=0, tid=1, block_on_device=true, block_on_worker_thread=true
    
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in ReplayTrace: %s\n", e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_ReleaseTrace(tt_Trace *trace) {
  try {
    if (!trace)
      return tt_Result_Error_InvalidArgument;
    
    // Delete the shared_ptr wrapper
    auto* trace_buffer = reinterpret_cast<std::shared_ptr<tt::tt_metal::TraceBuffer>*>(trace);
    delete trace_buffer;
    
    printf("ZLUDA DEBUG: Trace %p released\n", trace);
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_LoadTrace(tt_Device *device, const char *trace_file,
                             tt_Trace **trace) {
  try {
    if (!device || !trace_file || !trace)
      return tt_Result_Error_InvalidArgument;
    
    printf("ZLUDA DEBUG: Loading trace from file '%s'\n", trace_file);
    
    // TT Metal doesn't have a direct load from file API, so this is not implemented
    printf("ZLUDA WARNING: LoadTrace from file not implemented in TT Metal\n");
    return tt_Result_Error_NotImplemented;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_LightMetalBeginCapture(tt_Device *device,
                                          const char *capture_name) {
  try {
    if (!device || !capture_name)
      return tt_Result_Error_InvalidArgument;
    
    printf("ZLUDA DEBUG: Beginning LightMetal capture '%s'\n", capture_name);
    
    // Light Metal capture is a higher-level API - for now just log
    printf("ZLUDA WARNING: LightMetal capture APIs not fully implemented\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_LightMetalEndCapture(tt_Device *device) {
  try {
    if (!device)
      return tt_Result_Error_InvalidArgument;
    
    printf("ZLUDA DEBUG: Ending LightMetal capture\n");
    return tt_Result_Success;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_DumpDeviceProfileResults(tt_Device *device,
                                            const char *output_file) {
  try {
    if (!device || !output_file)
      return tt_Result_Error_InvalidArgument;
    
    printf("ZLUDA DEBUG: Dumping device profile results to '%s'\n", output_file);
    
    // Profiling results dump - not directly available in TT Metal API
    printf("ZLUDA WARNING: DumpDeviceProfileResults not implemented\n");
    return tt_Result_Error_NotImplemented;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

tt_Result tt_metal_GetCompileTimeArgValue(tt_Program *program,
                                          const char *kernel_name,
                                          const char *arg_name, void *value,
                                          size_t value_size) {
  try {
    if (!program || !kernel_name || !arg_name || !value)
      return tt_Result_Error_InvalidArgument;
    
    printf("ZLUDA DEBUG: Getting compile-time arg '%s' for kernel '%s'\n", arg_name, kernel_name);
    
    // TT Metal doesn't expose compile-time args directly through the API
    printf("ZLUDA WARNING: GetCompileTimeArgValue not implemented\n");
    return tt_Result_Error_NotImplemented;
  } catch (...) {
    return tt_Result_Error_RuntimeError;
  }
}

// Additional debugging function to inspect kernel properties
tt_Result tt_metal_GetKernelInfo(tt_Kernel *kernel, char *info_buffer,
                                 size_t buffer_size) {
  try {
    if (!kernel) {
      printf("ZLUDA ERROR: kernel is NULL in GetKernelInfo\n");
      return tt_Result_Error_InvalidArgument;
    }
    if (!info_buffer || buffer_size == 0) {
      printf("ZLUDA ERROR: invalid info_buffer in GetKernelInfo\n");
      return tt_Result_Error_InvalidArgument;
    }

    // Cast back to kernel wrapper
    KernelWrapper *wrapper = reinterpret_cast<KernelWrapper *>(kernel);

    // Format kernel information
    int written = snprintf(
        info_buffer, buffer_size,
        "Kernel Handle: %u\n"
        "Kernel Name: %s\n"
        "Program: %p\n",
        wrapper->handle,
        wrapper->name.empty() ? "N/A" : wrapper->name.c_str(),
        wrapper->program);

    if (written < 0 || static_cast<size_t>(written) >= buffer_size) {
      printf("ZLUDA WARNING: kernel info truncated\n");
    }

    printf("ZLUDA DEBUG: Retrieved kernel info for kernel %p (Handle: %u)\n",
           wrapper, wrapper->handle);

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_GetKernelInfo: %s\n", e.what());
    return tt_Result_Error_Unknown;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_GetKernelInfo\n");
    return tt_Result_Error_Unknown;
  }
}

tt_Result tt_metal_DestroyEvent(tt_Event *event) {
  try {
    if (!event) {
      printf("ZLUDA ERROR: event is NULL in DestroyEvent\n");
      return tt_Result_Error_InvalidArgument;
    }

    printf("ZLUDA DEBUG: Destroying event %p\n", event);
    delete[] reinterpret_cast<char *>(event);

    printf("ZLUDA DEBUG: Event destroyed successfully\n");
    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_DestroyEvent: %s\n", e.what());
    return tt_Result_Error_Unknown;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_DestroyEvent\n");
    return tt_Result_Error_Unknown;
  }
}

tt_EventStatus tt_metal_EventQuery(tt_Event *event) {
  try {
    if (!event) {
      printf("ZLUDA ERROR: event is NULL in EventQuery\n");
      return tt_EventStatus_Error;
    }

    printf("ZLUDA DEBUG: Querying event %p\n", event);

    // For now, always return Complete
    return tt_EventStatus_Complete;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EventQuery: %s\n", e.what());
    return tt_EventStatus_Error;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EventQuery\n");
    return tt_EventStatus_Error;
  }
}

tt_Result tt_metal_EventSynchronize(tt_Event *event) {
  try {
    if (!event) {
      printf("ZLUDA ERROR: event is NULL in EventSynchronize\n");
      return tt_Result_Error_InvalidArgument;
    }

    printf("ZLUDA DEBUG: Synchronizing with event %p\n", event);

    // Stub implementation for now

    return tt_Result_Success;
  } catch (const std::exception &e) {
    printf("ZLUDA ERROR: Exception in tt_metal_EventSynchronize: %s\n",
           e.what());
    return tt_Result_Error_RuntimeError;
  } catch (...) {
    printf("ZLUDA ERROR: Unknown exception in tt_metal_EventSynchronize\n");
    return tt_Result_Error_RuntimeError;
  }
}

} // extern "C"