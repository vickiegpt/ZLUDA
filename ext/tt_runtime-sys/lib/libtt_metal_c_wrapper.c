// C wrapper stub functions for TT Metal
#include <stdint.h>
#include <stdlib.h>

// Simple stub implementations of the C functions that the Rust code expects
void* tt_metal_CreateDevice(int device_id) {
    return malloc(1); // Return dummy pointer
}

int tt_metal_CloseDevice(void* device) {
    if (device) free(device);
    return 0; // Success
}

void* tt_metal_CreateProgram(void* device) {
    return malloc(1); // Return dummy pointer
}

int tt_metal_DestroyProgram(void* program) {
    if (program) free(program);
    return 0; // Success
}

void* tt_metal_CreateBuffer(void* device, uint64_t size) {
    return malloc(1); // Return dummy pointer
}

int tt_metal_DestroyBuffer(void* buffer) {
    if (buffer) free(buffer);
    return 0; // Success
}

int tt_metal_LoadFromLLVM(void* program, const char* llvm_ir) {
    return 0; // Success stub
}

void* tt_metal_CreateKernel(void* program, const char* kernel_name) {
    return malloc(1); // Return dummy pointer
}

int tt_metal_SetRuntimeArgs(void* program, const char* kernel_name, void** args, int32_t num_args) {
    return 0; // Success stub
}

int tt_metal_LaunchProgram(void* program) {
    return 0; // Success stub
}

int tt_metal_WaitForCompletion(void* program) {
    return 0; // Success stub
}

int tt_metal_WriteToBuffer(void* buffer, const void* data, uint64_t size) {
    return 0; // Success stub
}

int tt_metal_ReadFromBuffer(void* buffer, void* data, uint64_t size) {
    return 0; // Success stub
}