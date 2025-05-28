#!/bin/bash
set -e
LLVM_IR_FILE="$1"
SPIRV_FILE="$2"
STRIP_NAME=$(basename "$LLVM_IR_FILE" .ll)
# Disassemble the LLVM IR
llvm-dis-18 "$LLVM_IR_FILE" -o /tmp/temp$STRIP_NAME.ll

# Fix address spaces - ensure all global variables have explicit address space 0
sed -i -E 's/@_([0-9]+) = internal global/@_\1 = internal addrspace(1) global/g' /tmp/temp$STRIP_NAME.ll
sed -i 's/addrspace(5)/addrspace(0)/g' /tmp/temp$STRIP_NAME.ll
sed -i 's/addrspace(4) byref/addrspace(1) byref/g' /tmp/temp$STRIP_NAME.ll

# 同时也需要修改加载这些参数的指令
sed -i 's/load i64, ptr addrspace(4) %"\([0-9]\+\)", align/load i64, ptr addrspace(1) %"\1", align/g' /tmp/temp$STRIP_NAME.ll

# 更新元数据中的函数签名（如果有）
sed -i 's/!0 = !{.*ptr addrspace(4)/!0 = !{ptr @vector_extract, !"vector_extract"}/g' /tmp/temp$STRIP_NAME.ll
sed -i 's/amdgpu_kernel/spir_kernel/g' /tmp/temp$STRIP_NAME.ll
# 为inttoptr到addrspace(1)的每个实例创建新的变量序列
sed -i 's/%"\([0-9]\+\)" = inttoptr \(.*\) to ptr addrspace(1)/%"\1_tmp" = inttoptr \2 to ptr addrspace(4)\n  %"\1" = addrspacecast ptr addrspace(4) %"\1_tmp" to ptr addrspace(1)/g' /tmp/temp$STRIP_NAME.ll
# Reassemble and convert to SPIR-V
llvm-as-18 /tmp/temp$STRIP_NAME.ll -o /tmp/temp$STRIP_NAME.bc
llvm-spirv-18 /tmp/temp$STRIP_NAME.bc -o "$SPIRV_FILE"
