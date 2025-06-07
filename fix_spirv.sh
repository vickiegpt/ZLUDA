#!/bin/bash
set -e
LLVM_IR_FILE="$1"
SPIRV_FILE="$2"
STRIP_NAME=$(basename "$LLVM_IR_FILE" .ll)

# Create a temporary file
TEMP_LL_FILE="/tmp/temp$STRIP_NAME.ll"

# Check if the input file is already in text format
if [[ "$LLVM_IR_FILE" == *.ll ]]; then
  # Already in text format, just copy it
  cp "$LLVM_IR_FILE" "$TEMP_LL_FILE"
else
  # Disassemble the LLVM IR
  llvm-dis-20 "$LLVM_IR_FILE" -o "$TEMP_LL_FILE"
fi
# Fix call
# Fix address spaces - ensure all global variables have explicit address space 0
sed -i -E 's/@_([0-9]+) = internal global/@_\1 = internal addrspace(1) global/g' "$TEMP_LL_FILE"
sed -i 's/addrspace(5)/addrspace(0)/g' "$TEMP_LL_FILE"
sed -i 's/addrspace(4) byref/addrspace(1) byref/g' "$TEMP_LL_FILE"

# 修复uinc_wrap原子操作 - 将其替换为add操作
sed -i 's/atomicrmw uinc_wrap/atomicrmw add/g' "$TEMP_LL_FILE" 

# Update metadata for function signatures
sed -i 's/amdgpu_kernel/spir_kernel/g' "$TEMP_LL_FILE"

# Fix inttoptr to addrspace(1) conversions
sed -i 's/%"\([0-9]\+\)" = inttoptr \(.*\) to ptr addrspace(1)/%"\1_tmp" = inttoptr \2 to ptr addrspace(4)\n  %"\1" = addrspacecast ptr addrspace(4) %"\1_tmp" to ptr addrspace(1)/g' "$TEMP_LL_FILE"

# Fix inttoptr with non-quoted variable names (%2 instead of %"2")
sed -i 's/\(%[0-9]\+\) = inttoptr \(.*\) to ptr$/\1 = inttoptr \2 to ptr addrspace(4)/g' "$TEMP_LL_FILE"

# Fix addrspacecast instructions for non-quoted variables
sed -i 's/\(%"[0-9]\+"\) = addrspacecast ptr \(%[0-9]\+\) to ptr/\1 = addrspacecast ptr addrspace(4) \2 to ptr/g' "$TEMP_LL_FILE"

# Then find all addrspacecast from ptr to ptr addrspace(1) and replace with double cast through addrspace(0)
sed -i 's/\(%"[0-9]\+"\) = addrspacecast ptr \(%[0-9]\+\) to ptr addrspace(1)/\1_gen = addrspacecast ptr addrspace(0) \2 to ptr addrspace(0)\n  \1 = addrspacecast ptr addrspace(0) \1_gen to ptr addrspace(1)/g' "$TEMP_LL_FILE"

sed -i 's/\(%[0-9]\+\) = addrspacecast ptr addrspace(3) \(%"[0-9]\+"\) to ptr\n\(%[0-9]\+\) = addrspacecast ptr addrspace(3) \(%"[0-9]\+"\) to ptr/\1 = addrspacecast ptr addrspace(3) \2 to ptr\n\3 = addrspacecast ptr addrspace(3) \4 to ptr/g' "$TEMP_LL_FILE"
sed -i 's/\(%[0-9]\+\) = addrspacecast ptr addrspace(3) \(%"[0-9]\+"\) to ptr/\1 = addrspacecast ptr addrspace(3) \2 to ptr addrspace(4)/g' "$TEMP_LL_FILE"

# Fix non-quoted variable inttoptr (already moved up to execute earlier)
# sed -i 's/\(%[0-9]\+\) = inttoptr \(.*\) to ptr$/\1 = inttoptr \2 to ptr addrspace(4)/g' "$TEMP_LL_FILE"
sed -i 's/%"75" = addrspacecast ptr addrspace(4) %2 to ptr addrspace(4)/%"75" = addrspacecast ptr %2 to ptr addrspace(4)/g' "$TEMP_LL_FILE"

# Reassemble
TEMP_BC_FILE="/tmp/temp$STRIP_NAME.bc"
llvm-as-20 "$TEMP_LL_FILE" -o "$TEMP_BC_FILE"

# 将库文件转换为文本格式，避免属性组不兼容问题
LIB_FILE="/home/try/Documents/ZLUDA/ptx/lib/zluda_ptx_ze_impl.bc"

# 尝试反汇编库文件，如果失败则跳过合并步骤
COMBINED_BC_FILE="/tmp/combined$STRIP_NAME.bc"
echo /opt/intel/oneapi/2025.1/bin/compiler/llvm-link "$LIB_FILE" "$TEMP_BC_FILE" -o "$COMBINED_BC_FILE" 

/opt/intel/oneapi/2025.1/bin/compiler/llvm-link "$LIB_FILE" "$TEMP_BC_FILE" -o "$COMBINED_BC_FILE" 
# 使用合并后的文件生成SPIR-V
/opt/intel/oneapi/2025.1/bin/compiler/llvm-spirv "$COMBINED_BC_FILE" -o "$SPIRV_FILE"  --spirv-ext=+all,-SPV_KHR_untyped_pointers --spirv-debug 
