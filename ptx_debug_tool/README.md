# PTX Debug Tool - State Recovery and Debugging

This tool provides comprehensive debugging and state recovery capabilities for PTX (Parallel Thread Execution) programs compiled with ZLUDA. It maintains mappings from PTX source code to target architectures (SASS/AMD GCN/Intel SPIRV) and enables program state recovery at arbitrary execution points.

## Features

- **DWARF Debug Information**: Generates DWARF debug info for PTX compilation
- **Multi-Architecture Support**: Supports SASS, AMD GCN, and Intel SPIRV targets
- **State Recovery**: Recover PTX program state from target architecture debugging information
- **Interactive Debugging**: GDB-like command-line debugging interface
- **Breakpoint Management**: Set and manage breakpoints at PTX source locations
- **Variable Inspection**: Inspect variable values and memory state
- **Call Stack Tracing**: Track function calls and maintain call stack
- **Export Capabilities**: Export debug information for external debuggers

## Installation

```bash
cd ptx_debug_tool
cargo build --release
```

## Usage

### Compile PTX with Debug Information

```bash
./target/release/ptx-debug compile \
    --input examples/test_kernel.ptx \
    --output test_kernel.debug.json \
    --target amd_gcn \
    --verbose
```

### Analyze Debug Mappings

```bash
# Basic analysis
./target/release/ptx-debug analyze \
    --mapping-file test_kernel.debug.json

# Detailed analysis with specific line filter
./target/release/ptx-debug analyze \
    --mapping-file test_kernel.debug.json \
    --detailed \
    --line 42
```

### Interactive Debugging Session

```bash
./target/release/ptx-debug debug \
    --mapping-file test_kernel.debug.json \
    --breakpoint test_kernel.ptx:42:10
```

#### Debug Commands

- `help` - Show available commands
- `info` - Display current execution state
- `break <file>:<line>:<column>` - Set breakpoint
- `delete <id>` - Remove breakpoint
- `print <variable>` - Print variable value
- `backtrace` - Show call stack
- `recover <address>` - Find PTX location from target address
- `quit` - Exit debugger

### State Recovery from Crash

```bash
./target/release/ptx-debug recover \
    --mapping-file test_kernel.debug.json \
    --address 0x1000 \
    --target amd_gcn \
    --memory-dump crash.dump
```

### Export Debug Information

```bash
# Export for GDB
./target/release/ptx-debug export \
    --mapping-file test_kernel.debug.json \
    --format gdb \
    --output debug.gdb

# Export for VS Code
./target/release/ptx-debug export \
    --mapping-file test_kernel.debug.json \
    --format vscode \
    --output .vscode/launch.json
```

## Architecture Overview

### Debug Information Flow

```
PTX Source → PTX Parser → Debug-Aware Compiler → Target Code + DWARF
                                                      ↓
Debug Mappings ← State Recovery ← Runtime Debugging ← Target Execution
```

### Key Components

1. **PtxDwarfBuilder**: Generates DWARF debug information during compilation
2. **PtxStateRecoveryManager**: Manages debugging state and breakpoints
3. **DebugAwarePtxContext**: Integrates debug info into compilation pipeline
4. **TargetDebugInfo**: Target-specific debug information handling

### Debug Mapping Structure

Each debug mapping entry contains:
- PTX source location (file, line, column)
- Target architecture instructions
- Variable location mappings
- Scope and call frame information

```json
{
  "ptx_location": {
    "file": "kernel.ptx",
    "line": 42,
    "column": 10,
    "instruction_offset": 100
  },
  "target_instructions": [
    {
      "AmdGcn": {
        "instruction": "v_add_f32 v0, v1, v2",
        "address": 4096,
        "register_state": {}
      }
    }
  ],
  "variable_mappings": {
    "tid": {"Register": "v10"},
    "result": {"Memory": {"address": 8192, "size": 4}}
  },
  "scope_id": 1
}
```

## Examples

### Example 1: Basic Kernel Debugging

```ptx
.entry vector_add(.param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr)
{
    .reg .u32 %tid;
    .reg .f32 %f<3>;
    
    mov.u32 %tid, %tid.x;           // Breakpoint here
    // ... rest of kernel
}
```

Set breakpoint and debug:
```bash
# Compile with debug info
ptx-debug compile -i kernel.ptx -o kernel.debug.json

# Start debugging
ptx-debug debug -m kernel.debug.json -b kernel.ptx:8:5

# In debugger:
(ptx-debug) print tid
tid = Integer(123)
(ptx-debug) info
Current Location: kernel.ptx:8:5
Thread ID: (123, 0, 0)
...
```

### Example 2: Crash Recovery

When a GPU kernel crashes at address `0x2000`:

```bash
# Recover PTX location
ptx-debug recover -m kernel.debug.json -a 0x2000 -t amd_gcn

# Output:
Crash occurred at PTX location:
  File: kernel.ptx
  Line: 42
  Column: 15
  Instruction: add.f32 %f3, %f1, %f2
```

### Example 3: Function Call Debugging

```ptx
.func (.reg .f32 result) multiply(.reg .f32 a, .reg .f32 b)
{
    mul.f32 result, a, b;           // Breakpoint in function
    ret;
}

.entry test_kernel()
{
    call (%f1), multiply, (%f2, %f3);  // Call function
}
```

Debug session:
```
(ptx-debug) backtrace
Call stack:
  #0: multiply at kernel.ptx:3:5
  #1: test_kernel at kernel.ptx:8:5
```

## Integration with External Debuggers

### GDB Integration

```bash
# Export GDB script
ptx-debug export -m kernel.debug.json -f gdb -o ptx_debug.gdb

# Use in GDB
(gdb) source ptx_debug.gdb
(gdb) ptx-break kernel.ptx:42
(gdb) ptx-info
```

### VS Code Integration

1. Export VS Code configuration:
```bash
ptx-debug export -m kernel.debug.json -f vscode -o .vscode/launch.json
```

2. Use VS Code debug features with PTX source-level debugging

## API Usage

### Programmatic Access

```rust
use ptx::debug::*;
use ptx::state_recovery::*;

// Create debug-aware compilation
let mut debug_context = DebugAwarePtxContext::new(true);
debug_context.initialize_debug_info(context, module, "kernel.ptx")?;

// Add debug location during compilation
debug_context.add_debug_location(builder, 42, 10, "add.f32")?;

// Create state recovery manager
let mut recovery = PtxStateRecoveryManager::new();
recovery.load_debug_mappings("kernel.debug.json")?;

// Set breakpoint
let bp_id = recovery.add_breakpoint(location, None);

// Recover state from target address
if let Some(ptx_loc) = recovery.find_ptx_location_from_target("amd_gcn", 0x1000) {
    println!("Crash at PTX {}:{}", ptx_loc.line, ptx_loc.column);
}
```

## Troubleshooting

### Common Issues

1. **No debug information generated**: Ensure debug compilation is enabled
2. **Address not found**: Check target architecture matches runtime
3. **Variable not available**: Variable may be optimized out or out of scope
4. **Mapping file corrupt**: Regenerate debug mappings from source

### Debug Tips

- Use `--verbose` flag for detailed compilation output
- Set multiple breakpoints to trace execution flow
- Export debug info for external tools when needed
- Check memory dumps for additional context

## Contributing

1. Add new target architecture support in `debug_integration.rs`
2. Extend debugger commands in `main.rs`
3. Improve DWARF generation in `debug.rs`
4. Add test cases for new features

## License

This tool is part of the ZLUDA project and follows the same licensing terms.