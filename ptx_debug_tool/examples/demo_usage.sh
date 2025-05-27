#!/bin/bash
# Demo script showing PTX debug tool usage

set -e

echo "=== PTX Debug Tool Demo ==="
echo

# Build the debug tool
echo "Building PTX debug tool..."
cd "$(dirname "$0")/.."
cargo build --release
cd examples

# Compile PTX with debug information
echo "1. Compiling PTX with debug information..."
../target/release/ptx-debug compile \
    --input test_kernel.ptx \
    --output test_kernel.debug.json \
    --target amd_gcn \
    --verbose

echo "   ✓ Debug mappings saved to test_kernel.debug.json"
echo

# Analyze debug mappings
echo "2. Analyzing debug mappings..."
../target/release/ptx-debug analyze \
    --mapping-file test_kernel.debug.json \
    --detailed \
    --line 42

echo
echo "   ✓ Found detailed mappings for line 42"
echo

# Export for GDB
echo "3. Exporting debug information for GDB..."
../target/release/ptx-debug export \
    --mapping-file test_kernel.debug.json \
    --format gdb \
    --output test_kernel.gdb

echo "   ✓ GDB script saved to test_kernel.gdb"
echo

# Export for VS Code
echo "4. Exporting debug configuration for VS Code..."
mkdir -p .vscode
../target/release/ptx-debug export \
    --mapping-file test_kernel.debug.json \
    --format vscode \
    --output .vscode/launch.json

echo "   ✓ VS Code config saved to .vscode/launch.json"
echo

# Demonstrate state recovery
echo "5. Demonstrating state recovery from crash address..."
../target/release/ptx-debug recover \
    --mapping-file test_kernel.debug.json \
    --address 0x1000 \
    --target amd_gcn

echo
echo "   ✓ PTX location recovered from target address"
echo

# Create a simple interactive session script
echo "6. Creating interactive debugging session example..."
cat > debug_session_commands.txt << 'EOF'
help
break test_kernel.ptx:42:10
break test_kernel.ptx:85:5
info
print tid
recover 0x1000
backtrace
quit
EOF

echo "   Starting interactive debugging session..."
echo "   (Commands will be read from debug_session_commands.txt)"
echo

# Note: This would be interactive in real usage
echo "   Example commands that would be run:"
cat debug_session_commands.txt | sed 's/^/   > /'

echo
echo "   ✓ Interactive debugging commands demonstrated"
echo

# Clean up
echo "7. Cleaning up demo files..."
rm -f debug_session_commands.txt

echo "   ✓ Demo completed successfully!"
echo
echo "=== Summary ==="
echo "Created files:"
echo "  - test_kernel.debug.json  (Debug mappings)"
echo "  - test_kernel.gdb        (GDB script)"
echo "  - .vscode/launch.json    (VS Code config)"
echo
echo "To run interactive debugging:"
echo "  ../target/release/ptx-debug debug --mapping-file test_kernel.debug.json"
echo
echo "To recover from a crash:"
echo "  ../target/release/ptx-debug recover --mapping-file test_kernel.debug.json --address 0x<address> --target amd_gcn"