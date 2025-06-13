#!/bin/bash
set -e
PTX_FILE="$1"

# Set PTX version to 8.5
sed -i 's/\.version [0-9]\+\.[0-9]\+/.version 8.5/g' $PTX_FILE

# Handle pointer references
sed -i 's/.ptr .const/.ptr/g' $PTX_FILE
sed -i 's/.ptr .shared/.ptr/g' $PTX_FILE

# Fix store instructions by replacing const with local
sed -i 's/st\.const\.u64/st\.local\.u64/g' $PTX_FILE
sed -i 's/st\.const\.u32/st\.local\.u32/g' $PTX_FILE
sed -i 's/st\.const\.u16/st\.local\.u16/g' $PTX_FILE
sed -i 's/st\.const\.u8/st\.local\.u8/g' $PTX_FILE
sed -i 's/st\.const\.s64/st\.local\.s64/g' $PTX_FILE
sed -i 's/st\.const\.s32/st\.local\.s32/g' $PTX_FILE
sed -i 's/st\.const\.s16/st\.local\.s16/g' $PTX_FILE
sed -i 's/st\.const\.s8/st\.local\.s8/g' $PTX_FILE
sed -i 's/st\.const\.b64/st\.local\.b64/g' $PTX_FILE
sed -i 's/st\.const\.b32/st\.local\.b32/g' $PTX_FILE
sed -i 's/st\.const\.b16/st\.local\.b16/g' $PTX_FILE
sed -i 's/st\.const\.b8/st\.local\.b8/g' $PTX_FILE
sed -i 's/st\.const\.f64/st\.local\.f64/g' $PTX_FILE
sed -i 's/st\.const\.f32/st\.local\.f32/g' $PTX_FILE
sed -i 's/st\.const\.f16/st\.local\.f16/g' $PTX_FILE

# Convert shared to local if needed
sed -i 's/st\.shared\.u64/st\.local\.u64/g' $PTX_FILE
sed -i 's/st\.shared\.u32/st\.local\.u32/g' $PTX_FILE
sed -i 's/st\.shared\.u16/st\.local\.u16/g' $PTX_FILE
sed -i 's/st\.shared\.u8/st\.local\.u8/g' $PTX_FILE
sed -i 's/st\.shared\.s64/st\.local\.s64/g' $PTX_FILE
sed -i 's/st\.shared\.s32/st\.local\.s32/g' $PTX_FILE
sed -i 's/st\.shared\.s16/st\.local\.s16/g' $PTX_FILE
sed -i 's/st\.shared\.s8/st\.local\.s8/g' $PTX_FILE
sed -i 's/st\.shared\.b64/st\.local\.b64/g' $PTX_FILE
sed -i 's/st\.shared\.b32/st\.local\.b32/g' $PTX_FILE
sed -i 's/st\.shared\.b16/st\.local\.b16/g' $PTX_FILE
sed -i 's/st\.shared\.b8/st\.local\.b8/g' $PTX_FILE
sed -i 's/st\.shared\.f64/st\.local\.f64/g' $PTX_FILE
sed -i 's/st\.shared\.f32/st\.local\.f32/g' $PTX_FILE
sed -i 's/st\.shared\.f16/st\.local\.f16/g' $PTX_FILE

# If we need to change SM target to something older
if [ "$2" = "sm_52" ]; then
    sed -i 's/\.target sm_[0-9]\+/.target sm_52/g' $PTX_FILE
fi

echo "Fixed PTX file: $PTX_FILE"
