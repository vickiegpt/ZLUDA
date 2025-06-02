// Integration layer for PTX debug information generation
// This module provides the integration between the PTX compilation pipeline
// and the DWARF debug information generation

use super::*;
use crate::debug::{
    self, DwarfMappingEntry, PtxDwarfBuilder, PtxSourceLocation, TargetInstruction,
    VariableLocation,
};
use llvm_zluda::core::*;
use llvm_zluda::debuginfo::*;
use llvm_zluda::prelude::*;
use std::collections::HashMap;
use std::ffi::CString;
use llvm_zluda::*;

/// Debug-enabled PTX compilation context
pub struct DebugAwarePtxContext {
    pub dwarf_builder: Option<PtxDwarfBuilder>,
    pub source_mappings: Vec<DwarfMappingEntry>,
    pub current_function_debug_info: Option<LLVMMetadataRef>,
    pub enable_debug: bool,
}

impl DebugAwarePtxContext {
    pub fn new(enable_debug: bool) -> Self {
        Self {
            dwarf_builder: None,
            source_mappings: Vec::new(),
            current_function_debug_info: None,
            enable_debug,
        }
    }

    /// Initialize debug information for PTX compilation
    pub unsafe fn initialize_debug_info(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        source_file: &str,
    ) -> Result<(), String> {
        if self.enable_debug {
            self.dwarf_builder = Some(PtxDwarfBuilder::new(
                context,
                module,
                source_file,
                "ZLUDA PTX Compiler with Debug Support",
            )?);
        }
        Ok(())
    }

    /// Add debug location for PTX instruction
    pub unsafe fn add_debug_location(
        &mut self,
        builder: LLVMBuilderRef,
        ptx_line: u32,
        ptx_column: u32,
        _instruction_name: &str,
    ) -> Result<(), String> {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            let debug_loc = dwarf_builder.create_debug_location(
                ptx_line,
                ptx_column,
                self.current_function_debug_info,
            );

            // Set debug location for current instruction
            // Convert metadata to value for setting debug location
            let debug_value = LLVMMetadataAsValue(
                dwarf_builder.context,
                debug_loc
            );
            LLVMSetCurrentDebugLocation(builder, debug_value);

            // Create a mapping entry for later state recovery
            let mapping = DwarfMappingEntry {
                ptx_location: PtxSourceLocation {
                    file: "input.ptx".to_string(), // TODO: get actual filename
                    line: ptx_line,
                    column: ptx_column,
                    instruction_offset: 0, // TODO: calculate offset
                },
                target_instructions: Vec::new(), // Will be filled by target backends
                variable_mappings: HashMap::new(),
                scope_id: 0, // TODO: generate proper scope ID
            };

            self.source_mappings.push(mapping);
        }
        Ok(())
    }

    /// Begin function debug info
    pub unsafe fn begin_function_debug_info(
        &mut self,
        function_name: &str,
        line: u32,
    ) -> Result<(), String> {
        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Create proper function type for debug info (void function with no parameters)
            let function_type = dwarf_builder.create_function_type(None, &[])?;

            let function_debug_info = dwarf_builder.create_function_debug_info(
                function_name,
                function_name, // linkage name same as function name
                line,
                function_type, // proper function type
                false,         // not local to unit
                true,          // is definition
            )?;

            self.current_function_debug_info = Some(function_debug_info);
        }
        Ok(())
    }

    /// Add variable debug info
    pub unsafe fn add_variable_debug_info(
        &mut self,
        builder: LLVMBuilderRef,
        var_name: &str,
        var_line: u32,
        var_type_name: &str,
        var_size_bits: u64,
        storage: LLVMValueRef,
        location: &VariableLocation,
    ) -> Result<(), String> {
        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Create debug type for variable
            let var_type = dwarf_builder.create_basic_type(
                var_type_name,
                var_size_bits,
                4, // DW_ATE_float - using a default encoding value
            )?;

            let var_debug_info =
                dwarf_builder.create_variable_debug_info(var_name, var_line, var_type, location)?;

            // Create debug location for variable declaration
            let debug_loc =
                dwarf_builder.create_debug_location(var_line, 0, self.current_function_debug_info);

            // Create empty debug expression (no complex location)
            let empty_expr =
                LLVMDIBuilderCreateExpression(dwarf_builder.get_builder(), std::ptr::null_mut(), 0);

            // Insert variable declaration using new debug format
            // Note: Temporarily disabled due to LLVM API changes
            // TODO: Re-enable when LLVM bindings are updated
            LLVMZludaDIBuilderInsertDeclareRecordAtEnd(
                dwarf_builder.get_builder(),
                storage,
                var_debug_info,
                empty_expr,
                debug_loc,
                LLVMGetInsertBlock(builder),
            );

            // Update current mapping with variable info
            if let Some(last_mapping) = self.source_mappings.last_mut() {
                last_mapping
                    .variable_mappings
                    .insert(var_name.to_string(), location.clone());
            }
        }
        Ok(())
    }

    /// Finalize debug information
    pub unsafe fn finalize_debug_info(&mut self) {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            dwarf_builder.finalize();
        }
    }

    /// Get debug mappings for state recovery
    pub fn get_debug_mappings(&self) -> &[DwarfMappingEntry] {
        &self.source_mappings
    }

    /// Export debug mapping table
    pub fn export_debug_mappings(&self) -> String {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            return dwarf_builder.export_mapping_table();
        }

        "Debug information not available".to_string()
    }
}

/// Helper function to determine PTX type encoding for DWARF
pub fn ptx_type_to_dwarf_encoding(ptx_type: &ast::ScalarType) -> u32 {
    // We'll use hardcoded constants instead of the enum values since they're not available
    match ptx_type {
        ast::ScalarType::S32 | ast::ScalarType::S64 => 5, // DW_ATE_signed
        ast::ScalarType::U32 | ast::ScalarType::U64 => 7, // DW_ATE_unsigned
        ast::ScalarType::F32 | ast::ScalarType::F64 => 4, // DW_ATE_float
        ast::ScalarType::Pred => 2,                       // DW_ATE_boolean
        // Handle all other types including the ones that were missing
        ast::ScalarType::B8
        | ast::ScalarType::B16
        | ast::ScalarType::B32
        | ast::ScalarType::B64
        | ast::ScalarType::B128 => 7, // DW_ATE_unsigned
        ast::ScalarType::U8 | ast::ScalarType::U16 => 7, // DW_ATE_unsigned
        ast::ScalarType::S8 | ast::ScalarType::S16 => 5, // DW_ATE_signed
        ast::ScalarType::F16 => 4,                       // DW_ATE_float
        ast::ScalarType::U16x2 | ast::ScalarType::S16x2 => 7, // DW_ATE_unsigned (vectors)
        ast::ScalarType::F16x2 | ast::ScalarType::BF16 | ast::ScalarType::BF16x2 => 4, // DW_ATE_float
    }
}

/// Helper function to get PTX type size in bits
pub fn ptx_type_size_bits(ptx_type: &ast::ScalarType) -> u64 {
    match ptx_type {
        ast::ScalarType::U8 | ast::ScalarType::S8 | ast::ScalarType::B8 => 8,
        ast::ScalarType::U16
        | ast::ScalarType::S16
        | ast::ScalarType::B16
        | ast::ScalarType::F16
        | ast::ScalarType::BF16 => 16,
        ast::ScalarType::U32
        | ast::ScalarType::S32
        | ast::ScalarType::B32
        | ast::ScalarType::F32 => 32,
        ast::ScalarType::U64
        | ast::ScalarType::S64
        | ast::ScalarType::B64
        | ast::ScalarType::F64 => 64,
        ast::ScalarType::B128 => 128,
        ast::ScalarType::U16x2
        | ast::ScalarType::S16x2
        | ast::ScalarType::F16x2
        | ast::ScalarType::BF16x2 => 32, // Two 16-bit values = 32 bits
        ast::ScalarType::Pred => 1,
    }
}

/// Integration point for target backend debug info
pub trait TargetDebugInfo {
    /// Add target-specific instruction mapping
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    );

    /// Export target-specific debug information
    fn export_target_debug_info(&self) -> Vec<u8>;
}

/// AMD GCN debug information integration
pub struct AmdGcnDebugInfo {
    mappings: HashMap<(u32, u32), Vec<TargetInstruction>>,
}

impl AmdGcnDebugInfo {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }
}

impl TargetDebugInfo for AmdGcnDebugInfo {
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    ) {
        let key = (ptx_location.line, ptx_location.column);
        self.mappings
            .entry(key)
            .or_insert_with(Vec::new)
            .push(target_instruction);
    }

    fn export_target_debug_info(&self) -> Vec<u8> {
        // Export as JSON for now, could be DWARF format later
        serde_json::to_vec(&self.mappings).unwrap_or_default()
    }
}

/// Intel SPIRV debug information integration
pub struct IntelSpirvDebugInfo {
    mappings: HashMap<(u32, u32), Vec<TargetInstruction>>,
}

impl IntelSpirvDebugInfo {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }
}

impl TargetDebugInfo for IntelSpirvDebugInfo {
    fn add_target_instruction_mapping(
        &mut self,
        ptx_location: &PtxSourceLocation,
        target_instruction: TargetInstruction,
    ) {
        let key = (ptx_location.line, ptx_location.column);
        self.mappings
            .entry(key)
            .or_insert_with(Vec::new)
            .push(target_instruction);
    }

    fn export_target_debug_info(&self) -> Vec<u8> {
        // Export SPIR-V debug information
        // For now, JSON format, but could be extended to use SPIR-V debug extensions
        serde_json::to_vec(&self.mappings).unwrap_or_default()
    }
}

fn ptx_type_to_dwarf_type(
    dwarf_builder: &debug::PtxDwarfBuilder,
    _builder: LLVMBuilderRef,
    ptx_type: &ast::ScalarType,
    _instruction_name: &str,
) -> LLVMMetadataRef {
    unsafe {
        // Get appropriate encoding from our helper function
        let encoding = ptx_type_to_dwarf_encoding(ptx_type);

        match ptx_type {
            ast::ScalarType::B32 => LLVMDIBuilderCreateBasicType(
                // Use public interface methods instead of accessing private field
                dwarf_builder.get_builder(),
                c"int".as_ptr(),
                3,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::U32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"unsigned int".as_ptr(),
                12,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::S32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"int".as_ptr(),
                3,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::F32 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"float".as_ptr(),
                5,
                32,
                encoding,
                0, // flags
            ),
            ast::ScalarType::F64 => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"double".as_ptr(),
                6,
                64,
                encoding,
                0, // flags
            ),
            ast::ScalarType::Pred => LLVMDIBuilderCreateBasicType(
                dwarf_builder.get_builder(),
                c"bool".as_ptr(),
                4,
                1,
                encoding,
                0, // flags
            ),
            // Handle other scalar types with a catch-all
            _ => {
                // Default to int type for now
                LLVMDIBuilderCreateBasicType(
                    dwarf_builder.get_builder(),
                    c"unknown".as_ptr(),
                    7,
                    32,
                    encoding,
                    0, // flags
                )
            }
        }
    }
}
