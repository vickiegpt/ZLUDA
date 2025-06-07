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
use llvm_zluda::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

/// Debug-enabled PTX compilation context
pub struct DebugAwarePtxContext {
    pub dwarf_builder: Option<PtxDwarfBuilder>,
    pub source_mappings: Vec<DwarfMappingEntry>,
    pub current_function_debug_info: Option<LLVMMetadataRef>,
    pub debug_enabled: bool,
}

impl DebugAwarePtxContext {
    pub fn new(debug_enabled: bool) -> Self {
        Self {
            dwarf_builder: None,
            source_mappings: Vec::new(),
            current_function_debug_info: None,
            debug_enabled,
        }
    }

    /// Initialize debug information for PTX compilation
    pub unsafe fn initialize_debug_info(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        source_file: &str,
    ) -> Result<(), String> {
        if self.debug_enabled {
            self.dwarf_builder = Some(PtxDwarfBuilder::new(
                context,
                module,
                source_file,
                "ZLUDA PTX Compiler with Debug Support",
            )?);
        }
        Ok(())
    }

    /// Initialize debug information with additional details
    pub unsafe fn initialize_debug_info_with_details(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
        optimization_level: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.debug_enabled {
            return Ok(());
        }

        // Make sure filename ends with .ptx
        let filename = if !filename.ends_with(".ptx") {
            format!("{}.ptx", filename)
        } else {
            filename.to_string()
        };

        // Create builder with basic parameters
        self.dwarf_builder = Some(crate::debug::PtxDwarfBuilder::new(
            context, module, &filename, producer,
        )?);

        Ok(())
    }

    /// Add debug location for PTX instruction
    pub unsafe fn add_debug_location(
        &mut self,
        builder: LLVMBuilderRef,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Only create debug locations if we have a valid function scope
            // Using compile unit as scope is not valid according to LLVM verification
            if let Some(function_scope) = self.current_function_debug_info {
                // Create debug location with valid function scope
                let debug_loc =
                    dwarf_builder.create_debug_location(line, column, Some(function_scope))?;

                // Set the debug location on the builder
                let debug_value = LLVMMetadataAsValue(dwarf_builder.context, debug_loc);
                LLVMSetCurrentDebugLocation(builder, debug_value);
            }

            // Store mapping for later use (regardless of whether we set debug location)
            self.source_mappings.push(debug::DwarfMappingEntry {
                ptx_location: debug::PtxSourceLocation {
                    file: "kernel.ptx".to_string(),
                    line,
                    column,
                    instruction_offset: 0,
                },
                target_instructions: Vec::new(),
                variable_mappings: std::collections::HashMap::new(),
                scope_id: 0,
            });
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

    /// Add variable debug info with enhanced PTX variable and memory address tracking
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
        // Get encoding first to avoid borrowing conflicts
        let encoding = self.get_ptx_dwarf_encoding(var_type_name);

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Create debug type for variable with PTX-specific encoding
            let var_type =
                dwarf_builder.create_basic_type(var_type_name, var_size_bits, encoding)?;

            let var_debug_info =
                dwarf_builder.create_variable_debug_info(var_name, var_line, var_type, location)?;

            // Create debug location for variable declaration
            let debug_loc = dwarf_builder.create_debug_location(
                var_line,
                0,
                self.current_function_debug_info,
            )?;

            // Create enhanced debug expression based on variable location
            let debug_expr = Self::create_ptx_variable_expression_static(
                location,
                var_size_bits,
                dwarf_builder,
            )?;

            // Insert variable declaration with enhanced location info
            LLVMZludaDIBuilderInsertDeclareRecordAtEnd(
                dwarf_builder.get_builder(),
                storage,
                var_debug_info,
                debug_expr,
                debug_loc,
                LLVMGetInsertBlock(builder),
            );

            // Create enhanced variable mapping with memory address info
            self.add_enhanced_variable_mapping(var_name, location, var_line, var_size_bits)?;
        }
        Ok(())
    }

    /// Get PTX-specific DWARF encoding for variable types
    fn get_ptx_dwarf_encoding(&self, var_type_name: &str) -> u32 {
        match var_type_name {
            "s8" | "s16" | "s32" | "s64" => 5, // DW_ATE_signed
            "u8" | "u16" | "u32" | "u64" => 7, // DW_ATE_unsigned
            "f16" | "f32" | "f64" => 4,        // DW_ATE_float
            "b8" | "b16" | "b32" | "b64" => 8, // DW_ATE_boolean
            "pred" => 2,                       // DW_ATE_boolean
            _ => 7,                            // Default to unsigned
        }
    }

    /// Create PTX variable expression with memory address information (static version)
    unsafe fn create_ptx_variable_expression_static(
        location: &VariableLocation,
        size_bits: u64,
        dwarf_builder: &debug::PtxDwarfBuilder,
    ) -> Result<LLVMMetadataRef, String> {
        let mut expr_ops = Vec::new();

        match location {
            VariableLocation::Register(reg_name) => {
                // For register variables, create register-based expression
                // DW_OP_reg0 + register_number
                let reg_num = Self::parse_ptx_register_number_static(reg_name);
                if reg_num < 32 {
                    expr_ops.push(0x50 + reg_num as u64); // DW_OP_reg0 + n
                } else {
                    expr_ops.push(0x90); // DW_OP_regx
                    expr_ops.push(reg_num as u64);
                }
            }
            VariableLocation::Memory {
                address: _,
                size: _,
            } => {
                // For memory variables, use empty expression to avoid LLVM validation errors
                // LLVM doesn't accept absolute memory addresses in debug expressions
                // The variable info will still be available, just without location tracking
            }
            VariableLocation::Constant(value) => {
                // For constant variables
                if let Ok(const_val) = value.parse::<i64>() {
                    if const_val >= 0 && const_val <= 31 {
                        expr_ops.push(0x30 + const_val as u64); // DW_OP_lit0 + n
                    } else {
                        expr_ops.push(0x10); // DW_OP_consts
                        expr_ops.push(const_val as u64);
                    }
                }
            }
        }

        // Create the debug expression (only if we have operations)
        if expr_ops.is_empty() {
            Ok(ptr::null_mut())
        } else {
            let expr = LLVMDIBuilderCreateExpression(
                dwarf_builder.get_builder(),
                expr_ops.as_mut_ptr() as *mut u64,
                expr_ops.len(),
            );
            Ok(expr)
        }
    }

    /// Parse PTX register name to get register number (static version)
    fn parse_ptx_register_number_static(reg_name: &str) -> u32 {
        // Extract register number from PTX register names like %r0, %f1, etc.
        if let Some(num_part) = reg_name
            .strip_prefix('%')
            .and_then(|s| s.chars().skip(1).collect::<String>().parse::<u32>().ok())
        {
            num_part
        } else {
            0 // Default to register 0
        }
    }

    /// Add enhanced variable mapping with PTX-specific information
    fn add_enhanced_variable_mapping(
        &mut self,
        var_name: &str,
        location: &VariableLocation,
        line: u32,
        size_bits: u64,
    ) -> Result<(), String> {
        // Create enhanced mapping entry
        let ptx_location = debug::PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = std::collections::HashMap::new();
        variable_mappings.insert(var_name.to_string(), location.clone());

        // Add memory address information to target instructions
        let target_instruction = match location {
            VariableLocation::Memory { address, size } => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("var_decl_{}", var_name),
                    opcode: 0x20, // OpVariable in SPIR-V
                    operands: vec![
                        format!("ptr_0x{:x}", address),
                        format!("size_{}", size),
                        format!("bits_{}", size_bits),
                    ],
                }
            }
            VariableLocation::Register(reg_name) => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("reg_assign_{}", var_name),
                    opcode: 0x3e, // OpLoad in SPIR-V
                    operands: vec![reg_name.clone(), format!("bits_{}", size_bits)],
                }
            }
            VariableLocation::Constant(value) => {
                debug::TargetInstruction::IntelSpirv {
                    instruction: format!("const_assign_{}", var_name),
                    opcode: 0x2b, // OpConstant in SPIR-V
                    operands: vec![value.clone(), format!("bits_{}", size_bits)],
                }
            }
        };

        let mapping_entry = debug::DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.current_function_debug_info.map(|_| 1).unwrap_or(0) as u64,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Finalize debug information
    pub unsafe fn finalize_debug_info(&mut self) {
        if let Some(ref dwarf_builder) = self.dwarf_builder {
            // Finalize the debug information
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
        // Handle any other types
        _ => 64, // Default size for unknown types
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

// Remove duplicate definition - use the one from debug.rs module

// Remove duplicate PtxDwarfBuilder definition since it's already in debug.rs

/// Debug context for managing debug information
pub struct DebugContext {
    /// Whether debug information is enabled
    pub debug_enabled: bool,
    /// DWARF builder for creating debug metadata
    pub dwarf_builder: Option<PtxDwarfBuilder>,
    /// Current function's debug information
    pub current_function_debug_info: Option<LLVMMetadataRef>,
    /// Source mappings for debug information
    pub source_mappings: Vec<DwarfMappingEntry>,
}

impl DebugContext {
    /// Create a new debug context
    pub fn new() -> Self {
        Self {
            // Disable debug for Intel SPIR-V backend due to incompatibility
            debug_enabled: true, // Enable basic debug info
            dwarf_builder: None, // No DWARF builder for now
            current_function_debug_info: None,
            source_mappings: Vec::new(),
        }
    }

    /// Check if debug information is enabled
    pub fn is_debug_enabled(&self) -> bool {
        self.debug_enabled
    }

    /// Initialize debug information with additional details
    pub unsafe fn initialize_debug_info_with_details(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
        producer: &str,
        optimization_level: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.debug_enabled {
            return Ok(());
        }

        // Create builder with basic parameters
        self.dwarf_builder = Some(unsafe {
            crate::debug::PtxDwarfBuilder::new(context, module, filename, producer)?
        });

        Ok(())
    }

    /// Initialize debug information
    pub unsafe fn initialize_debug_info(
        &mut self,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.debug_enabled = true;

        // Make sure filename ends with .ptx
        let filename = if !filename.ends_with(".ptx") {
            format!("{}.ptx", filename)
        } else {
            filename.to_string()
        };

        // Create DWARF builder
        let producer = "ZLUDA PTX Compiler";
        self.dwarf_builder = Some(crate::debug::PtxDwarfBuilder::new(
            context, module, &filename, producer,
        )?);

        Ok(())
    }

    /// Add debug location for PTX instruction
    pub unsafe fn add_debug_location(
        &mut self,
        builder: LLVMBuilderRef,
        line: u32,
        column: u32,
        instruction_name: &str,
    ) -> Result<(), String> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return Ok(());
        }

        if let Some(ref mut dwarf_builder) = self.dwarf_builder {
            // Only create debug locations if we have a valid function scope
            // Using compile unit as scope is not valid according to LLVM verification
            if let Some(function_scope) = self.current_function_debug_info {
                // Create debug location with valid function scope
                let debug_loc =
                    dwarf_builder.create_debug_location(line, column, Some(function_scope))?;

                // Set the debug location on the builder
                let debug_value = LLVMMetadataAsValue(dwarf_builder.context, debug_loc);
                LLVMSetCurrentDebugLocation(builder, debug_value);
            }

            // Store mapping for later use (regardless of whether we set debug location)
            self.source_mappings.push(debug::DwarfMappingEntry {
                ptx_location: debug::PtxSourceLocation {
                    file: "kernel.ptx".to_string(),
                    line,
                    column,
                    instruction_offset: 0,
                },
                target_instructions: Vec::new(),
                variable_mappings: std::collections::HashMap::new(),
                scope_id: 0,
            });
        }

        Ok(())
    }

    /// Set current function's debug info
    pub fn set_function_debug_info(&mut self, metadata: LLVMMetadataRef) {
        self.current_function_debug_info = Some(metadata);
    }

    /// Finalize debug information
    pub unsafe fn finalize_debug_info(&mut self) {
        if let Some(builder) = &self.dwarf_builder {
            builder.finalize();
        }
    }

    /// Get the current debug location
    pub fn get_current_debug_location(&self) -> Option<LLVMMetadataRef> {
        if !self.debug_enabled || self.dwarf_builder.is_none() {
            return None;
        }

        // If we have any source mappings, get the last one's debug location
        if let Some(mapping) = self.source_mappings.last() {
            if let Some(ref dwarf_builder) = self.dwarf_builder {
                // Try to create a debug location from the last mapping
                match unsafe {
                    dwarf_builder.create_debug_location(
                        mapping.ptx_location.line,
                        mapping.ptx_location.column,
                        self.current_function_debug_info,
                    )
                } {
                    Ok(loc) => return Some(loc),
                    Err(_) => return None,
                }
            }
        }

        None
    }
}
