// DWARF debug information generation for PTX to target architecture mapping
// This module provides functionality to maintain mappings from PTX source to
// compiled target code (SASS/AMD GCN/Intel SPIRV) for program state recovery

use super::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use llvm_zluda::core::*;
use llvm_zluda::debuginfo::*;
use llvm_zluda::prelude::*;
use llvm_zluda::*;
use serde::{Deserialize, Serialize};

/// PTX source location information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtxSourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub instruction_offset: usize,
}

/// Target architecture instruction mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetInstruction {
    AmdGcn {
        instruction: String,
        address: u64,
        register_state: HashMap<String, String>,
    },
    IntelSpirv {
        instruction: String,
        opcode: u32,
        operands: Vec<String>,
    },
    Sass {
        instruction: String,
        address: u64,
        predicate: Option<String>,
    },
}

/// DWARF mapping entry that connects PTX source to target instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DwarfMappingEntry {
    pub ptx_location: PtxSourceLocation,
    pub target_instructions: Vec<TargetInstruction>,
    pub variable_mappings: HashMap<String, VariableLocation>,
    pub scope_id: u64,
}

/// Variable location in target architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableLocation {
    Register(String),
    Memory { address: u64, size: u32 },
    Constant(String),
}

/// DWARF debug info builder for PTX compilation
pub struct PtxDwarfBuilder {
    pub context: LLVMContextRef,
    pub module: LLVMModuleRef,
    di_builder: *mut llvm_zluda::LLVMOpaqueDIBuilder,
    pub compile_unit: LLVMMetadataRef,
    file: LLVMMetadataRef,
    source_mappings: Vec<DwarfMappingEntry>,
    current_scope: LLVMMetadataRef,
    variable_counter: u64,
}

impl PtxDwarfBuilder {
    /// Create a new DWARF builder for PTX compilation
    pub unsafe fn new(
        context: LLVMContextRef,
        module: LLVMModuleRef,
        source_file: &str,
        producer: &str,
    ) -> Result<Self, String> {
        let di_builder = LLVMCreateDIBuilder(module);
        if di_builder.is_null() {
            return Err("Failed to create DIBuilder".to_string());
        }

        let producer_cstr = CString::new(producer).map_err(|_| "Invalid producer string")?;
        let directory_cstr = CString::new(".").map_err(|_| "Invalid directory string")?;
        let filename_cstr = CString::new(source_file).map_err(|_| "Invalid filename string")?;

        // Create debug info file
        let di_file = LLVMDIBuilderCreateFile(
            di_builder,
            filename_cstr.as_ptr(),
            filename_cstr.as_bytes().len(),
            directory_cstr.as_ptr(),
            directory_cstr.as_bytes().len(),
        );

        // Create compile unit with proper parameters
        let di_compile_unit = LLVMDIBuilderCreateCompileUnit(
            di_builder,
            llvm_zluda::debuginfo::LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageC, // Use C for compatibility
            di_file,
            producer_cstr.as_ptr(),
            producer_cstr.as_bytes().len(),
            0,           // not optimized initially
            ptr::null(), // no flags
            0,           // flags length
            0,           // runtime version
            ptr::null(), // split name
            0,           // split name length
            llvm_zluda::debuginfo::LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
            0,           // DWO ID
            1,           // split debug inlining
            0,           // debug info for profiling
            ptr::null(), // sysroot
            0,           // sysroot length
            ptr::null(), // SDK
            0,           // SDK length
        );

        // Set DWARF version metadata to fix "invalid version (0)" error
        // Only add module flags if they don't already exist
        let dwarf_version_str = CString::new("Dwarf Version").unwrap();
        let existing_dwarf_flag = LLVMGetModuleFlag(
            module,
            dwarf_version_str.as_ptr(),
            dwarf_version_str.as_bytes().len(),
        );
        if existing_dwarf_flag.is_null() {
            let version_val = LLVMConstInt(LLVMInt32TypeInContext(context), 4, 0); // DWARF version 4
            let version_metadata = LLVMValueAsMetadata(version_val);
            LLVMAddModuleFlag(
                module,
                LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError,
                dwarf_version_str.as_ptr(),
                dwarf_version_str.as_bytes().len(),
                version_metadata,
            );
        }

        // Also set Debug Info Version
        let debug_info_version_str = CString::new("Debug Info Version").unwrap();
        let existing_debug_flag = LLVMGetModuleFlag(
            module,
            debug_info_version_str.as_ptr(),
            debug_info_version_str.as_bytes().len(),
        );
        if existing_debug_flag.is_null() {
            let debug_version_val = LLVMConstInt(LLVMInt32TypeInContext(context), 3, 0); // Debug Info Version 3
            let debug_version_metadata = LLVMValueAsMetadata(debug_version_val);
            LLVMAddModuleFlag(
                module,
                LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError,
                debug_info_version_str.as_ptr(),
                debug_info_version_str.as_bytes().len(),
                debug_version_metadata,
            );
        }

        Ok(Self {
            context,
            module,
            di_builder,
            compile_unit: di_compile_unit,
            file: di_file,
            source_mappings: Vec::new(),
            current_scope: di_compile_unit, // Use compile unit as initial scope
            variable_counter: 0,
        })
    }

    /// Add a PTX source to target instruction mapping
    pub fn add_mapping(&mut self, mapping: DwarfMappingEntry) {
        self.source_mappings.push(mapping);
    }

    /// Create debug types for PTX scalar types
    pub unsafe fn create_ptx_debug_type(&self, ptx_type: &ptx_parser::ScalarType) -> LLVMMetadataRef {
        let (name, size_bits, encoding) = match ptx_type {
            ptx_parser::ScalarType::U8 => ("u8", 8, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U16 => ("u16", 16, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U32 => ("u32", 32, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::U64 => ("u64", 64, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::S8 => ("s8", 8, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S16 => ("s16", 16, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S32 => ("s32", 32, 5), // DW_ATE_signed
            ptx_parser::ScalarType::S64 => ("s64", 64, 5), // DW_ATE_signed
            ptx_parser::ScalarType::F16 => ("f16", 16, 4), // DW_ATE_float
            ptx_parser::ScalarType::F32 => ("f32", 32, 4), // DW_ATE_float
            ptx_parser::ScalarType::F64 => ("f64", 64, 4), // DW_ATE_float
            ptx_parser::ScalarType::Pred => ("pred", 1, 2), // DW_ATE_boolean
            ptx_parser::ScalarType::B8 => ("b8", 8, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B16 => ("b16", 16, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B32 => ("b32", 32, 7), // DW_ATE_unsigned
            ptx_parser::ScalarType::B64 => ("b64", 64, 7), // DW_ATE_unsigned
            _ => ("unknown", 32, 7), // Default fallback
        };
        
        let name_cstr = CString::new(name).unwrap();
        LLVMDIBuilderCreateBasicType(
            self.di_builder,
            name_cstr.as_ptr(),
            name.len(),
            size_bits,
            encoding,
            0, // flags
        )
    }

    /// Create debug info for function parameters
    pub unsafe fn create_parameter_debug_info(
        &self,
        function_scope: LLVMMetadataRef,
        param_name: &str,
        param_type: &ptx_parser::ScalarType,
        arg_num: u32,
        line: u32,
    ) -> LLVMMetadataRef {
        let param_name_cstr = CString::new(param_name).unwrap();
        let param_debug_type = self.create_ptx_debug_type(param_type);
        
        LLVMDIBuilderCreateParameterVariable(
            self.di_builder,
            function_scope,
            param_name_cstr.as_ptr(),
            param_name.len(),
            arg_num,
            self.file,
            line,
            param_debug_type,
            1, // alwaysPreserve
            0, // flags
        )
    }

    /// Create debug info for local variables
    pub unsafe fn create_local_variable_debug_info(
        &self,
        scope: LLVMMetadataRef,
        var_name: &str,
        var_type: &ptx_parser::ScalarType,
        line: u32,
    ) -> LLVMMetadataRef {
        let var_name_cstr = CString::new(var_name).unwrap();
        let var_debug_type = self.create_ptx_debug_type(var_type);
        
        LLVMDIBuilderCreateAutoVariable(
            self.di_builder,
            scope,
            var_name_cstr.as_ptr(),
            var_name.len(),
            self.file,
            line,
            var_debug_type,
            1, // alwaysPreserve
            0, // flags
            0, // alignInBits
        )
    }

    /// Create lexical block for improved scope tracking
    pub unsafe fn create_lexical_block(
        &self,
        parent_scope: LLVMMetadataRef,
        line: u32,
        column: u32,
    ) -> LLVMMetadataRef {
        LLVMDIBuilderCreateLexicalBlock(
            self.di_builder,
            parent_scope,
            self.file,
            line,
            column,
        )
    }

    /// Create function debug info with parameters
    pub unsafe fn create_function_debug_info(
        &self,
        function_name: &str,
        linkage_name: &str,
        line: u32,
        is_definition: bool,
        param_types: &[ptx_parser::ScalarType],
    ) -> LLVMMetadataRef {
        let function_name_cstr = CString::new(function_name).unwrap();
        let linkage_name_cstr = CString::new(linkage_name).unwrap();
        
        // Create function type
        let void_type = LLVMDIBuilderCreateBasicType(
            self.di_builder,
            c"void".as_ptr(),
            4,
            0,
            0,
            0,
        );
        
        // Create parameter types array
        let mut param_debug_types = vec![void_type]; // Return type first
        for param_type in param_types {
            param_debug_types.push(self.create_ptx_debug_type(param_type));
        }
        
        let function_type = LLVMDIBuilderCreateSubroutineType(
            self.di_builder,
            self.file,
            param_debug_types.as_mut_ptr(),
            param_debug_types.len() as u32,
            0, // flags
        );
        
        let sp_flags = if is_definition {
            0x40 // DISPFlagDefinition
        } else {
            0
        };
        
        LLVMDIBuilderCreateFunction(
            self.di_builder,
            self.file, // scope
            function_name_cstr.as_ptr(),
            function_name.len(),
            linkage_name_cstr.as_ptr(),
            linkage_name.len(),
            self.file,
            line,
            function_type,
            0, // isLocalToUnit
            is_definition as i32,
            line, // scopeLine
            0, // flags
            0, // isOptimized
        )
    }

    /// Create debug location for PTX source line
    pub unsafe fn create_debug_location(
        &self,
        line: u32,
        column: u32,
        scope: Option<LLVMMetadataRef>,
    ) -> Result<LLVMMetadataRef, String> {
        let scope_ref = scope.unwrap_or(self.current_scope);

        let debug_loc = LLVMDIBuilderCreateDebugLocation(
            self.context,
            line,
            column,
            scope_ref,
            ptr::null_mut(), // no inlined_at
        );

        if debug_loc.is_null() {
            return Err("Failed to create debug location".to_string());
        }

        Ok(debug_loc)
    }


    /// Create variable debug info with enhanced PTX variable and memory address tracking
    pub unsafe fn create_variable_debug_info(
        &mut self,
        name: &str,
        line: u32,
        var_type: LLVMMetadataRef,
        location: &VariableLocation,
        function_scope: Option<LLVMMetadataRef>,
    ) -> Result<LLVMMetadataRef, String> {
        let name_cstr = CString::new(name).map_err(|_| "Invalid variable name")?;

        // Use function scope if provided, otherwise try to use current_scope only if it's a valid local scope
        let valid_scope = function_scope.unwrap_or_else(|| {
            // If no function scope provided, we can't create local variables safely
            // Return null to indicate this variable should be skipped
            ptr::null_mut()
        });

        if valid_scope.is_null() {
            return Err("No valid function scope available for local variable".to_string());
        }

        // Create enhanced variable with PTX-specific attributes based on location
        let di_variable = match location {
            VariableLocation::Memory { address, size } => {
                // Create memory-based variable with address annotation
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for memory variable
                self.add_memory_variable_mapping(name, line, *address, *size)?;
                var
            }
            VariableLocation::Register(reg_name) => {
                // Create register-based variable with register annotation
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for register variable
                self.add_register_variable_mapping(name, line, reg_name)?;
                var
            }
            VariableLocation::Constant(value) => {
                // Create constant variable
                let var = LLVMDIBuilderCreateAutoVariable(
                    self.di_builder,
                    valid_scope,
                    name_cstr.as_ptr(),
                    name_cstr.as_bytes().len(),
                    self.file,
                    line,
                    var_type,
                    1, // always preserve
                    0, // flags
                    0, // align in bits
                );

                // Create and add mapping for constant variable
                self.add_constant_variable_mapping(name, line, value)?;
                var
            }
        };

        self.variable_counter += 1;
        Ok(di_variable)
    }

    /// Add memory variable mapping with address tracking
    fn add_memory_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        address: u64,
        size: u32,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Memory { address, size },
        );

        // Create target instruction with memory address information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpVariable_{}", var_name),
            opcode: 0x3B, // OpVariable in SPIR-V
            operands: vec![
                format!("ptr_0x{:016x}", address),
                format!("size_{}", size),
                format!("type_memory"),
                format!("storage_class_function"),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Add register variable mapping with register tracking
    fn add_register_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        reg_name: &str,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Register(reg_name.to_string()),
        );

        // Create target instruction with register information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpLoad_{}", var_name),
            opcode: 0x3D, // OpLoad in SPIR-V
            operands: vec![
                format!("reg_{}", reg_name),
                format!("type_register"),
                self.parse_register_type(reg_name),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Add constant variable mapping
    fn add_constant_variable_mapping(
        &mut self,
        var_name: &str,
        line: u32,
        value: &str,
    ) -> Result<(), String> {
        let ptx_location = PtxSourceLocation {
            file: "kernel.ptx".to_string(),
            line,
            column: 0,
            instruction_offset: 0,
        };

        let mut variable_mappings = HashMap::new();
        variable_mappings.insert(
            var_name.to_string(),
            VariableLocation::Constant(value.to_string()),
        );

        // Create target instruction with constant information
        let target_instruction = TargetInstruction::IntelSpirv {
            instruction: format!("OpConstant_{}", var_name),
            opcode: 0x2B, // OpConstant in SPIR-V
            operands: vec![
                format!("value_{}", value),
                format!("type_constant"),
                self.parse_constant_type(value),
            ],
        };

        let mapping_entry = DwarfMappingEntry {
            ptx_location,
            target_instructions: vec![target_instruction],
            variable_mappings,
            scope_id: self.variable_counter,
        };

        self.source_mappings.push(mapping_entry);
        Ok(())
    }

    /// Parse PTX register type from register name
    fn parse_register_type(&self, reg_name: &str) -> String {
        if reg_name.starts_with("%r") {
            "int32".to_string()
        } else if reg_name.starts_with("%f") {
            "float32".to_string()
        } else if reg_name.starts_with("%d") {
            "float64".to_string()
        } else if reg_name.starts_with("%p") {
            "predicate".to_string()
        } else if reg_name.starts_with("%h") {
            "int16".to_string()
        } else if reg_name.starts_with("%c") {
            "int8".to_string()
        } else {
            "unknown".to_string()
        }
    }

    /// Parse constant type from value
    fn parse_constant_type(&self, value: &str) -> String {
        if value.contains('.') {
            "float".to_string()
        } else if value.starts_with('-') || value.parse::<i64>().is_ok() {
            "integer".to_string()
        } else {
            "string".to_string()
        }
    }

    /// Create basic type debug info (for PTX types)
    pub unsafe fn create_basic_type(
        &self,
        name: &str,
        size_in_bits: u64,
        encoding: u32,
    ) -> Result<LLVMMetadataRef, String> {
        let name_cstr = CString::new(name).map_err(|_| "Invalid type name")?;

        Ok(LLVMDIBuilderCreateBasicType(
            self.di_builder,
            name_cstr.as_ptr(),
            name_cstr.as_bytes().len(),
            size_in_bits,
            encoding,
            0, // flags
        ))
    }

    /// Create a function/subroutine type for debug info
    pub unsafe fn create_function_type(
        &self,
        return_type: Option<LLVMMetadataRef>,
        parameter_types: &[LLVMMetadataRef],
    ) -> Result<LLVMMetadataRef, String> {
        // Create array of parameter types
        let mut all_types = Vec::new();

        // Add return type as first element (LLVM convention)
        if let Some(ret_type) = return_type {
            all_types.push(ret_type);
        } else {
            // Create void type for no return
            let void_type = self.create_basic_type("void", 0, 0)?;
            all_types.push(void_type);
        }

        // Add parameter types
        all_types.extend_from_slice(parameter_types);

        Ok(LLVMDIBuilderCreateSubroutineType(
            self.di_builder,
            self.file, // file
            all_types.as_mut_ptr(),
            all_types.len() as u32,
            0, // flags
        ))
    }

    /// Create a compile unit for the module
    pub unsafe fn create_compile_unit(&mut self) -> Result<(), String> {
        // The compile unit is already created in the constructor, so this is a no-op
        Ok(())
    }

    /// Get all source mappings for state recovery
    pub fn get_mappings(&self) -> &[DwarfMappingEntry] {
        &self.source_mappings
    }

    /// Find mapping by PTX source location
    pub fn find_mapping_by_location(&self, line: u32, column: u32) -> Option<&DwarfMappingEntry> {
        self.source_mappings.iter().find(|mapping| {
            mapping.ptx_location.line == line && mapping.ptx_location.column == column
        })
    }

    /// Export mappings for external debugger integration
    pub fn export_mapping_table(&self) -> String {
        let mut output = String::new();
        output.push_str("# PTX to Target Architecture Debug Mapping\n");
        output.push_str("# Format: ptx_line:ptx_col -> target_instructions\n\n");

        for mapping in &self.source_mappings {
            output.push_str(&format!(
                "{}:{}:{} -> [\n",
                mapping.ptx_location.file, mapping.ptx_location.line, mapping.ptx_location.column
            ));

            for (i, target_inst) in mapping.target_instructions.iter().enumerate() {
                match target_inst {
                    TargetInstruction::AmdGcn {
                        instruction,
                        address,
                        ..
                    } => {
                        output.push_str(&format!(
                            "  AMD_GCN[{}]: {} @ 0x{:x}\n",
                            i, instruction, address
                        ));
                    }
                    TargetInstruction::IntelSpirv {
                        instruction,
                        opcode,
                        ..
                    } => {
                        output.push_str(&format!(
                            "  SPIRV[{}]: {} (opcode: {})\n",
                            i, instruction, opcode
                        ));
                    }
                    TargetInstruction::Sass {
                        instruction,
                        address,
                        predicate,
                    } => {
                        let pred_str = predicate
                            .as_ref()
                            .map(|p| format!(" [{}]", p))
                            .unwrap_or_default();
                        output.push_str(&format!(
                            "  SASS[{}]: {} @ 0x{:x}{}\n",
                            i, instruction, address, pred_str
                        ));
                    }
                }
            }
            output.push_str("]\n\n");
        }

        output
    }

    /// Get the underlying DIBuilder
    pub fn get_builder(&self) -> *mut llvm_zluda::LLVMOpaqueDIBuilder {
        self.di_builder
    }

    /// Finalize debug information generation
    pub unsafe fn finalize(&self) {
        if !self.di_builder.is_null() {
            LLVMDIBuilderFinalize(self.di_builder);
        }
    }

    /// Finalize the compilation unit
    pub unsafe fn finalize_compile_unit(&self) -> Result<(), String> {
        // For LLVM, we don't need to do anything specific to finalize the compile unit
        // The debug info is finalized automatically when the module is compiled
        Ok(())
    }

    /// Clear all debug locations to prevent invalid records
    pub unsafe fn clear_debug_locations(&self) -> Result<(), String> {
        // Set null debug location on all DIBuilders
        LLVMDIBuilderFinalize(self.di_builder);

        // Clear any pending nodes by finalizing
        LLVMDIBuilderFinalize(self.di_builder);

        Ok(())
    }
}

impl Drop for PtxDwarfBuilder {
    fn drop(&mut self) {
        unsafe {
            if !self.di_builder.is_null() {
                LLVMDisposeDIBuilder(self.di_builder);
            }
        }
    }
}

/// State recovery mechanism using DWARF mappings
pub struct PtxStateRecovery {
    mappings: Vec<DwarfMappingEntry>,
    current_execution_point: Option<PtxSourceLocation>,
}

impl PtxStateRecovery {
    pub fn new(mappings: Vec<DwarfMappingEntry>) -> Self {
        Self {
            mappings,
            current_execution_point: None,
        }
    }

    /// Set current execution point in PTX source
    pub fn set_execution_point(&mut self, location: PtxSourceLocation) {
        self.current_execution_point = Some(location);
    }

    /// Recover PTX state from target architecture debugging information
    pub fn recover_ptx_state(&self, target_address: u64) -> Option<PtxSourceLocation> {
        for mapping in &self.mappings {
            for target_inst in &mapping.target_instructions {
                match target_inst {
                    TargetInstruction::AmdGcn { address, .. }
                    | TargetInstruction::Sass { address, .. } => {
                        if *address == target_address {
                            return Some(mapping.ptx_location.clone());
                        }
                    }
                    TargetInstruction::IntelSpirv { .. } => {
                        // SPIRV doesn't have direct address mapping, use opcode matching
                        // This would need runtime integration for proper address translation
                    }
                }
            }
        }
        None
    }

    /// Get variable locations at current execution point
    pub fn get_variable_state(&self) -> Option<&HashMap<String, VariableLocation>> {
        if let Some(ref current_location) = self.current_execution_point {
            for mapping in &self.mappings {
                if mapping.ptx_location == *current_location {
                    return Some(&mapping.variable_mappings);
                }
            }
        }
        None
    }

    /// Export current state for debugging
    pub fn export_state_dump(&self) -> String {
        let mut dump = String::new();

        if let Some(ref location) = self.current_execution_point {
            dump.push_str(&format!(
                "Current PTX execution point: {}:{}:{}\n",
                location.file, location.line, location.column
            ));

            if let Some(var_state) = self.get_variable_state() {
                dump.push_str("Variable state:\n");
                for (name, location) in var_state {
                    match location {
                        VariableLocation::Register(reg) => {
                            dump.push_str(&format!("  {} -> register {}\n", name, reg));
                        }
                        VariableLocation::Memory { address, size } => {
                            dump.push_str(&format!(
                                "  {} -> memory 0x{:x} (size: {})\n",
                                name, address, size
                            ));
                        }
                        VariableLocation::Constant(value) => {
                            dump.push_str(&format!("  {} -> constant {}\n", name, value));
                        }
                    }
                }
            }
        } else {
            dump.push_str("No current execution point set\n");
        }

        dump
    }
}

/// Integration point for adding debug info to PTX compilation pipeline
pub fn integrate_debug_info_generation(
    context: LLVMContextRef,
    module: LLVMModuleRef,
    source_file: &str,
) -> Result<PtxDwarfBuilder, String> {
    unsafe { PtxDwarfBuilder::new(context, module, source_file, "ZLUDA PTX Compiler") }
}
