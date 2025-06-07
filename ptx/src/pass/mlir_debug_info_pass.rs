/// MLIR Debug Info Pass for ZLUDA PTX compilation
/// Inspired by flang/lib/Optimizer/Transforms/AddDebugInfo.cpp
///
/// This pass adds DWARF debug information to MLIR operations to ensure
/// location information survives translation to LLVM-IR.
use crate::pass::mlir_debug_framework::{UniversalDebugLocation, UniversalMlirDebugContext};
use std::collections::HashMap;

/// Debug Info Pass for PTX to MLIR compilation
pub struct ZludaDebugInfoPass {
    debug_context: UniversalMlirDebugContext,
    file_attrs: HashMap<String, String>,
    compile_unit_attr: Option<String>,
}

impl ZludaDebugInfoPass {
    pub fn new(debug_context: UniversalMlirDebugContext) -> Self {
        Self {
            debug_context,
            file_attrs: HashMap::new(),
            compile_unit_attr: None,
        }
    }

    /// Generate MLIR debug attributes that survive LLVM-IR translation
    pub fn process_module(&mut self, module_content: &mut String) {
        // Add debug info infrastructure similar to Flang
        self.add_compile_unit_attributes(module_content);
        self.add_file_attributes(module_content);
        self.transform_function_locations(module_content);
        self.add_variable_debug_declarations(module_content);
    }

    /// Add DWARF compile unit information
    fn add_compile_unit_attributes(&mut self, module_content: &mut String) {
        let compile_unit = format!(
            r#"#di_compile_unit = #llvm.di_compile_unit<
  id = distinct[0]<>, 
  sourceLanguage = DW_LANG_C_plus_plus, 
  file = #di_file, 
  producer = "ZLUDA PTX Compiler", 
  isOptimized = false, 
  emissionKind = Full
>"#
        );

        let file_attr = format!(
            r#"#di_file = #llvm.di_file<"input.ptx" in "{}">"#,
            std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
        );

        // Insert at the beginning of the module
        let module_start = module_content.find("module").unwrap_or(0);
        let insert_pos = module_content[..module_start].rfind('\n').unwrap_or(0);

        let debug_attrs = format!("{}\n{}\n\n", file_attr, compile_unit);
        module_content.insert_str(insert_pos, &debug_attrs);

        self.compile_unit_attr = Some("#di_compile_unit".to_string());
    }

    /// Add file attributes for each source file
    fn add_file_attributes(&mut self, _module_content: &mut String) {
        for (filename, _info) in &self.debug_context.source_files {
            let file_attr = format!(
                r#"#di_file_{} = #llvm.di_file<"{}" in "{}">"#,
                filename.replace(".", "_").replace("/", "_"),
                filename,
                std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
            self.file_attrs.insert(filename.clone(), file_attr);
        }
    }

    /// Transform function locations to use fused locations with debug info
    fn transform_function_locations(&mut self, module_content: &mut String) {
        // Find function declarations and add debug subprogram attributes
        let func_regex = regex::Regex::new(r"func\.func @(\w+)\((.*?)\).*?\{").unwrap();

        for captures in func_regex.captures_iter(module_content.clone().as_str()) {
            let func_name = &captures[1];
            let full_match = &captures[0];

            // Create subprogram attribute
            let subprogram_attr = self.create_subprogram_attribute(func_name);

            // Create fused location
            let fused_location = format!(
                r#" attributes {{
  llvm.debug_subprogram = {}
}}"#,
                subprogram_attr
            );

            // Replace the function declaration
            let new_func_decl = full_match.replace(" {", &format!("{} {{", fused_location));
            *module_content = module_content.replace(full_match, &new_func_decl);
        }
    }

    /// Create subprogram debug attribute for a function
    fn create_subprogram_attribute(&self, func_name: &str) -> String {
        format!(
            r#"#llvm.di_subprogram<
  id = distinct[{}]<>,
  compileUnit = {},
  scope = #di_file,
  name = "{}",
  linkageName = "{}",
  file = #di_file,
  line = 1,
  scopeLine = 1,
  subprogramFlags = "Definition",
  type = #di_subroutine_type
>"#,
            self.get_next_distinct_id(),
            self.compile_unit_attr
                .as_ref()
                .unwrap_or(&"#di_compile_unit".to_string()),
            func_name,
            func_name
        )
    }

    /// Add variable debug declarations
    fn add_variable_debug_declarations(&mut self, module_content: &mut String) {
        // Add debug declare intrinsics for variables
        let debug_declares = self.generate_debug_declare_intrinsics();

        // Insert after function entry
        for declare in debug_declares {
            if let Some(pos) = module_content.find("  %") {
                module_content.insert_str(pos, &format!("    {}\n", declare));
            }
        }
    }

    /// Generate debug declare intrinsics for variables
    fn generate_debug_declare_intrinsics(&self) -> Vec<String> {
        let mut declares = Vec::new();

        for (_var_id, var_info) in &self.debug_context.variables {
            let declare = format!(
                r#"llvm.intr.dbg.declare #llvm.di_local_variable<
  scope = #di_subprogram,
  name = "{}",
  file = #di_file,
  line = {},
  type = #di_basic_type
> = %{} : !llvm.ptr"#,
                var_info.name, var_info.location.line, var_info.name
            );
            declares.push(declare);
        }

        declares
    }

    /// Generate unique distinct IDs for debug info
    fn get_next_distinct_id(&self) -> usize {
        // Simple counter - in real implementation should be more sophisticated
        static mut COUNTER: usize = 1;
        unsafe {
            COUNTER += 1;
            COUNTER
        }
    }

    /// Add required debug type definitions
    pub fn add_debug_type_definitions(&self, module_content: &mut String) {
        let type_defs = r#"
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
"#;

        if let Some(pos) = module_content.find("module") {
            module_content.insert_str(pos, &format!("{}\n", type_defs));
        }
    }
}

/// Integration with existing TOSA/TTIR converters
pub trait DebugInfoIntegration {
    fn apply_debug_info_pass(
        &mut self,
        debug_context: &UniversalMlirDebugContext,
    ) -> Result<(), String>;
}

/// Apply debug info transformations to preserve location information
pub fn apply_zluda_debug_info_pass(
    mlir_content: &mut String,
    debug_context: UniversalMlirDebugContext,
) -> Result<(), String> {
    let mut debug_pass = ZludaDebugInfoPass::new(debug_context);

    // Add debug type definitions
    debug_pass.add_debug_type_definitions(mlir_content);

    // Process the module to add debug info
    debug_pass.process_module(mlir_content);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::mlir_debug_framework::{MlirDialect, OptimizationLevel};

    #[test]
    fn test_debug_info_pass() {
        let mut debug_context =
            UniversalMlirDebugContext::new(MlirDialect::TOSA, OptimizationLevel::Default);

        debug_context.add_source_file("test.ptx", Some("// Test PTX".to_string()));

        let mut mlir_content = r#"
module {
  func.func @test_kernel() {
    return
  }
}
"#
        .to_string();

        let result = apply_zluda_debug_info_pass(&mut mlir_content, debug_context);
        assert!(result.is_ok());
        assert!(mlir_content.contains("di_compile_unit"));
        assert!(mlir_content.contains("di_file"));
    }

    #[test]
    fn test_subprogram_generation() {
        let debug_context =
            UniversalMlirDebugContext::new(MlirDialect::TOSA, OptimizationLevel::Default);

        let debug_pass = ZludaDebugInfoPass::new(debug_context);
        let subprogram = debug_pass.create_subprogram_attribute("test_func");

        assert!(subprogram.contains("di_subprogram"));
        assert!(subprogram.contains("test_func"));
        assert!(subprogram.contains("Definition"));
    }
}
