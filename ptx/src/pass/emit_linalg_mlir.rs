// emit_linalg_mlir.rs - Direct PTX to Linalg MLIR conversion
// This pass converts PTX AST directly to MLIR using linalg, memref, and arith dialects
// for better compatibility with various backends, especially the Tenstorrent backend.

use std::collections::HashMap;
use std::fmt::Write;
use super::*;
use ptx_parser as ast;

pub fn run<'input>(
    id_defs: GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<'input, ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<String, TranslateError> {
    let mut converter = PtxToLinalgConverter::new(&id_defs);
    converter.convert_module(directives)
}

struct PtxToLinalgConverter<'a, 'input> {
    id_defs: &'a GlobalStringIdentResolver2<'input>,
    output: String,
    indent_level: usize,
    ssa_counter: u32,
    memref_counter: u32,
    value_map: HashMap<SpirvWord, String>,
    tensor_shapes: HashMap<SpirvWord, Vec<i64>>,
}

impl<'a, 'input> PtxToLinalgConverter<'a, 'input> {
    fn new(id_defs: &'a GlobalStringIdentResolver2<'input>) -> Self {
        Self {
            id_defs,
            output: String::new(),
            indent_level: 0,
            ssa_counter: 0,
            memref_counter: 0,
            value_map: HashMap::new(),
            tensor_shapes: HashMap::new(),
        }
    }

    fn write_line(&mut self, line: &str) {
        for _ in 0..self.indent_level {
            self.output.push_str("  ");
        }
        self.output.push_str(line);
        self.output.push('\n');
    }

    fn next_ssa_value(&mut self) -> String {
        let name = format!("%{}", self.ssa_counter);
        self.ssa_counter += 1;
        name
    }

    fn next_memref(&mut self) -> String {
        let name = format!("%memref{}", self.memref_counter);
        self.memref_counter += 1;
        name
    }

    fn convert_module(&mut self, directives: Vec<Directive2<'input, ast::Instruction<SpirvWord>, SpirvWord>>) -> Result<String, TranslateError> {
        self.write_line("module {");
        self.indent_level += 1;

        for directive in directives {
            match directive {
                Directive2::Variable(linking, variable) => {
                    self.convert_global_variable(linking, variable)?;
                }
                Directive2::Method(method) => {
                    self.convert_function(method)?;
                }
            }
        }

        self.indent_level -= 1;
        self.write_line("}");

        Ok(self.output.clone())
    }

    fn convert_global_variable(&mut self, _linking: ast::LinkingDirective, variable: ast::Variable<SpirvWord>) -> Result<(), TranslateError> {
        let var_name = self.get_variable_name(variable.name)?;
        let memref_type = self.get_memref_type(&variable.v_type)?;
        
        self.write_line(&format!("memref.global \"{}\" : {}", var_name, memref_type));
        Ok(())
    }

    fn convert_function(&mut self, method: Function2<'input, ast::Instruction<SpirvWord>, SpirvWord>) -> Result<(), TranslateError> {
        let func_name = match &method.func_decl.name {
            ast::MethodName::Kernel(name) => name.to_string(),
            ast::MethodName::Func(id) => {
                self.id_defs.ident_map.get(id)
                    .and_then(|entry| entry.name.as_ref())
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("func_{}", id.0))
            }
        };

        // Generate function signature
        let mut signature = format!("func.func @{}(", func_name);
        
        // Input parameters
        for (i, param) in method.func_decl.input_arguments.iter().enumerate() {
            if i > 0 {
                signature.push_str(", ");
            }
            let param_type = self.convert_type_to_linalg(&param.v_type)?;
            signature.push_str(&format!("%arg{}: {}", i, param_type));
            
            // Map parameter to SSA value
            let param_ssa = format!("%arg{}", i);
            self.value_map.insert(param.name, param_ssa);
        }
        
        signature.push_str(")");

        // Return type
        if !method.func_decl.return_arguments.is_empty() {
            signature.push_str(" -> ");
            for (i, ret_arg) in method.func_decl.return_arguments.iter().enumerate() {
                if i > 0 {
                    signature.push_str(", ");
                }
                let ret_type = self.convert_type_to_linalg(&ret_arg.v_type)?;
                signature.push_str(&ret_type);
            }
        }

        signature.push_str(" {");
        self.write_line(&signature);
        self.indent_level += 1;

        // Add common constants
        self.write_line("%c0 = arith.constant 0 : index");
        self.write_line("%c1 = arith.constant 1 : index");

        // Convert function body
        if let Some(body) = method.body {
            for statement in body {
                self.convert_statement(statement)?;
            }
        }

        // Generate appropriate return statement
        if !method.func_decl.return_arguments.is_empty() {
            // Function has return values - need to return something
            // For special register functions, return a dummy constant value  
            let ret_type = &method.func_decl.return_arguments[0].v_type;
            let ret_scalar_type = self.convert_scalar_type_to_linalg(ret_type.scalar_type())?;
            let dummy_ret = self.next_ssa_value();
            self.write_line(&format!("{} = arith.constant 0 : {}", dummy_ret, ret_scalar_type));
            self.write_line(&format!("func.return {} : {}", dummy_ret, ret_scalar_type));
        } else {
            // Function returns void
            self.write_line("func.return");
        }
        self.indent_level -= 1;
        self.write_line("}");

        Ok(())
    }

    fn convert_statement(&mut self, statement: Statement<ast::Instruction<SpirvWord>, SpirvWord>) -> Result<(), TranslateError> {
        match statement {
            Statement::Label(label) => {
                self.write_line(&format!("// Label: {}", label.0));
            }
            Statement::Variable(var) => {
                self.convert_local_variable(var)?;
            }
            Statement::Instruction(inst) => {
                self.convert_instruction(inst)?;
            }
            Statement::Constant(const_def) => {
                self.convert_constant(const_def)?;
            }
            _ => {
                self.write_line(&format!("// Unsupported statement type"));
            }
        }
        Ok(())
    }

    fn convert_local_variable(&mut self, var: ast::Variable<SpirvWord>) -> Result<(), TranslateError> {
        let memref_type = self.get_memref_type(&var.v_type)?;
        let var_ssa = self.next_ssa_value();
        
        self.write_line(&format!("{} = memref.alloc() : {}", var_ssa, memref_type));
        self.value_map.insert(var.name, var_ssa);
        
        Ok(())
    }

    fn convert_constant(&mut self, const_def: ConstantDefinition) -> Result<(), TranslateError> {
        let value_str = self.format_immediate_value(&const_def.value);
        let type_str = self.convert_scalar_type_to_linalg(const_def.typ)?;
        let const_ssa = self.next_ssa_value();
        
        self.write_line(&format!("{} = arith.constant {} : {}", const_ssa, value_str, type_str));
        self.value_map.insert(const_def.dst, const_ssa);
        
        Ok(())
    }

    fn convert_instruction(&mut self, inst: ast::Instruction<SpirvWord>) -> Result<(), TranslateError> {
        match inst {
            ast::Instruction::Ld { data, arguments, .. } => {
                self.convert_load_instruction(data, arguments.dst, arguments.src)?;
            }
            ast::Instruction::St { data, arguments, .. } => {
                self.convert_store_instruction(data, arguments.src1, arguments.src2)?;
            }
            ast::Instruction::Add { data, arguments, .. } => {
                self.convert_add_instruction(data, arguments.dst, arguments.src1, arguments.src2)?;
            }
            ast::Instruction::Sub { data, arguments, .. } => {
                self.convert_sub_instruction(data, arguments.dst, arguments.src1, arguments.src2)?;
            }
            ast::Instruction::Mul { data, arguments, .. } => {
                self.convert_mul_instruction(data, arguments.dst, arguments.src1, arguments.src2)?;
            }
            ast::Instruction::Mov { data, arguments, .. } => {
                self.convert_mov_instruction(data, arguments.dst, arguments.src)?;
            }
            _ => {
                self.write_line("// Unsupported instruction");
            }
        }
        Ok(())
    }

    fn convert_load_instruction(&mut self, data: ast::LdDetails, dst: SpirvWord, src: SpirvWord) -> Result<(), TranslateError> {
        // Check if the source is a function parameter (starts with %arg)
        if let Ok(src_ssa) = self.get_ssa_value(src) {
            if src_ssa.starts_with("%arg") {
                // For function parameter loads, directly use the parameter value
                self.value_map.insert(dst, src_ssa);
                return Ok(());
            }
        }
        
        let src_ssa = self.get_ssa_value(src)?;
        
        // Check if this is a parameter load (ld.param)
        if data.state_space == ast::StateSpace::Param {
            // For parameter loads, the source is already an SSA value (function parameter)
            // We can directly use the parameter value
            self.value_map.insert(dst, src_ssa);
            return Ok(());
        }
        
        // For other state spaces (global, shared, etc.), use memref.load
        let dst_ssa = self.next_ssa_value();
        let memref_type = self.get_memref_type(&data.typ)?;
        self.write_line(&format!("{} = memref.load {}[%c0] : {}", dst_ssa, src_ssa, memref_type));
        self.value_map.insert(dst, dst_ssa);
        
        Ok(())
    }

    fn convert_store_instruction(&mut self, data: ast::StData, src1: SpirvWord, src2: SpirvWord) -> Result<(), TranslateError> {
        // Check if this is a parameter store (st.param)
        if data.state_space == ast::StateSpace::Param {
            // For parameter stores, we don't need to generate any MLIR code
            // Parameters are passed as function arguments and handled directly
            return Ok(());
        }
        
        let src_ssa = self.get_ssa_value(src2)?;  // src2 is the value to store
        let dst_ssa = self.get_ssa_value(src1)?;  // src1 is the destination address
        
        let memref_type = self.get_memref_type(&data.typ)?;
        self.write_line(&format!("memref.store {}, {}[%c0] : {}", src_ssa, dst_ssa, memref_type));
        
        Ok(())
    }

    fn convert_add_instruction(&mut self, data: ast::ArithDetails, dst: SpirvWord, src1: SpirvWord, src2: SpirvWord) -> Result<(), TranslateError> {
        let dst_ssa = self.next_ssa_value();
        let src1_ssa = self.get_ssa_value(src1)?;
        let src2_ssa = self.get_ssa_value(src2)?;
        
        let arith_type = self.convert_scalar_type_to_linalg(data.type_())?;
        
        // Use linalg.generic for element-wise operations on tensors
        if self.is_tensor_type(&ast::Type::Scalar(data.type_())) {
            self.convert_tensor_elementwise_op("arith.addi", &dst_ssa, &src1_ssa, &src2_ssa, &arith_type)?;
        } else {
            // Scalar arithmetic
            self.write_line(&format!("{} = arith.addi {}, {} : {}", dst_ssa, src1_ssa, src2_ssa, arith_type));
        }
        
        self.value_map.insert(dst, dst_ssa);
        Ok(())
    }

    fn convert_sub_instruction(&mut self, data: ast::ArithDetails, dst: SpirvWord, src1: SpirvWord, src2: SpirvWord) -> Result<(), TranslateError> {
        let dst_ssa = self.next_ssa_value();
        let src1_ssa = self.get_ssa_value(src1)?;
        let src2_ssa = self.get_ssa_value(src2)?;
        
        let arith_type = self.convert_scalar_type_to_linalg(data.type_())?;
        
        if self.is_tensor_type(&ast::Type::Scalar(data.type_())) {
            self.convert_tensor_elementwise_op("arith.subi", &dst_ssa, &src1_ssa, &src2_ssa, &arith_type)?;
        } else {
            self.write_line(&format!("{} = arith.subi {}, {} : {}", dst_ssa, src1_ssa, src2_ssa, arith_type));
        }
        
        self.value_map.insert(dst, dst_ssa);
        Ok(())
    }

    fn convert_mul_instruction(&mut self, data: ast::MulDetails, dst: SpirvWord, src1: SpirvWord, src2: SpirvWord) -> Result<(), TranslateError> {
        let dst_ssa = self.next_ssa_value();
        let src1_ssa = self.get_ssa_value(src1)?;
        let src2_ssa = self.get_ssa_value(src2)?;
        
        let arith_type = self.convert_scalar_type_to_linalg(data.type_())?;
        
        if self.is_tensor_type(&ast::Type::Scalar(data.type_())) {
            self.convert_tensor_elementwise_op("arith.muli", &dst_ssa, &src1_ssa, &src2_ssa, &arith_type)?;
        } else {
            self.write_line(&format!("{} = arith.muli {}, {} : {}", dst_ssa, src1_ssa, src2_ssa, arith_type));
        }
        
        self.value_map.insert(dst, dst_ssa);
        Ok(())
    }

    fn convert_mov_instruction(&mut self, data: ast::MovDetails, dst: SpirvWord, src: SpirvWord) -> Result<(), TranslateError> {
        let dst_ssa = self.next_ssa_value();
        let src_ssa = self.get_ssa_value(src)?;
        
        // For memref types, create a copy using linalg.copy
        if self.is_memref_move(&data.typ) {
            let memref_type = self.convert_type_to_linalg(&data.typ)?;
            self.write_line(&format!("{} = memref.alloc() : {}", dst_ssa, memref_type));
            self.write_line(&format!("linalg.copy ins({} : {}) outs({} : {})", 
                src_ssa, memref_type, dst_ssa, memref_type));
        } else {
            // Simple scalar move - create a zero constant of the appropriate type and add it
            let scalar_type = self.convert_scalar_type_to_linalg(data.typ.scalar_type())?;
            let zero_const = self.next_ssa_value();
            self.write_line(&format!("{} = arith.constant 0 : {}", zero_const, scalar_type));
            self.write_line(&format!("{} = arith.addi {}, {} : {}", dst_ssa, src_ssa, zero_const, scalar_type));
        }
        
        self.value_map.insert(dst, dst_ssa);
        Ok(())
    }

    fn convert_tensor_elementwise_op(&mut self, op: &str, dst: &str, src1: &str, src2: &str, element_type: &str) -> Result<(), TranslateError> {
        // Get tensor shape for this operation
        let shape = self.get_inferred_tensor_shape();
        let tensor_type = format!("tensor<{}x{}>", shape[0], element_type);
        
        self.write_line(&format!("{} = linalg.generic {{", dst));
        self.indent_level += 1;
        self.write_line("indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],");
        self.write_line("iterator_types = [\"parallel\"]");
        self.indent_level -= 1;
        self.write_line(&format!("}} ins({}, {} : {}, {}) outs(%init : {}) {{", src1, src2, tensor_type, tensor_type, tensor_type));
        self.indent_level += 1;
        self.write_line(&format!("^bb0(%arg0: {}, %arg1: {}, %arg2: {}):", element_type, element_type, element_type));
        self.write_line(&format!("  %result = {} %arg0, %arg1 : {}", op, element_type));
        self.write_line(&format!("  linalg.yield %result : {}", element_type));
        self.indent_level -= 1;
        self.write_line(&format!("}} -> {}", tensor_type));
        
        Ok(())
    }

    fn convert_type_to_linalg(&self, typ: &ast::Type) -> Result<String, TranslateError> {
        match typ {
            ast::Type::Scalar(scalar_type) => self.convert_scalar_type_to_linalg(*scalar_type),
            ast::Type::Vector(len, scalar_type) => {
                let element_type = self.convert_scalar_type_to_linalg(*scalar_type)?;
                Ok(format!("memref<{}x{}>", len, element_type))
            }
            ast::Type::Array(_, scalar_type, dimensions) => {
                let element_type = self.convert_scalar_type_to_linalg(*scalar_type)?;
                let dims: Vec<String> = dimensions.iter().map(|d| d.to_string()).collect();
                Ok(format!("memref<{}x{}>", dims.join("x"), element_type))
            }
            ast::Type::Pointer(_, _) => {
                Ok("memref<1xi64>".to_string()) // Represent pointers as memref of i64
            }
        }
    }

    fn convert_scalar_type_to_linalg(&self, scalar_type: ast::ScalarType) -> Result<String, TranslateError> {
        Ok(match scalar_type {
            ast::ScalarType::U8 | ast::ScalarType::B8 => "i8",
            ast::ScalarType::U16 | ast::ScalarType::B16 => "i16", 
            ast::ScalarType::U32 | ast::ScalarType::B32 => "i32",
            ast::ScalarType::U64 | ast::ScalarType::B64 => "i64",
            ast::ScalarType::S8 => "i8",
            ast::ScalarType::S16 => "i16",
            ast::ScalarType::S32 => "i32", 
            ast::ScalarType::S64 => "i64",
            ast::ScalarType::F16 => "f16",
            ast::ScalarType::F32 => "f32",
            ast::ScalarType::F64 => "f64",
            ast::ScalarType::Pred => "i1",
            _ => return Err(TranslateError::UntypedSymbol),
        }.to_string())
    }

    fn get_memref_type(&self, typ: &ast::Type) -> Result<String, TranslateError> {
        match typ {
            ast::Type::Scalar(scalar_type) => {
                let element_type = self.convert_scalar_type_to_linalg(*scalar_type)?;
                Ok(format!("memref<1x{}>", element_type))
            }
            ast::Type::Vector(len, scalar_type) => {
                let element_type = self.convert_scalar_type_to_linalg(*scalar_type)?;
                Ok(format!("memref<{}x{}>", len, element_type))
            }
            ast::Type::Array(_, scalar_type, dimensions) => {
                let element_type = self.convert_scalar_type_to_linalg(*scalar_type)?;
                let dims: Vec<String> = dimensions.iter().map(|d| d.to_string()).collect();
                Ok(format!("memref<{}x{}>", dims.join("x"), element_type))
            }
            ast::Type::Pointer(_, _) => {
                Ok("memref<1xi64>".to_string())
            }
        }
    }

    fn get_variable_name(&self, var_id: SpirvWord) -> Result<String, TranslateError> {
        self.id_defs.ident_map.get(&var_id)
            .and_then(|entry| entry.name.as_ref())
            .map(|name| name.to_string())
            .ok_or(TranslateError::UnknownSymbol)
    }

    fn get_ssa_value(&self, var_id: SpirvWord) -> Result<String, TranslateError> {
        self.value_map.get(&var_id)
            .cloned()
            .ok_or(TranslateError::UnknownSymbol)
    }

    fn format_immediate_value(&self, value: &ast::ImmediateValue) -> String {
        match value {
            ast::ImmediateValue::U64(v) => v.to_string(),
            ast::ImmediateValue::S64(v) => v.to_string(),
            ast::ImmediateValue::F32(v) => v.to_string(),
            ast::ImmediateValue::F64(v) => v.to_string(),
        }
    }

    fn is_tensor_type(&self, _typ: &ast::Type) -> bool {
        // For now, we'll treat arrays and vectors as tensors in some contexts
        false // Simplified for initial implementation
    }

    fn is_memref_move(&self, _typ: &ast::Type) -> bool {
        // Determine if this move operation should use memref operations
        false // Simplified for initial implementation
    }

    fn get_inferred_tensor_shape(&self) -> Vec<i64> {
        // Return a default shape - in practice this would be inferred from context
        vec![1] // Default to 1D tensor with size 1
    }
}

// Alternative public function for direct PTX to MLIR conversion
pub fn run_direct<'input>(
    id_defs: GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<'input, ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<String, TranslateError> {
    run(id_defs, directives)
}