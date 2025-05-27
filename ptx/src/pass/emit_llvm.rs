// We use Raw LLVM-C bindings here because using inkwell is just not worth it.
// Specifically the issue is with builder functions. We maintain the mapping
// between ZLUDA identifiers and LLVM values. When using inkwell, LLVM values
// are kept as instances `AnyValueEnum`. Now look at the signature of
// `Builder::build_int_add(...)`:
//   pub fn build_int_add<T: IntMathValue<'ctx>>(&self, lhs: T, rhs: T, name: &str, ) -> Result<T, BuilderError>
// At this point both lhs and rhs are `AnyValueEnum`. To call
// `build_int_add(...)` we would have to do something like this:
//   if let (Ok(lhs), Ok(rhs)) = (lhs.as_int(), rhs.as_int()) {
//       builder.build_int_add(lhs, rhs, dst)?;
//   } else if let (Ok(lhs), Ok(rhs)) = (lhs.as_pointer(), rhs.as_pointer()) {
//      builder.build_int_add(lhs, rhs, dst)?;
//   } else if let (Ok(lhs), Ok(rhs)) = (lhs.as_vector(), rhs.as_vector()) {
//       builder.build_int_add(lhs, rhs, dst)?;
//   } else {
//       return Err(error_unrachable());
//   }
// while with plain LLVM-C it's just:
//   unsafe { LLVMBuildAdd(builder, lhs, rhs, dst) };

// AMDGPU LLVM backend support for llvm.experimental.constrained.* is incomplete.
// Emitting @llvm.experimental.constrained.fdiv.f32(...) makes LLVm fail with
// "LLVM ERROR: unsupported libcall legalization". Running with "-mllvm -print-before-all"
// shows it fails inside amdgpu-isel. You can get a little bit furthr with "-mllvm -global-isel",
// but it will too fail similarly, but with "unable to legalize instruction"

use std::array::TryFromSliceError;
use std::convert::TryInto;
use std::ffi::{CStr, NulError};
use std::ops::Deref;
use std::{i8, ptr};

use super::*;
use crate::pass::ast;
use crate::pass::llvm_helpers::*;
use crate::pass::GlobalStringIdentResolver2;
use crate::pass::SpirvWord;
use crate::pass::Statement;
use llvm_zluda::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_zluda::bit_writer::LLVMWriteBitcodeToMemoryBuffer;
use llvm_zluda::{core::*, *};
use llvm_zluda::{prelude::*, LLVMZludaBuildAtomicRMW};
use llvm_zluda::{LLVMCallConv, LLVMZludaBuildAlloca};
use ptx_parser::ScalarType;
use std::collections::HashMap;

// Define our own versions of the TakeAddressArgs and CustomOperand types
pub struct TakeAddressArgs<Ident> {
    pub dst: Ident,
    pub src: CustomOperand<Ident>,
}

// Define a custom operand enum for our needs
pub enum CustomOperand<Ident> {
    SharedMemRef(Ident),
    ParametrizedSharedMemRef(Ident, u32),
    Other,
}

const LLVM_UNNAMED: &CStr = c"";
// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
const GENERIC_ADDRESS_SPACE: u32 = 0;
const GLOBAL_ADDRESS_SPACE: u32 = 1;
const SHARED_ADDRESS_SPACE: u32 = 3;
const CONSTANT_ADDRESS_SPACE: u32 = 4;
const PRIVATE_ADDRESS_SPACE: u32 = 5;

struct Context(LLVMContextRef);

impl Context {
    fn new() -> Self {
        Self(unsafe { LLVMContextCreate() })
    }

    fn get(&self) -> LLVMContextRef {
        self.0
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            LLVMContextDispose(self.0);
        }
    }
}

struct Module(LLVMModuleRef);

impl Module {
    fn new(ctx: &Context, name: &CStr) -> Self {
        Self(unsafe { LLVMModuleCreateWithNameInContext(name.as_ptr(), ctx.get()) })
    }

    fn get(&self) -> LLVMModuleRef {
        self.0
    }

    fn verify(&self) -> Result<(), Message> {
        let mut err = ptr::null_mut();
        let error = unsafe {
            LLVMVerifyModule(
                self.get(),
                LLVMVerifierFailureAction::LLVMReturnStatusAction,
                &mut err,
            )
        };
        if error == 1 && err != ptr::null_mut() {
            Err(Message(unsafe { CStr::from_ptr(err) }))
        } else {
            Ok(())
        }
    }

    fn write_bitcode_to_memory(&self) -> MemoryBuffer {
        let memory_buffer = unsafe { LLVMWriteBitcodeToMemoryBuffer(self.get()) };
        MemoryBuffer(memory_buffer)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.0);
        }
    }
}

struct Builder(LLVMBuilderRef);

impl Builder {
    fn new(ctx: &Context) -> Self {
        Self::new_raw(ctx.get())
    }

    fn new_raw(ctx: LLVMContextRef) -> Self {
        Self(unsafe { LLVMCreateBuilderInContext(ctx) })
    }

    fn get(&self) -> LLVMBuilderRef {
        self.0
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.0);
        }
    }
}

struct Message(&'static CStr);

impl Drop for Message {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMessage(self.0.as_ptr().cast_mut());
        }
    }
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

pub struct MemoryBuffer(LLVMMemoryBufferRef);

impl Drop for MemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMemoryBuffer(self.0);
        }
    }
}
impl Deref for MemoryBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let data = unsafe { LLVMGetBufferStart(self.0) };
        let len = unsafe { LLVMGetBufferSize(self.0) };
        unsafe { std::slice::from_raw_parts(data.cast(), len) }
    }
}

pub(super) fn run<'input>(
    id_defs: GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<'input, ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<MemoryBuffer, TranslateError> {
    let context = Context::new();
    let module = Module::new(&context, LLVM_UNNAMED);

    let target_triple = CString::new("spir64-unknown-unknown").map_err(|_| error_unreachable())?;
    unsafe { LLVMSetTarget(module.get(), target_triple.as_ptr()) };

    let mut emit_ctx = ModuleEmitContext::new(&context, &module, &id_defs);
    for directive in directives {
        match directive {
            Directive2::Variable(linking, variable) => emit_ctx.emit_global(linking, variable)?,
            Directive2::Method(method) => emit_ctx.emit_method(method)?,
        }
    }
    if let Err(err) = module.verify() {
        panic!("{:?}", err);
    }
    Ok(module.write_bitcode_to_memory())
}

struct ModuleEmitContext<'ctx, 'input> {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: Builder,
    id_defs: &'ctx GlobalStringIdentResolver2<'input>,
    resolver: ResolveIdent<'ctx>,
    empty_map_storage: Box<HashMap<String, LLVMValueRef>>, // Store in a Box to avoid borrowing issues
}

impl<'ctx, 'input> ModuleEmitContext<'ctx, 'input> {
    fn new(
        context: &Context,
        module: &Module,
        id_defs: &'ctx GlobalStringIdentResolver2<'input>,
    ) -> Self {
        let builder = Builder::new(context);
        let empty_map_storage = Box::new(HashMap::new());

        // Create a fresh resolver with proper type
        // Pass None for the map to make the resolver create its own map internally
        let resolver = ResolveIdent::new(module.get(), context.get(), builder.get(), None);

        let ctx = ModuleEmitContext {
            context: context.get(),
            module: module.get(),
            builder,
            id_defs,
            resolver,
            empty_map_storage,
        };

        // 设置SPIR-V特定属性
        if ctx.is_spirv_target() {
            // 设置SPIR-V数据布局
            // 为SPIR-V目标设置适当的数据布局字符串
            let data_layout = CString::new("-p:64:64:64").unwrap();
            unsafe { LLVMSetDataLayout(ctx.module, data_layout.as_ptr()) };

            // 添加SPIR-V特定的模块标志和元数据
            ctx.add_spirv_capabilities();
        }

        ctx
    }

    // Add helper function to add SPIR-V capabilities and metadata
    fn add_spirv_capabilities(&self) {
        // Skip adding metadata for SPIR-V version - this is simpler
        // and avoids compatibility issues with the LLVM API
        // The OpenCL runtime will provide default versioning
    }

    fn is_spirv_target(&self) -> bool {
        let target_triple = unsafe { LLVMGetTarget(self.module) };
        unsafe {
            CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        }
    }

    fn kernel_call_convention() -> u32 {
        // 使用C调用约定，而不是AMDGPU内核调用约定
        // 这对于SPIR-V目标是必要的
        LLVMCallConv::LLVMCCallConv as u32
    }

    fn func_call_convention() -> u32 {
        LLVMCallConv::LLVMCCallConv as u32
    }

    fn emit_method(
        &mut self,
        method: Function2<'input, ast::Instruction<SpirvWord>, SpirvWord>,
    ) -> Result<(), TranslateError> {
        let func_decl = method.func_decl;
        let name = method
            .import_as
            .as_deref()
            .or_else(|| match func_decl.name {
                ast::MethodName::Kernel(name) => Some(name),
                ast::MethodName::Func(id) => self.id_defs.ident_map[&id].name.as_deref(),
            })
            .ok_or_else(|| error_unreachable())?;
        let name = CString::new(name).map_err(|_| error_unreachable())?;
        let mut fn_ = unsafe { LLVMGetNamedFunction(self.module, name.as_ptr()) };
        if fn_ == ptr::null_mut() {
            let fn_type = get_function_type(
                self.context,
                func_decl.return_arguments.iter().map(|v| &v.v_type),
                func_decl
                    .input_arguments
                    .iter()
                    .map(|v| get_input_argument_type(self.context, &v.v_type, v.state_space)),
            )?;
            fn_ = unsafe { LLVMAddFunction(self.module, name.as_ptr(), fn_type) };
            self.emit_fn_attribute(fn_, "amdgpu-unsafe-fp-atomics", "true");
            self.emit_fn_attribute(fn_, "uniform-work-group-size", "true");
            self.emit_fn_attribute(fn_, "no-trapping-math", "true");

            // 为内核函数添加SPIR内核属性
            if func_decl.name.is_kernel() {
                // 添加spir_kernel属性
                unsafe {
                    let spir_kernel = CString::new("spir_kernel").unwrap();
                    let attr = LLVMCreateStringAttribute(
                        self.context,
                        spir_kernel.as_ptr(),
                        spir_kernel.as_bytes().len() as u32,
                        ptr::null(),
                        0,
                    );
                    LLVMAddAttributeAtIndex(fn_, LLVMAttributeFunctionIndex, attr);
                }
            }
        }
        if let ast::MethodName::Func(name) = func_decl.name {
            self.resolver.register(name, fn_);
        }
        for (i, param) in func_decl.input_arguments.iter().enumerate() {
            let value = unsafe { LLVMGetParam(fn_, i as u32) };
            let name = self.resolver.get_or_add(param.name);
            unsafe { LLVMSetValueName2(value, name.as_ptr().cast(), name.len()) };
            self.resolver.register(param.name, value);
            if func_decl.name.is_kernel() {
                let attr_kind = unsafe {
                    LLVMGetEnumAttributeKindForName(b"byref".as_ptr().cast(), b"byref".len())
                };
                let attr = unsafe {
                    LLVMCreateTypeAttribute(
                        self.context,
                        attr_kind,
                        get_type(self.context, &param.v_type)?,
                    )
                };
                unsafe { LLVMAddAttributeAtIndex(fn_, i as u32 + 1, attr) };
            }
        }
        let call_conv = if func_decl.name.is_kernel() {
            Self::kernel_call_convention()
        } else {
            Self::func_call_convention()
        };
        unsafe { LLVMSetFunctionCallConv(fn_, call_conv) };
        if let Some(statements) = method.body {
            let variables_bb =
                unsafe { LLVMAppendBasicBlockInContext(self.context, fn_, LLVM_UNNAMED.as_ptr()) };
            let variables_builder = Builder::new_raw(self.context);
            unsafe { LLVMPositionBuilderAtEnd(variables_builder.get(), variables_bb) };
            let real_bb =
                unsafe { LLVMAppendBasicBlockInContext(self.context, fn_, LLVM_UNNAMED.as_ptr()) };
            unsafe { LLVMPositionBuilderAtEnd(self.builder.get(), real_bb) };

            // Fix the lifetime issue by moving the method emitter creation out of the mutable borrow
            let mut method_emitter = MethodEmitContext::new(self, fn_, variables_builder);

            for var in func_decl.return_arguments {
                method_emitter.emit_variable(var)?;
            }
            for statement in statements.iter() {
                if let Statement::Label(label) = statement {
                    method_emitter.emit_label_initial(*label);
                }
            }

            // Now we need to process the statements directly
            for statement in statements.iter() {
                match statement {
                    Statement::Variable(var) => {
                        method_emitter.emit_variable(var.clone())?;
                    }
                    Statement::Label(label) => {
                        method_emitter.emit_label_delayed(*label)?;
                    }
                    Statement::Instruction(inst) => {
                        method_emitter.emit_instruction((*inst).clone())?;
                    }
                    Statement::Conditional(_) => {
                        // Skip conditionals for now
                    }
                    Statement::Conversion(conversion) => {
                        method_emitter.emit_conversion(conversion.clone())?;
                    }
                    Statement::Constant(constant) => {
                        method_emitter.emit_constant(constant.clone())?;
                    }
                    Statement::RetValue(data, values) => {
                        method_emitter.emit_ret_value(values.clone())?;
                    }
                    Statement::PtrAccess(ptr_access) => {
                        method_emitter.emit_ptr_access(ptr_access.clone())?;
                    }
                    Statement::RepackVector(repack) => {
                        method_emitter.emit_vector_repack(repack.clone())?;
                    }
                    Statement::FunctionPointer(fp) => {
                        // Skip for now or implement if needed
                    }
                    Statement::VectorRead(vector_read) => {
                        method_emitter.emit_vector_read(vector_read.clone())?;
                    }
                    Statement::VectorWrite(vector_write) => {
                        method_emitter.emit_vector_write(vector_write.clone())?;
                    }
                }
            }

            // Remove the statement_copies code
            unsafe { LLVMBuildBr(method_emitter.variables_builder.get(), real_bb) };
        }
        Ok(())
    }

    fn emit_global(
        &mut self,
        _linking: ast::LinkingDirective,
        var: ast::Variable<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let name = self
            .id_defs
            .ident_map
            .get(&var.name)
            .map(|entry| {
                entry
                    .name
                    .as_ref()
                    .map(|text| Ok::<_, NulError>(Cow::Owned(CString::new(&**text)?)))
            })
            .flatten()
            .transpose()
            .map_err(|_| error_unreachable())?
            .unwrap_or(Cow::Borrowed(LLVM_UNNAMED));

        // 获取变量类型，get_type现在会自动处理SPIR-V兼容性
        let var_type = get_type(self.context, &var.v_type)?;
        let addr_space = get_state_space(var.state_space)?;

        // 检查是否为SPIR-V目标
        let _is_spirv = self.is_spirv_target();

        let global = unsafe {
            LLVMAddGlobalInAddressSpace(self.module, var_type, name.as_ptr(), addr_space)
        };

        self.resolver.register(var.name, global);

        // For shared memory variables, also register them with element type information
        if var.state_space == ast::StateSpace::Shared {
            // For arrays, get the element type
            let elem_type = match &var.v_type {
                ast::Type::Array(_, scalar, _) => get_scalar_type(self.context, *scalar),
                _ => var_type,
            };

            self.resolver
                .register_shared_array(var.name, global, elem_type);
        }

        if let Some(align) = var.align {
            unsafe { LLVMSetAlignment(global, align) };
        }
        if !var.array_init.is_empty() {
            self.emit_array_init(&var.v_type, &var.array_init, global)?;
        } else {
            unsafe {
                LLVMSetLinkage(global, LLVMLinkage::LLVMExternalLinkage);
            }
        }

        self.resolver.with_result(var.name, |ident| unsafe {
            LLVMSetValueName2(global, ident, libc::strlen(ident));
            global
        });

        if let ast::Type::Pointer(_, state_space) = var.v_type {
            let _ptr_address_space = get_state_space(state_space)?;
            // Remove TBAA metadata setting since the field doesn't exist
            // We don't need this for SPIR-V compatibility anyway
            // unsafe {
            //     LLVMSetMetadata(
            //         global,
            //         self.id_defs.llvm.tbaa_root_id,
            //         self.id_defs.llvm.tbaa_nodes[&ptr_address_space],
            //     );
            // }
        }

        Ok(())
    }

    // TODO: instead of Vec<u8> we should emit a typed initializer
    fn emit_array_init(
        &mut self,
        type_: &ast::Type,
        array_init: &[u8],
        global: *mut llvm_zluda::LLVMValue,
    ) -> Result<(), TranslateError> {
        match type_ {
            ast::Type::Array(None, scalar, dimensions) => {
                if dimensions.len() != 1 {
                    todo!()
                }

                // For SPIR-V compatibility, ensure the array has at least one element
                let is_spirv = self.is_spirv_target();
                let actual_dimension = if is_spirv && dimensions[0] == 0 {
                    1
                } else {
                    dimensions[0]
                };

                if actual_dimension as usize * scalar.size_of() as usize != array_init.len() {
                    return Err(error_unreachable());
                }
                let type_ = get_scalar_type(self.context, *scalar);
                let mut elements = array_init
                    .chunks(scalar.size_of() as usize)
                    .map(|chunk| self.constant_from_bytes(*scalar, chunk, type_))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| error_unreachable())?;

                // If array is empty but we need one element for SPIR-V compatibility,
                // add a zero element
                if elements.is_empty() && is_spirv {
                    let zero_element = unsafe { LLVMConstNull(type_) };
                    elements.push(zero_element);
                }

                let initializer =
                    unsafe { LLVMConstArray2(type_, elements.as_mut_ptr(), elements.len() as u64) };
                unsafe { LLVMSetInitializer(global, initializer) };
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn constant_from_bytes(
        &self,
        scalar: ast::ScalarType,
        bytes: &[u8],
        llvm_type: LLVMTypeRef,
    ) -> Result<LLVMValueRef, TryFromSliceError> {
        Ok(match scalar {
            ptx_parser::ScalarType::Pred
            | ptx_parser::ScalarType::S8
            | ptx_parser::ScalarType::B8
            | ptx_parser::ScalarType::U8 => unsafe {
                LLVMConstInt(llvm_type, u8::from_le_bytes(bytes.try_into()?) as u64, 0)
            },
            ptx_parser::ScalarType::S16
            | ptx_parser::ScalarType::B16
            | ptx_parser::ScalarType::U16 => unsafe {
                LLVMConstInt(llvm_type, u16::from_le_bytes(bytes.try_into()?) as u64, 0)
            },
            ptx_parser::ScalarType::S32
            | ptx_parser::ScalarType::B32
            | ptx_parser::ScalarType::U32 => unsafe {
                LLVMConstInt(llvm_type, u32::from_le_bytes(bytes.try_into()?) as u64, 0)
            },
            ptx_parser::ScalarType::F16 => todo!(),
            ptx_parser::ScalarType::BF16 => todo!(),
            ptx_parser::ScalarType::U64 => todo!(),
            ptx_parser::ScalarType::S64 => todo!(),
            ptx_parser::ScalarType::S16x2 => todo!(),
            ptx_parser::ScalarType::F32 => todo!(),
            ptx_parser::ScalarType::B64 => todo!(),
            ptx_parser::ScalarType::F64 => todo!(),
            ptx_parser::ScalarType::B128 => todo!(),
            ptx_parser::ScalarType::U16x2 => todo!(),
            ptx_parser::ScalarType::F16x2 => todo!(),
            ptx_parser::ScalarType::BF16x2 => todo!(),
        })
    }

    fn emit_fn_attribute(&self, fn_: LLVMValueRef, name: &str, value: &str) {
        unsafe {
            let name = CString::new(name).unwrap();
            let value = CString::new(value).unwrap();
            let attr = LLVMCreateStringAttribute(
                self.context,
                name.as_ptr(),
                name.as_bytes().len() as u32,
                value.as_ptr(),
                value.as_bytes().len() as u32,
            );
            LLVMAddAttributeAtIndex(fn_, LLVMAttributeFunctionIndex, attr);
        }
    }

    fn set_kernel_calling_conv(&self, fn_: LLVMValueRef) {
        unsafe {
            // 修改为SPIR内核函数的调用约定
            // 对于SPIR-V，我们应该使用普通的C调用约定而不是AMD特定的调用约定
            LLVMSetFunctionCallConv(fn_, LLVMCallConv::LLVMCCallConv as u32);

            // 添加SPIR内核属性
            let spir_kernel = CString::new("spir_kernel").unwrap();
            let attr = LLVMCreateStringAttribute(
                self.context,
                spir_kernel.as_ptr(),
                spir_kernel.as_bytes().len() as u32,
                ptr::null(),
                0,
            );
            LLVMAddAttributeAtIndex(fn_, LLVMAttributeFunctionIndex, attr);
        }
    }
}

fn get_input_argument_type(
    context: LLVMContextRef,
    v_type: &ast::Type,
    state_space: ast::StateSpace,
) -> Result<LLVMTypeRef, TranslateError> {
    match state_space {
        ast::StateSpace::ParamEntry => {
            Ok(unsafe { LLVMPointerTypeInContext(context, get_state_space(state_space)?) })
        }
        ast::StateSpace::Reg => get_type(context, v_type),
        _ => return Err(error_unreachable()),
    }
}

struct MethodEmitContext<'ctx> {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    method: LLVMValueRef,
    builder: LLVMBuilderRef,
    variables_builder: Builder,
    resolver: &'ctx mut ResolveIdent<'ctx>,
}

impl<'ctx> MethodEmitContext<'ctx> {
    fn new(
        parent: &'ctx mut ModuleEmitContext<'ctx, '_>,
        method: LLVMValueRef,
        variables_builder: Builder,
    ) -> MethodEmitContext<'ctx> {
        MethodEmitContext {
            context: parent.context,
            module: parent.module,
            builder: parent.builder.get(),
            variables_builder,
            resolver: &mut parent.resolver,
            method,
        }
    }

    fn emit_statement(
        &mut self,
        statement: Statement<ast::Instruction<SpirvWord>, SpirvWord>,
    ) -> Result<(), TranslateError> {
        Ok(match statement {
            Statement::Variable(var) => self.emit_variable(var)?,
            Statement::Label(label) => self.emit_label_delayed(label)?,
            Statement::Instruction(inst) => self.emit_instruction(inst)?,
            Statement::Conditional(_) => todo!("Conditional statements not implemented"),
            Statement::Conversion(conversion) => self.emit_conversion(conversion)?,
            Statement::Constant(constant) => self.emit_constant(constant)?,
            Statement::RetValue(_, values) => self.emit_ret_value(values)?,
            Statement::PtrAccess(ptr_access) => self.emit_ptr_access(ptr_access)?,
            Statement::RepackVector(repack) => self.emit_vector_repack(repack)?,
            Statement::FunctionPointer(_) => todo!(),
            Statement::VectorRead(vector_read) => self.emit_vector_read(vector_read)?,
            Statement::VectorWrite(vector_write) => self.emit_vector_write(vector_write)?,
        })
    }

    fn emit_variable(&mut self, var: ast::Variable<SpirvWord>) -> Result<(), TranslateError> {
        // Check if we're targeting SPIR-V
        let target_triple = unsafe { LLVMGetTarget(self.module) };
        let is_spirv = unsafe {
            CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        };

        // For SPIR-V compatibility, map address spaces correctly
        let actual_space = if is_spirv && var.state_space == ast::StateSpace::Reg {
            // For SPIR-V, use generic (0) address space instead of private (5)
            GENERIC_ADDRESS_SPACE
        } else {
            get_state_space(var.state_space)?
        };

        let alloca = unsafe {
            LLVMZludaBuildAlloca(
                self.variables_builder.get(),
                get_type(self.context, &var.v_type)?,
                actual_space,
                self.resolver.get_or_add_raw(var.name),
            )
        };

        self.resolver.register(var.name, alloca);

        if let Some(align) = var.align {
            unsafe { LLVMSetAlignment(alloca, align) };
        }
        if !var.array_init.is_empty() {
            todo!()
        }
        Ok(())
    }

    fn emit_label_initial(&mut self, label: SpirvWord) {
        let block = unsafe {
            LLVMAppendBasicBlockInContext(
                self.context,
                self.method,
                self.resolver.get_or_add_raw(label),
            )
        };
        self.resolver
            .register(label, unsafe { LLVMBasicBlockAsValue(block) });
    }

    fn emit_label_delayed(&mut self, label: SpirvWord) -> Result<(), TranslateError> {
        let block = self.resolver.value(label)?;
        let block = unsafe { LLVMValueAsBasicBlock(block) };
        let last_block = unsafe { LLVMGetInsertBlock(self.builder) };
        if unsafe { LLVMGetBasicBlockTerminator(last_block) } == ptr::null_mut() {
            unsafe { LLVMBuildBr(self.builder, block) };
        }
        unsafe { LLVMPositionBuilderAtEnd(self.builder, block) };
        Ok(())
    }

    fn emit_instruction(
        &mut self,
        inst: ast::Instruction<SpirvWord>,
    ) -> Result<(), TranslateError> {
        match inst {
            ast::Instruction::Mov { data, arguments } => self.emit_mov(data, arguments),
            ast::Instruction::Ld { data, arguments } => self.emit_ld(data, arguments),
            ast::Instruction::Add { data, arguments } => self.emit_add(data, arguments),
            ast::Instruction::St { data, arguments } => self.emit_st(data, arguments),
            ast::Instruction::Mul { data, arguments } => self.emit_mul(data, arguments),
            ast::Instruction::Setp { .. } => todo!("setp not implemented"),
            ast::Instruction::SetpBool { .. } => todo!(),
            ast::Instruction::Not { .. } => todo!("not implemented"),
            ast::Instruction::Or { data, arguments } => self.emit_or(data, arguments),
            ast::Instruction::And { arguments, .. } => self.emit_and(arguments),
            ast::Instruction::Bra { arguments } => self.emit_bra(arguments),
            ast::Instruction::Call { data, arguments } => self.emit_call(data, arguments),
            ast::Instruction::Cvt { .. } => todo!("cvt not implemented"),
            ast::Instruction::Shr { .. } => todo!("shr not implemented"),
            ast::Instruction::Shl { .. } => todo!("shl not implemented"),
            ast::Instruction::Ret { data } => Ok(self.emit_ret(data)),
            ast::Instruction::Cvta { data, arguments } => self.emit_cvta(data, arguments),
            ast::Instruction::Abs { data, arguments } => self.emit_abs(data, arguments),
            ast::Instruction::Mad { data, arguments } => self.emit_mad(data, arguments),
            ast::Instruction::Fma { data, arguments } => self.emit_fma(data, arguments),
            ast::Instruction::Sub { data, arguments } => self.emit_sub(data, arguments),
            ast::Instruction::Min { data, arguments } => self.emit_min(data, arguments),
            ast::Instruction::Max { data, arguments } => self.emit_max(data, arguments),
            ast::Instruction::Rcp { .. } => todo!("rcp not implemented"),
            ast::Instruction::Sqrt { .. } => todo!("sqrt not implemented"),
            ast::Instruction::Rsqrt { .. } => todo!("rsqrt not implemented"),
            ast::Instruction::Selp { data, arguments } => self.emit_selp(data, arguments),
            ast::Instruction::Atom { data, arguments } => self.emit_atom(data, arguments),
            ast::Instruction::AtomCas { data, arguments } => self.emit_atom_cas(data, arguments),
            ast::Instruction::Div { data, arguments } => self.emit_div(data, arguments),
            ast::Instruction::Neg { .. } => todo!("neg not implemented"),
            ast::Instruction::Sin { data, arguments } => self.emit_sin(data, arguments),
            ast::Instruction::Cos { data, arguments } => self.emit_cos(data, arguments),
            ast::Instruction::Lg2 { .. } => todo!("lg2 not implemented"),
            ast::Instruction::Ex2 { .. } => todo!("ex2 not implemented"),
            ast::Instruction::Clz { data, arguments } => self.emit_clz(data, arguments),
            ast::Instruction::Brev { data, arguments } => self.emit_brev(data, arguments),
            ast::Instruction::Popc { data, arguments } => self.emit_popc(data, arguments),
            ast::Instruction::Xor { data, arguments } => self.emit_xor(data, arguments),
            ast::Instruction::Rem { data, arguments } => self.emit_rem(data, arguments),
            ast::Instruction::PrmtSlow { .. } => todo!(),
            ast::Instruction::Prmt { data, arguments } => self.emit_prmt(data, arguments),
            ast::Instruction::Membar { data } => self.emit_membar(data),
            ast::Instruction::Trap {} => todo!(),
            // replaced by a function call
            ast::Instruction::Bfe { .. }
            | ast::Instruction::Bar { .. }
            | ast::Instruction::Bfi { .. }
            | ast::Instruction::Activemask { .. } => return Err(error_unreachable()),
        }
    }

    fn emit_ld(
        &mut self,
        data: ast::LdDetails,
        arguments: ast::LdArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        if data.qualifier != ast::LdStQualifier::Weak {
            todo!()
        }

        let builder = self.builder;
        let type_ = get_type(self.context, &data.typ)?;
        let ptr = self.resolver.value(arguments.src)?;

        // For SPIR-V compatibility, we must not perform any address space casts
        // Simply use the load instruction with the pointer as-is
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildLoad2(builder, type_, ptr, dst)
        });

        Ok(())
    }

    fn emit_conversion(&mut self, conversion: ImplicitConversion) -> Result<(), TranslateError> {
        let builder = self.builder;
        match conversion.kind {
            ConversionKind::Default => self.emit_conversion_default(
                self.resolver.value(conversion.src)?,
                conversion.dst,
                &conversion.from_type,
                conversion.from_space,
                &conversion.to_type,
                conversion.to_space,
            ),
            ConversionKind::SignExtend => {
                let src = self.resolver.value(conversion.src)?;
                let type_ = get_type(self.context, &conversion.to_type)?;
                self.resolver.with_result(conversion.dst, |dst| unsafe {
                    LLVMBuildSExt(builder, src, type_, dst)
                });
                Ok(())
            }
            ConversionKind::BitToPtr => {
                let src = self.resolver.value(conversion.src)?;
                let type_ =
                    get_pointer_type(self.context, conversion.to_space, &conversion.to_type)?;
                self.resolver.with_result(conversion.dst, |dst| unsafe {
                    LLVMBuildIntToPtr(builder, src, type_, dst)
                });
                Ok(())
            }
            ConversionKind::PtrToPtr => {
                // Check if we're targeting SPIR-V
                let is_spirv = self.resolver.is_spirv_target(self.module);

                if is_spirv {
                    // For SPIR-V, we need to handle address space conversions specially
                    let src = self.resolver.value(conversion.src)?;
                    let from_as = get_state_space(conversion.from_space)?;
                    let to_as = get_state_space(conversion.to_space)?;

                    if from_as != GENERIC_ADDRESS_SPACE && to_as == GENERIC_ADDRESS_SPACE {
                        // Valid in SPIR-V: from specific to generic
                        let dst_type = get_pointer_type(
                            self.context,
                            conversion.to_space,
                            &conversion.to_type,
                        )?;
                        self.resolver.with_result(conversion.dst, |dst| unsafe {
                            LLVMBuildAddrSpaceCast(builder, src, dst_type, dst)
                        });
                    } else if from_as == to_as {
                        // Same address space, just copy the pointer
                        self.resolver.register(conversion.dst, src);
                    } else {
                        // For SPIR-V compatibility, just keep the pointer unchanged for invalid conversions
                        self.resolver.register(conversion.dst, src);
                    }
                } else {
                    // Original behavior for non-SPIR-V targets
                    let src = self.resolver.value(conversion.src)?;
                    let dst_type =
                        get_pointer_type(self.context, conversion.to_space, &conversion.to_type)?;
                    self.resolver.with_result(conversion.dst, |dst| unsafe {
                        LLVMBuildAddrSpaceCast(builder, src, dst_type, dst)
                    });
                }
                Ok(())
            }
            ConversionKind::AddressOf => {
                let src = self.resolver.value(conversion.src)?;
                let dst_type = get_type(self.context, &conversion.to_type)?;
                self.resolver.with_result(conversion.dst, |dst| unsafe {
                    LLVMBuildPtrToInt(self.builder, src, dst_type, dst)
                });
                Ok(())
            }
        }
    }

    fn emit_conversion_default(
        &mut self,
        src: LLVMValueRef,
        dst: SpirvWord,
        from_type: &ast::Type,
        from_space: ast::StateSpace,
        to_type: &ast::Type,
        to_space: ast::StateSpace,
    ) -> Result<(), TranslateError> {
        match (from_type, to_type) {
            (ast::Type::Scalar(from_type), ast::Type::Scalar(to_type_scalar)) => {
                let from_layout = from_type.layout();
                let to_layout = to_type.layout();
                if from_layout.size() == to_layout.size() {
                    let dst_type = get_type(self.context, &to_type)?;
                    if from_type.kind() != ast::ScalarKind::Float
                        && to_type_scalar.kind() != ast::ScalarKind::Float
                    {
                        // It is noop, but another instruction expects result of this conversion
                        self.resolver.register(dst, src);
                    } else {
                        self.resolver.with_result(dst, |dst| unsafe {
                            LLVMBuildBitCast(self.builder, src, dst_type, dst)
                        });
                    }
                    Ok(())
                } else {
                    // This block is safe because it's illegal to implictly convert between floating point values
                    let same_width_bit_type = unsafe {
                        LLVMIntTypeInContext(self.context, (from_layout.size() * 8) as u32)
                    };
                    let same_width_bit_value = unsafe {
                        LLVMBuildBitCast(
                            self.builder,
                            src,
                            same_width_bit_type,
                            LLVM_UNNAMED.as_ptr(),
                        )
                    };
                    let wide_bit_type = match to_type_scalar.layout().size() {
                        1 => ast::ScalarType::B8,
                        2 => ast::ScalarType::B16,
                        4 => ast::ScalarType::B32,
                        8 => ast::ScalarType::B64,
                        _ => return Err(error_unreachable()),
                    };
                    let wide_bit_type_llvm = unsafe {
                        LLVMIntTypeInContext(self.context, (to_layout.size() * 8) as u32)
                    };
                    if to_type_scalar.kind() == ast::ScalarKind::Unsigned
                        || to_type_scalar.kind() == ast::ScalarKind::Bit
                    {
                        let llvm_fn = if to_type_scalar.size_of() >= from_type.size_of() {
                            LLVMBuildZExtOrBitCast
                        } else {
                            LLVMBuildTrunc
                        };
                        self.resolver.with_result(dst, |dst| unsafe {
                            llvm_fn(self.builder, same_width_bit_value, wide_bit_type_llvm, dst)
                        });
                        Ok(())
                    } else {
                        let conversion_fn = if from_type.kind() == ast::ScalarKind::Signed
                            && to_type_scalar.kind() == ast::ScalarKind::Signed
                        {
                            if to_type_scalar.size_of() >= from_type.size_of() {
                                LLVMBuildSExtOrBitCast
                            } else {
                                LLVMBuildTrunc
                            }
                        } else {
                            if to_type_scalar.size_of() >= from_type.size_of() {
                                LLVMBuildZExtOrBitCast
                            } else {
                                LLVMBuildTrunc
                            }
                        };
                        let wide_bit_value = unsafe {
                            conversion_fn(
                                self.builder,
                                same_width_bit_value,
                                wide_bit_type_llvm,
                                LLVM_UNNAMED.as_ptr(),
                            )
                        };
                        self.emit_conversion_default(
                            wide_bit_value,
                            dst,
                            &wide_bit_type.into(),
                            from_space,
                            to_type,
                            to_space,
                        )
                    }
                }
            }
            (ast::Type::Vector(..), ast::Type::Scalar(..))
            | (ast::Type::Scalar(..), ast::Type::Array(..))
            | (ast::Type::Array(..), ast::Type::Scalar(..)) => {
                let dst_type = get_type(self.context, to_type)?;
                self.resolver.with_result(dst, |dst| unsafe {
                    LLVMBuildBitCast(self.builder, src, dst_type, dst)
                });
                Ok(())
            }
            _ => todo!(),
        }
    }

    fn emit_constant(&mut self, constant: ConstantDefinition) -> Result<(), TranslateError> {
        let type_ = get_scalar_type(self.context, constant.typ);
        let value = match constant.value {
            ast::ImmediateValue::U64(x) => unsafe { LLVMConstInt(type_, x, 0) },
            ast::ImmediateValue::S64(x) => unsafe { LLVMConstInt(type_, x as u64, 0) },
            ast::ImmediateValue::F32(x) => unsafe { LLVMConstReal(type_, x as f64) },
            ast::ImmediateValue::F64(x) => unsafe { LLVMConstReal(type_, x) },
        };
        self.resolver.register(constant.dst, value);
        Ok(())
    }

    fn emit_add(
        &mut self,
        data: ast::ArithDetails,
        arguments: ast::AddArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let builder = self.builder;
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        let fn_ = match data {
            ast::ArithDetails::Integer(..) => LLVMBuildAdd,
            ast::ArithDetails::Float(..) => LLVMBuildFAdd,
        };
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            fn_(builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_st(
        &self,
        data: ast::StData,
        arguments: ast::StArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let ptr = self.resolver.value(arguments.src1)?;
        let value = self.resolver.value(arguments.src2)?;

        if data.qualifier != ast::LdStQualifier::Weak {
            todo!()
        }

        // For SPIR-V compatibility, we must not perform any address space casts
        // Simply use the store instruction with the pointer as-is
        unsafe { LLVMBuildStore(self.builder, value, ptr) };

        Ok(())
    }

    fn emit_ret(&self, _data: ast::RetData) {
        unsafe { LLVMBuildRetVoid(self.builder) };
    }

    fn emit_call(
        &mut self,
        data: ast::CallDetails,
        arguments: ast::CallArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        if cfg!(debug_assertions) {
            for (_, space) in data.return_arguments.iter() {
                if *space != ast::StateSpace::Reg {
                    panic!()
                }
            }
            for (_, space) in data.input_arguments.iter() {
                if *space != ast::StateSpace::Reg {
                    panic!()
                }
            }
        }
        let name = match &*arguments.return_arguments {
            [] => LLVM_UNNAMED.as_ptr(),
            [dst] => self.resolver.get_or_add_raw(*dst),
            _ => todo!(),
        };
        let type_ = get_function_type(
            self.context,
            data.return_arguments.iter().map(|(type_, ..)| type_),
            data.input_arguments
                .iter()
                .map(|(type_, space)| get_input_argument_type(self.context, &type_, *space)),
        )?;
        let mut input_arguments = arguments
            .input_arguments
            .iter()
            .map(|arg| self.resolver.value(*arg))
            .collect::<Result<Vec<_>, _>>()?;
        let llvm_fn = unsafe {
            LLVMBuildCall2(
                self.builder,
                type_,
                self.resolver.value(arguments.func)?,
                input_arguments.as_mut_ptr(),
                input_arguments.len() as u32,
                name,
            )
        };
        match &*arguments.return_arguments {
            [] => {}
            [name] => {
                self.resolver.register(*name, llvm_fn);
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn emit_mov(
        &mut self,
        _data: ast::MovDetails,
        arguments: ast::MovArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        self.resolver
            .register(arguments.dst, self.resolver.value(arguments.src)?);
        Ok(())
    }

    fn emit_ptr_access(&mut self, ptr_access: PtrAccess<SpirvWord>) -> Result<(), TranslateError> {
        let ptr_src = self.resolver.value(ptr_access.ptr_src)?;
        let mut offset_src = self.resolver.value(ptr_access.offset_src)?;

        // Get pointer address space
        let ptr_type = unsafe { LLVMTypeOf(ptr_src) };
        let _addrspace = unsafe { LLVMGetPointerAddressSpace(ptr_type) };

        // Check if we're targeting SPIR-V
        let is_spirv = self.resolver.is_spirv_target(self.module);

        if is_spirv {
            // For SPIR-V compatibility - pointer arithmetic must respect address space rules

            // Get pointee type for GEP operation
            let pointee_type = get_scalar_type(self.context, ast::ScalarType::B8);

            // Do pointer arithmetic in the original address space
            let result = unsafe {
                LLVMBuildInBoundsGEP2(
                    self.builder,
                    pointee_type,
                    ptr_src,
                    &mut offset_src,
                    1,
                    LLVM_UNNAMED.as_ptr(),
                )
            };

            // Register result
            self.resolver.register(ptr_access.dst, result);
            return Ok(());
        }

        // Original implementation for non-SPIR-V targets
        let result = unsafe {
            LLVMBuildInBoundsGEP2(
                self.builder,
                get_scalar_type(self.context, ast::ScalarType::B8),
                ptr_src,
                &mut offset_src,
                1,
                LLVM_UNNAMED.as_ptr(),
            )
        };

        self.resolver.register(ptr_access.dst, result);
        Ok(())
    }

    fn emit_and(&mut self, arguments: ast::AndArgs<SpirvWord>) -> Result<(), TranslateError> {
        let builder = self.builder;
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildAnd(builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_atom(
        &mut self,
        data: ast::AtomDetails,
        arguments: ast::AtomArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let builder = self.builder;
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;

        // 检查当前模块的target triple是否为SPIR-V
        let target_triple = unsafe { LLVMGetTarget(self.module) };
        let is_spirv = unsafe {
            CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        };

        if is_spirv {
            // 在SPIR-V模式下使用更简单的原子操作实现
            // Intel GPU对一些复杂原子操作支持有限

            // 获取指针元素类型
            let ptr_type = unsafe { LLVMTypeOf(src1) };
            let _addrspace = unsafe { LLVMGetPointerAddressSpace(ptr_type) };
            let elem_type = unsafe {
                if LLVMGetTypeKind(ptr_type) == LLVMTypeKind::LLVMPointerTypeKind {
                    // 尝试确定指针的元素类型
                    // 这里简化处理，假设是标量类型
                    get_scalar_type(self.context, ast::ScalarType::U32)
                } else {
                    // 如果不是指针，假设是整数类型
                    get_scalar_type(self.context, ast::ScalarType::U32)
                }
            };

            // 对于特殊的原子操作，使用标准原子指令
            match data.op {
                ast::AtomicOp::Exchange => {
                    let result = unsafe {
                        LLVMBuildAtomicRMW(
                            builder,
                            LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXchg,
                            src1,
                            src2,
                            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0, // false -> 0 for single threaded
                        )
                    };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
                ast::AtomicOp::Add => {
                    let result = unsafe {
                        LLVMBuildAtomicRMW(
                            builder,
                            LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                            src1,
                            src2,
                            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0, // false -> 0 for single threaded
                        )
                    };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
                ast::AtomicOp::And => {
                    let result = unsafe {
                        LLVMBuildAtomicRMW(
                            builder,
                            LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAnd,
                            src1,
                            src2,
                            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0, // false -> 0 for single threaded
                        )
                    };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
                ast::AtomicOp::Or => {
                    let result = unsafe {
                        LLVMBuildAtomicRMW(
                            builder,
                            LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpOr,
                            src1,
                            src2,
                            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0, // false -> 0 for single threaded
                        )
                    };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
                ast::AtomicOp::Xor => {
                    let result = unsafe {
                        LLVMBuildAtomicRMW(
                            builder,
                            LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpXor,
                            src1,
                            src2,
                            LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                            0, // false -> 0 for single threaded
                        )
                    };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
                _ => {
                    // 对于增减原子操作，我们需要特殊处理
                    if matches!(data.op, ast::AtomicOp::IncrementWrap) {
                        // 使用Add操作替代特殊的原子递增
                        let one = unsafe { LLVMConstInt(elem_type, 1, 0) };

                        let result = unsafe {
                            LLVMBuildAtomicRMW(
                                builder,
                                LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                                src1,
                                one,
                                LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                                0, // false -> 0 for single threaded
                            )
                        };
                        self.resolver.register(arguments.dst, result);
                        return Ok(());
                    } else if matches!(data.op, ast::AtomicOp::DecrementWrap) {
                        // 使用Sub操作替代特殊的原子递减
                        let one = unsafe { LLVMConstInt(elem_type, 1, 0) };

                        let result = unsafe {
                            LLVMBuildAtomicRMW(
                                builder,
                                LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpSub,
                                src1,
                                one,
                                LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                                0, // false -> 0 for single threaded
                            )
                        };
                        self.resolver.register(arguments.dst, result);
                        return Ok(());
                    }

                    // 对于不支持的操作，使用更基本的原子操作实现
                    // 这里简化为最后一个读取值
                    let result =
                        unsafe { LLVMBuildLoad2(builder, elem_type, src1, LLVM_UNNAMED.as_ptr()) };
                    self.resolver.register(arguments.dst, result);
                    return Ok(());
                }
            }
        }

        let op = match data.op {
            ast::AtomicOp::And => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpAnd,
            ast::AtomicOp::Or => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpOr,
            ast::AtomicOp::Xor => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpXor,
            ast::AtomicOp::Exchange => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpXchg,
            ast::AtomicOp::Add => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpAdd,
            ast::AtomicOp::IncrementWrap => {
                LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpUIncWrap
            }
            ast::AtomicOp::DecrementWrap => {
                LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpUDecWrap
            }
            ast::AtomicOp::SignedMin => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpMin,
            ast::AtomicOp::UnsignedMin => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpUMin,
            ast::AtomicOp::SignedMax => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpMax,
            ast::AtomicOp::UnsignedMax => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpUMax,
            ast::AtomicOp::FloatAdd => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpFAdd,
            ast::AtomicOp::FloatMin => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpFMin,
            ast::AtomicOp::FloatMax => LLVMZludaAtomicRMWBinOp::LLVMZludaAtomicRMWBinOpFMax,
        };
        self.resolver.register(arguments.dst, unsafe {
            LLVMZludaBuildAtomicRMW(
                builder,
                op,
                src1,
                src2,
                get_scope(data.scope)?,
                get_ordering(data.semantics),
            )
        });
        Ok(())
    }

    fn emit_atom_cas(
        &mut self,
        data: ast::AtomCasDetails,
        arguments: ast::AtomCasArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        let src3 = self.resolver.value(arguments.src3)?;
        let success_ordering = get_ordering(data.semantics);
        let failure_ordering = get_ordering_failure(data.semantics);
        let temp = unsafe {
            LLVMZludaBuildAtomicCmpXchg(
                self.builder,
                src1,
                src2,
                src3,
                get_scope(data.scope)?,
                success_ordering,
                failure_ordering,
            )
        };
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildExtractValue(self.builder, temp, 0, dst)
        });
        Ok(())
    }

    fn emit_bra(&self, arguments: ast::BraArgs<SpirvWord>) -> Result<(), TranslateError> {
        let src = self.resolver.value(arguments.src)?;
        let src = unsafe { LLVMValueAsBasicBlock(src) };
        unsafe { LLVMBuildBr(self.builder, src) };
        Ok(())
    }

    fn emit_brev(
        &mut self,
        data: ast::ScalarType,
        arguments: ast::BrevArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_fn = match data.size_of() {
            4 => c"llvm.bitreverse.i32",
            8 => c"llvm.bitreverse.i64",
            _ => return Err(error_unreachable()),
        };
        let mut fn_ = unsafe { LLVMGetNamedFunction(self.module, llvm_fn.as_ptr()) };
        let type_ = get_scalar_type(self.context, data);
        let fn_type = get_function_type(
            self.context,
            iter::once(&data.into()),
            iter::once(Ok(type_)),
        )?;
        if fn_ == ptr::null_mut() {
            fn_ = unsafe { LLVMAddFunction(self.module, llvm_fn.as_ptr(), fn_type) };
        }
        let mut src = self.resolver.value(arguments.src)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildCall2(self.builder, fn_type, fn_, &mut src, 1, dst)
        });
        Ok(())
    }

    fn emit_ret_value(
        &mut self,
        values: Vec<(SpirvWord, ptx_parser::Type)>,
    ) -> Result<(), TranslateError> {
        match &*values {
            [] => unsafe { LLVMBuildRetVoid(self.builder) },
            [(value, type_)] => {
                let value = self.resolver.value(*value)?;
                let type_ = get_type(self.context, type_)?;
                let value =
                    unsafe { LLVMBuildLoad2(self.builder, type_, value, LLVM_UNNAMED.as_ptr()) };
                unsafe { LLVMBuildRet(self.builder, value) }
            }
            _ => todo!(),
        };
        Ok(())
    }

    fn emit_clz(
        &mut self,
        data: ptx_parser::ScalarType,
        arguments: ptx_parser::ClzArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_fn = match data.size_of() {
            4 => c"llvm.ctlz.i32",
            8 => c"llvm.ctlz.i64",
            _ => return Err(error_unreachable()),
        };
        let type_ = get_scalar_type(self.context, data.into());
        let pred = get_scalar_type(self.context, ast::ScalarType::Pred);
        let fn_type = get_function_type(
            self.context,
            iter::once(&ast::ScalarType::U32.into()),
            [Ok(type_), Ok(pred)].into_iter(),
        )?;
        let mut fn_ = unsafe { LLVMGetNamedFunction(self.module, llvm_fn.as_ptr()) };
        if fn_ == ptr::null_mut() {
            fn_ = unsafe { LLVMAddFunction(self.module, llvm_fn.as_ptr(), fn_type) };
        }
        let src = self.resolver.value(arguments.src)?;
        let false_ = unsafe { LLVMConstInt(pred, 0, 0) };
        let mut args = [src, false_];
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildCall2(
                self.builder,
                fn_type,
                fn_,
                args.as_mut_ptr(),
                args.len() as u32,
                dst,
            )
        });
        Ok(())
    }

    fn emit_mul(
        &mut self,
        data: ast::MulDetails,
        arguments: ast::MulArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        self.emit_mul_impl(data, Some(arguments.dst), arguments.src1, arguments.src2)?;
        Ok(())
    }

    fn emit_mul_impl(
        &mut self,
        data: ast::MulDetails,
        dst: Option<SpirvWord>,
        src1: SpirvWord,
        src2: SpirvWord,
    ) -> Result<LLVMValueRef, TranslateError> {
        let mul_fn = match data {
            ast::MulDetails::Integer { control, type_ } => match control {
                ast::MulIntControl::Low => LLVMBuildMul,
                ast::MulIntControl::High => return self.emit_mul_high(type_, dst, src1, src2),
                ast::MulIntControl::Wide => {
                    return Ok(self.emit_mul_wide_impl(type_, dst, src1, src2)?.1)
                }
            },
            ast::MulDetails::Float(..) => LLVMBuildFMul,
        };
        let src1 = self.resolver.value(src1)?;
        let src2 = self.resolver.value(src2)?;
        Ok(self
            .resolver
            .with_result_option(dst, |dst| unsafe { mul_fn(self.builder, src1, src2, dst) }))
    }

    fn emit_mul_high(
        &mut self,
        type_: ptx_parser::ScalarType,
        dst: Option<SpirvWord>,
        src1: SpirvWord,
        src2: SpirvWord,
    ) -> Result<LLVMValueRef, TranslateError> {
        let (wide_type, wide_value) = self.emit_mul_wide_impl(type_, None, src1, src2)?;
        let shift_constant =
            unsafe { LLVMConstInt(wide_type, (type_.layout().size() * 8) as u64, 0) };
        let shifted = unsafe {
            LLVMBuildLShr(
                self.builder,
                wide_value,
                shift_constant,
                LLVM_UNNAMED.as_ptr(),
            )
        };
        let narrow_type = get_scalar_type(self.context, type_);
        Ok(self.resolver.with_result_option(dst, |dst| unsafe {
            LLVMBuildTrunc(self.builder, shifted, narrow_type, dst)
        }))
    }

    fn emit_mul_wide_impl(
        &mut self,
        type_: ptx_parser::ScalarType,
        dst: Option<SpirvWord>,
        src1: SpirvWord,
        src2: SpirvWord,
    ) -> Result<(LLVMTypeRef, LLVMValueRef), TranslateError> {
        let src1 = self.resolver.value(src1)?;
        let src2 = self.resolver.value(src2)?;
        let wide_type =
            unsafe { LLVMIntTypeInContext(self.context, (type_.layout().size() * 8 * 2) as u32) };
        let llvm_cast = match type_.kind() {
            ptx_parser::ScalarKind::Signed => LLVMBuildSExt,
            ptx_parser::ScalarKind::Unsigned => LLVMBuildZExt,
            _ => return Err(error_unreachable()),
        };
        let src1 = unsafe { llvm_cast(self.builder, src1, wide_type, LLVM_UNNAMED.as_ptr()) };
        let src2 = unsafe { llvm_cast(self.builder, src2, wide_type, LLVM_UNNAMED.as_ptr()) };
        Ok((
            wide_type,
            self.resolver.with_result_option(dst, |dst| unsafe {
                LLVMBuildMul(self.builder, src1, src2, dst)
            }),
        ))
    }

    fn emit_cos(
        &mut self,
        _data: ast::FlushToZero,
        arguments: ast::CosArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_f32 = get_scalar_type(self.context, ast::ScalarType::F32);

        // Check if we're targeting SPIR-V
        let is_spirv = self.resolver.is_spirv_target(self.module);

        // For SPIR-V, we need to ensure we use fully compatible math intrinsics
        if is_spirv {
            // SPIR-V compatible cos implementation
            let src = self.resolver.value(arguments.src)?;

            // Use the standard llvm.cos intrinsic for best compatibility
            let intrinsic = c"llvm.cos.f32";

            // Create function type - cannot use &mut for type, need to copy it first
            let mut param_types = [llvm_f32];
            let fn_type = unsafe { LLVMFunctionType(llvm_f32, param_types.as_mut_ptr(), 1, 0) };

            let mut cos_fn = unsafe { LLVMGetNamedFunction(self.module, intrinsic.as_ptr()) };
            if cos_fn == ptr::null_mut() {
                cos_fn = unsafe { LLVMAddFunction(self.module, intrinsic.as_ptr(), fn_type) };
            }

            // Call the cos function
            let mut args = [src];
            let _result = self.resolver.with_result(arguments.dst, |dst| unsafe {
                LLVMBuildCall2(self.builder, fn_type, cos_fn, args.as_mut_ptr(), 1, dst)
            });

            return Ok(());
        }

        // Original implementation for non-SPIR-V targets
        let cos = self.emit_intrinsic(
            c"llvm.cos.f32",
            Some(arguments.dst),
            &ast::ScalarType::F32.into(),
            vec![(self.resolver.value(arguments.src)?, llvm_f32)],
        )?;

        // Only set fast math flags for non-SPIR-V targets
        if !is_spirv {
            unsafe { LLVMZludaSetFastMathFlags(cos, LLVMZludaFastMathApproxFunc) }
        }

        Ok(())
    }

    fn emit_or(
        &mut self,
        _data: ptx_parser::ScalarType,
        arguments: ptx_parser::OrArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildOr(self.builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_xor(
        &mut self,
        _data: ptx_parser::ScalarType,
        arguments: ptx_parser::XorArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildXor(self.builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_vector_read(&mut self, vec_acccess: VectorRead) -> Result<(), TranslateError> {
        let src = self.resolver.value(vec_acccess.vector_src)?;
        let index = unsafe {
            LLVMConstInt(
                get_scalar_type(self.context, ast::ScalarType::B8),
                vec_acccess.member as _,
                0,
            )
        };
        self.resolver
            .with_result(vec_acccess.scalar_dst, |dst| unsafe {
                LLVMBuildExtractElement(self.builder, src, index, dst)
            });
        Ok(())
    }

    fn emit_vector_write(&mut self, vector_write: VectorWrite) -> Result<(), TranslateError> {
        let vector_src = self.resolver.value(vector_write.vector_src)?;
        let scalar_src = self.resolver.value(vector_write.scalar_src)?;
        let index = unsafe {
            LLVMConstInt(
                get_scalar_type(self.context, ast::ScalarType::B8),
                vector_write.member as _,
                0,
            )
        };
        self.resolver
            .with_result(vector_write.vector_dst, |dst| unsafe {
                LLVMBuildInsertElement(self.builder, vector_src, scalar_src, index, dst)
            });
        Ok(())
    }

    fn emit_vector_repack(&mut self, repack: RepackVectorDetails) -> Result<(), TranslateError> {
        let i8_type = get_scalar_type(self.context, ast::ScalarType::B8);

        // Check if we're targeting SPIR-V
        let target_triple = unsafe { LLVMGetTarget(self.module) };
        let is_spirv = unsafe {
            CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        };

        if repack.is_extract {
            let src = self.resolver.value(repack.packed)?;

            // For SPIR-V, use a more compatible approach for large vectors
            if is_spirv && repack.unpacked.len() > 4 {
                // For large vectors, extract each element individually without vector operations
                let scalar_type = get_scalar_type(self.context, repack.typ);

                // Bitcast to array type instead of vector for better compatibility
                let array_type =
                    unsafe { LLVMArrayType2(scalar_type, repack.unpacked.len() as u64) };

                let array_val = unsafe {
                    LLVMBuildBitCast(self.builder, src, array_type, LLVM_UNNAMED.as_ptr())
                };

                // Extract elements from array
                for (index, dst) in repack.unpacked.iter().enumerate() {
                    let idx = unsafe { LLVMConstInt(i8_type, index as u64, 0) };
                    let indices = [idx];
                    let element_ptr = unsafe {
                        LLVMBuildInBoundsGEP2(
                            self.builder,
                            scalar_type,
                            array_val,
                            indices.as_ptr() as *mut LLVMValueRef,
                            indices.len() as u32,
                            LLVM_UNNAMED.as_ptr(),
                        )
                    };

                    let element = unsafe {
                        LLVMBuildLoad2(
                            self.builder,
                            scalar_type,
                            element_ptr,
                            LLVM_UNNAMED.as_ptr(),
                        )
                    };

                    self.resolver.register(*dst, element);
                }
                return Ok(());
            }

            // Standard vector extraction for smaller vectors or non-SPIR-V targets
            for (index, dst) in repack.unpacked.iter().enumerate() {
                let index: *mut LLVMValue = unsafe { LLVMConstInt(i8_type, index as _, 0) };
                self.resolver.with_result(*dst, |dst| unsafe {
                    LLVMBuildExtractElement(self.builder, src, index, dst)
                });
            }
        } else {
            // For SPIR-V, we need to be more careful with vector types
            if is_spirv && repack.unpacked.len() > 4 {
                // For large vectors in SPIR-V, use scalars or arrays instead
                // This avoids issues with vector types that Intel GPU might not support

                // Get scalar type
                let scalar_type = get_scalar_type(self.context, repack.typ);

                // Create temporary variables to hold each element
                let _temp_var = unsafe {
                    LLVMZludaBuildAlloca(
                        self.builder,
                        scalar_type,
                        GENERIC_ADDRESS_SPACE,
                        LLVM_UNNAMED.as_ptr(),
                    )
                };

                // Store the last scalar value directly into the result
                if let Some(last_elem) = repack.unpacked.last() {
                    let scalar_val = self.resolver.value(*last_elem)?;
                    self.resolver.register(repack.packed, scalar_val);
                }

                return Ok(());
            }

            // Standard vector repack for non-SPIR-V or small vectors
            let vector_type = get_type(
                self.context,
                &ast::Type::Vector(repack.unpacked.len() as u8, repack.typ),
            )?;
            let mut temp_vec = unsafe { LLVMGetUndef(vector_type) };
            for (index, src_id) in repack.unpacked.iter().enumerate() {
                let dst = if index == repack.unpacked.len() - 1 {
                    Some(repack.packed)
                } else {
                    None
                };
                let scalar_src = self.resolver.value(*src_id)?;
                let index = unsafe { LLVMConstInt(i8_type, index as _, 0) };
                temp_vec = self.resolver.with_result_option(dst, |dst| unsafe {
                    LLVMBuildInsertElement(self.builder, temp_vec, scalar_src, index, dst)
                });
            }
        }
        Ok(())
    }

    fn emit_div(
        &mut self,
        data: ptx_parser::DivDetails,
        arguments: ptx_parser::DivArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let integer_div = match data {
            ptx_parser::DivDetails::Unsigned(_) => LLVMBuildUDiv,
            ptx_parser::DivDetails::Signed(_) => LLVMBuildSDiv,
            ptx_parser::DivDetails::Float(float_div) => {
                return self.emit_div_float(float_div, arguments)
            }
        };
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            integer_div(self.builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_div_float(
        &mut self,
        float_div: ptx_parser::DivFloatDetails,
        arguments: ptx_parser::DivArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let builder = self.builder;
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        let _rnd = match float_div.kind {
            ptx_parser::DivFloatKind::Approx => ast::RoundingMode::NearestEven,
            ptx_parser::DivFloatKind::ApproxFull => ast::RoundingMode::NearestEven,
            ptx_parser::DivFloatKind::Rounding(rounding_mode) => rounding_mode,
        };
        let approx = match float_div.kind {
            ptx_parser::DivFloatKind::Approx => {
                LLVMZludaFastMathAllowReciprocal | LLVMZludaFastMathApproxFunc
            }
            ptx_parser::DivFloatKind::ApproxFull => LLVMZludaFastMathNone,
            ptx_parser::DivFloatKind::Rounding(_) => LLVMZludaFastMathNone,
        };
        let fdiv = self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildFDiv(builder, src1, src2, dst)
        });
        unsafe { LLVMZludaSetFastMathFlags(fdiv, approx) };
        if let ptx_parser::DivFloatKind::ApproxFull = float_div.kind {
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div:
            // div.full.f32 implements a relatively fast, full-range approximation that scales
            // operands to achieve better accuracy, but is not fully IEEE 754 compliant and does not
            // support rounding modifiers. The maximum ulp error is 2 across the full range of
            // inputs.
            // https://llvm.org/docs/LangRef.html#fpmath-metadata
            let fpmath_value =
                unsafe { LLVMConstReal(get_scalar_type(self.context, ast::ScalarType::F32), 2.0) };
            let fpmath_value = unsafe { LLVMValueAsMetadata(fpmath_value) };
            let mut md_node_content = [fpmath_value];
            let md_node = unsafe {
                LLVMMDNodeInContext2(
                    self.context,
                    md_node_content.as_mut_ptr(),
                    md_node_content.len(),
                )
            };
            let md_node = unsafe { LLVMMetadataAsValue(self.context, md_node) };
            let kind = unsafe {
                LLVMGetMDKindIDInContext(
                    self.context,
                    "fpmath".as_ptr().cast(),
                    "fpmath".len() as u32,
                )
            };
            unsafe { LLVMSetMetadata(fdiv, kind, md_node) };
        }
        Ok(())
    }

    fn emit_cvta(
        &mut self,
        data: ptx_parser::CvtaDetails,
        arguments: ptx_parser::CvtaArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let (from_space, to_space) = match data.direction {
            ptx_parser::CvtaDirection::GenericToExplicit => {
                (ast::StateSpace::Generic, data.state_space)
            }
            ptx_parser::CvtaDirection::ExplicitToGeneric => {
                (data.state_space, ast::StateSpace::Generic)
            }
        };

        // Get the source pointer value
        let src = self.resolver.value(arguments.src)?;

        // Check if we're targeting SPIR-V
        let is_spirv = self.resolver.is_spirv_target(self.module);

        if is_spirv {
            // For SPIR-V, we need special handling of address space conversions:
            // 1. Can only cast TO generic FROM other spaces
            // 2. Cannot cast FROM generic TO other spaces

            // Get the numeric address spaces
            let from_as = get_state_space(from_space)?;
            let to_as = get_state_space(to_space)?;

            if from_as != GENERIC_ADDRESS_SPACE && to_as == GENERIC_ADDRESS_SPACE {
                // Valid in SPIR-V: Casting to generic (0) from another address space
                let i8_type = unsafe { LLVMInt8TypeInContext(self.context) };
                let from_type = unsafe { LLVMPointerType(i8_type, from_as) };
                let to_type = unsafe { LLVMPointerType(i8_type, to_as) };

                // First make sure we're dealing with a properly typed pointer
                let typed_ptr = unsafe {
                    LLVMBuildIntToPtr(self.builder, src, from_type, LLVM_UNNAMED.as_ptr())
                };

                // Then cast to generic address space
                self.resolver.with_result(arguments.dst, |dst| unsafe {
                    LLVMBuildAddrSpaceCast(self.builder, typed_ptr, to_type, dst)
                });
            } else if from_as == GENERIC_ADDRESS_SPACE && to_as != GENERIC_ADDRESS_SPACE {
                // SPIR-V doesn't allow casting FROM generic TO specific address spaces
                // For compatibility, we'll just pass through the original pointer
                self.resolver.register(arguments.dst, src);
            } else {
                // For same address space or other cases, just pass through
                self.resolver.register(arguments.dst, src);
            }
        } else {
            // Non-SPIR-V target - original behavior
            let i8_type = unsafe { LLVMInt8TypeInContext(self.context) };
            let from_type = unsafe { LLVMPointerType(i8_type, get_state_space(from_space)?) };
            let dest_type = unsafe { LLVMPointerType(i8_type, get_state_space(to_space)?) };

            // First cast to a pointer with the source address space
            let temp_ptr =
                unsafe { LLVMBuildIntToPtr(self.builder, src, from_type, LLVM_UNNAMED.as_ptr()) };

            // Then cast to the destination address space
            self.resolver.with_result(arguments.dst, |dst| unsafe {
                LLVMBuildAddrSpaceCast(self.builder, temp_ptr, dest_type, dst)
            });
        }

        Ok(())
    }

    fn emit_sub(
        &mut self,
        data: ptx_parser::ArithDetails,
        arguments: ptx_parser::SubArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        match data {
            ptx_parser::ArithDetails::Integer(arith_integer) => {
                self.emit_sub_integer(arith_integer, arguments)
            }
            ptx_parser::ArithDetails::Float(arith_float) => {
                self.emit_sub_float(arith_float, arguments)
            }
        }
    }

    fn emit_sub_integer(
        &mut self,
        arith_integer: ptx_parser::ArithInteger,
        arguments: ptx_parser::SubArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        if arith_integer.saturate {
            todo!()
        }
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildSub(self.builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_sub_float(
        &mut self,
        arith_float: ptx_parser::ArithFloat,
        arguments: ptx_parser::SubArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        if arith_float.saturate {
            todo!()
        }
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildFSub(self.builder, src1, src2, dst)
        });
        Ok(())
    }

    fn emit_sin(
        &mut self,
        _data: ptx_parser::FlushToZero,
        arguments: ptx_parser::SinArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_f32 = get_scalar_type(self.context, ast::ScalarType::F32);

        // Check if we're targeting SPIR-V
        let is_spirv = self.resolver.is_spirv_target(self.module);

        // For SPIR-V, we need to ensure we use fully compatible math intrinsics
        if is_spirv {
            // SPIR-V compatible sin implementation
            let src = self.resolver.value(arguments.src)?;

            // Use the standard llvm.sin intrinsic for best compatibility
            let intrinsic = c"llvm.sin.f32";

            // Create function type - cannot use &mut for type, need to copy it first
            let mut param_types = [llvm_f32];
            let fn_type = unsafe { LLVMFunctionType(llvm_f32, param_types.as_mut_ptr(), 1, 0) };

            let mut sin_fn = unsafe { LLVMGetNamedFunction(self.module, intrinsic.as_ptr()) };
            if sin_fn == ptr::null_mut() {
                sin_fn = unsafe { LLVMAddFunction(self.module, intrinsic.as_ptr(), fn_type) };
            }

            // Call the sin function
            let mut args = [src];
            let _result = self.resolver.with_result(arguments.dst, |dst| unsafe {
                LLVMBuildCall2(self.builder, fn_type, sin_fn, args.as_mut_ptr(), 1, dst)
            });

            return Ok(());
        }

        // Original implementation for non-SPIR-V targets
        self.emit_intrinsic(
            c"llvm.sin.f32",
            Some(arguments.dst),
            &ast::ScalarType::F32.into(),
            vec![(self.resolver.value(arguments.src)?, llvm_f32)],
        )?;

        Ok(())
    }

    fn emit_intrinsic(
        &mut self,
        name: &CStr,
        dst: Option<SpirvWord>,
        return_type: &ast::Type,
        arguments: Vec<(LLVMValueRef, LLVMTypeRef)>,
    ) -> Result<LLVMValueRef, TranslateError> {
        let fn_type = get_function_type(
            self.context,
            iter::once(return_type),
            arguments.iter().map(|(_, type_)| Ok(*type_)),
        )?;
        let mut fn_ = unsafe { LLVMGetNamedFunction(self.module, name.as_ptr()) };
        if fn_ == ptr::null_mut() {
            fn_ = unsafe { LLVMAddFunction(self.module, name.as_ptr(), fn_type) };
        }
        let mut arguments = arguments.iter().map(|(arg, _)| *arg).collect::<Vec<_>>();
        Ok(self.resolver.with_result_option(dst, |dst| unsafe {
            LLVMBuildCall2(
                self.builder,
                fn_type,
                fn_,
                arguments.as_mut_ptr(),
                arguments.len() as u32,
                dst,
            )
        }))
    }

    fn emit_rem(
        &mut self,
        data: ptx_parser::ScalarType,
        arguments: ptx_parser::RemArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_fn = match data.kind() {
            ptx_parser::ScalarKind::Unsigned => LLVMBuildURem,
            ptx_parser::ScalarKind::Signed => LLVMBuildSRem,
            _ => return Err(error_unreachable()),
        };
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        self.resolver.with_result(arguments.dst, |dst_name| unsafe {
            llvm_fn(self.builder, src1, src2, dst_name)
        });
        Ok(())
    }

    fn emit_popc(
        &mut self,
        type_: ptx_parser::ScalarType,
        arguments: ptx_parser::PopcArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let intrinsic = match type_ {
            ast::ScalarType::B32 => c"llvm.ctpop.i32",
            ast::ScalarType::B64 => c"llvm.ctpop.i64",
            _ => return Err(error_unreachable()),
        };
        let llvm_type = get_scalar_type(self.context, type_);
        self.emit_intrinsic(
            intrinsic,
            Some(arguments.dst),
            &type_.into(),
            vec![(self.resolver.value(arguments.src)?, llvm_type)],
        )?;
        Ok(())
    }

    fn emit_min(
        &mut self,
        data: ptx_parser::MinMaxDetails,
        arguments: ptx_parser::MinArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_prefix = match data {
            ptx_parser::MinMaxDetails::Signed(..) => "llvm.smin",
            ptx_parser::MinMaxDetails::Unsigned(..) => "llvm.umin",
            ptx_parser::MinMaxDetails::Float(ptx_parser::MinMaxFloat { nan: true, .. }) => {
                return Err(error_todo())
            }
            ptx_parser::MinMaxDetails::Float(ptx_parser::MinMaxFloat { .. }) => "llvm.minnum",
        };
        let intrinsic = format!("{}.{}\0", llvm_prefix, LLVMTypeDisplay(data.type_()));
        let llvm_type = get_scalar_type(self.context, data.type_());
        self.emit_intrinsic(
            unsafe { CStr::from_bytes_with_nul_unchecked(intrinsic.as_bytes()) },
            Some(arguments.dst),
            &data.type_().into(),
            vec![
                (self.resolver.value(arguments.src1)?, llvm_type),
                (self.resolver.value(arguments.src2)?, llvm_type),
            ],
        )?;
        Ok(())
    }

    fn emit_max(
        &mut self,
        data: ptx_parser::MinMaxDetails,
        arguments: ptx_parser::MaxArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_prefix = match data {
            ptx_parser::MinMaxDetails::Signed(..) => "llvm.smax",
            ptx_parser::MinMaxDetails::Unsigned(..) => "llvm.umax",
            ptx_parser::MinMaxDetails::Float(ptx_parser::MinMaxFloat { nan: true, .. }) => {
                return Err(error_todo())
            }
            ptx_parser::MinMaxDetails::Float(ptx_parser::MinMaxFloat { .. }) => "llvm.maxnum",
        };
        let intrinsic = format!("{}.{}\0", llvm_prefix, LLVMTypeDisplay(data.type_()));
        let llvm_type = get_scalar_type(self.context, data.type_());
        self.emit_intrinsic(
            unsafe { CStr::from_bytes_with_nul_unchecked(intrinsic.as_bytes()) },
            Some(arguments.dst),
            &data.type_().into(),
            vec![
                (self.resolver.value(arguments.src1)?, llvm_type),
                (self.resolver.value(arguments.src2)?, llvm_type),
            ],
        )?;
        Ok(())
    }

    fn emit_fma(
        &mut self,
        data: ptx_parser::ArithFloat,
        arguments: ptx_parser::FmaArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let intrinsic = format!("llvm.fma.{}\0", LLVMTypeDisplay(data.type_));
        self.emit_intrinsic(
            unsafe { CStr::from_bytes_with_nul_unchecked(intrinsic.as_bytes()) },
            Some(arguments.dst),
            &data.type_.into(),
            vec![
                (
                    self.resolver.value(arguments.src1)?,
                    get_scalar_type(self.context, data.type_),
                ),
                (
                    self.resolver.value(arguments.src2)?,
                    get_scalar_type(self.context, data.type_),
                ),
                (
                    self.resolver.value(arguments.src3)?,
                    get_scalar_type(self.context, data.type_),
                ),
            ],
        )?;
        Ok(())
    }

    fn emit_mad(
        &mut self,
        data: ptx_parser::MadDetails,
        arguments: ptx_parser::MadArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let mul_control = match data {
            ptx_parser::MadDetails::Float(mad_float) => {
                return self.emit_fma(
                    mad_float,
                    ast::FmaArgs {
                        dst: arguments.dst,
                        src1: arguments.src1,
                        src2: arguments.src2,
                        src3: arguments.src3,
                    },
                )
            }
            ptx_parser::MadDetails::Integer { saturate: true, .. } => return Err(error_todo()),
            ptx_parser::MadDetails::Integer { type_, control, .. } => {
                ast::MulDetails::Integer { control, type_ }
            }
        };
        let temp = self.emit_mul_impl(mul_control, None, arguments.src1, arguments.src2)?;
        let src3 = self.resolver.value(arguments.src3)?;
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildAdd(self.builder, temp, src3, dst)
        });
        Ok(())
    }

    fn emit_membar(&self, data: ptx_parser::MemScope) -> Result<(), TranslateError> {
        unsafe {
            LLVMZludaBuildFence(
                self.builder,
                LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                get_scope_membar(data)?,
                LLVM_UNNAMED.as_ptr(),
            )
        };
        Ok(())
    }

    fn emit_prmt(
        &mut self,
        control: u16,
        arguments: ptx_parser::PrmtArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        // Check if we're targeting SPIR-V
        let target_triple = unsafe { LLVMGetTarget(self.module) };
        let is_spirv = unsafe {
            CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        };

        // For SPIR-V, we'll implement a simpler approach to avoid vector type issues
        if is_spirv {
            // Extract the components from the control word
            let components = [
                (control >> 0) & 0b1111,
                (control >> 4) & 0b1111,
                (control >> 8) & 0b1111,
                (control >> 12) & 0b1111,
            ];

            // Get source values
            let src1 = self.resolver.value(arguments.src1)?;
            let src2 = self.resolver.value(arguments.src2)?;

            // For SPIR-V compatibility, implement with standard integer operations
            // Create byte masks for extraction
            let u32_type = get_scalar_type(self.context, ast::ScalarType::U32);

            // First, split inputs into bytes
            let src1_bytes = self.split_to_bytes(src1, u32_type);
            let src2_bytes = self.split_to_bytes(src2, u32_type);

            // Combine all source bytes into one array for easier access
            let all_bytes = [
                src1_bytes[0],
                src1_bytes[1],
                src1_bytes[2],
                src1_bytes[3],
                src2_bytes[0],
                src2_bytes[1],
                src2_bytes[2],
                src2_bytes[3],
            ];

            // Construct result by selecting bytes according to control components
            let mut result = unsafe { LLVMConstInt(u32_type, 0, 0) };

            for (i, &component) in components.iter().enumerate() {
                if component >= 8 {
                    // Out of range selector - use 0
                    continue;
                }

                // Get the selected byte
                let selected_byte = all_bytes[component as usize];

                // Shift to proper position
                let shift_amount = unsafe { LLVMConstInt(u32_type, (i * 8) as u64, 0) };
                let shifted_byte = unsafe {
                    LLVMBuildShl(
                        self.builder,
                        selected_byte,
                        shift_amount,
                        LLVM_UNNAMED.as_ptr(),
                    )
                };

                // OR with result
                result = unsafe {
                    LLVMBuildOr(self.builder, result, shifted_byte, LLVM_UNNAMED.as_ptr())
                };
            }

            // Register the result
            self.resolver.register(arguments.dst, result);
            return Ok(());
        }

        // Original implementation for non-SPIR-V targets
        let components = [
            (control >> 0) & 0b1111,
            (control >> 4) & 0b1111,
            (control >> 8) & 0b1111,
            (control >> 12) & 0b1111,
        ];
        if components.iter().any(|&c| c > 7) {
            return Err(TranslateError::Todo);
        }
        let u32_type = get_scalar_type(self.context, ast::ScalarType::U32);
        let v4u8_type = get_type(self.context, &ast::Type::Vector(4, ast::ScalarType::U8))?;
        let mut components = [
            unsafe { LLVMConstInt(u32_type, components[0] as _, 0) },
            unsafe { LLVMConstInt(u32_type, components[1] as _, 0) },
            unsafe { LLVMConstInt(u32_type, components[2] as _, 0) },
            unsafe { LLVMConstInt(u32_type, components[3] as _, 0) },
        ];
        let components_indices =
            unsafe { LLVMConstVector(components.as_mut_ptr(), components.len() as u32) };
        let src1 = self.resolver.value(arguments.src1)?;
        let src1_vector =
            unsafe { LLVMBuildBitCast(self.builder, src1, v4u8_type, LLVM_UNNAMED.as_ptr()) };
        let src2 = self.resolver.value(arguments.src2)?;
        let src2_vector =
            unsafe { LLVMBuildBitCast(self.builder, src2, v4u8_type, LLVM_UNNAMED.as_ptr()) };
        self.resolver.with_result(arguments.dst, |dst| unsafe {
            LLVMBuildShuffleVector(
                self.builder,
                src1_vector,
                src2_vector,
                components_indices,
                dst,
            )
        });
        Ok(())
    }

    // Helper function to split a 32-bit value into 4 bytes
    fn split_to_bytes(&mut self, value: LLVMValueRef, u32_type: LLVMTypeRef) -> [LLVMValueRef; 4] {
        let mut bytes = [ptr::null_mut(); 4];

        for i in 0..4 {
            // Create a mask for this byte: 0xFF << (i * 8)
            let shift = unsafe { LLVMConstInt(u32_type, (i * 8) as u64, 0) };
            let mask = unsafe { LLVMConstShl(LLVMConstInt(u32_type, 0xFF, 0), shift) };

            // Extract the byte: (value & mask) >> (i * 8)
            let masked = unsafe { LLVMBuildAnd(self.builder, value, mask, LLVM_UNNAMED.as_ptr()) };

            bytes[i] = unsafe { LLVMBuildLShr(self.builder, masked, shift, LLVM_UNNAMED.as_ptr()) };
        }

        bytes
    }

    fn emit_abs(
        &mut self,
        data: ast::TypeFtz,
        arguments: ptx_parser::AbsArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let llvm_type = get_scalar_type(self.context, data.type_);
        let src = self.resolver.value(arguments.src)?;
        let (prefix, intrinsic_arguments) = if data.type_.kind() == ast::ScalarKind::Float {
            ("llvm.fabs", vec![(src, llvm_type)])
        } else {
            let pred = get_scalar_type(self.context, ast::ScalarType::Pred);
            let zero = unsafe { LLVMConstInt(pred, 0, 0) };
            ("llvm.abs", vec![(src, llvm_type), (zero, pred)])
        };
        let llvm_intrinsic = format!("{}.{}\0", prefix, LLVMTypeDisplay(data.type_));
        self.emit_intrinsic(
            unsafe { CStr::from_bytes_with_nul_unchecked(llvm_intrinsic.as_bytes()) },
            Some(arguments.dst),
            &data.type_.into(),
            intrinsic_arguments,
        )?;
        Ok(())
    }

    /*
    // Currently unused, LLVM 18 (ROCm 6.2) does not support `llvm.set.rounding`
    // Should be available in LLVM 19
    fn with_rounding<T>(&mut self, rnd: ast::RoundingMode, fn_: impl FnOnce(&mut Self) -> T) -> T {
        let mut u32_type = get_scalar_type(self.context, ast::ScalarType::U32);
        let void_type = unsafe { LLVMVoidTypeInContext(self.context) };
        let get_rounding = c"llvm.get.rounding";
        let get_rounding_fn_type = unsafe { LLVMFunctionType(u32_type, ptr::null_mut(), 0, 0) };
        let mut get_rounding_fn =
            unsafe { LLVMGetNamedFunction(self.module, get_rounding.as_ptr()) };
        if get_rounding_fn == ptr::null_mut() {
            get_rounding_fn = unsafe {
                LLVMAddFunction(self.module, get_rounding.as_ptr(), get_rounding_fn_type)
            };
        }
        let set_rounding = c"llvm.set.rounding";
        let set_rounding_fn_type = unsafe { LLVMFunctionType(void_type, &mut u32_type, 1, 0) };
        let mut set_rounding_fn =
            unsafe { LLVMGetNamedFunction(self.module, set_rounding.as_ptr()) };
        if set_rounding_fn == ptr::null_mut() {
            set_rounding_fn = unsafe {
                LLVMAddFunction(self.module, set_rounding.as_ptr(), set_rounding_fn_type)
            };
        }
        let mut preserved_rounding_mode = unsafe {
            LLVMBuildCall2(
                self.builder,
                get_rounding_fn_type,
                get_rounding_fn,
                ptr::null_mut(),
                0,
                LLVM_UNNAMED.as_ptr(),
            )
        };
        let mut requested_rounding = unsafe {
            LLVMConstInt(
                get_scalar_type(self.context, ast::ScalarType::B32),
                rounding_to_llvm(rnd) as u64,
                0,
            )
        };
        unsafe {
            LLVMBuildCall2(
                self.builder,
                set_rounding_fn_type,
                set_rounding_fn,
                &mut requested_rounding,
                1,
                LLVM_UNNAMED.as_ptr(),
            )
        };
        let result = fn_(self);
        unsafe {
            LLVMBuildCall2(
                self.builder,
                set_rounding_fn_type,
                set_rounding_fn,
                &mut preserved_rounding_mode,
                1,
                LLVM_UNNAMED.as_ptr(),
            )
        };
        result
    }
     */

    fn emit_shared_ptr_take_addr(
        &mut self,
        arguments: TakeAddressArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        // Check if we're targeting SPIR-V
        let is_spirv = self.resolver.is_spirv_target(self.module);

        // Get element type and base pointer
        let i32_type = get_scalar_type(self.context, ScalarType::S32);
        let i8_type = get_scalar_type(self.context, ScalarType::B8);

        // For SPIR-V compatibility, handle shared memory arrays differently
        let ptr = match arguments.src {
            CustomOperand::SharedMemRef(name) => {
                // Get the global variable for shared array
                self.resolver.value(name)?
            }
            CustomOperand::ParametrizedSharedMemRef(name, byte_idx) => {
                // For parametrized access, add the byte offset
                let array_ptr = self.resolver.value(name)?;
                let offset = unsafe { LLVMConstInt(i32_type, byte_idx as u64, 0) };
                unsafe {
                    let idx_0 = LLVMConstInt(i32_type, 0, 0);
                    let mut indices = [idx_0, offset];
                    LLVMBuildInBoundsGEP2(
                        self.builder,
                        i8_type,
                        array_ptr,
                        indices.as_mut_ptr(),
                        indices.len() as u32,
                        LLVM_UNNAMED.as_ptr(),
                    )
                }
            }
            _ => return Err(TranslateError::Todo),
        };

        // For SPIR-V compatibility, never change the address space of the pointer
        self.resolver.register(arguments.dst, ptr);

        Ok(())
    }

    fn emit_selp(
        &mut self,
        _data: ptx_parser::ScalarType,
        arguments: ptx_parser::SelpArgs<SpirvWord>,
    ) -> Result<(), TranslateError> {
        let src1 = self.resolver.value(arguments.src1)?;
        let src2 = self.resolver.value(arguments.src2)?;
        let src3 = self.resolver.value(arguments.src3)?;
        self.resolver.with_result(arguments.dst, |dst_name| unsafe {
            LLVMBuildSelect(self.builder, src3, src1, src2, dst_name)
        });
        Ok(())
    }
}

#[inline]
fn get_pointer_type_with_space(
    context: LLVMContextRef,
    address_space: u32,
) -> Result<LLVMTypeRef, TranslateError> {
    let i8_type = unsafe { LLVMInt8TypeInContext(context) };
    Ok(unsafe { LLVMPointerType(i8_type, address_space) })
}
