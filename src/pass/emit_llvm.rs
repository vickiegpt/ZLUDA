impl<'a, 'input> ModuleEmitContext<'a, 'input> {
    fn new(
        context: &Context,
        module: &Module,
        id_defs: &'a GlobalStringIdentResolver2<'input>,
    ) -> Self {
        // Create separate builders
        let main_builder = Builder::new(context);
        let resolver_builder = Builder::new(context);

        // Create an empty HashMap that will be moved into the ModuleEmitContext
        let empty_map = Box::new(HashMap::new());
        let id_defs_map_ref = Box::leak(empty_map);

        // Create the ModuleEmitContext
        let mut ctx = ModuleEmitContext {
            context: context.get(),
            module: module.get(),
            builder: main_builder,
            id_defs,
            id_defs_map: HashMap::new(),
            resolver: ResolveIdent::new(
                module.get(),
                context.get(),
                resolver_builder.get(),
                id_defs_map_ref, // Pass the leaked reference
            ),
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

    // Fixed SPIR-V capabilities and metadata
    fn add_spirv_capabilities(&self) {
        // Skip adding metadata for SPIR-V version - this is simpler
        // and avoids compatibility issues with the LLVM API
        // The OpenCL runtime will provide default versioning
    }

    fn kernel_call_convention() -> u32 {
        // 使用C调用约定，而不是AMDGPU内核调用约定
        // 这对于SPIR-V目标是必要的
        LLVMCallConv::LLVMCCallConv as u32
    }

    fn func_call_convention() -> u32 {
        LLVMCallConv::LLVMCCallConv as u32
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

struct MethodEmitContext<'a, 'input> {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    method: LLVMValueRef,
    builder: LLVMBuilderRef,
    variables_builder: Builder,
    module_context: &'a mut ModuleEmitContext<'a, 'input>,
}

impl<'a, 'input> MethodEmitContext<'a, 'input> {
    fn new(
        parent: &'a mut ModuleEmitContext<'a, 'input>,
        method: LLVMValueRef,
        variables_builder: Builder,
    ) -> Self {
        MethodEmitContext {
            context: parent.context,
            module: parent.module,
            builder: parent.builder.get(),
            variables_builder,
            module_context: parent,
            method,
        }
    }

    // Helper function to check if target is SPIR-V
    fn is_spirv_target(&self) -> bool {
        self.module_context.is_spirv_target()
    }

    // ... existing code ...
}

impl Clone for Builder {
    fn clone(&self) -> Self {
        // Create a new builder in the same context
        unsafe {
            // Get the LLVMContextRef from the builder
            let bb = LLVMGetInsertBlock(self.0);
            let context = if !bb.is_null() {
                // If we have a valid insert block, get the context from there
                let fn_ = LLVMGetBasicBlockParent(bb);
                let module = LLVMGetGlobalParent(fn_);
                LLVMGetModuleContext(module)
            } else {
                // Fallback - create a new context (not ideal but prevents crash)
                LLVMContextCreate()
            };

            Self(LLVMCreateBuilderInContext(context))
        }
    }
}

struct ModuleEmitContext<'a, 'input> {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    builder: Builder,
    id_defs: &'a GlobalStringIdentResolver2<'input>,
    id_defs_map: HashMap<String, LLVMValueRef>,
    resolver: ResolveIdent<'static>,
}
