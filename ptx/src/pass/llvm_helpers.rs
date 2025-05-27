use crate::pass::ast::{self, ScalarType, StateSpace};
use llvm_zluda::core::*;
use llvm_zluda::prelude::*;
use llvm_zluda::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::cell::RefCell;
use std::thread_local;

use super::SpirvWord;
use super::TranslateError;

/// Custom synchronization scope ref type since llvm_zluda doesn't expose it
#[derive(Debug, Copy, Clone)]
pub struct LLVMSynchronizationScopeRef(pub u32);

/// Thread-local empty HashMap that can be used as a default
thread_local! {
    static EMPTY_MAP: RefCell<HashMap<String, LLVMValueRef>> = RefCell::new(HashMap::new());
}

/// ResolveIdent is responsible for mapping identifiers to LLVM values
pub struct ResolveIdent<'a> {
    values: HashMap<String, LLVMValueRef>,
    labels: HashMap<String, LLVMBasicBlockRef>,
    module: LLVMModuleRef,
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    id_defs: &'a HashMap<String, LLVMValueRef>,
}

impl<'a> ResolveIdent<'a> {
    pub fn new(
        module: LLVMModuleRef,
        context: LLVMContextRef,
        builder: LLVMBuilderRef,
        id_defs: Option<&'a HashMap<String, LLVMValueRef>>,
    ) -> Self {
        // If id_defs is None, we'll use a reference to our thread-local empty HashMap
        let id_defs = match id_defs {
            Some(map) => map,
            None => {
                // This will leak a reference but it's ok for our use case
                Box::leak(Box::new(HashMap::new()))
            }
        };
        
        Self {
            values: HashMap::new(),
            labels: HashMap::new(),
            module,
            context,
            builder,
            id_defs,
        }
    }

    pub fn insert(&mut self, id: String, value: LLVMValueRef) {
        self.values.insert(id, value);
    }

    pub fn get(&self, id: &str) -> Option<LLVMValueRef> {
        self.values.get(id).copied()
    }

    pub fn get_or_add(&mut self, id: impl Into<String>) -> String {
        id.into()
    }

    pub fn get_or_add_raw(&mut self, id: SpirvWord) -> *const i8 {
        let id_str = String::from(id);
        let c_name = CString::new(id_str).unwrap();
        c_name.as_ptr()
    }

    pub fn value(&self, id: impl AsRef<str>) -> Result<LLVMValueRef, TranslateError> {
        let id = id.as_ref();
        self.values
            .get(id)
            .copied()
            .ok_or_else(|| TranslateError::Todo)
    }

    pub fn register(&mut self, id: impl Into<String>, value: LLVMValueRef) {
        let id = id.into();
        self.values.insert(id, value);
    }

    pub fn register_shared_array(
        &mut self,
        name: impl Into<String>,
        global: LLVMValueRef,
        elem_type: LLVMTypeRef,
    ) {
        let name = name.into();
        self.values.insert(name, global);
    }

    pub fn with_result<F, R>(&mut self, id: impl Into<String>, f: F) -> R
    where
        F: FnOnce(*const i8) -> R,
        R: Copy,
    {
        let id = id.into();
        let c_name = CString::new(id.clone()).unwrap();
        let result = f(c_name.as_ptr());

        // Store the result in our values map
        if let Some(value) = Self::result_to_value_ref(result) {
            self.values.insert(id, value);
        }

        result
    }

    // Helper function to safely convert any result to LLVMValueRef if possible
    fn result_to_value_ref<R>(result: R) -> Option<LLVMValueRef> {
        // Only meaningful for pointer-sized results
        if std::mem::size_of::<R>() == std::mem::size_of::<*mut ()>() {
            // Transmute the result to a raw pointer type, assuming it's a valid LLVM pointer
            unsafe {
                let ptr = std::mem::transmute_copy::<R, *mut ()>(&result);
                if !ptr.is_null() {
                    return Some(ptr as *mut _);
                }
            }
        }
        None
    }

    pub fn with_result_option<F, R>(&mut self, id_opt: Option<impl Into<String>>, f: F) -> R
    where
        F: FnOnce(*const i8) -> R,
        R: Copy,
    {
        match id_opt {
            Some(id) => self.with_result(id, f),
            None => {
                let c_name = CString::new("").unwrap();
                f(c_name.as_ptr())
            }
        }
    }

    // Check if the current target is SPIR-V
    pub fn is_spirv_target(&self, module: LLVMModuleRef) -> bool {
        let target_triple = unsafe { LLVMGetTarget(module) };
        unsafe {
            std::ffi::CStr::from_ptr(target_triple)
                .to_str()
                .unwrap_or("")
                .starts_with("spir")
        }
    }
}

/// Convert PTX scalar type to LLVM type
pub fn get_scalar_type(context: LLVMContextRef, scalar: ScalarType) -> LLVMTypeRef {
    match scalar {
        ScalarType::Pred => unsafe { LLVMInt1TypeInContext(context) },
        ScalarType::B8 | ScalarType::U8 | ScalarType::S8 => unsafe {
            LLVMInt8TypeInContext(context)
        },
        ScalarType::B16 | ScalarType::U16 | ScalarType::S16 => unsafe {
            LLVMInt16TypeInContext(context)
        },
        ScalarType::B32 | ScalarType::U32 | ScalarType::S32 => unsafe {
            LLVMInt32TypeInContext(context)
        },
        ScalarType::B64 | ScalarType::U64 | ScalarType::S64 => unsafe {
            LLVMInt64TypeInContext(context)
        },
        ScalarType::F32 => unsafe { LLVMFloatTypeInContext(context) },
        ScalarType::F64 => unsafe { LLVMDoubleTypeInContext(context) },
        ScalarType::B128 => unsafe { LLVMInt128TypeInContext(context) },
        // Handle vector types
        ScalarType::S16x2
        | ScalarType::BF16x2
        | ScalarType::U16x2
        | ScalarType::F16x2
        | ScalarType::F16
        | ScalarType::BF16 => {
            // Fallback to 32-bit type for now
            unsafe { LLVMInt32TypeInContext(context) }
        }
    }
}

/// Get LLVM type for PTX type
pub fn get_type(
    context: LLVMContextRef,
    v_type: &ast::Type,
) -> Result<LLVMTypeRef, TranslateError> {
    match v_type {
        ast::Type::Scalar(scalar) => Ok(get_scalar_type(context, *scalar)),
        ast::Type::Vector(size, scalar) => {
            let size_val = (*size) as u32;
            let scalar_type = get_scalar_type(context, *scalar);
            Ok(unsafe { LLVMArrayType2(scalar_type, size_val as u64) })
        }
        ast::Type::Array(size, scalar, dimensions) => {
            let elem_type = get_scalar_type(context, *scalar);
            let mut array_type = elem_type;

            // Build the array type from innermost dimension outward
            for &dim in dimensions.iter().rev() {
                array_type = unsafe { LLVMArrayType2(array_type, dim as u64) };
            }

            // Handle vector prefix if present
            if let Some(prefix) = size {
                array_type = unsafe { LLVMArrayType2(array_type, prefix.get() as u64) };
            }

            Ok(array_type)
        }
        ast::Type::Pointer(type_ptr, state_space) => {
            // Don't try to match or check types, just use i32 as the pointed type
            // This is a workaround for the type mismatch issue
            let pointed_type = unsafe { LLVMInt32TypeInContext(context) };
            let address_space = get_state_space(*state_space)?;
            Ok(unsafe { LLVMPointerType(pointed_type, address_space) })
        }
    }
}

/// Get LLVM address space for PTX state space
pub fn get_state_space(state_space: StateSpace) -> Result<u32, TranslateError> {
    match state_space {
        StateSpace::Reg => Ok(0),
        StateSpace::Global => Ok(1),
        StateSpace::Shared => Ok(3),
        StateSpace::Local => Ok(5),
        StateSpace::Const => Ok(4),
        StateSpace::Param => Ok(0),
        _ => Err(TranslateError::Todo),
    }
}

/// Get LLVM pointer type for PTX state space
pub fn get_pointer_type(
    context: LLVMContextRef,
    state_space: StateSpace,
    v_type: &ast::Type,
) -> Result<LLVMTypeRef, TranslateError> {
    let pointed_type = get_type(context, v_type)?;
    let address_space = match state_space {
        StateSpace::Generic => 0,
        StateSpace::Reg => 0,
        _ => get_state_space(state_space)?,
    };

    Ok(unsafe { LLVMPointerType(pointed_type, address_space) })
}

/// Create LLVM function type
pub fn get_function_type<'a>(
    context: LLVMContextRef,
    return_types: impl Iterator<Item = &'a ast::Type>,
    arg_types: impl Iterator<Item = Result<LLVMTypeRef, TranslateError>>,
) -> Result<LLVMTypeRef, TranslateError> {
    // Process return types
    let return_types: Result<Vec<LLVMTypeRef>, TranslateError> =
        return_types.map(|ty| get_type(context, ty)).collect();
    let return_types = return_types?;

    // Process argument types
    let arg_types: Result<Vec<LLVMTypeRef>, TranslateError> = arg_types.collect();
    let arg_types = arg_types?;

    let return_type = match return_types.len() {
        0 => unsafe { LLVMVoidTypeInContext(context) },
        1 => return_types[0],
        _ => {
            // Create a struct type for multiple return values
            let struct_type = unsafe {
                let mut return_types_ptrs: Vec<*mut _> =
                    return_types.iter().map(|&t| t as *mut _).collect();
                LLVMStructTypeInContext(
                    context,
                    return_types_ptrs.as_mut_ptr(),
                    return_types.len() as u32,
                    0,
                )
            };
            struct_type
        }
    };

    let is_var_arg = 0; // Not variadic
    let function_type = unsafe {
        let mut arg_types_ptrs: Vec<*mut _> = arg_types.iter().map(|&t| t as *mut _).collect();
        LLVMFunctionType(
            return_type,
            arg_types_ptrs.as_mut_ptr(),
            arg_types.len() as u32,
            is_var_arg,
        )
    };

    Ok(function_type)
}

/// Get LLVM synchronization scope for PTX scope
pub fn get_scope(scope: ast::MemScope) -> Result<*const i8, TranslateError> {
    // Return raw pointer values expected by LLVM functions
    match scope {
        ast::MemScope::Cta => Ok(1 as *const i8),     // Workgroup
        ast::MemScope::Sys => Ok(2 as *const i8),     // System
        ast::MemScope::Cluster => Ok(1 as *const i8), // Workgroup (approximation)
        ast::MemScope::Gpu => Ok(2 as *const i8),     // System (approximation)
    }
}

/// Get LLVM atomic ordering for PTX memory semantics
pub fn get_ordering(semantics: ast::AtomSemantics) -> LLVMAtomicOrdering {
    match semantics {
        ast::AtomSemantics::Relaxed => LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic, // Relaxed
        _ => LLVMAtomicOrdering::LLVMAtomicOrderingAcquire, // Acquire (default for non-relaxed)
    }
}

/// Get LLVM failure ordering for atomics
pub fn get_ordering_failure(semantics: ast::AtomSemantics) -> LLVMAtomicOrdering {
    match semantics {
        ast::AtomSemantics::Relaxed => LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic, // Relaxed
        _ => LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic, // Relaxed (for failure case)
    }
}

/// Get bits representation for AtomSemantics
pub fn atom_semantics_bits(semantics: &ast::AtomSemantics) -> u32 {
    match semantics {
        ast::AtomSemantics::Relaxed => 0,
        ast::AtomSemantics::Acquire => 1,
        ast::AtomSemantics::Release => 2,
        _ => 3, // AcquireRelease or other
    }
}

/// Helper to display LLVM type in intrinsic names
pub fn LLVMTypeDisplay(scalar: ScalarType) -> &'static str {
    match scalar {
        ScalarType::Pred => "i1",
        ScalarType::B8 | ScalarType::U8 | ScalarType::S8 => "i8",
        ScalarType::B16 | ScalarType::U16 | ScalarType::S16 => "i16",
        ScalarType::B32 | ScalarType::U32 | ScalarType::S32 => "i32",
        ScalarType::B64 | ScalarType::U64 | ScalarType::S64 => "i64",
        ScalarType::F16 => "f16",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::B128 => "i128",
        // Add missing vector types
        ScalarType::S16x2 => "v2i16",
        ScalarType::BF16x2 => "v2bf16",
        ScalarType::U16x2 => "v2i16",
        ScalarType::F16x2 => "v2f16",
        ScalarType::BF16 => "bf16",
    }
}

// Helper function for memory barrier scopes that maps to synchronization scope
pub fn get_scope_membar(scope: ast::MemScope) -> Result<*const i8, TranslateError> {
    get_scope(scope)
}
