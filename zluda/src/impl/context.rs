use super::{driver, FromCuda, ZludaObject};
use cuda_types::cuda::*;
#[cfg(feature = "amd")]
use hip_runtime_sys::*;
use rustc_hash::FxHashSet;
#[cfg(feature = "intel")]
use std::os::raw::c_void;
use std::{cell::RefCell, ptr, sync::Mutex};
#[cfg(feature = "intel")]
use ze_runtime_sys::*;
use std::ffi::c_uint;
// 添加Result转换特性，用于ze_result_t到CUerror的转换
#[cfg(feature = "intel")]
trait ResultExt {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror>;
}

#[cfg(feature = "intel")]
impl ResultExt for ze_result_t {
    fn to_cuda_result<T>(self, value: T) -> Result<T, CUerror> {
        match self {
            ze_result_t::ZE_RESULT_SUCCESS => Ok(value),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY => Err(CUerror::OUT_OF_MEMORY),
            _ => Err(CUerror::UNKNOWN),
        }
    }
}

thread_local! {
    #[cfg(feature = "amd")]
    pub(crate) static CONTEXT_STACK: RefCell<Vec<(CUcontext, hipDevice_t)>> = RefCell::new(Vec::new());
    #[cfg(feature = "intel")]
    pub(crate) static CONTEXT_STACK: RefCell<Vec<(CUcontext, ze_device_handle_t)>> = RefCell::new(Vec::new());
}
#[cfg(feature = "amd")]
pub(crate) struct Context {
    pub(crate) device: hipDevice_t,
    pub(crate) mutable: Mutex<OwnedByContext>,
}
#[cfg(feature = "amd")]
impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            device: self.device,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _memory: FxHashSet::default(),
                _streams: FxHashSet::default(),
                _modules: FxHashSet::default(),
            }),
        }
    }
}
#[cfg(feature = "amd")]
pub(crate) struct OwnedByContext {
    pub(crate) ref_count: usize, // only used by primary context
    pub(crate) _memory: FxHashSet<hipDeviceptr_t>,
    pub(crate) _streams: FxHashSet<hipStream_t>,
    pub(crate) _modules: FxHashSet<CUmodule>,
}

impl ZludaObject for Context {
    const COOKIE: usize = 0x5f867c6d9cb73315;

    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}
#[cfg(feature = "amd")]
impl Context {
    pub(crate) fn new(device: hipDevice_t) -> Self {
        Self {
            device,
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _memory: FxHashSet::default(),
                _streams: FxHashSet::default(),
                _modules: FxHashSet::default(),
            }),
        }
    }

    pub(crate) fn get_device(&self) -> hipDevice_t {
        self.device
    }

    pub(crate) fn increment_ref_count(&self) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable.ref_count += 1;
        }
    }

    pub(crate) fn decrement_ref_count(&self) -> usize {
        if let Ok(mut mutable) = self.mutable.lock() {
            if mutable.ref_count > 0 {
                mutable.ref_count -= 1;
            }
            mutable.ref_count
        } else {
            0
        }
    }

    pub(crate) fn add_memory(&self, ptr: hipDeviceptr_t) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._memory.insert(ptr);
        }
    }

    pub(crate) fn remove_memory(&self, ptr: hipDeviceptr_t) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._memory.remove(&ptr)
        } else {
            false
        }
    }

    pub(crate) fn add_stream(&self, stream: hipStream_t) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._streams.insert(stream);
        }
    }

    pub(crate) fn remove_stream(&self, stream: hipStream_t) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._streams.remove(&stream)
        } else {
            false
        }
    }

    pub(crate) fn add_module(&self, module: CUmodule) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._modules.insert(module);
        }
    }

    pub(crate) fn remove_module(&self, module: CUmodule) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._modules.remove(&module)
        } else {
            false
        }
    }
}
#[cfg(feature = "amd")]
pub(crate) unsafe fn get_limit(pvalue: *mut usize, limit: hipLimit_t) -> hipError_t {
    unsafe { hipDeviceGetLimit(pvalue, limit) }
}
#[cfg(feature = "amd")]
pub(crate) fn set_limit(limit: hipLimit_t, value: usize) -> hipError_t {
    unsafe { hipDeviceSetLimit(limit, value) }
}
#[cfg(feature = "amd")]
pub(crate) fn synchronize() -> hipError_t {
    unsafe { hipDeviceSynchronize() }
}
#[cfg(feature = "amd")]
pub(crate) fn get_primary(hip_dev: hipDevice_t) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device(hip_dev)?;
    Ok(dev.primary_context())
}
#[cfg(feature = "amd")]
pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let new_device = if raw_ctx.0 == ptr::null_mut() {
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, old_device)) = stack.pop() {
                if let Some((_, new_device)) = stack.last() {
                    if old_device != *new_device {
                        return Some(*new_device);
                    }
                }
            }
            None
        })
    } else {
        let ctx: &Context = FromCuda::from_cuda(&raw_ctx).unwrap();
        let device = ctx.device;
        CONTEXT_STACK.with(move |stack| {
            let mut stack = stack.borrow_mut();
            let last_device = stack.last().map(|(_, dev)| *dev);
            stack.push((raw_ctx, device));
            match last_device {
                None => Some(device),
                Some(last_device) if last_device != device => Some(device),
                _ => None,
            }
        })
    };

    if let Some(dev) = new_device {
        unsafe { hipSetDevice(dev).unwrap() };
    }

    Ok(())
}

pub(crate) fn get_current() -> Option<CUcontext> {
    CONTEXT_STACK.with(|stack| {
        let stack = stack.borrow();
        stack.last().map(|(ctx, _)| *ctx)
    })
}

pub(crate) fn pop_current() -> Option<CUcontext> {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.pop().map(|(ctx, _)| ctx)
    })
}
#[cfg(feature = "amd")]
pub(crate) fn push(ctx: CUcontext, device: hipDevice_t) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push((ctx, device));
    });
}

#[cfg(feature = "amd")]
pub(crate) fn get_device_properties(device: hipDevice_t) -> Result<hipDeviceArch_t, CUerror> {
    let mut props = unsafe { std::mem::zeroed() };
    unsafe { hipGetDevicePropertiesR0600(&mut props, device).unwrap() };
    Ok(props.arch)
}
// Intel Level Zero implementations
#[cfg(feature = "intel")]
pub(crate) struct Context {
    pub(crate) device: ze_device_handle_t,
    pub(crate) context: ze_context_handle_t,
    pub(crate) mutable: Mutex<OwnedByContext>,
}
#[cfg(feature = "intel")]
impl Clone for Context {
    fn clone(&self) -> Self {
        let guard = self.mutable.lock().unwrap();
        Self {
            device: self.device,
            context: self.context,
            mutable: Mutex::new(OwnedByContext {
                ref_count: guard.ref_count,
                _command_queues: guard._command_queues.clone(),
                _command_lists: guard._command_lists.clone(),
                _modules: guard._modules.clone(),
                _allocations: guard._allocations.clone(),
            }),
        }
    }
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
#[cfg(feature = "intel")]
pub(crate) struct OwnedByContext {
    pub(crate) ref_count: usize,
    pub(crate) _command_queues: FxHashSet<ze_command_queue_handle_t>,
    pub(crate) _command_lists: FxHashSet<ze_command_list_handle_t>,
    pub(crate) _modules: FxHashSet<ze_module_handle_t>,
    pub(crate) _allocations: FxHashSet<*mut c_void>,
}

#[cfg(feature = "intel")]
impl Context {
    pub(crate) fn new(device: ze_device_handle_t) -> Self {
        // Create an empty context that will be properly initialized later when needed
        Self {
            device,
            context: ze_context_handle_t(std::ptr::null_mut()),
            mutable: Mutex::new(OwnedByContext {
                ref_count: 0,
                _command_queues: FxHashSet::default(),
                _command_lists: FxHashSet::default(),
                _modules: FxHashSet::default(),
                _allocations: FxHashSet::default(),
            }),
        }
    }

    pub(crate) fn initialize(&mut self) -> Result<(), CUerror> {
        if !self.context.0.is_null() {
            return Ok(());
        }

        // Initialize Level Zero
        unsafe { zeInit(0) }.to_cuda_result(())?;

        // Get driver for this device
        let mut driver_count = 0;
        unsafe { zeDriverGet(&mut driver_count, std::ptr::null_mut()).to_cuda_result(())? };

        let mut drivers = vec![std::ptr::null_mut(); driver_count as usize];
        unsafe { zeDriverGet(&mut driver_count, *drivers.as_mut_ptr()).to_cuda_result(())? };

        // Create context descriptor
        let mut context_desc = ze_context_desc_t {
            stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            pNext: std::ptr::null(),
            flags: 0,
        };

        let mut context = std::ptr::null_mut();

        // Create context for the device
        unsafe {
            zeContextCreate(*drivers[0], &mut context_desc, *&mut context).to_cuda_result(())?;

            self.context = *context;
        }

        // Initialize the reference count to 1 for the primary context
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable.ref_count = 1;
        }

        Ok(())
    }

    pub(crate) fn get_device(&self) -> ze_device_handle_t {
        self.device
    }

    pub(crate) fn get_context(&self) -> ze_context_handle_t {
        self.context
    }

    pub(crate) fn increment_ref_count(&self) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable.ref_count += 1;
        }
    }

    pub(crate) fn decrement_ref_count(&self) -> usize {
        if let Ok(mut mutable) = self.mutable.lock() {
            if mutable.ref_count > 0 {
                mutable.ref_count -= 1;
            }
            mutable.ref_count
        } else {
            0
        }
    }

    pub(crate) fn add_module(&self, module: ze_module_handle_t) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._modules.insert(module);
        }
    }

    pub(crate) fn remove_module(&self, module: ze_module_handle_t) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._modules.remove(&module)
        } else {
            false
        }
    }

    pub(crate) fn add_command_queue(&self, queue: ze_command_queue_handle_t) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._command_queues.insert(queue);
        }
    }

    pub(crate) fn remove_command_queue(&self, queue: ze_command_queue_handle_t) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._command_queues.remove(&queue)
        } else {
            false
        }
    }

    pub(crate) fn add_command_list(&self, list: ze_command_list_handle_t) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._command_lists.insert(list);
        }
    }

    pub(crate) fn remove_command_list(&self, list: ze_command_list_handle_t) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._command_lists.remove(&list)
        } else {
            false
        }
    }

    pub(crate) fn add_allocation(&self, ptr: *mut c_void) {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._allocations.insert(ptr);
        }
    }

    pub(crate) fn remove_allocation(&self, ptr: *mut c_void) -> bool {
        if let Ok(mut mutable) = self.mutable.lock() {
            mutable._allocations.remove(&ptr)
        } else {
            false
        }
    }

    pub(crate) fn destroy(&self) -> Result<(), CUerror> {
        // Destroy all tracked resources
        if let Ok(mutable) = self.mutable.lock() {
            // Clean up modules
            for module in &mutable._modules {
                unsafe { zeModuleDestroy(*module) }.to_cuda_result(())?;
            }

            // Clean up command lists
            for cmd_list in &mutable._command_lists {
                unsafe { zeCommandListDestroy(*cmd_list) }.to_cuda_result(())?;
            }

            // Clean up command queues
            for queue in &mutable._command_queues {
                unsafe { zeCommandQueueDestroy(*queue) }.to_cuda_result(())?;
            }

            // Free memory allocations
            for allocation in &mutable._allocations {
                unsafe { zeMemFree(self.context, *allocation) }.to_cuda_result(())?;
            }
        }

        // Destroy context
        unsafe { zeContextDestroy(self.context) }.to_cuda_result(())?;

        Ok(())
    }
}

#[cfg(feature = "intel")]
pub(crate) unsafe fn get_limit(_pvalue: *mut usize, _limit: c_uint) -> ze_result_t {
    // Level Zero doesn't have direct equivalents for all HIP/CUDA limits
    // For now, return success with default values
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "intel")]
pub(crate) fn set_limit(_limit: c_uint, _value: usize) -> ze_result_t {
    // Level Zero doesn't have direct equivalents for all HIP/CUDA limits
    // For now, return success
    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "intel")]
pub(crate) fn synchronize() -> ze_result_t {
    // Get current context
    let ctx = match get_current() {
        Some(ctx) => ctx,
        None => return ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT,
    };

    // Get Context from CUcontext
    let ze_ctx = match from_cuda_to(&ctx) {
        Ok(ctx) => ctx,
        Err(_) => return ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT,
    };

    // For full synchronization, synchronize all command queues
    if let Ok(mutable) = ze_ctx.mutable.lock() {
        for queue in &mutable._command_queues {
            let result = unsafe { zeCommandQueueSynchronize(*queue, u64::MAX) };

            if result != ze_result_t::ZE_RESULT_SUCCESS {
                return ze_result_t::ZE_RESULT_ERROR_UNKNOWN;
            }
        }
    }

    ze_result_t::ZE_RESULT_SUCCESS
}

#[cfg(feature = "intel")]
pub(crate) fn from_cuda_to(ctx: &CUcontext) -> Result<&Context, CUerror> {
    FromCuda::from_cuda(ctx)
}

#[cfg(feature = "intel")]
pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    let new_device = if raw_ctx.0 == ptr::null_mut() {
        CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, _)) = stack.pop() {
                if let Some((_, new_device)) = stack.last() {
                    return Some(*new_device);
                }
            }
            None
        })
    } else {
        let ze_ctx = from_cuda_to(&raw_ctx)?;
        let device = ze_ctx.device;
        CONTEXT_STACK.with(move |stack| {
            let mut stack = stack.borrow_mut();
            let last_device = stack.last().map(|(_, dev)| *dev);
            stack.push((raw_ctx, device));
            match last_device {
                None => Some(device),
                Some(last_device) if last_device != device => Some(device),
                _ => None,
            }
        })
    };

    // No direct equivalent to hipSetDevice in Level Zero
    // Device switching would be handled at command queue/list creation

    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn push(ctx: CUcontext, device: ze_device_handle_t) {
    CONTEXT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push((ctx, device));
    });
}

#[cfg(feature = "intel")]
pub(crate) fn get_device_properties(
    device: ze_device_handle_t,
) -> Result<ze_device_properties_t, CUerror> {
    let mut props: ze_device_properties_t = unsafe { std::mem::zeroed() };
    props.stype = ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    unsafe { zeDeviceGetProperties(device, &mut props) }.to_cuda_result(props)?;

    Ok(props)
}

// Intel Level Zero memory management functions
#[cfg(feature = "intel")]
pub(crate) fn ze_malloc(
    size: usize,
    alignment: usize,
    ze_ctx: &Context,
) -> Result<*mut c_void, CUerror> {
    let device_desc = ze_device_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        pNext: ptr::null(),
        flags: 0,
        ordinal: 0,
    };

    let mut ptr = ptr::null_mut();
    unsafe {
        zeMemAllocDevice(
            ze_ctx.context,
            &device_desc,
            size,
            alignment,
            ze_ctx.device,
            &mut ptr,
        )
        .to_cuda_result(())?;
    }

    // Track the allocation
    ze_ctx.add_allocation(ptr);

    Ok(ptr)
}

#[cfg(feature = "intel")]
pub(crate) fn ze_malloc_host(
    size: usize,
    alignment: usize,
    ze_ctx: &Context,
) -> Result<*mut c_void, CUerror> {
    let host_desc = ze_host_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        pNext: ptr::null(),
        flags: 0,
    };

    let mut ptr = ptr::null_mut();
    unsafe {
        zeMemAllocHost(ze_ctx.context, &host_desc, size, alignment, &mut ptr).to_cuda_result(())?;
    }

    // Track the allocation
    ze_ctx.add_allocation(ptr);

    Ok(ptr)
}

#[cfg(feature = "intel")]
pub(crate) fn ze_malloc_shared(
    size: usize,
    alignment: usize,
    ze_ctx: &Context,
) -> Result<*mut c_void, CUerror> {
    let device_desc = ze_device_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        pNext: ptr::null(),
        flags: 0,
        ordinal: 0,
    };

    let host_desc = ze_host_mem_alloc_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        pNext: ptr::null(),
        flags: 0,
    };

    let mut ptr = ptr::null_mut();
    unsafe {
        zeMemAllocShared(
            ze_ctx.context,
            &device_desc,
            &host_desc,
            size,
            alignment,
            ze_ctx.device,
            &mut ptr,
        )
        .to_cuda_result(())?;
    }

    // Track the allocation
    ze_ctx.add_allocation(ptr);

    Ok(ptr)
}

#[cfg(feature = "intel")]
pub(crate) fn ze_free(ptr: *mut c_void, ze_ctx: &Context) -> Result<(), CUerror> {
    unsafe {
        zeMemFree(ze_ctx.context, ptr).to_cuda_result(())?;
    }

    // Remove from tracked allocations
    ze_ctx.remove_allocation(ptr);

    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn ze_memcpy(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    ze_ctx: &Context,
) -> Result<(), CUerror> {
    // Get or create a command list for the copy operation
    let command_list = ze_get_command_list(ze_ctx)?;

    unsafe {
        // Append copy to command list
        zeCommandListAppendMemoryCopy(
            command_list,
            dst,
            src,
            size,
            *ptr::null_mut(),
            0,
            &mut ze_event_handle_t(ptr::null_mut()),
        )
        .to_cuda_result(())?;

        // Close the command list
        zeCommandListClose(command_list).to_cuda_result(())?;

        // Execute the command list
        let command_queue = ze_get_command_queue(ze_ctx)?;
        zeCommandQueueExecuteCommandLists(
            command_queue,
            1,
            &command_list,
            ze_fence_handle_t(ptr::null_mut()),
        )
        .to_cuda_result(())?;

        // Synchronize the command queue
        zeCommandQueueSynchronize(command_queue, u64::MAX).to_cuda_result(())?;
    }

    Ok(())
}

#[cfg(feature = "intel")]
pub(crate) fn ze_memset(
    dst: *mut c_void,
    value: i32,
    size: usize,
    ze_ctx: &Context,
) -> Result<(), CUerror> {
    // Get or create a command list for the set operation
    let command_list = ze_get_command_list(ze_ctx)?;

    unsafe {
        // Append fill to command list
        zeCommandListAppendMemoryFill(
            command_list,
            dst,
            &value as *const i32 as *const c_void,
            std::mem::size_of::<i32>(),
            size,
            *ptr::null_mut(),
            0,
            &mut ze_event_handle_t(ptr::null_mut()),
        )
        .to_cuda_result(())?;

        // Close the command list
        zeCommandListClose(command_list).to_cuda_result(())?;

        // Execute the command list
        let command_queue = ze_get_command_queue(ze_ctx)?;
        zeCommandQueueExecuteCommandLists(
            command_queue,
            1,
            &command_list,
            ze_fence_handle_t(ptr::null_mut()),
        )
        .to_cuda_result(())?;

        // Synchronize the command queue
        zeCommandQueueSynchronize(command_queue, u64::MAX).to_cuda_result(())?;
    }

    Ok(())
}

#[cfg(feature = "intel")]
fn ze_get_command_list(ze_ctx: &Context) -> Result<ze_command_list_handle_t, CUerror> {
    // In a real implementation, you might maintain a pool of command lists
    // or reuse existing ones, but for simplicity we create a new one

    let mut cmd_list_desc = ze_command_list_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        pNext: ptr::null(),
        commandQueueGroupOrdinal: 0, // Use default queue group
        flags: 0,
    };

    let mut command_list = ptr::null_mut();
    unsafe {
        zeCommandListCreate(
            ze_ctx.context,
            ze_ctx.device,
            &mut cmd_list_desc,
            *&mut command_list,
        )
        .to_cuda_result(())?;
    }

    let handle = unsafe { *command_list };

    // Track the command list
    ze_ctx.add_command_list(handle);

    Ok(handle)
}

#[cfg(feature = "intel")]
fn ze_get_command_queue(ze_ctx: &Context) -> Result<ze_command_queue_handle_t, CUerror> {
    // In a real implementation, you might maintain a pool of command queues
    // or reuse existing ones, but for simplicity we create a new one

    let mut queue_desc = ze_command_queue_desc_t {
        stype: ze_structure_type_t::ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        pNext: ptr::null(),
        ordinal: 0,
        index: 0,
        flags: 0,
        mode: ze_command_queue_mode_t::ZE_COMMAND_QUEUE_MODE_DEFAULT,
        priority: ze_command_queue_priority_t::ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };

    let mut command_queue = ptr::null_mut();
    unsafe {
        zeCommandQueueCreate(
            ze_ctx.context,
            ze_ctx.device,
            &mut queue_desc,
            *&mut command_queue,
        )
        .to_cuda_result(())?;
    }

    let handle = unsafe { *command_queue };

    // Track the command queue
    ze_ctx.add_command_queue(handle);

    Ok(handle)
}

#[cfg(feature = "intel")]
pub(crate) fn get_current_ze() -> Result<&'static Context, CUerror> {
    // Get the current CUcontext from the context stack
    let current_ctx = CONTEXT_STACK
        .with(|stack| stack.borrow().last().map(|(ctx, _)| *ctx))
        .ok_or(CUerror::INVALID_CONTEXT)?;

    // Convert the CUcontext to a &Context
    let context = from_cuda_to(&current_ctx)?;

    // This is a bit unsafe but necessary since we need to return a 'static reference
    // In reality, the context is stored in thread_local storage so it will live
    // for the duration of the program
    Ok(unsafe { std::mem::transmute(context) })
}

#[cfg(feature = "intel")]
pub(crate) fn get_primary_ze(
    device: ze_device_handle_t,
) -> Result<(&'static Context, CUcontext), CUerror> {
    let dev = driver::device_ze(device)?;
    Ok(dev.primary_context())
}
