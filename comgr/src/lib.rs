#[cfg(feature = "intel")]
use intel_comgr_sys::*;
#[cfg(feature = "amd")]
use amd_comgr_sys::*;
use std::{ffi::CStr, mem, ptr};

#[cfg(feature = "amd")]
struct Data(amd_comgr_data_t);
#[cfg(feature = "amd")]
impl Data {
    fn new(
        kind: amd_comgr_data_kind_t,
        name: &CStr,
        content: &[u8],
    ) -> Result<Self, amd_comgr_status_s> {
        let mut data = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_data(kind, &mut data) }?;
        unsafe { amd_comgr_set_data_name(data, name.as_ptr()) }?;
        unsafe { amd_comgr_set_data(data, content.len(), content.as_ptr().cast()) }?;
        Ok(Self(data))
    }

    fn get(&self) -> amd_comgr_data_t {
        self.0
    }

    fn copy_content(&self) -> Result<Vec<u8>, amd_comgr_status_s> {
        let mut size = unsafe { mem::zeroed() };
        unsafe { amd_comgr_get_data(self.get(), &mut size, ptr::null_mut()) }?;
        let mut result: Vec<u8> = Vec::with_capacity(size);
        unsafe { result.set_len(size) };
        unsafe { amd_comgr_get_data(self.get(), &mut size, result.as_mut_ptr().cast()) }?;
        Ok(result)
    }
}
#[cfg(feature = "amd")]
struct DataSet(amd_comgr_data_set_t);
#[cfg(feature = "amd")]
impl DataSet {
    fn new() -> Result<Self, amd_comgr_status_s> {
        let mut data_set = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_data_set(&mut data_set) }?;
        Ok(Self(data_set))
    }

    fn add(&self, data: &Data) -> Result<(), amd_comgr_status_s> {
        unsafe { amd_comgr_data_set_add(self.get(), data.get()) }
    }

    fn get(&self) -> amd_comgr_data_set_t {
        self.0
    }

    fn get_data(
        &self,
        kind: amd_comgr_data_kind_t,
        index: usize,
    ) -> Result<Data, amd_comgr_status_s> {
        let mut data = unsafe { mem::zeroed() };
        unsafe { amd_comgr_action_data_get_data(self.get(), kind, index, &mut data) }?;
        Ok(Data(data))
    }
}
#[cfg(feature = "amd")]
impl Drop for DataSet {
    fn drop(&mut self) {
        unsafe { amd_comgr_destroy_data_set(self.get()).ok() };
    }
}
#[cfg(feature = "amd")]
struct ActionInfo(amd_comgr_action_info_t);

#[cfg(feature = "amd")]
impl ActionInfo {
    fn new() -> Result<Self, amd_comgr_status_s> {
        let mut action = unsafe { mem::zeroed() };
        unsafe { amd_comgr_create_action_info(&mut action) }?;
        Ok(Self(action))
    }

    fn set_isa_name(&self, isa: &CStr) -> Result<(), amd_comgr_status_s> {
        let mut full_isa = "amdgcn-amd-amdhsa--".to_string().into_bytes();
        full_isa.extend(isa.to_bytes_with_nul());
        unsafe { amd_comgr_action_info_set_isa_name(self.get(), full_isa.as_ptr().cast()) }
    }

    fn set_language(&self, language: amd_comgr_language_t) -> Result<(), amd_comgr_status_s> {
        unsafe { amd_comgr_action_info_set_language(self.get(), language) }
    }

    fn set_options<'a>(
        &self,
        options: impl Iterator<Item = &'a CStr>,
    ) -> Result<(), amd_comgr_status_s> {
        let options = options.map(|x| x.as_ptr()).collect::<Vec<_>>();
        unsafe {
            amd_comgr_action_info_set_option_list(
                self.get(),
                options.as_ptr().cast_mut(),
                options.len(),
            )
        }
    }

    fn get(&self) -> amd_comgr_action_info_t {
        self.0
    }
}

#[cfg(feature = "amd")]
impl Drop for ActionInfo {
    fn drop(&mut self) {
        unsafe { amd_comgr_destroy_action_info(self.get()).ok() };
    }
}
#[cfg(feature = "amd")]
pub fn compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, amd_comgr_status_s> {
    let bitcode_data_set = DataSet::new()?;
    let main_bitcode_data = Data::new(
        amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
        c"zluda.bc",
        main_buffer,
    )?;
    bitcode_data_set.add(&main_bitcode_data)?;
    let stdlib_bitcode_data = Data::new(
        amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_BC,
        c"ptx_impl.bc",
        ptx_impl,
    )?;
    bitcode_data_set.add(&stdlib_bitcode_data)?;
    let linking_info = ActionInfo::new()?;
    let linked_data_set = do_action(
        &bitcode_data_set,
        &linking_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_BC_TO_BC,
    )?;
    let link_with_device_libs_info = ActionInfo::new()?;
    link_with_device_libs_info.set_isa_name(gcn_arch)?;
    link_with_device_libs_info.set_language(amd_comgr_language_t::AMD_COMGR_LANGUAGE_LLVM_IR)?;
    // This makes no sense, but it makes ockl linking work
    link_with_device_libs_info
        .set_options([c"-Xclang", c"-mno-link-builtin-bitcode-postopt"].into_iter())?;
    let with_device_libs = do_action(
        &linked_data_set,
        &link_with_device_libs_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
    )?;
    let compile_action_info = ActionInfo::new()?;
    compile_action_info.set_isa_name(gcn_arch)?;
    let common_options = [c"-O3", c"-mno-wavefrontsize64", c"-mcumode"].into_iter();
    let opt_options = if cfg!(debug_assertions) {
        [c"-g", c"", c"", c"", c""]
    } else {
        [
            c"-g0",
            // default inlining threshold times 10
            c"-mllvm",
            c"-inline-threshold=2250",
            c"-mllvm",
            c"-inlinehint-threshold=3250",
        ]
    };
    compile_action_info.set_options(common_options.chain(opt_options))?;
    let reloc_data_set = do_action(
        &with_device_libs,
        &compile_action_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
    )?;
    let exec_data_set = do_action(
        &reloc_data_set,
        &compile_action_info,
        amd_comgr_action_kind_t::AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
    )?;
    let executable =
        exec_data_set.get_data(amd_comgr_data_kind_t::AMD_COMGR_DATA_KIND_EXECUTABLE, 0)?;
    executable.copy_content()
}

#[cfg(feature = "amd")]
fn do_action(
    data_set: &DataSet,
    action: &ActionInfo,
    kind: amd_comgr_action_kind_t,
) -> Result<DataSet, amd_comgr_status_s> {
    let result = DataSet::new()?;
    unsafe { amd_comgr_do_action(kind, action.get(), data_set.get(), result.get()) }?;
    Ok(result)
}

#[cfg(feature = "intel")]
pub fn compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, intel_comgr_status_s> {
    let bitcode_data_set = DataSet::new()?;
    
    // Add main bitcode
    let mut main_data = unsafe { mem::zeroed() };
    unsafe { intel_comgr_create_data(intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC, &mut main_data) }?;
    unsafe { intel_comgr_data_set_name(main_data, c"zluda.bc".as_ptr()) }?;
    unsafe { intel_comgr_data_set_bytes(main_data, main_buffer.as_ptr() as _, main_buffer.len()) }?;
    unsafe { intel_comgr_data_set_add(bitcode_data_set.0, main_data) }?;
    
    // Add stdlib bitcode
    let mut stdlib_data = unsafe { mem::zeroed() };
    unsafe { intel_comgr_create_data(intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC, &mut stdlib_data) }?;
    unsafe { intel_comgr_data_set_name(stdlib_data, c"ptx_impl.bc".as_ptr()) }?;
    unsafe { intel_comgr_data_set_bytes(stdlib_data, ptx_impl.as_ptr() as _, ptx_impl.len()) }?;
    unsafe { intel_comgr_data_set_add(bitcode_data_set.0, stdlib_data) }?;
    
    // Linking action info
    let mut linking_info = unsafe { mem::zeroed() };
    unsafe { intel_comgr_create_action_info(&mut linking_info) }?;
    
    // Link BC to BC
    let linked_data_set = do_action(
        &bitcode_data_set,
        &ActionInfo(linking_info),
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_LINK_BC_TO_BC,
    )?;
    
    // Setup action info for adding device libraries
    let mut device_libs_info = unsafe { mem::zeroed() };
    unsafe { intel_comgr_create_action_info(&mut device_libs_info) }?;
    unsafe { intel_comgr_action_info_set_language(device_libs_info, intel_comgr_language_s::INTEL_COMGR_LANGUAGE_LLVM_IR) }?;
    
    // Set target if needed
    // The gcn_arch in Intel's case would typically be a specific GPU target
    let target = format!("intel-gpu-{}", gcn_arch.to_str().unwrap_or("gen9"));
    let target_cstr = std::ffi::CString::new(target).unwrap();
    unsafe { intel_comgr_action_info_set_target(device_libs_info, target_cstr.as_ptr()) }?;
    
    // Set options
    let options = ["-Xclang", "-mno-link-builtin-bitcode"];
    let options_cstr = options
        .iter()
        .map(|opt| std::ffi::CString::new(*opt).unwrap())
        .collect::<Vec<_>>();
    let option_ptrs = options_cstr
        .iter()
        .map(|opt| opt.as_ptr())
        .collect::<Vec<_>>();
    
    unsafe { intel_comgr_action_info_set_option_list(device_libs_info, option_ptrs.as_ptr(), option_ptrs.len()) }?;
    
    // Add device libraries to the bitcode
    let with_device_libs = do_action(
        &linked_data_set,
        &ActionInfo(device_libs_info),
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
    )?;
    
    // Setup compiler action info for optimization
    let mut compile_info = unsafe { mem::zeroed() };
    unsafe { intel_comgr_create_action_info(&mut compile_info) }?;
    
    // Common compiler options
    let common_options = ["-O3"];
    // Add debug options if in debug mode
    let debug_options = if cfg!(debug_assertions) {
        vec!["-g"]
    } else {
        vec![
            "-g0",
            // Optimization flags 
            "-mllvm", "-inline-threshold=2250",
            "-mllvm", "-inlinehint-threshold=3250",
        ]
    };
    
    // Combine options
    let mut all_options = common_options.to_vec();
    all_options.extend(debug_options);
    
    // Convert options to C strings
    let options_cstr = all_options
        .iter()
        .map(|opt| std::ffi::CString::new(*opt).unwrap())
        .collect::<Vec<_>>();
    let option_ptrs = options_cstr
        .iter()
        .map(|opt| opt.as_ptr())
        .collect::<Vec<_>>();
    
    unsafe { intel_comgr_action_info_set_option_list(compile_info, option_ptrs.as_ptr(), option_ptrs.len()) }?;
    
    // Optimize bitcode
    let optimized_bc = do_action(
        &with_device_libs,
        &ActionInfo(compile_info),
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_OPTIMIZE_BC_TO_BC,
    )?;
    
    // Generate relocatable code
    let reloc_data_set = do_action(
        &optimized_bc,
        &ActionInfo(compile_info),
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
    )?;
    
    // Get the output data
    let mut count = 0;
    unsafe { intel_comgr_get_data_count(reloc_data_set.0, &mut count) }?;
    
    if count == 0 {
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
    }
    
    // Get the data
    let mut data = unsafe { mem::zeroed() };
    unsafe { intel_comgr_get_data(reloc_data_set.0, 0, &mut data) }?;
    
    // Get the data size and copy contents
    let mut size = 0;
    let mut buffer = Vec::new();
    
    // First get size
    unsafe {
        // Get the size of the data
        intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size)?;
        
        if size > 0 {
            // Allocate buffer and get bytes
            buffer.reserve(size);
            buffer.set_len(size);
            intel_comgr_data_get_bytes(data, buffer.as_mut_ptr() as *mut std::os::raw::c_void, &mut size)?;
        }
        
        // Release resources
        intel_comgr_release_data(data)?;
    }
    
    Ok(buffer)
}

#[cfg(feature = "intel")]
fn do_action(
    data_set: &DataSet,
    action: &ActionInfo,
    kind: intel_comgr_action_kind_s,
) -> Result<DataSet, intel_comgr_status_s> {
    let mut result = DataSet::new()?;
    unsafe { intel_comgr_do_action(kind, action.0, data_set.0, result.0) }?;
    Ok(result)
}

// Helper implementation for DataSet with Intel support
#[cfg(feature = "intel")]
pub struct DataSet(intel_comgr_data_set_t);
#[cfg(feature = "intel")]
impl DataSet {
    fn new() -> Result<Self, intel_comgr_status_s> {
        let mut data_set = unsafe { mem::zeroed() };
        unsafe { intel_comgr_create_data_set(&mut data_set) }?;
        Ok(Self(data_set))
    }
    
    fn get(&self) -> intel_comgr_data_set_t {
        self.0
    }
}

// Drop implementation for Intel DataSet
#[cfg(feature = "intel")]
impl Drop for DataSet {
    fn drop(&mut self) {
        unsafe { intel_comgr_release_data_set(self.0).ok() };
    }
}

// Implementation of ActionInfo for Intel
#[cfg(feature = "intel")]
impl ActionInfo {
    fn new() -> Result<Self, intel_comgr_status_s> {
        let mut action = unsafe { mem::zeroed() };
        unsafe { intel_comgr_create_action_info(&mut action) }?;
        Ok(Self(action))
    }
    
    fn set_language(&self, language: intel_comgr_language_t) -> Result<(), intel_comgr_status_s> {
        unsafe { intel_comgr_action_info_set_language(self.0, language) }
    }
    
    fn set_options<'a>(
        &self,
        options: impl Iterator<Item = &'a CStr>,
    ) -> Result<(), intel_comgr_status_s> {
        let options = options.map(|x| x.as_ptr()).collect::<Vec<_>>();
        unsafe {
            intel_comgr_action_info_set_option_list(
                self.0,
                options.as_ptr(),
                options.len(),
            )
        }
    }
    
    fn get(&self) -> intel_comgr_action_info_t {
        self.0
    }
}

// Drop implementation for Intel ActionInfo
#[cfg(feature = "intel")]
impl Drop for ActionInfo {
    fn drop(&mut self) {
        unsafe { intel_comgr_release_action_info(self.0).ok() };
    }
}