use amd_comgr_sys::*;
#[cfg(feature = "amd")]
use amd_comgr_sys::*;
#[cfg(feature = "intel")]
use intel_comgr_sys::*;
#[cfg(feature = "intel")]
use std::ffi::CString;
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
    // Optional debug log
    eprintln!("ZLUDA DEBUG: Compiling bitcode for Intel GPU target");
    eprintln!(
        "ZLUDA DEBUG: Main buffer size: {} bytes, PTX impl size: {} bytes",
        main_buffer.len(),
        ptx_impl.len()
    );
    eprintln!(
        "ZLUDA DEBUG: Target architecture: {:?}",
        gcn_arch.to_string_lossy()
    );

    // Directly try to compile - no fallback
    let result = try_compile_bitcode(gcn_arch, main_buffer, ptx_impl);

    match &result {
        Ok(buffer) => {
            eprintln!(
                "ZLUDA DEBUG: Compilation succeeded, generated SPIR-V size: {} bytes",
                buffer.len()
            );
        }
        Err(e) => {
            eprintln!("ZLUDA DEBUG: Compilation failed with error: {:?}", e);
        }
    }

    result
}

#[cfg(feature = "intel")]
fn try_compile_bitcode(
    gcn_arch: &CStr,
    main_buffer: &[u8],
    ptx_impl: &[u8],
) -> Result<Vec<u8>, intel_comgr_status_s> {
    eprintln!(
        "ZLUDA VERBOSE: Creating relocatable with buffer size = {}, ptx_impl size = {}",
        main_buffer.len(),
        ptx_impl.len()
    );

    // Create new DataSet for inputs
    let bitcode_data_set = DataSet::new()?;

    // Create the main bitcode data
    let mut main_data = unsafe { mem::zeroed() };
    match unsafe {
        intel_comgr_create_data(
            intel_comgr_data_kind_s::INTEL_COMGR_DATA_KIND_BC,
            &mut main_data,
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Created main bitcode data input"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to create main bitcode data: {:?}", e);
            return Err(e);
        }
    }

    match unsafe {
        intel_comgr_data_set_bytes(
            main_data,
            main_buffer.as_ptr() as *const std::os::raw::c_void,
            main_buffer.len(),
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set main bitcode data content"),
        Err(e) => {
            eprintln!(
                "ZLUDA ERROR: Failed to set main bitcode data content: {:?}",
                e
            );
            return Err(e);
        }
    }

    match unsafe { intel_comgr_data_set_name(main_data, c"combined_module.bc".as_ptr()) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set main bitcode data name"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set main bitcode data name: {:?}", e);
            return Err(e);
        }
    }

    // Add the main bitcode data to the input DataSet
    match unsafe { intel_comgr_data_set_add(bitcode_data_set.0, main_data) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Added main bitcode data to input DataSet"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to add main bitcode data: {:?}", e);
            return Err(e);
        }
    }

    // Setup compilation options
    let mut compile_info = unsafe { mem::zeroed() };
    match unsafe { intel_comgr_create_action_info(&mut compile_info) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Created compile action info"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to create compile action info: {:?}", e);
            return Err(e);
        }
    }

    match unsafe {
        intel_comgr_action_info_set_language(
            compile_info,
            intel_comgr_language_s::INTEL_COMGR_LANGUAGE_OPENCL_2_0,
        )
    } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set compile language to OpenCL 2.0"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set compile language: {:?}", e);
            return Err(e);
        }
    }

    // Set the target architecture
    let target_cstr =
        CString::new(format!("skl-{}", "64")).expect("failed to create target string");
    match unsafe { intel_comgr_action_info_set_target(compile_info, target_cstr.as_ptr()) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Set compile target to SKL-64"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to set compile target: {:?}", e);
            return Err(e);
        }
    }

    // Perform the BC to relocatable action
    let action_info = ActionInfo(compile_info);
    let reloc_data_set = do_action(
        &bitcode_data_set,
        &action_info,
        intel_comgr_action_kind_s::INTEL_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
    )?;

    // Get the output relocatable object data
    let mut count = 0;
    match unsafe { intel_comgr_get_data_count(reloc_data_set.0, &mut count) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Found {} output data objects", count),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to get output data count: {:?}", e);
            return Err(e);
        }
    }

    if count == 0 {
        eprintln!("ZLUDA ERROR: No output data objects found");
        return Err(intel_comgr_status_s::INTEL_COMGR_STATUS_ERROR);
    }

    let mut data = unsafe { mem::zeroed() };
    match unsafe { intel_comgr_get_data(reloc_data_set.0, 0, &mut data) } {
        Ok(_) => eprintln!("ZLUDA VERBOSE: Successfully got output data"),
        Err(e) => {
            eprintln!("ZLUDA ERROR: Failed to get output data: {:?}", e);
            return Err(e);
        }
    }

    // Get the content of the output data
    let mut buffer = Vec::new();
    let mut size = 0;
    unsafe {
        // Get the size of the data
        match intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size) {
            Ok(_) => eprintln!("ZLUDA VERBOSE: Output data size: {} bytes", size),
            Err(e) => {
                eprintln!("ZLUDA ERROR: Failed to get output data size: {:?}", e);
                return Err(e);
            }
        }

        if size > 0 {
            // Allocate buffer and get bytes
            buffer.reserve(size);
            buffer.set_len(size);
            match intel_comgr_data_get_bytes(
                data,
                buffer.as_mut_ptr() as *mut std::os::raw::c_void,
                &mut size,
            ) {
                Ok(_) => {
                    eprintln!(
                        "ZLUDA VERBOSE: Successfully copied output data of {} bytes",
                        size
                    );

                    // Check if the buffer contains our mock marker
                    let marker = b"ZLUDA_MOCK_RELOCATABLE\0";
                    if buffer.len() > marker.len()
                        && buffer.windows(marker.len()).any(|window| window == marker)
                    {
                        eprintln!("ZLUDA VERBOSE: Found mock relocatable marker in output");
                    }

                    // Output some file structure info to help debug
                    if buffer.len() >= 4 && &buffer[0..4] == b"\x7fELF" {
                        eprintln!("ZLUDA VERBOSE: Output has valid ELF header");
                    } else {
                        eprintln!("ZLUDA WARNING: Output does not have valid ELF header");

                        // Try to display the first 32 bytes for debugging
                        if buffer.len() >= 32 {
                            let prefix: Vec<_> =
                                buffer[0..32].iter().map(|b| format!("{:02x}", b)).collect();
                            eprintln!("ZLUDA DEBUG: First 32 bytes: {}", prefix.join(" "));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("ZLUDA ERROR: Failed to copy output data: {:?}", e);
                    return Err(e);
                }
            }
        } else {
            eprintln!("ZLUDA WARNING: Output data size is 0");
        }

        // Release resources
        match intel_comgr_release_data(data) {
            Ok(_) => eprintln!("ZLUDA VERBOSE: Released output data resources"),
            Err(e) => {
                eprintln!(
                    "ZLUDA ERROR: Failed to release output data resources: {:?}",
                    e
                );
                return Err(e);
            }
        }
    }

    eprintln!(
        "ZLUDA VERBOSE: Compilation completed successfully, output size: {} bytes",
        buffer.len()
    );
    Ok(buffer)
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
#[cfg(feature = "intel")]
struct ActionInfo(intel_comgr_action_info_t);

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
        unsafe { intel_comgr_action_info_set_option_list(self.0, options.as_ptr(), options.len()) }
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

#[cfg(feature = "intel")]
fn do_action(
    data_set: &DataSet,
    action: &ActionInfo,
    kind: intel_comgr_action_kind_s,
) -> Result<DataSet, intel_comgr_status_s> {
    let mut result = DataSet::new()?;
    let action_kind_name = match kind.0 {
        0 => "INTEL_COMGR_ACTION_SOURCE_TO_PREPROCESSED",
        1 => "INTEL_COMGR_ACTION_ADD_PRECOMPILED_HEADERS",
        2 => "INTEL_COMGR_ACTION_COMPILE_SOURCE_TO_BC",
        3 => "INTEL_COMGR_ACTION_ADD_DEVICE_LIBRARIES",
        4 => "INTEL_COMGR_ACTION_LINK_BC_TO_BC",
        5 => "INTEL_COMGR_ACTION_OPTIMIZE_BC_TO_BC",
        6 => "INTEL_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE",
        7 => "INTEL_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY",
        8 => "INTEL_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE",
        9 => "INTEL_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN",
        _ => "UNKNOWN_ACTION",
    };

    eprintln!("ZLUDA VERBOSE: Executing action: {}", action_kind_name);

    let status = unsafe { intel_comgr_do_action(kind, action.0, data_set.0, result.0) };

    match status {
        Ok(_) => {
            eprintln!(
                "ZLUDA VERBOSE: Action {} completed successfully",
                action_kind_name
            );

            // Try to get log data if available
            let mut data_count = 0;
            if let Ok(_) = unsafe { intel_comgr_get_data_count(result.0, &mut data_count) } {
                eprintln!("ZLUDA VERBOSE: Action produced {} data objects", data_count);

                // Log retrieval is simplified as Intel doesn't have the data_get_kind function
                for i in 0..data_count {
                    let mut data = unsafe { mem::zeroed() };
                    if let Ok(_) = unsafe { intel_comgr_get_data(result.0, i, &mut data) } {
                        // Try to get data size - if successful, attempt to read it as log
                        let mut size = 0;
                        if let Ok(_) = unsafe {
                            intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size)
                        } {
                            if size > 0 {
                                let mut content = vec![0u8; size];
                                if let Ok(_) = unsafe {
                                    intel_comgr_data_get_bytes(
                                        data,
                                        content.as_mut_ptr() as *mut std::os::raw::c_void,
                                        &mut size,
                                    )
                                } {
                                    if let Ok(text) = String::from_utf8(content) {
                                        if text.contains("error:") || text.contains("warning:") {
                                            eprintln!(
                                                "ZLUDA COMPILER LOG for {}: \n{}",
                                                action_kind_name, text
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Release the data
                        unsafe { intel_comgr_release_data(data).ok() };
                    }
                }
            }

            Ok(result)
        }
        Err(e) => {
            eprintln!(
                "ZLUDA ERROR: Action {} failed with error: {:?}",
                action_kind_name, e
            );

            // Check if we can get any logs even in case of failure
            let mut data_count = 0;
            if let Ok(_) = unsafe { intel_comgr_get_data_count(result.0, &mut data_count) } {
                if data_count > 0 {
                    eprintln!("ZLUDA VERBOSE: Found {} logs for failed action", data_count);
                    for i in 0..data_count {
                        let mut data = unsafe { mem::zeroed() };
                        if let Ok(_) = unsafe { intel_comgr_get_data(result.0, i, &mut data) } {
                            let mut size = 0;
                            if let Ok(_) = unsafe {
                                intel_comgr_data_get_bytes(data, std::ptr::null_mut(), &mut size)
                            } {
                                if size > 0 {
                                    let mut content = vec![0u8; size];
                                    if let Ok(_) = unsafe {
                                        intel_comgr_data_get_bytes(
                                            data,
                                            content.as_mut_ptr() as *mut std::os::raw::c_void,
                                            &mut size,
                                        )
                                    } {
                                        if let Ok(text) = String::from_utf8(content) {
                                            eprintln!(
                                                "ZLUDA FAILURE LOG for {}: \n{}",
                                                action_kind_name, text
                                            );
                                        }
                                    }
                                }
                            }
                            unsafe { intel_comgr_release_data(data).ok() };
                        }
                    }
                }
            }

            Err(e)
        }
    }
}
