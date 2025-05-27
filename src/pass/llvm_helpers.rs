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
