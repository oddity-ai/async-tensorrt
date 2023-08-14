use cpp::cpp;

/// Get last error message produced by TensorRT.
///
/// # Thread-safety
///
/// This function could return an error produced by a function that executed on a different thread.
pub fn get_last_error_message() -> String {
    // SAFETY: This is safe because first, we copy the error out of `GLOBAL_LOGGER`, which is thread-safe,
    // and then we take ownership of the string before destroying it on the C++ side.
    let error_boxed_ptr = cpp!(unsafe [] -> *mut std::ffi::c_void as "void*" {
        const std::string lastError = GLOBAL_LOGGER.getLastError();
        const char* lastErrorCstr = lastError.c_str();
        void* lastErrorPtr = rust!(Logger_takeLastError [
            lastErrorCstr : *const std::os::raw::c_char as "const char*"
        ] -> *mut std::ffi::c_void as "void*" {
            let error_boxed = Box::new(
                std::ffi::CStr::from_ptr(lastErrorCstr)
                    .to_str()
                    .unwrap_or_default()
                    .to_owned()
            );
            Box::into_raw(error_boxed) as *mut std::ffi::c_void
        });
        return lastErrorPtr;
    });
    // SAFETY: This is safe because we boxed the error ourselves earlier and used `Box::into_raw`
    // to get this pointer.
    let error_boxed = unsafe { Box::from_raw(error_boxed_ptr as *mut String) };
    let error = *error_boxed;
    if !error.is_empty() {
        error
    } else {
        "unknown error".to_string()
    }
}
