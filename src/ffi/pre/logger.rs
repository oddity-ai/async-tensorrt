use cpp::cpp;

cpp! {{
    #ifndef ODDITY_FFI_LOGGER
    #define ODDITY_FFI_LOGGER

    #include <mutex>

    // The custom logger is required for TensorRT. We can use it to intercept error messages and
    // diagnostics. TensorRT is very verbose when verbose.
    class Logger : public ILogger
    {
    public:
        // Implements the `log` function for the logger. This will be invoked
        // by TensorRT for every log message.
        void log(Severity severity, const char* msg) noexcept override {
            // If severity is `ERROR` or worse, then store the error message in
            // `m_lastError`.
            if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
                std::lock_guard<std::mutex> _lastErrorGuard(m_lastErrorMutex);
                m_lastError = std::string(msg);
            }
            // Pass message on to Rust handler.
            std::int32_t severity_val = static_cast<std::int32_t>(severity);
            rust!(Logger_handleLogMessage [
                severity_val : i32 as "std::int32_t",
                msg : *const std::os::raw::c_char as "const char*"
            ] {
                handle_log_message_raw(severity_val, msg);
            });
        }

        // Get last logged error message.
        const std::string getLastError() {
            std::lock_guard<std::mutex> _lastErrorGuard(m_lastErrorMutex);
            return m_lastError;
        }
    private:
        std::mutex m_lastErrorMutex {};
        std::string m_lastError = "";
    }
    GLOBAL_LOGGER;

    #endif // ODDITY_FFI_LOGGER
}}

/// TensorRT logging message severity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    /// An internal error has occurred. Execution is unrecoverable.
    InternalError,
    /// An application error has occurred.
    Error,
    /// An application error has been discovered, but TensorRT has recovered or fallen back to a default.
    Warning,
    /// Informational messages with instructional information.
    Info,
    /// Verbose messages with debugging information.
    Verbose,
    /// A severity code was provied by TensorRT that was not recognized.
    Unknown,
}

impl From<i32> for Severity {
    /// Convert from raw log level integer to [`Severity`].
    fn from(value: i32) -> Self {
        match value {
            0 => Severity::InternalError,
            1 => Severity::Error,
            2 => Severity::Warning,
            3 => Severity::Info,
            4 => Severity::Verbose,
            _ => Severity::Unknown,
        }
    }
}

/// Raw handler for log messages.
///
/// This function redirects logging to `tracing`, with the following rules:
/// * `InternalError` and `Error` become `error`.
/// * `Warning` becomes `warn`.
/// * `Info` becomes `trace`.
/// * All other logging is ignored.
///
/// # Arguments
///
/// * `severity` - Integer severity value of log message.
/// * `msg` - Raw C string log message.
///
/// # Safety
///
/// The caller must ensure that the message in `msg` is a valid pointer to a C string.
unsafe fn handle_log_message_raw(severity: i32, msg: *const std::os::raw::c_char) {
    let msg_c_str: &std::ffi::CStr = std::ffi::CStr::from_ptr(msg);
    let msg = msg_c_str.to_str().unwrap_or("");
    if !msg.is_empty() {
        match severity.into() {
            Severity::InternalError | Severity::Error => {
                tracing::error!(target: "tensorrt", "{msg}");
            }
            Severity::Warning => {
                tracing::warn!(target: "tensorrt", "{msg}");
            }
            Severity::Info => {
                tracing::trace!(target: "tensorrt", "{msg}");
            }
            _ => {}
        }
    }
}
