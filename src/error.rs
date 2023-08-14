/// An error that occurred in TensorRT.
#[derive(Debug, Clone)]
pub enum Error {
    /// TensorRT error described by error message.
    TensorRt { message: String },
    /// Error in CUDA backend.
    Cuda(async_cuda_core::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::TensorRt { message } => write!(f, "{message}"),
            Error::Cuda(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<async_cuda_core::Error> for Error {
    #[inline]
    fn from(err: async_cuda_core::Error) -> Self {
        Error::Cuda(err)
    }
}

/// Create a TensorRT error from the last recorded error produced by the logger.
///
/// # Thread-safety
///
/// This function might return an error that was produced by an invocation to some TensorRT function
/// in a different thread.
///
/// # Return value
///
/// TensorRT error with corresponding error message.
#[inline]
pub(crate) fn last_error() -> Error {
    Error::TensorRt {
        message: crate::ffi::error::get_last_error_message(),
    }
}
