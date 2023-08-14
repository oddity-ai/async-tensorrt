// The order of these includes is important due to the nature of how the `cpp` crate works. We want
// the C++ includes, helpers and other global stuff to be initialized in order or it might not
// compile correctly.
#[rustfmt::skip]
mod pre {
    mod includes;
    mod helpers;
    mod logger;
}

pub mod builder_config;
pub mod error;
pub mod memory;
pub mod network;
pub mod parser;
pub mod sync;

/// Convenience macro for turning TensorRT error code into a `std::result::Result`.
///
/// # Usage
///
/// There are two possible uses of the macro:
///
/// (1) Shorthand to return `Ok(something)` or the most recent TensorRT error:
///
/// ```ignore
/// result!(return_value);
/// ```
///
/// (2) Shorthand to return `Ok(())` or the most recent TensorRT error:
///
/// ```ignore
/// result!(code)
/// ```
macro_rules! result {
    ($ptr:expr, $ok:expr) => {
        if !$ptr.is_null() {
            Ok($ok)
        } else {
            Err($crate::error::last_error())
        }
    };
    ($ptr:expr) => {
        result!($ptr, ())
    };
}

use result;
