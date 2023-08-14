use cpp::cpp;

use crate::error::last_error;
use crate::ffi::network::NetworkDefinition;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// For parsing an ONNX model into a TensorRT network definition ([`crate::NetworkDefinition`]).
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvonnxparser_1_1_i_parser.html)
pub struct Parser(*mut std::ffi::c_void);

impl Parser {
    /// Create new parser, parse ONNX file and return a [`crate::NetworkDefinition`].
    ///
    /// Note that this function is CPU-intensive. Callers should not use it in async context or
    /// spawn a blocking task for it.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvonnxparser_1_1_i_parser.html#a973ac2ed682f18c4c6258ed93fc8efa3)
    ///
    /// # Arguments
    ///
    /// * `network_definition` - Network definition to use.
    /// * `path` - Path to file to parse.
    ///
    /// # Return value
    ///
    /// Parsed network definition.
    pub fn parse_network_definition_from_file(
        mut network_definition: NetworkDefinition,
        path: &impl AsRef<std::path::Path>,
    ) -> Result<NetworkDefinition> {
        // SAFETY: The call to `Parser::new` is unsafe because we must ensure that the new parser
        // outlives `network_definition`. We manually make sure of that here by putting the parser
        // inside `NetworkDefinition` and such it will only be destroyed when `network_definition`
        // is.
        unsafe {
            let mut parser = Self::new(&mut network_definition);
            parser.parse_from_file(path)?;
            // Put parser object in `network_definition` because destroying the parser before the
            // network definition is not allowed.
            network_definition._parser = Some(parser);
        }
        Ok(network_definition)
    }

    /// Parse ONNX file.
    ///
    /// Note that this function is CPU-intensive. Callers should not use it in async context or
    /// spawn a blocking task for it.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvonnxparser_1_1_i_parser.html#a973ac2ed682f18c4c6258ed93fc8efa3)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to file to parse.
    fn parse_from_file(&mut self, path: &impl AsRef<std::path::Path>) -> Result<()> {
        let internal = self.as_mut_ptr();
        let path_ffi = std::ffi::CString::new(path.as_ref().as_os_str().to_str().unwrap()).unwrap();
        let path_ptr = path_ffi.as_ptr();
        let ret = cpp!(unsafe [
            internal as "void*",
            path_ptr as "const char*"
        ] -> bool as "bool" {
            return ((IParser*) internal)->parseFromFile(
                path_ptr,
                // Set to `VERBOSE` and let Rust code handle what message are passed on based on
                // logger configuration.
                static_cast<int>(ILogger::Severity::kVERBOSE)
            );
        });
        if ret {
            Ok(())
        } else {
            Err(last_error())
        }
    }

    /// Create new parser.
    ///
    /// # Arguments
    ///
    /// * Reference to network definition to attach to parser.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the [`Parser`] outlives the given [`NetworkDefinition`].
    unsafe fn new(network_definition: &mut NetworkDefinition) -> Self {
        let network_definition_internal = network_definition.as_ptr();
        let internal = cpp!(unsafe [
            network_definition_internal as "void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return createParser(
                *((INetworkDefinition*) network_definition_internal),
                GLOBAL_LOGGER
            );
        });
        Parser(internal)
    }

    /// Get internal readonly pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let Parser(internal) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let Parser(internal) = *self;
        internal
    }
}

impl Drop for Parser {
    fn drop(&mut self) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IParser*) internal);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::onnx::*;
    use crate::{Builder, NetworkDefinitionCreationFlags};

    #[tokio::test]
    async fn test_parser_parses_onnx_file() {
        let simple_onnx_file = simple_onnx_file!();
        let mut builder = Builder::new().await;
        let network = builder.network_definition(NetworkDefinitionCreationFlags::ExplicitBatchSize);
        assert!(
            Parser::parse_network_definition_from_file(network, &simple_onnx_file.path()).is_ok()
        );
    }
}
