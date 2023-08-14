use cpp::cpp;

use crate::ffi::parser::Parser;

/// A network definition for input to the builder.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html)
pub struct NetworkDefinition {
    internal: *mut std::ffi::c_void,
    pub(crate) _parser: Option<Parser>,
}

/// Implements [`Send`] for [`NetworkDefinition`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`NetworkDefinition`].
unsafe impl Send for NetworkDefinition {}

/// Implements [`Sync`] for [`NetworkDefinition`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`NetworkDefinition`].
unsafe impl Sync for NetworkDefinition {}

impl NetworkDefinition {
    /// Wrap internal pointer as [`NetworkDefinition`].
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `INetworkDefinition` object.
    pub(crate) fn wrap(internal: *mut std::ffi::c_void) -> Self {
        Self {
            internal,
            _parser: None,
        }
    }

    /// Get network inputs.
    pub fn inputs(&self) -> Vec<Tensor> {
        let mut inputs = Vec::with_capacity(self.num_inputs());
        for index in 0..self.num_inputs() {
            inputs.push(self.input(index));
        }
        inputs
    }

    /// Get number of inputs.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a715d0ea103f1978c5b5e9173af2994a4)
    pub fn num_inputs(&self) -> usize {
        let internal = self.as_ptr();
        let num_inputs = cpp!(unsafe [
            internal as "const void*"
        ] -> std::os::raw::c_int as "int" {
            return ((const INetworkDefinition*) internal)->getNbInputs();
        });
        num_inputs as usize
    }

    /// Get network input at given index.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a3142a780be319b7f6a9e9e7f6ed12ca4)
    ///
    /// # Arguments
    ///
    /// * `index` - Input index.
    pub fn input(&self, index: usize) -> Tensor<'_> {
        let internal = self.as_ptr();
        let index = index as std::os::raw::c_int;
        let tensor_internal = cpp!(unsafe [
            internal as "const void*",
            index as "int"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((const INetworkDefinition*) internal)->getInput(index);
        });
        Tensor::wrap(tensor_internal)
    }

    /// Get network outputs.
    pub fn outputs(&self) -> Vec<Tensor<'_>> {
        let mut outputs = Vec::with_capacity(self.num_outputs());
        for index in 0..self.num_outputs() {
            outputs.push(self.output(index));
        }
        outputs
    }

    /// Get number of outputs.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#aef477421510ad25a342ecd950736a59a)
    pub fn num_outputs(&self) -> usize {
        let internal = self.as_ptr();
        let num_outputs = cpp!(unsafe [
            internal as "const void*"
        ] -> std::os::raw::c_int as "int" {
            return ((const INetworkDefinition*) internal)->getNbOutputs();
        });
        num_outputs as usize
    }

    /// Get network output at given index.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_network_definition.html#a2cb7b6ee73a876fc73076a559fa9e955)
    ///
    /// # Arguments
    ///
    /// * `index` - Output index.
    pub fn output(&self, index: usize) -> Tensor<'_> {
        let internal = self.as_ptr();
        let index = index as std::os::raw::c_int;
        let tensor_internal = cpp!(unsafe [
            internal as "const void*",
            index as "int"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((const INetworkDefinition*) internal)->getOutput(index);
        });
        Tensor::wrap(tensor_internal)
    }

    /// Get internal readonly pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let NetworkDefinition { internal, .. } = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let NetworkDefinition { internal, .. } = *self;
        internal
    }
}

impl Drop for NetworkDefinition {
    fn drop(&mut self) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((INetworkDefinition*) internal);
        });
    }
}

/// Specifies immutable properties of [`NetworkDefinition`] expressed at creation time.
///
/// [TensorRT documentation of `NetworkDefinitionCreationFlags`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a77b643e855bcc302b30348276fa36504)
/// [TensorRT documentation of `NetworkDefinitionCreationFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#aa8f406be96c14b7dbea548cf19f09a08a85b8fdd336af67a4aa147b3430064945)
#[derive(Copy, Clone)]
pub enum NetworkDefinitionCreationFlags {
    None,
    ExplicitBatchSize,
}

/// A tensor in a [`NetworkDefinition`].
///
/// [TensorRT documenation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html)
pub struct Tensor<'parent> {
    internal: *mut std::ffi::c_void,
    _phantom: std::marker::PhantomData<&'parent ()>,
}

/// Implements [`Send`] for [`Tensor`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Tensor`].
unsafe impl<'parent> Send for Tensor<'parent> {}

/// Implements [`Sync`] for [`Tensor`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Tensor`].
unsafe impl<'parent> Sync for Tensor<'parent> {}

impl<'parent> Tensor<'parent> {
    /// Wrap internal pointer as [`Tensor`].
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `ITensor` object.
    #[inline]
    pub(crate) fn wrap(internal: *mut std::ffi::c_void) -> Self {
        Self {
            internal,
            _phantom: Default::default(),
        }
    }

    /// Get the tensor name.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html#a684fd842a172ad300dbb31270fc675a2)
    pub fn name(&self) -> String {
        let internal = self.as_ptr();
        let name = cpp!(unsafe [
            internal as "const void*"
        ] -> *const std::os::raw::c_char as "const char*" {
            return ((const ITensor*) internal)->getName();
        });
        // SAFETY: This is safe because:
        // * The pointer is valid because we just got it from TensorRT.
        // * The pointer isn't kept after this block (we copy the string instead).
        unsafe { std::ffi::CStr::from_ptr(name).to_string_lossy().to_string() }
    }

    /// Set the tensor name.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html#a44ffc55db1d6e68908859596c4e4ef49)
    ///
    /// # Arguments
    ///
    /// * `name` - Name to set.
    pub fn set_name(&mut self, name: &str) {
        let internal = self.as_mut_ptr();
        let name_ffi = std::ffi::CString::new(name).unwrap();
        let name_ptr = name_ffi.as_ptr();
        cpp!(unsafe [
            internal as "void*",
            name_ptr as "const char*"
        ] {
            return ((ITensor*) internal)->setName(name_ptr);
        });
    }

    /// Get internal readonly pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let Tensor { internal, .. } = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let Tensor { internal, .. } = *self;
        internal
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::utils::*;

    #[tokio::test]
    async fn test_network_inputs_and_outputs() {
        let (_, network) = simple_network!();
        assert_eq!(network.num_inputs(), 1);
        assert_eq!(network.num_outputs(), 1);
        let inputs = network.inputs();
        let input = inputs.first().unwrap();
        assert_eq!(input.name(), "X");
        let outputs = network.outputs();
        let output = outputs.first().unwrap();
        assert_eq!(output.name(), "Y");
    }

    #[tokio::test]
    async fn test_tensor_set_name() {
        let (_, network) = simple_network!();
        network.outputs()[0].set_name("Z");
        assert_eq!(network.outputs()[0].name(), "Z");
    }
}
