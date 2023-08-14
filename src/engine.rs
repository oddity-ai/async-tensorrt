use async_cuda_core::runtime::Future;
use async_cuda_core::{DeviceBuffer, Stream};

use crate::ffi::memory::HostBuffer;
use crate::ffi::sync::engine::Engine as InnerEngine;
use crate::ffi::sync::engine::ExecutionContext as InnerExecutionContext;

pub use crate::ffi::sync::engine::TensorIoMode;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Engine for executing inference on a built network.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html)
pub struct Engine {
    inner: InnerEngine,
}

impl Engine {
    /// Create [`Engine`] from its inner object.
    pub fn from_inner(inner: InnerEngine) -> Self {
        Self { inner }
    }

    /// Serialize the network.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#ab42c2fde3292f557ed17aae6f332e571)
    ///
    /// # Return value
    ///
    /// A [`HostBuffer`] that contains the serialized engine.
    #[inline(always)]
    pub fn serialize(&self) -> Result<HostBuffer> {
        self.inner.serialize()
    }

    /// Get the number of IO tensors.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#af2018924cbea2fa84808040e60c58405)
    #[inline(always)]
    pub fn num_io_tensors(&self) -> usize {
        self.inner.num_io_tensors()
    }

    /// Retrieve the name of an IO tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#a0b1e9e3f82724be40f0ab74742deaf92)
    ///
    /// # Arguments
    ///
    /// * `io_tensor_index` - IO tensor index.
    #[inline(always)]
    pub fn io_tensor_name(&self, io_tensor_index: usize) -> String {
        self.inner.io_tensor_name(io_tensor_index)
    }

    /// Get the shape of a tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#af96a2ee402ab47b7e0b7f0becb63d693)
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - Tensor name.
    #[inline(always)]
    pub fn tensor_shape(&self, tensor_name: &str) -> Vec<usize> {
        self.inner.tensor_shape(tensor_name)
    }

    /// Get the IO mode of a tensor.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#ae236a14178df506070cd39a9ef3775e7)
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - Tensor name.
    #[inline(always)]
    pub fn tensor_io_mode(&self, tensor_name: &str) -> TensorIoMode {
        self.inner.tensor_io_mode(tensor_name)
    }
}

/// Context for executing inference using an engine.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html)
pub struct ExecutionContext<'engine> {
    inner: InnerExecutionContext<'engine>,
}

impl ExecutionContext<'static> {
    /// Create an execution context from an [`Engine`].
    ///
    /// This is the owned version of [`ExecutionContext::new()`]. It consumes the engine. In
    /// exchange, it produces an execution context with a `'static` lifetime.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#ac7a34cf3b59aa633a35f66f07f22a617)
    ///
    /// # Arguments
    ///
    /// * `engine` - Parent engine.
    pub async fn from_engine(engine: Engine) -> Result<Self> {
        Future::new(move || {
            InnerExecutionContext::from_engine(engine.inner).map(ExecutionContext::from_inner_owned)
        })
        .await
    }

    /// Create multiple execution contexts from an [`Engine`].
    ///
    /// This is the owned version of [`ExecutionContext::new()`]. It consumes the engine. In
    /// exchange, it produces a set of execution contexts with a `'static` lifetime.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#ac7a34cf3b59aa633a35f66f07f22a617)
    ///
    /// # Arguments
    ///
    /// * `engine` - Parent engine.
    /// * `num` - Number of execution contexsts to produce.
    pub async fn from_engine_many(engine: Engine, num: usize) -> Result<Vec<Self>> {
        Future::new(move || {
            Ok(InnerExecutionContext::from_engine_many(engine.inner, num)?
                .into_iter()
                .map(Self::from_inner_owned)
                .collect())
        })
        .await
    }

    /// Create [`ExecutionContext`] from its inner object.
    fn from_inner_owned(inner: InnerExecutionContext<'static>) -> Self {
        Self { inner }
    }
}

impl<'engine> ExecutionContext<'engine> {
    /// Create [`ExecutionContext`] from its inner object.
    fn from_inner(inner: InnerExecutionContext<'engine>) -> Self {
        Self { inner }
    }

    /// Create an execution context from an [`Engine`].
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_cuda_engine.html#ac7a34cf3b59aa633a35f66f07f22a617)
    ///
    /// # Arguments
    ///
    /// * `engine` - Parent engine.
    pub async fn new(engine: &mut Engine) -> Result<ExecutionContext> {
        Future::new(move || {
            InnerExecutionContext::new(&mut engine.inner).map(ExecutionContext::from_inner)
        })
        .await
    }

    /// Asynchronously execute inference.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_execution_context.html#a63cd95430852038ce864e17c670e0b36)
    ///
    /// # Stream ordered semantics
    ///
    /// This function exhibits stream ordered semantics. This means that it is only guaranteed to
    /// complete serially with respect to other operations on the same stream.
    ///
    /// # Thread-safety
    ///
    /// Calling this function from the same context with a different CUDA stream concurrently
    /// results in undefined behavior. To perform inference concurrently in multiple streams, use
    /// one execution context per stream.
    ///
    /// # Arguments
    ///
    /// * `io_buffers` - Input and output buffers.
    /// * `stream` - CUDA stream to execute on.
    pub async fn enqueue<T: Copy>(
        &mut self,
        io_buffers: &mut std::collections::HashMap<&str, &mut DeviceBuffer<T>>,
        stream: &Stream,
    ) -> Result<()> {
        let mut io_buffers_inner = io_buffers
            .iter_mut()
            .map(|(name, buffer)| (*name, buffer.inner_mut()))
            .collect::<std::collections::HashMap<_, _>>();
        Future::new(move || self.inner.enqueue(&mut io_buffers_inner, stream.inner())).await
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::memory::*;
    use crate::tests::utils::*;

    use super::*;

    #[tokio::test]
    async fn test_engine_serialize() {
        let engine = simple_engine!();
        let serialized_engine = engine.serialize().unwrap();
        let serialized_engine_bytes = serialized_engine.as_bytes();
        assert!(serialized_engine_bytes.len() > 1800);
        assert!(serialized_engine_bytes.len() < 2500);
        assert_eq!(
            &serialized_engine_bytes[..8],
            &[102_u8, 116_u8, 114_u8, 116_u8, 0_u8, 0_u8, 0_u8, 0_u8],
        );
    }

    #[tokio::test]
    async fn test_engine_tensor_info() {
        let engine = simple_engine!();
        assert_eq!(engine.num_io_tensors(), 2);
        assert_eq!(engine.io_tensor_name(0), "X");
        assert_eq!(engine.io_tensor_name(1), "Y");
        assert_eq!(engine.tensor_io_mode("X"), TensorIoMode::Input);
        assert_eq!(engine.tensor_io_mode("Y"), TensorIoMode::Output);
        assert_eq!(engine.tensor_shape("X"), &[1, 2]);
        assert_eq!(engine.tensor_shape("Y"), &[2, 3]);
    }

    #[tokio::test]
    async fn test_execution_context_new() {
        let mut engine = simple_engine!();
        assert!(ExecutionContext::new(&mut engine).await.is_ok());
        assert!(ExecutionContext::new(&mut engine).await.is_ok());
    }

    #[tokio::test]
    async fn test_execution_context_enqueue() {
        let stream = Stream::new().await.unwrap();
        let mut engine = simple_engine!();
        let mut context = ExecutionContext::new(&mut engine).await.unwrap();
        let mut io_buffers = std::collections::HashMap::from([
            ("X", to_device!(&[2.0, 4.0], &stream)),
            ("Y", to_device!(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &stream)),
        ]);
        let mut io_buffers_ref = io_buffers
            .iter_mut()
            .map(|(name, buffer)| (*name, buffer))
            .collect();
        context.enqueue(&mut io_buffers_ref, &stream).await.unwrap();
        let output = to_host!(io_buffers["Y"], &stream);
        assert_eq!(&output, &[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
