use async_cuda_core::runtime::Future;

use crate::engine::Engine;
use crate::ffi::memory::HostBuffer;
use crate::ffi::sync::runtime::Runtime as InnerRuntime;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Allows a serialized engine to be serialized.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html)
pub struct Runtime {
    inner: InnerRuntime,
}

impl Runtime {
    /// Create a new [`Runtime`].
    pub async fn new() -> Self {
        let inner = Future::new(InnerRuntime::new).await;
        Self { inner }
    }

    /// Deserialize engine from a plan (a [`HostBuffer`]).
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#ad0dc765e77cab99bfad901e47216a767)
    ///
    /// # Arguments
    ///
    /// * `plan` - Plan to deserialize from.
    pub async fn deserialize_engine_from_plan(self, plan: &HostBuffer) -> Result<Engine> {
        Future::new(move || {
            self.inner
                .deserialize_engine_from_plan(plan)
                .map(Engine::from_inner)
        })
        .await
    }

    /// Deserialize engine from a slice buffer.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#ad0dc765e77cab99bfad901e47216a767)
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer slice to read from.
    pub async fn deserialize_engine(self, buffer: &[u8]) -> Result<Engine> {
        Future::new(move || {
            self.inner
                .deserialize_engine(buffer)
                .map(Engine::from_inner)
        })
        .await
    }
}
