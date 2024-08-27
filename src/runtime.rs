use async_cuda::runtime::Future;

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

    /// Set whether the runtime is allowed to deserialize engines with host executable code.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#a5a19c2524f74179cd9b781c6240eb3ce)
    ///
    /// # Arguments
    ///
    /// * `allowed` - Whether the runtime is allowed to deserialize engines with host executable code.
    pub fn set_engine_host_code_allowed(&mut self, allowed: bool) {
        self.inner.set_engine_host_code_allowed(allowed);
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
