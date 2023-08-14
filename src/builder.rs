use async_cuda_core::runtime::Future;

use crate::ffi::builder_config::BuilderConfig;
use crate::ffi::memory::HostBuffer;
use crate::ffi::network::{NetworkDefinition, NetworkDefinitionCreationFlags};
use crate::ffi::sync::builder::Builder as InnerBuilder;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Builds an engine from a network definition.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html)
pub struct Builder {
    inner: InnerBuilder,
}

impl Builder {
    /// Create a new [`Builder`].
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1_1_1anonymous__namespace_02_nv_infer_8h_03.html)
    pub async fn new() -> Self {
        let inner = Future::new(InnerBuilder::new).await;
        Builder { inner }
    }

    /// Create a new optimization profile.
    ///
    /// Note that the official TensorRT documentation states:
    ///
    /// "If the network has any dynamic input tensors, the appropriate calls to setDimensions() must
    /// be made."
    ///
    /// But this part of the API has not been implemented yet in `async-tensorrt`.
    ///
    /// This function is still useful to allocate the optimization profile in the [`Builder`], which
    /// may or may not actually affect the building process later.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a68a8b59fbf86e42762b7087e6ffe6fb4)
    #[inline(always)]
    pub fn add_optimization_profile(&mut self) -> Result<()> {
        self.inner.add_optimization_profile()
    }

    /// Create a new optimization profile.
    ///
    /// Note that the official TensorRT documentation states:
    ///
    /// "If the network has any dynamic input tensors, the appropriate calls to setDimensions() must
    /// be made."
    ///
    /// But this part of the API has not been implemented yet in `async-tensorrt`.
    ///
    /// This function is still useful to allocate the optimization profile in the [`Builder`], which
    /// may or may not actually affect the building process later.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a68a8b59fbf86e42762b7087e6ffe6fb4)
    #[inline(always)]
    pub fn with_optimization_profile(mut self) -> Result<Self> {
        self.add_optimization_profile()?;
        Ok(self)
    }

    /// Create a builder configuration object.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a8fac4203e688430dff87483fc9db6bf2)
    ///
    /// # Return value
    ///
    /// A [`BuilderConfig`] that can later be passed to `build_serialized_network`.
    #[inline(always)]
    pub async fn config(&mut self) -> BuilderConfig {
        Future::new(|| self.inner.config()).await
    }

    /// Create a network definition object.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a853122d044b70383b2c9ebe7fdf11e07)
    ///
    /// # Arguments
    ///
    /// * `flags` - Flags for specifying network properties.
    #[inline(always)]
    pub fn network_definition(
        &mut self,
        flags: NetworkDefinitionCreationFlags,
    ) -> NetworkDefinition {
        self.inner.network_definition(flags)
    }

    /// Builds and serializes a network for the provided [`crate::ffi::network::NetworkDefinition`]
    /// and [`BuilderConfig`].
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#ab25ed4aec280df7d64e82930aa6b41c7)
    ///
    /// # Arguments
    ///
    /// * `network_definition` - Network definition.
    /// * `config` - Builder configuration.
    pub async fn build_serialized_network(
        &mut self,
        network_definition: &mut NetworkDefinition,
        config: BuilderConfig,
    ) -> Result<HostBuffer> {
        Future::new(move || {
            self.inner
                .build_serialized_network(network_definition, config)
        })
        .await
    }

    /// Determine whether the platform has fast native INT8.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#ab09433c57e3ef02f7aad672ec4235ea4)
    #[inline(always)]
    pub fn platform_has_fast_int8(&self) -> bool {
        self.inner.platform_has_fast_int8()
    }

    /// Determine whether the platform has fast native FP16.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder.html#a6e42dd3ecb449ba54ffb823685a7ac47)
    #[inline(always)]
    pub fn platform_has_fast_fp16(&self) -> bool {
        self.inner.platform_has_fast_fp16()
    }
}
