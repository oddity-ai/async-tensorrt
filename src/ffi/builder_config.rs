use cpp::cpp;

/// Holds properties for configuring a builder to produce an engine.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html)
pub struct BuilderConfig(*mut std::ffi::c_void);

/// Implements [`Send`] for [`BuilderConfig`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`BuilderConfig`].
unsafe impl Send for BuilderConfig {}

/// Implements [`Sync`] for [`BuilderConfig`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`BuilderConfig`].
unsafe impl Sync for BuilderConfig {}

impl BuilderConfig {
    /// Wrap internal pointer as [`BuilderConfig`].
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `IBuilderConfig` object.
    pub(crate) fn wrap(internal: *mut std::ffi::c_void) -> Self {
        Self(internal)
    }

    /// Set the maximum workspace size.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#a8209999988ab480c60c8a905dfd2654d)
    ///
    /// # Arguments
    ///
    /// * `size` - The maximum GPU temporary memory which the engine can use at execution time in
    ///   bytes.
    pub fn with_max_workspace_size(mut self, size: usize) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*",
            size as "std::size_t"
        ] {
            ((IBuilderConfig*) internal)->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, size);
        });
        self
    }

    /// Set the `kSTRICT_TYPES` flag.
    ///
    /// [TensorRT documentation for `setFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ac9821504ae7a11769e48b0e62761837e)
    /// [TensorRT documentation for `kSTRICT_TYPES`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1ad3ff8ff39475957d8676c2cda337add7)
    pub fn with_strict_types(mut self) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kSTRICT_TYPES);
        });
        self
    }

    /// Set the `kFP16` flag.
    ///
    /// [TensorRT documentation for `setFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ac9821504ae7a11769e48b0e62761837e)
    /// [TensorRT documentation for `kFP16`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1a56e4ef5e47a48568bd24c4e0aaabcead)
    pub fn with_fp16(mut self) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kFP16);
        });
        self
    }

    /// Get internal readonly pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let BuilderConfig(internal) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let BuilderConfig(internal) = *self;
        internal
    }
}

impl Drop for BuilderConfig {
    fn drop(&mut self) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IBuilderConfig*) internal);
        });
    }
}
