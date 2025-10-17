use cpp::cpp;

use crate::OptimizationProfile;

type Result<T> = std::result::Result<T, crate::error::Error>;

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
            #if NV_TENSORRT_MAJOR >= 10
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kDIRECT_IO);
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
            #else
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kSTRICT_TYPES);
            #endif
        });
        self
    }

    /// Set the `kVERSION_COMPATIBLE` flag.
    ///
    /// [TensorRT documentation for `setFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ac9821504ae7a11769e48b0e62761837e)
    /// [TensorRT documentation for `kVERSION_COMPATIBLE`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1a64917aa1f8d9238c555a46fa1d4e83b7)
    pub fn with_version_compability(mut self) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kVERSION_COMPATIBLE);
        });
        self
    }

    /// Set the `kEXCLUDE_LEAN_RUNTIME` flag.
    ///
    /// [TensorRT documentation for `setFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ac9821504ae7a11769e48b0e62761837e)
    /// [TensorRT documentation for `kEXCLUDE_LEAN_RUNTIME`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1a239d59ead8393beeecaadd21ce3b3502)
    pub fn with_exclude_lean_runtime(mut self) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kEXCLUDE_LEAN_RUNTIME);
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

    /// Set the `kINT8` flag.
    ///
    /// [TensorRT documentation for `setFlag`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ac9821504ae7a11769e48b0e62761837e)
    /// [TensorRT documentation for `kINT8`](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#abdc74c40fe7a0c3d05d2caeccfbc29c1a69c1a4a69db0e50820cf63122f90ad09)
    pub fn with_int8(mut self) -> Self {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            ((IBuilderConfig*) internal)->setFlag(BuilderFlag::kINT8);
        });
        self
    }

    /// Add an optimization profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ab97fa40c85fa8afab65fc2659e38da82)
    pub fn with_optimization_profile(
        mut self,
        optimization_profile: OptimizationProfile,
    ) -> Result<Self> {
        self.add_optimization_profile(optimization_profile)?;
        Ok(self)
    }

    /// Add an optimization profile.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#ab97fa40c85fa8afab65fc2659e38da82)
    pub fn add_optimization_profile(
        &mut self,
        optimization_profile: OptimizationProfile,
    ) -> Result<()> {
        let internal = self.as_mut_ptr();
        let optimization_profile = optimization_profile.as_ptr();
        let index = cpp!(unsafe [
            internal as "void*",
            optimization_profile as "const IOptimizationProfile*"
        ] -> i32 as "std::int32_t" {
           return ((IBuilderConfig*) internal)->addOptimizationProfile(optimization_profile);
        });
        if index >= 0 {
            Ok(())
        } else {
            Err(crate::error::last_error())
        }
    }

    /// Get internal readonly pointer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let BuilderConfig(internal) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline(always)]
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
