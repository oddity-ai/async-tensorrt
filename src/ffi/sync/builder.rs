use cpp::cpp;

use crate::ffi::builder_config::BuilderConfig;
use crate::ffi::memory::HostBuffer;
use crate::ffi::network::{NetworkDefinition, NetworkDefinitionCreationFlags};
use crate::ffi::result;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::Builder`].
///
/// Refer to [`crate::Builder`] for documentation.
pub struct Builder(*mut std::ffi::c_void);

/// Implements [`Send`] for [`Builder`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Builder`].
unsafe impl Send for Builder {}

/// Implements [`Sync`] for [`Builder`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Builder`].
unsafe impl Sync for Builder {}

impl Builder {
    pub fn new() -> Self {
        let internal = cpp!(unsafe [] -> *mut std::ffi::c_void as "void*" {
            return createInferBuilder(GLOBAL_LOGGER);
        });
        Builder(internal)
    }

    pub fn add_optimization_profile(&mut self) -> Result<()> {
        let internal = self.as_mut_ptr();
        let optimization_profile_internal = cpp!(unsafe [
            internal as "void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((IBuilder*) internal)->createOptimizationProfile();
        });
        result!(optimization_profile_internal)
    }

    pub fn with_optimization_profile(mut self) -> Result<Self> {
        self.add_optimization_profile()?;
        Ok(self)
    }

    pub fn config(&mut self) -> BuilderConfig {
        let internal = self.as_mut_ptr();
        let internal = cpp!(unsafe [
            internal as "void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((IBuilder*) internal)->createBuilderConfig();
        });
        BuilderConfig::wrap(internal)
    }

    pub fn build_serialized_network(
        &mut self,
        network_definition: &mut NetworkDefinition,
        config: BuilderConfig,
    ) -> Result<HostBuffer> {
        let internal = self.as_mut_ptr();
        let internal_network_definition = network_definition.as_ptr();
        let internal_builder_config = config.as_ptr();
        let plan_internal = cpp!(unsafe [
            internal as "void*",
            internal_network_definition as "void*",
            internal_builder_config as "void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((IBuilder*) internal)->buildSerializedNetwork(
                *((INetworkDefinition*) internal_network_definition),
                *((IBuilderConfig*) internal_builder_config)
            );
        });
        result!(plan_internal, HostBuffer::wrap(plan_internal))
    }

    pub fn network_definition(
        &mut self,
        flags: NetworkDefinitionCreationFlags,
    ) -> NetworkDefinition {
        let internal = self.as_mut_ptr();
        let set_explicit_batch_size = match flags {
            NetworkDefinitionCreationFlags::None => false,
            NetworkDefinitionCreationFlags::ExplicitBatchSize => true,
        };
        let internal = cpp!(unsafe [
            internal as "void*",
            set_explicit_batch_size as "bool"
        ] -> *mut std::ffi::c_void as "void*" {
            std::uint32_t flags = 0;
            if (set_explicit_batch_size) {
                flags |= (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
            }
            return ((IBuilder*) internal)->createNetworkV2(flags);
        });
        NetworkDefinition::wrap(internal)
    }

    pub fn platform_has_fast_int8(&self) -> bool {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> bool as "bool" {
            return ((const IBuilder*) internal)->platformHasFastInt8();
        })
    }

    pub fn platform_has_fast_fp16(&self) -> bool {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> bool as "bool" {
            return ((const IBuilder*) internal)->platformHasFastFp16();
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let Builder(internal) = *self;
        internal
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let Builder(internal) = *self;
        internal
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IBuilder*) internal);
        });
    }
}

impl Default for Builder {
    fn default() -> Self {
        Builder::new()
    }
}
