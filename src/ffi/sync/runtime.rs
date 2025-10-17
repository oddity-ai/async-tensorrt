use cpp::cpp;

use async_cuda::device::DeviceId;
use async_cuda::ffi::device::Device;

use crate::ffi::memory::HostBuffer;
use crate::ffi::result;
use crate::ffi::sync::engine::Engine;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::Runtime`].
///
/// Refer to [`crate::Runtime`] for documentation.
pub struct Runtime {
    addr: *mut std::ffi::c_void,
    device: DeviceId,
}

/// Implements [`Send`] for [`Runtime`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Runtime`].
unsafe impl Send for Runtime {}

/// Implements [`Sync`] for [`Runtime`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Runtime`].
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new() -> Self {
        let device = Device::get_or_panic();
        let addr = cpp!(unsafe [] -> *mut std::ffi::c_void as "void*" {
            return createInferRuntime(GLOBAL_LOGGER);
        });
        Runtime { addr, device }
    }

    pub fn deserialize_engine_from_plan(self, plan: &HostBuffer) -> Result<Engine> {
        unsafe {
            // SAFETY: Since we have a reference to the buffer for the duration of this call, we
            // know the internal pointers will be and remain valid until the end of the block.
            self.deserialize_engine_raw(plan.data(), plan.size())
        }
    }

    pub fn deserialize_engine(self, buffer: &[u8]) -> Result<Engine> {
        unsafe {
            // SAFETY: Since we have a reference to the slice for the duration of this call, we
            // know the internal pointers will be and remain valid until the end of the block.
            self.deserialize_engine_raw(buffer.as_ptr() as *const std::ffi::c_void, buffer.len())
        }
    }

    /// Deserialize an engine from a buffer.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_runtime.html#ad0dc765e77cab99bfad901e47216a767)
    ///
    /// # Safety
    ///
    /// Both provided pointers must be valid pointers.
    ///
    /// # Arguments
    ///
    /// * `buffer_ptr` - Pointer to buffer to read from.
    /// * `buffer_size` - Size of buffer to read from.
    unsafe fn deserialize_engine_raw(
        mut self,
        buffer_ptr: *const std::ffi::c_void,
        buffer_size: usize,
    ) -> Result<Engine> {
        Device::set(self.device)?;
        let internal = self.as_mut_ptr();
        let internal_engine = cpp!(unsafe [
            internal as "void*",
            buffer_ptr as "const void*",
            buffer_size as "std::size_t"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((IRuntime*) internal)->deserializeCudaEngine(buffer_ptr, buffer_size);
        });
        result!(internal_engine, Engine::wrap(internal_engine, self))
    }

    pub fn set_engine_host_code_allowed(&mut self, allowed: bool) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*",
            allowed as "bool"
        ] {
            ((IRuntime*) internal)->setEngineHostCodeAllowed(allowed);
        });
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.addr
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.addr
    }

    #[inline(always)]
    pub fn device(&self) -> DeviceId {
        self.device
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        Device::set_or_panic(self.device);
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IRuntime*) internal);
        });
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Runtime::new()
    }
}
