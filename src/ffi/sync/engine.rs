use cpp::cpp;

use crate::error::last_error;
use crate::ffi::memory::HostBuffer;
use crate::ffi::result;
use crate::ffi::sync::runtime::Runtime;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::Engine`].
///
/// Refer to [`crate::Engine`] for documentation.
pub struct Engine {
    internal: *mut std::ffi::c_void,
    _runtime: Runtime,
}

/// Implements [`Send`] for [`Engine`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Engine`].
unsafe impl Send for Engine {}

/// Implements [`Sync`] for [`Engine`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`Engine`].
unsafe impl Sync for Engine {}

impl Engine {
    #[inline]
    pub(crate) fn wrap(internal: *mut std::ffi::c_void, runtime: Runtime) -> Self {
        Engine {
            internal,
            _runtime: runtime,
        }
    }

    pub fn serialize(&self) -> Result<HostBuffer> {
        let internal = self.as_ptr();
        let internal_buffer = cpp!(unsafe [
            internal as "const void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return (void*) ((const ICudaEngine*) internal)->serialize();
        });
        result!(internal_buffer, HostBuffer::wrap(internal_buffer))
    }

    pub fn num_io_tensors(&self) -> usize {
        let internal = self.as_ptr();
        let num_io_tensors = cpp!(unsafe [
            internal as "const void*"
        ] -> std::os::raw::c_int as "int" {
            return ((const ICudaEngine*) internal)->getNbIOTensors();
        });
        num_io_tensors as usize
    }

    pub fn io_tensor_name(&self, io_tensor_index: usize) -> String {
        let internal = self.as_ptr();
        let io_tensor_index = io_tensor_index as std::os::raw::c_int;
        let io_tensor_name_ptr = cpp!(unsafe [
            internal as "const void*",
            io_tensor_index as "int"
        ] -> *const std::os::raw::c_char as "const char*" {
            return ((const ICudaEngine*) internal)->getIOTensorName(io_tensor_index);
        });

        // SAFETY: This is safe because:
        // * The pointer is valid because we just got it from TensorRT.
        // * The pointer isn't kept after this block (we copy the string instead).
        unsafe {
            std::ffi::CStr::from_ptr(io_tensor_name_ptr)
                .to_string_lossy()
                .to_string()
        }
    }

    pub fn tensor_shape(&self, tensor_name: &str) -> Vec<usize> {
        let internal = self.as_ptr();
        let tensor_name_cstr = std::ffi::CString::new(tensor_name).unwrap();
        let tensor_name_ptr = tensor_name_cstr.as_ptr();
        let tensor_dimensions = cpp!(unsafe [
            internal as "const void*",
            tensor_name_ptr as "const char*"
        ] -> Dims as "Dims32" {
            return ((const ICudaEngine*) internal)->getTensorShape(tensor_name_ptr);
        });

        let mut dimensions = Vec::with_capacity(tensor_dimensions.nbDims as usize);
        for i in 0..tensor_dimensions.nbDims {
            dimensions.push(tensor_dimensions.d[i as usize] as usize);
        }

        dimensions
    }

    pub fn tensor_io_mode(&self, tensor_name: &str) -> TensorIoMode {
        let internal = self.as_ptr();
        let tensor_name_cstr = std::ffi::CString::new(tensor_name).unwrap();
        let tensor_name_ptr = tensor_name_cstr.as_ptr();
        let tensor_io_mode = cpp!(unsafe [
            internal as "const void*",
            tensor_name_ptr as "const char*"
        ] -> i32 as "std::int32_t" {
            return (std::int32_t) ((const ICudaEngine*) internal)->getTensorIOMode(tensor_name_ptr);
        });
        TensorIoMode::from_i32(tensor_io_mode)
    }

    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let Engine { internal, .. } = *self;
        internal
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let Engine { internal, .. } = *self;
        internal
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let Engine { internal, .. } = *self;
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((ICudaEngine*) internal);
        });
    }
}

/// Synchronous implementation of [`crate::ExecutionContext`].
///
/// Refer to [`crate::ExecutionContext`] for documentation.
pub struct ExecutionContext<'engine> {
    internal: *mut std::ffi::c_void,
    _parent: Option<std::sync::Arc<Engine>>,
    _phantom: std::marker::PhantomData<&'engine ()>,
}

/// Implements [`Send`] for `ExecutionContext`.
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`ExecutionContext`].
unsafe impl<'engine> Send for ExecutionContext<'engine> {}

/// Implements [`Sync`] for `ExecutionContext`.
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`ExecutionContext`].
unsafe impl<'engine> Sync for ExecutionContext<'engine> {}

impl ExecutionContext<'static> {
    pub fn from_engine(mut engine: Engine) -> Result<Self> {
        let internal = unsafe { Self::new_internal(&mut engine) };
        result!(
            internal,
            Self {
                internal,
                _parent: Some(std::sync::Arc::new(engine)),
                _phantom: Default::default(),
            }
        )
    }

    pub fn from_engine_many(mut engine: Engine, num: usize) -> Result<Vec<Self>> {
        let mut internals = Vec::with_capacity(num);
        for _ in 0..num {
            internals.push(unsafe { Self::new_internal(&mut engine) });
        }
        let parent = std::sync::Arc::new(engine);
        internals
            .into_iter()
            .map(|internal| {
                result!(
                    internal,
                    Self {
                        internal,
                        _parent: Some(parent.clone()),
                        _phantom: Default::default(),
                    }
                )
            })
            .collect()
    }
}

impl<'engine> ExecutionContext<'engine> {
    pub fn new(engine: &'engine mut Engine) -> Result<Self> {
        let internal = unsafe { Self::new_internal(engine) };
        result!(
            internal,
            Self {
                internal,
                _parent: None,
                _phantom: Default::default(),
            }
        )
    }

    pub fn enqueue<T: Copy>(
        &mut self,
        io_tensors: &mut std::collections::HashMap<
            &str,
            &mut async_cuda_core::ffi::memory::DeviceBuffer<T>,
        >,
        stream: &async_cuda_core::ffi::stream::Stream,
    ) -> Result<()> {
        let internal = self.as_mut_ptr();
        for (tensor_name, buffer) in io_tensors {
            unsafe {
                self.set_tensor_address(tensor_name, buffer)?;
            }
        }
        let stream_ptr = stream.as_internal().as_ptr();
        let success = cpp!(unsafe [
            internal as "void*",
            stream_ptr as "const void*"
        ] -> bool as "bool" {
            return ((IExecutionContext*) internal)->enqueueV3((cudaStream_t) stream_ptr);
        });
        if success {
            Ok(())
        } else {
            Err(last_error())
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let ExecutionContext { internal, .. } = *self;
        internal
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let ExecutionContext { internal, .. } = *self;
        internal
    }

    unsafe fn new_internal(engine: &mut Engine) -> *mut std::ffi::c_void {
        let internal_engine = engine.as_mut_ptr();
        let internal = cpp!(unsafe [
            internal_engine as "void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return (void*) ((ICudaEngine*) internal_engine)->createExecutionContext();
        });
        internal
    }

    unsafe fn set_tensor_address<T: Copy>(
        &mut self,
        tensor_name: &str,
        buffer: &mut async_cuda_core::ffi::memory::DeviceBuffer<T>,
    ) -> Result<()> {
        let internal = self.as_mut_ptr();
        let tensor_name_cstr = std::ffi::CString::new(tensor_name).unwrap();
        let tensor_name_ptr = tensor_name_cstr.as_ptr();
        let buffer_ptr = buffer.as_mut_internal().as_mut_ptr();
        let success = cpp!(unsafe [
            internal as "const void*",
            tensor_name_ptr as "const char*",
            buffer_ptr as "void*"
        ] -> bool as "bool" {
            return ((IExecutionContext*) internal)->setTensorAddress(
                tensor_name_ptr,
                buffer_ptr
            );
        });
        if success {
            Ok(())
        } else {
            Err(last_error())
        }
    }
}

impl<'engine> Drop for ExecutionContext<'engine> {
    fn drop(&mut self) {
        let ExecutionContext { internal, .. } = *self;
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IExecutionContext*) internal);
        });
    }
}

/// Tensor IO mode.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TensorIoMode {
    None,
    Input,
    Output,
}

impl TensorIoMode {
    /// Create [`IoTensorMode`] from `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer representation of IO mode.
    fn from_i32(value: i32) -> Self {
        match value {
            1 => TensorIoMode::Input,
            2 => TensorIoMode::Output,
            _ => TensorIoMode::None,
        }
    }
}

/// Internal representation of the `Dims32` struct in TensorRT.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
struct Dims {
    pub nbDims: i32,
    pub d: [i32; 8usize],
}
