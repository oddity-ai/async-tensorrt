use cpp::cpp;

use async_cuda::device::DeviceId;
use async_cuda::ffi::device::Device;

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
    runtime: Runtime,
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
        Engine { internal, runtime }
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
        ] -> Dims as "Dims64" {
            #if NV_TENSORRT_MAJOR >= 10
            return ((const ICudaEngine*) internal)->getTensorShape(tensor_name_ptr);
            #else
            Dims32 dims32 = ((const ICudaEngine*) internal)->getTensorShape(tensor_name_ptr);
            Dims64 dims64;
            dims64.nbDims = dims32.nbDims;
            for (int i = 0; i < dims32.nbDims; i++) {
                dims64.d[i] = dims32.d[i];
            }
            return dims64;
            #endif
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

    pub fn tensor_data_type(&self, tensor_name: &str) -> TensorDataType {
        let internal = self.as_ptr();
        let tensor_name_cstr = std::ffi::CString::new(tensor_name).unwrap();
        let tensor_name_ptr = tensor_name_cstr.as_ptr();
        let tensor_data_type = cpp!(unsafe [
            internal as "const void*",
            tensor_name_ptr as "const char*"
        ] -> i32 as "std::int32_t" {
            return (std::int32_t) ((const ICudaEngine*) internal)->getTensorDataType(tensor_name_ptr);
        });
        TensorDataType::from_i32(tensor_data_type)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let Engine { internal, .. } = *self;
        internal
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let Engine { internal, .. } = *self;
        internal
    }

    #[inline(always)]
    pub fn device(&self) -> DeviceId {
        self.runtime.device()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        Device::set_or_panic(self.runtime.device());
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
    device: DeviceId,
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
                device: engine.device(),
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
        let device = engine.device();
        let parent = std::sync::Arc::new(engine);
        internals
            .into_iter()
            .map(|internal| {
                result!(
                    internal,
                    Self {
                        internal,
                        device,
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
                device: engine.device(),
                _parent: None,
                _phantom: Default::default(),
            }
        )
    }

    pub fn enqueue<T: Copy>(
        &mut self,
        io_tensors: &mut std::collections::HashMap<
            &str,
            &mut async_cuda::ffi::memory::DeviceBuffer<T>,
        >,
        stream: &async_cuda::ffi::stream::Stream,
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

    #[inline(always)]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let ExecutionContext { internal, .. } = *self;
        internal
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let ExecutionContext { internal, .. } = *self;
        internal
    }

    #[inline(always)]
    pub fn device(&self) -> DeviceId {
        self.device
    }

    unsafe fn new_internal(engine: &mut Engine) -> *mut std::ffi::c_void {
        Device::set_or_panic(engine.device());
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
        buffer: &mut async_cuda::ffi::memory::DeviceBuffer<T>,
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
        Device::set_or_panic(self.device);
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

/// Internal representation of the `Dims64` struct in TensorRT.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
struct Dims {
    pub nbDims: i32,
    pub d: [i64; 8usize],
}

/// Tensor DataType.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TensorDataType {
    FLOAT,
    HALF,
    INT8,
    INT32,
    BOOL,
    UINT8,
    FP8,
    BF16,
    INT64,
    INT4,
}

impl TensorDataType {
    /// Create [`TensorDataType`] from `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - Integer representation of IO mode.
    fn from_i32(value: i32) -> Self {
        match value {
            0 => TensorDataType::FLOAT,
            1 => TensorDataType::HALF,
            2 => TensorDataType::INT8,
            3 => TensorDataType::INT32,
            4 => TensorDataType::BOOL,
            5 => TensorDataType::UINT8,
            6 => TensorDataType::FP8,
            7 => TensorDataType::BF16,
            8 => TensorDataType::INT64,
            9 => TensorDataType::INT4,
            _ => panic!("Unknown data type {}", value),
        }
    }
}
