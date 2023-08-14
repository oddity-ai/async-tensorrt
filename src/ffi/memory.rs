use cpp::cpp;

pub struct HostBuffer(*mut std::ffi::c_void);

/// Implements [`Send`] for [`HostBuffer`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`HostBuffer`].
unsafe impl Send for HostBuffer {}

/// Implements [`Sync`] for [`HostBuffer`].
///
/// # Safety
///
/// The TensorRT API is thread-safe with regards to all operations on [`HostBuffer`].
unsafe impl Sync for HostBuffer {}

/// Handle TensorRT-related memory accesible to caller.
///
/// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_host_memory.html)
impl HostBuffer {
    /// Wrap internal pointer as [`HostBuffer`].
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid `IHostMemory` object.
    #[inline]
    pub(crate) fn wrap(internal: *mut std::ffi::c_void) -> Self {
        HostBuffer(internal)
    }

    /// Get data slice pointing to the host buffer.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        let data = self.data() as *const u8;
        let size = self.size();
        // SAFETY: This is safe because:
        // * The pointer is valid because we just got it from TensorRT.
        // * The pointer will remain valid as long as `HostBuffer` remains around.
        unsafe { std::slice::from_raw_parts(data, size) }
    }

    /// Get readonly pointer to host buffer data.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_host_memory.html#a95d49ae9b0a5479af9433cb101a26782)
    #[inline]
    pub fn data(&self) -> *const std::ffi::c_void {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> *mut std::ffi::c_void as "void*" {
            return ((const IHostMemory*) internal)->data();
        })
    }

    /// Get size of host buffer data.
    ///
    /// [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_host_memory.html#adede91569ebccd258b357f29ba706e8e)
    #[inline]
    pub fn size(&self) -> usize {
        let internal = self.as_ptr();
        cpp!(unsafe [
            internal as "const void*"
        ] -> usize as "std::size_t" {
            return ((const IHostMemory*) internal)->size();
        })
    }

    /// Get internal readonly pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        let HostBuffer(internal) = *self;
        internal
    }

    /// Get internal mutable pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        let HostBuffer(internal) = *self;
        internal
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let internal = self.as_mut_ptr();
        cpp!(unsafe [
            internal as "void*"
        ] {
            destroy((IHostMemory*) internal);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::utils::*;

    #[tokio::test]
    async fn test_host_buffer_data_and_size() {
        let network_plan = simple_network_plan!();
        assert!(!network_plan.data().is_null());
        assert!(network_plan.size() > 0);
        let bytes = network_plan.as_bytes();
        assert_eq!(unsafe { *(network_plan.data() as *const u8) }, bytes[0]);
        assert_eq!(network_plan.size(), bytes.len());
    }
}
