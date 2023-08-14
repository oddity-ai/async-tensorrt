/// Convenience macro for testing to take a memory slice and put it on the device and return the
/// [`async_cuda_core::DeviceBuffer`] that refers to it.
macro_rules! to_device {
    ($slice:expr, $stream:expr) => {{
        let host_buffer = async_cuda_core::HostBuffer::from_slice($slice).await;
        let mut device_buffer = async_cuda_core::DeviceBuffer::new($slice.len(), $stream).await;
        device_buffer
            .copy_from(&host_buffer, $stream)
            .await
            .unwrap();
        device_buffer
    }};
}

/// Convenience macro for testing to take a [`async_cuda_core::DeviceBuffer`] and copy it back to
/// the host, then return a [`Vec`] of that memory.
macro_rules! to_host {
    ($device_buffer:expr, $stream:expr) => {{
        let mut host_buffer = async_cuda_core::HostBuffer::new($device_buffer.num_elements()).await;
        $device_buffer
            .copy_to(&mut host_buffer, $stream)
            .await
            .unwrap();
        host_buffer.to_vec()
    }};
}

pub(crate) use to_device;
pub(crate) use to_host;
