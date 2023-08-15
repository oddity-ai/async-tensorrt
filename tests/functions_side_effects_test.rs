use async_cuda::ffi::device::Device;

use async_tensorrt::ffi::network::NetworkDefinitionCreationFlags;
use async_tensorrt::ffi::parser::Parser;
use async_tensorrt::ffi::sync::builder::Builder;
use async_tensorrt::ffi::sync::engine::ExecutionContext;
use async_tensorrt::ffi::sync::runtime::Runtime;

/// This integration test helps determine which ffi functions affect the GPU state, or local thread
/// state.
///
/// This information is important to determine which function need to be executed on the runtime
/// thread, and which functions can be executed directly by the caller (and don't need to be async).
///
/// We only test functions where it is not immediately apparent whether or not the function has
/// side-effects.
///
/// # Find GPU side-effects
///
/// Run this integration test under the Nsight profile with the following command:
///
/// ```bash
/// nsys profile --output /tmp/side_effects_trace --force-overwrite true cargo test --release --test functions_side_effects_test
/// ```
///
/// Use the `nsys-ui` utility to inspect the report produced in `/tmp/side_effects_trace.qdstrm` and
/// determine for each function call if one or more CUDA API functions were invoked, and if the GPU
/// was affected in any way. Function calls are separated by device synchronization markers in the
/// trace.
///
/// # Find thread-local side-effects
///
/// These need to inferred from documentation or usage (or an educated guess).
///
/// # Results
///
/// | Function                                     | Side-effect: GPU | Side-effect: thread-local | Notes
/// | -------------------------------------------- | ---------------- | ------------------------- | ---------------
/// | `Builder::new`                               | ✅               | ❓                        |
/// | `Builder::add_optimization_profile`          | ❌               | ❌                        |
/// | `Builder::with_optimization_profile`         | ❌               | ❌                        |
/// | `Builder::config`                            | ✅               | ❓                        | Calls `cudaGetDeviceProperties_v2` internally.
/// | `Builder::network_definition`                | ❌               | ❌                        |
/// | `BuilderConfig::*`                           | ❌               | ❌                        | Since no device allocation happens in `Builder::config` we can assume there are no GPU effects.
/// | `NetworkDefinition::*`                       | ❌               | ❌                        | Since no device allocation happens in `Builder::network_definition` we can assume there are no GPU effects.
/// | `Tensor::*`                                  | ❌               | ❌                        | Since no device allocation happens in `Builder::network_definition` we can assume there are no GPU effects.
/// | `HostBuffer::*`                              | ❌               | ❌                        | Assuming based on its name.
/// | `Parser::parse_network_definition_from_file` | ❌               | ❌                        |
/// | `Builder::build_serialized_network`          | ✅               | ❓                        |
/// | `Runtime::new`                               | ✅               | ❓                        | Just a `cudaFree` for some reason (expected more).
/// | `Runtime::deserialize_engine_from_plan`      | ✅               | ❓                        |
/// | `Runtime::deserialize_engine`                | ✅               | ❓                        |
/// | `Engine::serialize`                          | ❌               | ❌                        |
/// | `Engine::num_io_tensors`                     | ❌               | ❌                        |
/// | `Engine::io_tensor_name`                     | ❌               | ❌                        |
/// | `Engine::tensor_shape`                       | ❌               | ❌                        |
/// | `Engine::tensor_io_mode`                     | ❌               | ❌                        |
/// | `ExecutionContext::from_engine`              | ✅               | ❓                        | Assumed (uses `createExecutionContext` internally).
/// | `ExecutionContext::from_engine_many`         | ✅               | ❓                        | Assumed (uses `createExecutionContext` internally).
/// | `ExecutionContext::new`                      | ✅               | ❓                        | Assumed (uses `createExecutionContext` internally).
/// | `ExecutionContext::enqueue`                  | ✅               | ❓                        | Assumed (uses `createExecutionContext` internally).
/// | `ExecutionContext::drop`                     | ✅               | ❓                        |
/// | `Engine::drop`                               | ❌               | ❌                        |
/// | `Runtime::drop`                              | ❌               | ❌                        |
/// | `Builder::drop`                              | ✅               | ❓                        |
#[tokio::test]
async fn test_stream_new_side_effects() {
    // First block contains stuff we are not interested in measuring...

    // Load simple dummy ONNX file.
    let onnx_file = {
        use std::io::Write;
        let mut simple_onnx_file = tempfile::NamedTempFile::new().unwrap();
        simple_onnx_file
            .as_file_mut()
            .write_all(SIMPLE_ONNX)
            .unwrap();
        simple_onnx_file
    };

    // A sequence of CUDA calls that is easy to find in the trace.
    Device::synchronize().unwrap();
    let _mem_info_1 = Device::memory_info().unwrap();
    let _mem_info_2 = Device::memory_info().unwrap();
    let _mem_info_3 = Device::memory_info().unwrap();
    let _mem_info_4 = Device::memory_info().unwrap();
    Device::synchronize().unwrap();

    let mut builder = Builder::new();
    Device::synchronize().unwrap();

    builder.add_optimization_profile().unwrap();
    Device::synchronize().unwrap();

    let mut builder = builder.with_optimization_profile().unwrap();
    Device::synchronize().unwrap();

    let builder_config = builder.config();
    Device::synchronize().unwrap();

    let network_definition =
        builder.network_definition(NetworkDefinitionCreationFlags::ExplicitBatchSize);
    Device::synchronize().unwrap();

    let mut network_definition =
        Parser::parse_network_definition_from_file(network_definition, &onnx_file.path()).unwrap();
    Device::synchronize().unwrap();

    let plan = builder
        .build_serialized_network(&mut network_definition, builder_config)
        .unwrap();
    Device::synchronize().unwrap();

    let runtime = Runtime::new();
    Device::synchronize().unwrap();

    let _engine = runtime.deserialize_engine(plan.as_bytes()).unwrap();
    Device::synchronize().unwrap();

    let runtime = Runtime::new();
    Device::synchronize().unwrap();

    let mut engine = runtime.deserialize_engine_from_plan(&plan).unwrap();
    Device::synchronize().unwrap();

    let _engine_serialized = engine.serialize().unwrap();
    Device::synchronize().unwrap();

    let _ = engine.num_io_tensors();
    Device::synchronize().unwrap();

    let first_tensor_name = engine.io_tensor_name(0);
    Device::synchronize().unwrap();

    let _ = engine.tensor_shape(&first_tensor_name);
    Device::synchronize().unwrap();

    let _ = engine.tensor_io_mode(&first_tensor_name);
    Device::synchronize().unwrap();

    let execution_context = ExecutionContext::new(&mut engine);
    Device::synchronize().unwrap();

    drop(execution_context);
    Device::synchronize().unwrap();

    drop(engine);
    Device::synchronize().unwrap();

    let runtime = Runtime::new();
    Device::synchronize().unwrap();

    drop(runtime);
    Device::synchronize().unwrap();

    drop(builder);
    Device::synchronize().unwrap();
}

/// Dummy ONNX file contents.
static SIMPLE_ONNX: &[u8; 155] = &[
    0x08, 0x07, 0x12, 0x0c, 0x6f, 0x6e, 0x6e, 0x78, 0x2d, 0x65, 0x78, 0x61, 0x6d, 0x70, 0x6c, 0x65,
    0x3a, 0x84, 0x01, 0x0a, 0x26, 0x0a, 0x01, 0x58, 0x0a, 0x04, 0x50, 0x61, 0x64, 0x73, 0x12, 0x01,
    0x59, 0x22, 0x03, 0x50, 0x61, 0x64, 0x2a, 0x13, 0x0a, 0x04, 0x6d, 0x6f, 0x64, 0x65, 0x22, 0x08,
    0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0xa0, 0x01, 0x03, 0x12, 0x0a, 0x74, 0x65, 0x73,
    0x74, 0x2d, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x2a, 0x10, 0x08, 0x04, 0x10, 0x07, 0x3a, 0x04, 0x00,
    0x00, 0x01, 0x01, 0x42, 0x04, 0x50, 0x61, 0x64, 0x73, 0x5a, 0x13, 0x0a, 0x01, 0x58, 0x12, 0x0e,
    0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x5a, 0x12,
    0x0a, 0x04, 0x50, 0x61, 0x64, 0x73, 0x12, 0x0a, 0x0a, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0a, 0x02,
    0x08, 0x04, 0x62, 0x13, 0x0a, 0x01, 0x59, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a,
    0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x04, 0x42, 0x02, 0x10, 0x0c,
];
