macro_rules! simple_network {
    () => {{
        let simple_onnx_file = $crate::tests::onnx::simple_onnx_file!();
        let mut builder = $crate::Builder::new()
            .await
            .with_optimization_profile()
            .unwrap();
        let network =
            builder.network_definition($crate::NetworkDefinitionCreationFlags::ExplicitBatchSize);
        let network =
            $crate::Parser::parse_network_definition_from_file(network, &simple_onnx_file.path())
                .unwrap();
        (builder, network)
    }};
}

macro_rules! simple_network_plan {
    () => {{
        let (mut builder, mut network) = $crate::tests::utils::simple_network!();
        let builder_config = builder.config().await;
        builder
            .build_serialized_network(&mut network, builder_config)
            .await
            .unwrap()
    }};
}

macro_rules! simple_engine {
    () => {{
        let network_plan = $crate::tests::utils::simple_network_plan!();
        let runtime = $crate::Runtime::new().await;
        runtime
            .deserialize_engine_from_plan(&network_plan)
            .await
            .unwrap()
    }};
}

pub(crate) use simple_engine;
pub(crate) use simple_network;
pub(crate) use simple_network_plan;
