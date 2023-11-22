use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static DROP_FAILED_TIMESTAMP_QUERY_SET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // https://github.com/gfx-rs/wgpu/issues/4139
            .expect_fail(FailureCase::always()),
    )
    .run_sync(|ctx| {
        // Creating this query set should fail, since we didn't include
        // TIMESTAMP_QUERY in our required features.
        let bad_query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("doomed query set"),
            ty: wgpu::QueryType::Timestamp,
            count: 1,
        });

        // Dropping this should not panic.
        assert!(bad_query_set.is_err());
    });
