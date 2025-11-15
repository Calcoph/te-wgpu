use std::iter;

use crate::ray_tracing::{acceleration_structure_limits, AsBuildContext};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    fail, fail_if, gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static UNBUILT_BLAS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(unbuilt_blas);

fn unbuilt_blas(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    // Build the TLAS package with an unbuilt BLAS.
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default())
        .unwrap();

    encoder.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    fail(
        || {
            ctx.queue.submit([encoder.finish().unwrap()]).map_err(|(_,e)|e)
        },
        None,
    );
}

#[gpu_test]
static UNBUILT_BLAS_COMPACTION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(unbuilt_blas_compaction);

fn unbuilt_blas_compaction(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::ALLOW_COMPACTION,
        AccelerationStructureFlags::empty(),
    );

    fail(
        || {
            // Prepare checks the BLAS has been built
            as_ctx.blas.prepare_compaction_async(|_| {})
        },
        None,
    );
}

#[gpu_test]
static BLAS_COMPACTION_WITHOUT_FLAGS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(blas_compaction_without_flags);

fn blas_compaction_without_flags(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default()).unwrap();

    encoder.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    fail(
        || {
            ctx.queue.submit([encoder.finish().unwrap()]).map_err(|(_,e)|e)
        },
        None,
    );
}

#[gpu_test]
static OUT_OF_ORDER_AS_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(out_of_order_as_build);

fn out_of_order_as_build(ctx: TestingContext) {
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    //
    // Encode the TLAS build before the BLAS build, but submit them in the right order.
    //

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        })
        .unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    ctx.queue
        .submit([encoder_blas.finish().unwrap(), encoder_tlas.finish().unwrap()]).unwrap();

    drop(as_ctx);

    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    //
    // Encode the BLAS build before the TLAS build, but submit them in the wrong order.
    //

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 2"),
        })
        .unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 2"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    fail(
        || {
            ctx.queue
                .submit([encoder_tlas.finish().unwrap(), encoder_blas.finish().unwrap()]).map_err(|(_, e)|e)
        },
        None,
    );
}

#[gpu_test]
static OUT_OF_ORDER_AS_BUILD_USE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            )
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(out_of_order_as_build_use);

fn out_of_order_as_build_use(ctx: TestingContext) {
    //
    // Create a clean `AsBuildContext`
    //

    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    //
    // Build in the right order, then rebuild the BLAS so the TLAS is invalid, then use the TLAS.
    //

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        })
        .unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    let mut encoder_blas2 = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 2"),
        }).unwrap();

    encoder_blas2.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    ctx.queue.submit([
        encoder_blas.finish().unwrap(),
        encoder_tlas.finish().unwrap(),
        encoder_blas2.finish().unwrap(),
    ]).unwrap();

    //
    // Create shader to use tlas with
    //

    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"))
        .unwrap();
    let compute_pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("basic_usage"),
            compilation_options: Default::default(),
            cache: None,
        })
        .unwrap();

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0).unwrap(),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
        }],
    }).unwrap();

    //
    // Use TLAS
    //

    let mut encoder_compute = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default())
        .unwrap();
    {
        let mut pass = encoder_compute.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        }).unwrap().unwrap();
        pass.set_pipeline(&compute_pipeline).unwrap();
        pass.set_bind_group(0, Some(&bind_group), &[]).unwrap();
        pass.dispatch_workgroups(1, 1, 1).unwrap()
    }

    fail(
        || {
            ctx.queue.submit(Some(encoder_compute.finish().unwrap())).map_err(|(_,e)|e)
        },
        None,
    );

    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::empty(),
    );

    //
    // Build in the right order, then rebuild the BLAS so the TLAS is invalid, then use the TLAS.
    //

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 3"),
        })
        .unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_blas2 = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 4"),
        })
        .unwrap();

    encoder_blas2.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 2"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    ctx.queue.submit([
        encoder_blas.finish().unwrap(),
        encoder_tlas.finish().unwrap(),
        encoder_blas2.finish().unwrap(),
    ]).unwrap();

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0).unwrap(),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
        }],
    }).unwrap();

    //
    // Use TLAS
    //

    let mut encoder_compute = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default())
        .unwrap();
    {
        let mut pass = encoder_compute.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        }).unwrap().unwrap();
        pass.set_pipeline(&compute_pipeline).unwrap();
        pass.set_bind_group(0, Some(&bind_group), &[]).unwrap();
        pass.dispatch_workgroups(1, 1, 1).unwrap()
    }

    fail(
        || {
            ctx.queue.submit(Some(encoder_compute.finish().unwrap())).map_err(|(_, err)| err)
        },
        None,
    );
}

#[gpu_test]
static EMPTY_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
    )
    .run_sync(empty_build);
fn empty_build(ctx: TestingContext) {
    let mut encoder_safe = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        })
        .unwrap();

    encoder_safe.build_acceleration_structures(iter::empty(), iter::empty()).unwrap();

    ctx.queue.submit([encoder_safe.finish().unwrap()]).unwrap();
}

#[gpu_test]
static BUILD_WITH_TRANSFORM: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(build_with_transform);

fn build_with_transform(ctx: TestingContext) {
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; size_of::<[[f32; 3]; 3]>()],
        usage: BufferUsages::BLAS_INPUT,
    }).unwrap();

    let transform = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&[
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        })
        .unwrap();

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
    };

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::USE_TRANSFORM,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    ).unwrap();

    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("TLAS"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    }).unwrap();
    tlas[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xFF,
    ));

    let mut encoder_build = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BUILD 1"),
        })
        .unwrap();

    encoder_build.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: Some(&transform),
                transform_buffer_offset: Some(0),
            }]),
        }],
        [&tlas],
    ).unwrap();
    ctx.queue.submit([encoder_build.finish().unwrap()]).unwrap();
}

#[gpu_test]
static ONLY_BLAS_VERTEX_RETURN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN,
            )
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(only_blas_vertex_return);

fn only_blas_vertex_return(ctx: TestingContext) {
    // Set up BLAS with TLAS
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN,
        AccelerationStructureFlags::empty(),
    );

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        })
        .unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();

    ctx.queue
        .submit([encoder_blas.finish().unwrap(), encoder_tlas.finish().unwrap()])
        .unwrap();

    // Create a bind-group containing a TLAS with a bind-group layout that requires vertex return,
    // because only the BLAS and not the TLAS has `AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN`
    // this is invalid.
    {
        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::AccelerationStructure {
                        vertex_return: true,
                    },
                    count: None,
                }],
            }).unwrap();
        fail(
            || {
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
                    }],
                })
            },
            None,
        );
        // drop these
    }

    // We then use it with a shader that does not require vertex return which should succeed.
    {
        //
        // Create shader to use tlas with
        //

        let shader = ctx
            .device
            .create_shader_module(include_wgsl!("shader.wgsl"))
            .unwrap();
        let compute_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("basic_usage"),
                compilation_options: Default::default(),
                cache: None,
            })
            .unwrap();

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &compute_pipeline.get_bind_group_layout(0).unwrap(),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(&as_ctx.tlas),
            }],
        }).unwrap();

        //
        // Use TLAS
        //

        let mut encoder_compute = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default())
            .unwrap();
        {
            let mut pass = encoder_compute.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            }).unwrap().unwrap();
            pass.set_pipeline(&compute_pipeline).unwrap();
            pass.set_bind_group(0, Some(&bind_group), &[]).unwrap();
            pass.dispatch_workgroups(1, 1, 1).unwrap()
        }

        ctx.queue.submit(Some(encoder_compute.finish().unwrap())).unwrap();
    }
}

#[gpu_test]
static ONLY_TLAS_VERTEX_RETURN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN,
            )
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(only_tlas_vertex_return);

fn only_tlas_vertex_return(ctx: TestingContext) {
    // Set up BLAS with TLAS
    let as_ctx = AsBuildContext::new(
        &ctx,
        AccelerationStructureFlags::empty(),
        AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN,
    );

    let mut encoder_blas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS 1"),
        }).unwrap();

    encoder_blas.build_acceleration_structures([&as_ctx.blas_build_entry()], []).unwrap();

    let mut encoder_tlas = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS 1"),
        })
        .unwrap();

    encoder_tlas.build_acceleration_structures([], [&as_ctx.tlas]).unwrap();
    fail(|| encoder_tlas.finish(), None);
}

#[gpu_test]
static EXTRA_FORMAT_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS,
            )
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    .run_sync(|ctx| test_as_build_format_stride(ctx, VertexFormat::Snorm16x4, 6, false));

#[gpu_test]
static MISALIGNED_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    // Larger than the minimum size, but not aligned as required
    .run_sync(|ctx| test_as_build_format_stride(ctx, VertexFormat::Float32x3, 13, true));

#[gpu_test]
static TOO_SMALL_STRIDE_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            // https://github.com/gfx-rs/wgpu/issues/6727
            .skip(FailureCase::backend_adapter(wgpu::Backends::VULKAN, "AMD")),
    )
    // Aligned as required, but smaller than minimum size
    .run_sync(|ctx| test_as_build_format_stride(ctx, VertexFormat::Float32x3, 8, true));

fn test_as_build_format_stride(
    ctx: TestingContext,
    format: VertexFormat,
    stride: BufferAddress,
    invalid_combination: bool,
) {
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &vec![0; (format.min_acceleration_structure_vertex_stride() * 3) as usize],
        usage: BufferUsages::BLAS_INPUT,
    }).unwrap();

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        // The fourth component is ignored, and it allows us to have a smaller stride.
        vertex_format: format,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::empty(),
    };

    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    ).unwrap();

    let mut command_encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BLAS_1"),
        }).unwrap();
    command_encoder.build_acceleration_structures(
        &[BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: stride,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        &[],
    ).unwrap();
    let command_buffer = fail_if(
        invalid_combination,
        || command_encoder.finish(),
        None,
    ).unwrap();
    if !invalid_combination {
        ctx.queue.submit([command_buffer]).unwrap();
    }
}
