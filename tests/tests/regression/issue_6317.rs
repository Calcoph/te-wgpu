use wgpu::{DownlevelFlags, Limits};
use wgpu_macros::gpu_test;
use wgpu_test::{fail, GpuTestConfiguration, TestParameters};

#[gpu_test]
static NON_FATAL_ERRORS_IN_QUEUE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        let shader_with_trivial_bind_group = concat!(
            "@group(0) @binding(0) var<storage, read_write> stuff: u32;\n",
            "\n",
            "@compute @workgroup_size(1) fn main() { stuff = 2u; }\n"
        );

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_with_trivial_bind_group.into()),
            }).unwrap();

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: Default::default(),
                }).unwrap();

        fail(
            || -> Result<(), String> {
                let mut command_encoder = ctx.device.create_command_encoder(&Default::default()).unwrap();
                {
                    let mut render_pass = command_encoder.begin_compute_pass(&Default::default()).unwrap();
                    render_pass.set_pipeline(&compute_pipeline).map_err(|err| err.to_string())?;

                    // NOTE: We deliberately don't set a bind group here, to provoke a validation
                    // error.

                    render_pass.dispatch_workgroups(1, 1, 1).map_err(|err| err.to_string())?;
                }

                ctx.queue.submit([command_encoder.finish().map_err(|err| err.to_string())?]).unwrap();
                Ok(())
            },
            Some(concat!(
                "The current set ComputePipeline with '' label ",
                "expects a BindGroup to be set at index 0"
            )),
        );
    });
