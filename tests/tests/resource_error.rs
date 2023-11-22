use wgpu_test::{fail, gpu_test, valid, GpuTestConfiguration};

#[gpu_test]
static BAD_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    // Create a buffer with bad parameters and call a few methods.
    // Validation should fail but there should be not panic.
    fail(|| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 99999999,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
});

#[gpu_test]
static BAD_TEXTURE: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    fail(|| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 0,
                height: 12345678,
                depth_or_array_layers: 9001,
            },
            mip_level_count: 2000,
            sample_count: 27,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::all(),
            view_formats: &[],
        })
    });
});
