//! Tests for texture copy bounds checks.

use wgpu_test::{fail_if, gpu_test, GpuTestConfiguration};

#[gpu_test]
static BAD_COPY_ORIGIN_TEST: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    let try_origin = |origin, size, should_panic| {
        let texture = ctx.device.create_texture(&TEXTURE_DESCRIPTOR).unwrap();
        let data = vec![255; BUFFER_SIZE as usize];

        let _ = fail_if(
            should_panic,
            || {
                ctx.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    BUFFER_COPY_LAYOUT,
                    size,
                )
            },
            None,
        );
    };

    try_origin(wgpu::Origin3d { x: 0, y: 0, z: 0 }, TEXTURE_SIZE, false);
    try_origin(wgpu::Origin3d { x: 1, y: 0, z: 0 }, TEXTURE_SIZE, true);
    try_origin(wgpu::Origin3d { x: 0, y: 1, z: 0 }, TEXTURE_SIZE, true);
    try_origin(wgpu::Origin3d { x: 0, y: 0, z: 1 }, TEXTURE_SIZE, true);

    try_origin(
        wgpu::Origin3d {
            x: TEXTURE_SIZE.width - 1,
            y: TEXTURE_SIZE.height - 1,
            z: TEXTURE_SIZE.depth_or_array_layers - 1,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        false,
    );
    try_origin(
        wgpu::Origin3d {
            x: u32::MAX,
            y: 0,
            z: 0,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        true,
    );
    try_origin(
        wgpu::Origin3d {
            x: u32::MAX,
            y: 0,
            z: 0,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        true,
    );
    try_origin(
        wgpu::Origin3d {
            x: u32::MAX,
            y: 0,
            z: 0,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        true,
    );
});

const TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 64,
    height: 64,
    depth_or_array_layers: 1,
};

const TEXTURE_DESCRIPTOR: wgpu::TextureDescriptor = wgpu::TextureDescriptor {
    label: Some("CopyOrigin"),
    size: TEXTURE_SIZE,
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: wgpu::TextureFormat::Rgba8UnormSrgb,
    usage: wgpu::TextureUsages::COPY_DST.union(wgpu::TextureUsages::COPY_SRC),
    view_formats: &[],
};

const BYTES_PER_PIXEL: u32 = 4;

const BUFFER_SIZE: u32 = TEXTURE_SIZE.width * TEXTURE_SIZE.height * BYTES_PER_PIXEL;

const BUFFER_COPY_LAYOUT: wgpu::TexelCopyBufferLayout = wgpu::TexelCopyBufferLayout {
    offset: 0,
    bytes_per_row: Some(TEXTURE_SIZE.width * BYTES_PER_PIXEL),
    rows_per_image: None,
};
