use std::fmt::Display;

use wgc::{
    device::queue::QueueWriteError,
    resource::{CreateBufferError, CreateTextureError},
};

/// Describes a [Buffer](crate::Buffer) when allocating.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferInitDescriptor<'a> {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: crate::Label<'a>,
    /// Contents of a buffer on creation.
    pub contents: &'a [u8],
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    pub usage: crate::BufferUsages,
}

#[derive(Debug)]
pub enum CreateTextureWithDataError {
    CTError(CreateTextureError),
    QWError(QueueWriteError),
}

impl From<CreateTextureError> for CreateTextureWithDataError {
    fn from(value: CreateTextureError) -> Self {
        CreateTextureWithDataError::CTError(value)
    }
}

impl From<QueueWriteError> for CreateTextureWithDataError {
    fn from(value: QueueWriteError) -> Self {
        CreateTextureWithDataError::QWError(value)
    }
}

impl Display for CreateTextureWithDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateTextureWithDataError::CTError(e) => write!(f, "CreateTextureWithDataError {e}"),
            CreateTextureWithDataError::QWError(e) => write!(f, "CreateTextureWithDataError {e}"),
        }
    }
}

impl std::error::Error for CreateTextureWithDataError {}

/// Utility methods not meant to be in the main API.
pub trait DeviceExt {
    /// Creates a [Buffer](crate::Buffer) with data to initialize it.
    fn create_buffer_init(
        &self,
        desc: &BufferInitDescriptor,
    ) -> Result<crate::Buffer, CreateBufferError>;

    /// Upload an entire texture and its mipmaps from a source buffer.
    ///
    /// Expects all mipmaps to be tightly packed in the data buffer.
    ///
    /// If the texture is a 2DArray texture, uploads each layer in order, expecting
    /// each layer and its mips to be tightly packed.
    ///
    /// Example:
    /// Layer0Mip0 Layer0Mip1 Layer0Mip2 ... Layer1Mip0 Layer1Mip1 Layer1Mip2 ...
    ///
    /// Implicitly adds the `COPY_DST` usage if it is not present in the descriptor,
    /// as it is required to be able to upload the data to the gpu.
    fn create_texture_with_data(
        &self,
        queue: &crate::Queue,
        desc: &crate::TextureDescriptor,
        data: &[u8],
    ) -> Result<crate::Texture, CreateTextureWithDataError>;
}

impl DeviceExt for crate::Device {
    fn create_buffer_init(
        &self,
        descriptor: &BufferInitDescriptor<'_>,
    ) -> Result<crate::Buffer, CreateBufferError> {
        // Skip mapping if the buffer is zero sized
        if descriptor.contents.is_empty() {
            let wgt_descriptor = crate::BufferDescriptor {
                label: descriptor.label,
                size: 0,
                usage: descriptor.usage,
                mapped_at_creation: false,
            };

            self.create_buffer(&wgt_descriptor)
        } else {
            let unpadded_size = descriptor.contents.len() as crate::BufferAddress;
            // Valid vulkan usage is
            // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
            // 2. buffer size must be greater than 0.
            // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
            let align_mask = crate::COPY_BUFFER_ALIGNMENT - 1;
            let padded_size =
                ((unpadded_size + align_mask) & !align_mask).max(crate::COPY_BUFFER_ALIGNMENT);

            let wgt_descriptor = crate::BufferDescriptor {
                label: descriptor.label,
                size: padded_size,
                usage: descriptor.usage,
                mapped_at_creation: true,
            };

            let buffer = self.create_buffer(&wgt_descriptor)?;

            buffer.slice(..).get_mapped_range_mut()[..unpadded_size as usize]
                .copy_from_slice(descriptor.contents);
            buffer.unmap()?;

            Ok(buffer)
        }
    }

    fn create_texture_with_data(
        &self,
        queue: &crate::Queue,
        desc: &crate::TextureDescriptor,
        data: &[u8],
    ) -> Result<crate::Texture, CreateTextureWithDataError> {
        // Implicitly add the COPY_DST usage
        let mut desc = desc.to_owned();
        desc.usage |= crate::TextureUsages::COPY_DST;
        let texture = self.create_texture(&desc)?;

        // Will return None only if it's a combined depth-stencil format
        // If so, default to 4, validation will fail later anyway since the depth or stencil
        // aspect needs to be written to individually
        let block_size = desc.format.block_copy_size(None).unwrap_or(4);
        let (block_width, block_height) = desc.format.block_dimensions();
        let layer_iterations = desc.array_layer_count();

        let mut binary_offset = 0;
        for layer in 0..layer_iterations {
            for mip in 0..desc.mip_level_count {
                let mut mip_size = desc.mip_level_size(mip).unwrap();
                // copying layers separately
                if desc.dimension != wgt::TextureDimension::D3 {
                    mip_size.depth_or_array_layers = 1;
                }

                // When uploading mips of compressed textures and the mip is supposed to be
                // a size that isn't a multiple of the block size, the mip needs to be uploaded
                // as its "physical size" which is the size rounded up to the nearest block size.
                let mip_physical = mip_size.physical_size(desc.format);

                // All these calculations are performed on the physical size as that's the
                // data that exists in the buffer.
                let width_blocks = mip_physical.width / block_width;
                let height_blocks = mip_physical.height / block_height;

                let bytes_per_row = width_blocks * block_size;
                let data_size = bytes_per_row * height_blocks * mip_size.depth_or_array_layers;

                let end_offset = binary_offset + data_size as usize;

                queue.write_texture(
                    crate::ImageCopyTexture {
                        texture: &texture,
                        mip_level: mip,
                        origin: crate::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                        aspect: wgt::TextureAspect::All,
                    },
                    &data[binary_offset..end_offset],
                    crate::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height_blocks),
                    },
                    mip_physical,
                )?;

                binary_offset = end_offset;
            }
        }

        Ok(texture)
    }
}
