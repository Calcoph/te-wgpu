//! Infrastructure for dispatching calls to the appropriate "backend". The "backends" are:
//!
//! - `wgpu_core`: An implementation of the the wgpu api on top of various native graphics APIs.
//! - `webgpu`: An implementation of the wgpu api which calls WebGPU directly.
//!
//! The interface traits are all object safe and listed in the `InterfaceTypes` trait.
//!
//! The method for dispatching should optimize well if only one backend is compiled in,
//! as-if there was no dispatching at all.

#![allow(drop_bounds)] // This exists to remind implementors to impl drop.
#![allow(clippy::too_many_arguments)] // It's fine.

use wgc::{binding_model::{self, CreateBindGroupError, CreateBindGroupLayoutError, CreatePipelineLayoutError}, command::{self, CommandEncoderError, ComputePassError}, device::{queue::{QueueSubmitError, QueueWriteError}, DeviceError}, pipeline::{self, CreateComputePipelineError, CreateRenderPipelineError, CreateShaderModuleError}, present::{self, SurfaceError}, ray_tracing::{CreateBlasError, CreateTlasError}, resource::{self, CreateBufferError, CreateTextureError, CreateTextureViewError}};

use crate::{WasmNotSend, WasmNotSendSync};

use std::{any::Any, fmt::Debug, future::Future, hash::Hash, ops::Range, pin::Pin, sync::Arc};

use crate::backend;

/// Create a single trait with the given supertraits and a blanket impl for all types that implement them.
///
/// This is useful for creating a trait alias as a shorthand.
macro_rules! trait_alias {
    ($name:ident: $($bound:tt)+) => {
        pub trait $name: $($bound)+ {}
        impl<T: $($bound)+> $name for T {}
    };
}

// Various return futures in the API.
trait_alias!(RequestAdapterFuture: Future<Output = Option<DispatchAdapter>> + WasmNotSend + 'static);
trait_alias!(RequestDeviceFuture: Future<Output = Result<(DispatchDevice, DispatchQueue), crate::RequestDeviceError>> + WasmNotSend + 'static);
trait_alias!(ShaderCompilationInfoFuture: Future<Output = crate::CompilationInfo> + WasmNotSend + 'static);

// We can't use trait aliases here, as you can't convert from a dyn Trait to dyn Supertrait _yet_.
#[cfg(send_sync)]
pub type BoxDeviceLostCallback = Box<dyn FnOnce(crate::DeviceLostReason, String) + Send + 'static>;
#[cfg(not(send_sync))]
pub type BoxDeviceLostCallback = Box<dyn FnOnce(crate::DeviceLostReason, String) + 'static>;
#[cfg(send_sync)]
pub type BoxSubmittedWorkDoneCallback = Box<dyn FnOnce() + Send + 'static>;
#[cfg(not(send_sync))]
pub type BoxSubmittedWorkDoneCallback = Box<dyn FnOnce() + 'static>;
#[cfg(send_sync)]
pub type BufferMapCallback = Box<dyn FnOnce(Result<(), crate::BufferAsyncError>) + Send + 'static>;
#[cfg(not(send_sync))]
pub type BufferMapCallback = Box<dyn FnOnce(Result<(), crate::BufferAsyncError>) + 'static>;

// Common traits on all the interface traits
trait_alias!(CommonTraits: Any + Debug + WasmNotSendSync);
// Non-object-safe traits that are added as a bound on InterfaceTypes.
trait_alias!(ComparisonTraits: PartialEq + Eq + PartialOrd + Ord + Hash);

/// Types that represent a "Backend" for the wgpu API.
pub trait InterfaceTypes {
    type Instance: InstanceInterface + ComparisonTraits;
    type Adapter: AdapterInterface + ComparisonTraits;
    type Device: DeviceInterface + ComparisonTraits;
    type Queue: QueueInterface + ComparisonTraits;
    type ShaderModule: ShaderModuleInterface + ComparisonTraits;
    type BindGroupLayout: BindGroupLayoutInterface + ComparisonTraits;
    type BindGroup: BindGroupInterface + ComparisonTraits;
    type TextureView: TextureViewInterface + ComparisonTraits;
    type Sampler: SamplerInterface + ComparisonTraits;
    type Buffer: BufferInterface + ComparisonTraits;
    type Texture: TextureInterface + ComparisonTraits;
    type Blas: BlasInterface + ComparisonTraits;
    type Tlas: TlasInterface + ComparisonTraits;
    type QuerySet: QuerySetInterface + ComparisonTraits;
    type PipelineLayout: PipelineLayoutInterface + ComparisonTraits;
    type RenderPipeline: RenderPipelineInterface + ComparisonTraits;
    type ComputePipeline: ComputePipelineInterface + ComparisonTraits;
    type PipelineCache: PipelineCacheInterface + ComparisonTraits;
    type CommandEncoder: CommandEncoderInterface + ComparisonTraits;
    type ComputePass: ComputePassInterface + ComparisonTraits;
    type RenderPass: RenderPassInterface + ComparisonTraits;
    type CommandBuffer: CommandBufferInterface + ComparisonTraits;
    type RenderBundleEncoder: RenderBundleEncoderInterface + ComparisonTraits;
    type RenderBundle: RenderBundleInterface + ComparisonTraits;
    type Surface: SurfaceInterface + ComparisonTraits;
    type SurfaceOutputDetail: SurfaceOutputDetailInterface + ComparisonTraits;
    type QueueWriteBuffer: QueueWriteBufferInterface + ComparisonTraits;
    type BufferMappedRange: BufferMappedRangeInterface + ComparisonTraits;
}

pub trait InstanceInterface: CommonTraits {
    fn new(desc: &wgt::InstanceDescriptor) -> Self
    where
        Self: Sized;

    unsafe fn create_surface(
        &self,
        target: crate::SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, crate::CreateSurfaceError>;

    fn request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Pin<Box<dyn RequestAdapterFuture>>;

    fn poll_all_devices(&self, force_wait: bool) -> bool;

    #[cfg(feature = "wgsl")]
    fn wgsl_language_features(&self) -> crate::WgslLanguageFeatures;
}

pub trait AdapterInterface: CommonTraits {
    fn request_device(
        &self,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn RequestDeviceFuture>>;

    fn is_surface_supported(&self, surface: &DispatchSurface) -> bool;

    fn features(&self) -> crate::Features;

    fn limits(&self) -> crate::Limits;

    fn downlevel_capabilities(&self) -> crate::DownlevelCapabilities;

    fn get_info(&self) -> crate::AdapterInfo;

    fn get_texture_format_features(
        &self,
        format: crate::TextureFormat,
    ) -> crate::TextureFormatFeatures;

    fn get_presentation_timestamp(&self) -> crate::PresentationTimestamp;
}

pub trait DeviceInterface: CommonTraits {
    fn features(&self) -> crate::Features;
    fn limits(&self) -> crate::Limits;

    fn create_shader_module(
        &self,
        desc: crate::ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderRuntimeChecks,
    ) -> Result<DispatchShaderModule, CreateShaderModuleError>;
    unsafe fn create_shader_module_spirv(
        &self,
        desc: &crate::ShaderModuleDescriptorSpirV<'_>,
    ) -> Result<DispatchShaderModule, CreateShaderModuleError>;
    fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor<'_>,
    ) -> Result<DispatchBindGroupLayout, CreateBindGroupLayoutError>;
    fn create_bind_group(&self, desc: &crate::BindGroupDescriptor<'_>) -> Result<DispatchBindGroup, CreateBindGroupError>;
    fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<'_>,
    ) -> Result<DispatchPipelineLayout, CreatePipelineLayoutError>;
    fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<'_>,
    ) -> Result<DispatchRenderPipeline, CreateRenderPipelineError>;
    fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<'_>,
    ) -> Result<DispatchComputePipeline, CreateComputePipelineError>;
    unsafe fn create_pipeline_cache(
        &self,
        desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> Result<DispatchPipelineCache, pipeline::CreatePipelineCacheError>;
    fn create_buffer(&self, desc: &crate::BufferDescriptor<'_>) -> Result<DispatchBuffer, CreateBufferError>;
    fn create_texture(&self, desc: &crate::TextureDescriptor<'_>) -> Result<DispatchTexture, CreateTextureError> ;
    fn create_blas(
        &self,
        desc: &crate::CreateBlasDescriptor<'_>,
        sizes: crate::BlasGeometrySizeDescriptors,
    ) -> Result<(u64, DispatchBlas), CreateBlasError>;
    fn create_tlas(&self, desc: &crate::CreateTlasDescriptor<'_>) -> Result<DispatchTlas, CreateTlasError>;
    fn create_sampler(&self, desc: &crate::SamplerDescriptor<'_>) -> Result<DispatchSampler, resource::CreateSamplerError>;
    fn create_query_set(&self, desc: &crate::QuerySetDescriptor<'_>) -> Result<DispatchQuerySet, resource::CreateQuerySetError> ;
    fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<'_>,
    ) -> Result<DispatchCommandEncoder, DeviceError>;
    fn create_render_bundle_encoder(
        &self,
        desc: &crate::RenderBundleEncoderDescriptor<'_>,
    ) -> DispatchRenderBundleEncoder;

    fn set_device_lost_callback(&self, device_lost_callback: BoxDeviceLostCallback);

    fn start_capture(&self);
    fn stop_capture(&self);

    fn poll(&self, maintain: crate::Maintain) -> crate::MaintainResult;

    fn get_internal_counters(&self) -> crate::InternalCounters;
    fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport>;

    fn destroy(&self);
}

pub trait QueueInterface: CommonTraits {
    fn write_buffer(&self, buffer: &DispatchBuffer, offset: crate::BufferAddress, data: &[u8]) -> Result<(), wgc::device::queue::QueueWriteError>;

    fn create_staging_buffer(&self, size: crate::BufferSize) -> Result<DispatchQueueWriteBuffer, QueueWriteError>;
    fn validate_write_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Result<(), wgc::device::queue::QueueWriteError>;
    fn write_staging_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: crate::BufferAddress,
        staging_buffer: &DispatchQueueWriteBuffer,
    ) -> Result<(), wgc::device::queue::QueueWriteError>;

    fn write_texture(
        &self,
        texture: crate::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: crate::TexelCopyBufferLayout,
        size: crate::Extent3d,
    ) -> Result<(), wgc::device::queue::QueueWriteError>;
    #[cfg(any(webgpu, webgl))]
    fn copy_external_image_to_texture(
        &self,
        source: &wgt::CopyExternalImageSourceInfo,
        dest: wgt::CopyExternalImageDestInfo<&crate::api::Texture>,
        size: crate::Extent3d,
    );

    fn submit(&self, command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>) -> Result<u64, (u64, QueueSubmitError)>;

    fn get_timestamp_period(&self) -> f32;
    fn on_submitted_work_done(&self, callback: BoxSubmittedWorkDoneCallback);
}

pub trait ShaderModuleInterface: CommonTraits {
    fn get_compilation_info(&self) -> Pin<Box<dyn ShaderCompilationInfoFuture>>;
}
pub trait BindGroupLayoutInterface: CommonTraits {}
pub trait BindGroupInterface: CommonTraits {}
pub trait TextureViewInterface: CommonTraits {}
pub trait SamplerInterface: CommonTraits {}
pub trait BufferInterface: CommonTraits {
    fn map_async(
        &self,
        mode: crate::MapMode,
        range: Range<crate::BufferAddress>,
        callback: BufferMapCallback,
    ) -> Result<(), resource::BufferAccessError>;
    fn get_mapped_range(&self, sub_range: Range<crate::BufferAddress>)
        -> DispatchBufferMappedRange;
    #[cfg(webgpu)]
    fn get_mapped_range_as_array_buffer(
        &self,
        sub_range: Range<wgt::BufferAddress>,
    ) -> Option<js_sys::ArrayBuffer>;

    fn unmap(&self) -> Result<(), wgc::resource::BufferAccessError>;

    fn destroy(&self);
}
pub trait TextureInterface: CommonTraits {
    fn create_view(&self, desc: &crate::TextureViewDescriptor<'_>) -> Result<DispatchTextureView, CreateTextureViewError>;

    fn destroy(&self);
}
pub trait BlasInterface: CommonTraits {}
pub trait TlasInterface: CommonTraits {}
pub trait QuerySetInterface: CommonTraits {}
pub trait PipelineLayoutInterface: CommonTraits {}
pub trait RenderPipelineInterface: CommonTraits {
    fn get_bind_group_layout(&self, index: u32) -> Result<DispatchBindGroupLayout, binding_model::GetBindGroupLayoutError>;
}
pub trait ComputePipelineInterface: CommonTraits {
    fn get_bind_group_layout(&self, index: u32) -> Result<DispatchBindGroupLayout, binding_model::GetBindGroupLayoutError>;
}
pub trait PipelineCacheInterface: CommonTraits {
    fn get_data(&self) -> Option<Vec<u8>>;
}
pub trait CommandEncoderInterface: CommonTraits {
    fn copy_buffer_to_buffer(
        &self,
        source: &DispatchBuffer,
        source_offset: crate::BufferAddress,
        destination: &DispatchBuffer,
        destination_offset: crate::BufferAddress,
        copy_size: crate::BufferAddress,
    ) -> Result<(), wgc::command::CopyError>;
    fn copy_buffer_to_texture(
        &self,
        source: crate::TexelCopyBufferInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError>;
    fn copy_texture_to_buffer(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyBufferInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError>;
    fn copy_texture_to_texture(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError>;

    fn begin_compute_pass(&self, desc: &crate::ComputePassDescriptor<'_>) -> Result<DispatchComputePass, CommandEncoderError>;
    fn begin_render_pass(&self, desc: &crate::RenderPassDescriptor<'_>) -> Result<DispatchRenderPass, CommandEncoderError>;
    fn finish(&mut self) -> Result<DispatchCommandBuffer, CommandEncoderError>;

    fn clear_texture(
        &self,
        texture: &DispatchTexture,
        subresource_range: &crate::ImageSubresourceRange,
    ) -> Result<(), wgc::command::ClearError>;
    fn clear_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferAddress>,
    ) -> Result<(), wgc::command::ClearError>;

    fn insert_debug_marker(&self, label: &str) -> Result<(), CommandEncoderError> ;
    fn push_debug_group(&self, label: &str) -> Result<(), CommandEncoderError> ;
    fn pop_debug_group(&self) -> Result<(), CommandEncoderError> ;

    fn write_timestamp(&self, query_set: &DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::QueryError>;
    fn resolve_query_set(
        &self,
        query_set: &DispatchQuerySet,
        first_query: u32,
        query_count: u32,
        destination: &DispatchBuffer,
        destination_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::QueryError>;

    fn build_acceleration_structures_unsafe_tlas<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a crate::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a crate::TlasBuildEntry<'a>>,
    )-> Result<(), wgc::ray_tracing::BuildAccelerationStructureError>;
    fn build_acceleration_structures<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a crate::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a crate::TlasPackage>,
    ) -> Result<(), wgc::ray_tracing::BuildAccelerationStructureError>;
}
pub trait ComputePassInterface: CommonTraits {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) -> Result<(), ComputePassError>;
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) -> Result<(), ComputePassError>;
    fn set_push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), ComputePassError>;

    fn insert_debug_marker(&mut self, label: &str) -> Result<(), ComputePassError>;
    fn push_debug_group(&mut self, group_label: &str) -> Result<(), ComputePassError>;
    fn pop_debug_group(&mut self) -> Result<(), ComputePassError>;

    fn write_timestamp(&mut self, query_set: &DispatchQuerySet, query_index: u32) -> Result<(), ComputePassError>;
    fn begin_pipeline_statistics_query(&mut self, query_set: &DispatchQuerySet, query_index: u32) -> Result<(), ComputePassError>;
    fn end_pipeline_statistics_query(&mut self) -> Result<(), ComputePassError>;

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) -> Result<(), ComputePassError>;
    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), ComputePassError>;
    fn end(&mut self) -> Result<(), ComputePassError>;
}
pub trait RenderPassInterface: CommonTraits {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline) -> Result<(), wgc::command::RenderPassError>;
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) -> Result<(), wgc::command::RenderPassError>;
    fn set_index_buffer(
        &mut self,
        buffer: &DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn set_push_constants(&mut self, stages: crate::ShaderStages, offset: u32, data: &[u8]) -> Result<(), wgc::command::RenderPassError>;
    fn set_blend_constant(&mut self, color: crate::Color) -> Result<(), wgc::command::RenderPassError>;
    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<(), wgc::command::RenderPassError>;
    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn set_stencil_reference(&mut self, reference: u32) -> Result<(), wgc::command::RenderPassError>;

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<(), wgc::command::RenderPassError>;
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) -> Result<(), wgc::command::RenderPassError>;
    fn draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::RenderPassError>;

    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) -> Result<(), wgc::command::RenderPassError>;
    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) -> Result<(), wgc::command::RenderPassError>;

    fn insert_debug_marker(&mut self, label: &str) -> Result<(), wgc::command::RenderPassError>;
    fn push_debug_group(&mut self, group_label: &str) -> Result<(), wgc::command::RenderPassError>;
    fn pop_debug_group(&mut self) -> Result<(), wgc::command::RenderPassError>;

    fn write_timestamp(&mut self, query_set: &DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::RenderPassError>;
    fn begin_occlusion_query(&mut self, query_index: u32) -> Result<(), wgc::command::RenderPassError>;
    fn end_occlusion_query(&mut self) -> Result<(), wgc::command::RenderPassError>;
    fn begin_pipeline_statistics_query(&mut self, query_set: &DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::RenderPassError>;
    fn end_pipeline_statistics_query(&mut self) -> Result<(), wgc::command::RenderPassError>;

    fn execute_bundles(&mut self, render_bundles: &mut dyn Iterator<Item = &DispatchRenderBundle>) -> Result<(), wgc::command::RenderPassError>;

    fn end(&mut self) -> Result<(), wgc::command::RenderPassError>;
}

pub trait RenderBundleEncoderInterface: CommonTraits {
    fn set_pipeline(&mut self, pipeline: &DispatchRenderPipeline);
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    );
    fn set_index_buffer(
        &mut self,
        buffer: &DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    );
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    );
    fn set_push_constants(&mut self, stages: crate::ShaderStages, offset: u32, data: &[u8]);

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    );
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    );

    fn finish(self, desc: &crate::RenderBundleDescriptor<'_>) -> Result<DispatchRenderBundle, command::RenderBundleError>
    where
        Self: Sized;
}

pub trait CommandBufferInterface: CommonTraits {}
pub trait RenderBundleInterface: CommonTraits {}

pub trait SurfaceInterface: CommonTraits {
    fn get_capabilities(&self, adapter: &DispatchAdapter) -> wgt::SurfaceCapabilities;

    fn configure(&self, device: &DispatchDevice, config: &crate::SurfaceConfiguration) -> Result<(), present::ConfigureSurfaceError>;
    fn get_current_texture(
        &self,
    ) -> Result<(
        Option<DispatchTexture>,
        crate::SurfaceStatus,
        DispatchSurfaceOutputDetail,
    ), SurfaceError>;
}

pub trait SurfaceOutputDetailInterface: CommonTraits {
    fn present(&self);
    fn texture_discard(&self);
}

pub trait QueueWriteBufferInterface: CommonTraits {
    fn slice(&self) -> &[u8];

    fn slice_mut(&mut self) -> &mut [u8];
}

pub trait BufferMappedRangeInterface: CommonTraits {
    fn slice(&self) -> &[u8];
    fn slice_mut(&mut self) -> &mut [u8];
}

/// Generates Dispatch types for each of the interfaces. Each type is a wrapper around the
/// wgpu_core and webgpu types, and derefs to the appropriate interface trait-object.
///
/// When there is only one backend, deviritualization fires and all dispatches should turn into
/// direct calls. If there are multiple, some dispatching will occur.
///
/// This also provides `as_*` methods so that the backend implementations can dereference other
/// arguments. These are similarly free when there is only one backend.
///
/// In the future, we may want a truly generic backend, which could be extended from this enum.
macro_rules! dispatch_types_inner {
    (
        wgpu_core = $wgpu_core_context:ty;
        webgpu = $webgpu_context:ty;
        {ref type $name:ident = InterfaceTypes::$subtype:ident: $trait:ident};
    ) => {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
        pub enum $name {
            #[cfg(wgpu_core)]
            Core(Arc<<$wgpu_core_context as InterfaceTypes>::$subtype>),
            #[cfg(webgpu)]
            WebGPU(Arc<<$webgpu_context as InterfaceTypes>::$subtype>),
        }

        impl $name {
            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core(&self) -> &<$wgpu_core_context as InterfaceTypes>::$subtype {
                match self {
                    Self::Core(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not core")),
                }
            }

            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core_opt(&self) -> Option<&<$wgpu_core_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::Core(value) => Some(value),
                    _ => None,
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu(&self) -> &<$webgpu_context as InterfaceTypes>::$subtype {
                match self {
                    Self::WebGPU(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not webgpu")),
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu_opt(&self) -> Option<&<$webgpu_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::WebGPU(value) => Some(value),
                    _ => None,
                }
            }
        }

        #[cfg(wgpu_core)]
        impl From<<$wgpu_core_context as InterfaceTypes>::$subtype> for $name {
            #[inline]
            fn from(value: <$wgpu_core_context as InterfaceTypes>::$subtype) -> Self {
                Self::Core(Arc::new(value))
            }
        }

        #[cfg(webgpu)]
        impl From<<$webgpu_context as InterfaceTypes>::$subtype> for $name {
            #[inline]
            fn from(value: <$webgpu_context as InterfaceTypes>::$subtype) -> Self {
                Self::WebGPU(Arc::new(value))
            }
        }

        impl std::ops::Deref for $name {
            type Target = dyn $trait;

            #[inline]
            fn deref(&self) -> &Self::Target {
                match self {
                    #[cfg(wgpu_core)]
                    Self::Core(value) => value.as_ref(),
                    #[cfg(webgpu)]
                    Self::WebGPU(value) => value.as_ref(),
                }
            }
        }
    };
    (
        wgpu_core = $wgpu_core_context:ty;
        webgpu = $webgpu_context:ty;
        {mut type $name:ident = InterfaceTypes::$subtype:ident: $trait:ident};
    ) => {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $name {
            #[cfg(wgpu_core)]
            Core(<$wgpu_core_context as InterfaceTypes>::$subtype),
            #[cfg(webgpu)]
            WebGPU(<$webgpu_context as InterfaceTypes>::$subtype),
        }

        impl $name {
            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core(&self) -> &<$wgpu_core_context as InterfaceTypes>::$subtype {
                match self {
                    Self::Core(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not core")),
                }
            }

            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core_mut(&mut self) -> &mut <$wgpu_core_context as InterfaceTypes>::$subtype {
                match self {
                    Self::Core(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not core")),
                }
            }

            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core_opt(&self) -> Option<&<$wgpu_core_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::Core(value) => Some(value),
                    _ => None,
                }
            }

            #[cfg(wgpu_core)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_core_mut_opt(
                &mut self,
            ) -> Option<&mut <$wgpu_core_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::Core(value) => Some(value),
                    _ => None,
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu(&self) -> &<$webgpu_context as InterfaceTypes>::$subtype {
                match self {
                    Self::WebGPU(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not webgpu")),
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu_mut(&mut self) -> &mut <$webgpu_context as InterfaceTypes>::$subtype {
                match self {
                    Self::WebGPU(value) => value,
                    _ => panic!(concat!(stringify!($name), " is not webgpu")),
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu_opt(&self) -> Option<&<$webgpu_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::WebGPU(value) => Some(value),
                    _ => None,
                }
            }

            #[cfg(webgpu)]
            #[inline]
            #[allow(clippy::allow_attributes, unused)]
            pub fn as_webgpu_mut_opt(
                &mut self,
            ) -> Option<&mut <$webgpu_context as InterfaceTypes>::$subtype> {
                match self {
                    Self::WebGPU(value) => Some(value),
                    _ => None,
                }
            }
        }

        #[cfg(wgpu_core)]
        impl From<<$wgpu_core_context as InterfaceTypes>::$subtype> for $name {
            #[inline]
            fn from(value: <$wgpu_core_context as InterfaceTypes>::$subtype) -> Self {
                Self::Core(value)
            }
        }

        #[cfg(webgpu)]
        impl From<<$webgpu_context as InterfaceTypes>::$subtype> for $name {
            #[inline]
            fn from(value: <$webgpu_context as InterfaceTypes>::$subtype) -> Self {
                Self::WebGPU(value)
            }
        }

        impl std::ops::Deref for $name {
            type Target = dyn $trait;

            #[inline]
            fn deref(&self) -> &Self::Target {
                match self {
                    #[cfg(wgpu_core)]
                    Self::Core(value) => value,
                    #[cfg(webgpu)]
                    Self::WebGPU(value) => value,
                }
            }
        }

        impl std::ops::DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                match self {
                    #[cfg(wgpu_core)]
                    Self::Core(value) => value,
                    #[cfg(webgpu)]
                    Self::WebGPU(value) => value,
                }
            }
        }
    };
}

macro_rules! dispatch_types {
    (
        wgpu_core = $wgpu_core_context:ty;
        webgpu = $webgpu_context:ty;
        {$(
            $type:tt;
        )*}
    ) => {
        $(
            dispatch_types_inner!{
                wgpu_core = backend::ContextWgpuCore;
                webgpu = backend::ContextWebGpu;
                $type;
            }
        )*
    };
}

dispatch_types! {
    wgpu_core = backend::ContextWgpuCore;
    webgpu = backend::ContextWebGpu;
    {
        {ref type DispatchInstance = InterfaceTypes::Instance: InstanceInterface};
        {ref type DispatchAdapter = InterfaceTypes::Adapter: AdapterInterface};
        {ref type DispatchDevice = InterfaceTypes::Device: DeviceInterface};
        {ref type DispatchQueue = InterfaceTypes::Queue: QueueInterface};
        {ref type DispatchShaderModule = InterfaceTypes::ShaderModule: ShaderModuleInterface};
        {ref type DispatchBindGroupLayout = InterfaceTypes::BindGroupLayout: BindGroupLayoutInterface};
        {ref type DispatchBindGroup = InterfaceTypes::BindGroup: BindGroupInterface};
        {ref type DispatchTextureView = InterfaceTypes::TextureView: TextureViewInterface};
        {ref type DispatchSampler = InterfaceTypes::Sampler: SamplerInterface};
        {ref type DispatchBuffer = InterfaceTypes::Buffer: BufferInterface};
        {ref type DispatchTexture = InterfaceTypes::Texture: TextureInterface};
        {ref type DispatchBlas = InterfaceTypes::Blas: BlasInterface};
        {ref type DispatchTlas = InterfaceTypes::Tlas: TlasInterface};
        {ref type DispatchQuerySet = InterfaceTypes::QuerySet: QuerySetInterface};
        {ref type DispatchPipelineLayout = InterfaceTypes::PipelineLayout: PipelineLayoutInterface};
        {ref type DispatchRenderPipeline = InterfaceTypes::RenderPipeline: RenderPipelineInterface};
        {ref type DispatchComputePipeline = InterfaceTypes::ComputePipeline: ComputePipelineInterface};
        {ref type DispatchPipelineCache = InterfaceTypes::PipelineCache: PipelineCacheInterface};
        {mut type DispatchCommandEncoder = InterfaceTypes::CommandEncoder: CommandEncoderInterface};
        {mut type DispatchComputePass = InterfaceTypes::ComputePass: ComputePassInterface};
        {mut type DispatchRenderPass = InterfaceTypes::RenderPass: RenderPassInterface};
        {ref type DispatchCommandBuffer = InterfaceTypes::CommandBuffer: CommandBufferInterface};
        {mut type DispatchRenderBundleEncoder = InterfaceTypes::RenderBundleEncoder: RenderBundleEncoderInterface};
        {ref type DispatchRenderBundle = InterfaceTypes::RenderBundle: RenderBundleInterface};
        {ref type DispatchSurface = InterfaceTypes::Surface: SurfaceInterface};
        {ref type DispatchSurfaceOutputDetail = InterfaceTypes::SurfaceOutputDetail: SurfaceOutputDetailInterface};
        {mut type DispatchQueueWriteBuffer = InterfaceTypes::QueueWriteBuffer: QueueWriteBufferInterface};
        {mut type DispatchBufferMappedRange = InterfaceTypes::BufferMappedRange: BufferMappedRangeInterface};
    }
}