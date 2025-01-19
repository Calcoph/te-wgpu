use crate::{
    api,
    dispatch::{self, BufferMappedRangeInterface, InterfaceTypes},
    BindingResource, BufferBinding, BufferDescriptor, CompilationInfo, CompilationMessage,
    CompilationMessageType, Features, LoadOp, MapMode, Operations,
    ShaderSource, SurfaceTargetUnsafe, TextureDescriptor,
};

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    borrow::Cow::Borrowed, fmt, future::ready, ops::Range, pin::Pin, ptr::NonNull,
    slice, sync::Arc,
};
use wgc::{binding_model::{self, CreateBindGroupError, CreateBindGroupLayoutError, CreatePipelineLayoutError}, command::{self, bundle_ffi::*, CommandEncoderError, ComputePassError}, device::{queue::{QueueSubmitError, QueueWriteError}, DeviceError}, pipeline::{self, CreateComputePipelineError, CreateRenderPipelineError, CreateShaderModuleError}, present::{self, SurfaceError}, ray_tracing::{CreateBlasError, CreateTlasError}, resource::{self, CreateBufferError, CreateTextureError, CreateTextureViewError}};

#[derive(Clone)]
pub struct ContextWgpuCore(Arc<wgc::global::Global>);

impl Drop for ContextWgpuCore {
    fn drop(&mut self) {
        //nothing
    }
}

impl fmt::Debug for ContextWgpuCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextWgpuCore")
            .field("type", &"Native")
            .finish()
    }
}

impl ContextWgpuCore {
    pub unsafe fn from_hal_instance<A: wgc::hal_api::HalApi>(hal_instance: A::Instance) -> Self {
        Self(unsafe {
            Arc::new(wgc::global::Global::from_hal_instance::<A>(
                "wgpu",
                hal_instance,
            ))
        })
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: wgc::hal_api::HalApi>(&self) -> Option<&A::Instance> {
        unsafe { self.0.instance_as_hal::<A>() }
    }

    pub unsafe fn from_core_instance(core_instance: wgc::instance::Instance) -> Self {
        Self(unsafe { Arc::new(wgc::global::Global::from_instance(core_instance)) })
    }

    #[cfg(native)]
    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<wgc::id::AdapterId> {
        self.0.enumerate_adapters(backends)
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        unsafe { self.0.create_adapter_from_hal(hal_adapter.into(), None) }
    }

    pub unsafe fn adapter_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Adapter>) -> R,
        R,
    >(
        &self,
        adapter: &CoreAdapter,
        hal_adapter_callback: F,
    ) -> R {
        unsafe {
            self.0
                .adapter_as_hal::<A, F, R>(adapter.id, hal_adapter_callback)
        }
    }

    pub unsafe fn buffer_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        buffer: &CoreBuffer,
        hal_buffer_callback: F,
    ) -> R {
        unsafe {
            self.0
                .buffer_as_hal::<A, F, R>(buffer.id, hal_buffer_callback)
        }
    }

    pub unsafe fn create_device_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        adapter: &CoreAdapter,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Result<(CoreDevice, CoreQueue), crate::RequestDeviceError> {
        if trace_dir.is_some() {
            log::error!("Feature 'trace' has been removed temporarily, see https://github.com/gfx-rs/wgpu/issues/5974");
        }
        let (device_id, queue_id) = unsafe {
            self.0.create_device_from_hal(
                adapter.id,
                hal_device.into(),
                &desc.map_label(|l| l.map(Borrowed)),
                None,
                None,
                None,
            )
        }?;
        let device = CoreDevice {
            context: self.clone(),
            id: device_id,
            features: desc.required_features,
        };
        let queue = CoreQueue {
            context: self.clone(),
            id: queue_id,
        };
        Ok((device, queue))
    }

    pub unsafe fn create_texture_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_texture: A::Texture,
        device: &CoreDevice,
        desc: &TextureDescriptor<'_>,
    ) -> Result<CoreTexture, CreateTextureError> {
        let descriptor = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let id = unsafe {
            self.0
                .create_texture_from_hal(Box::new(hal_texture), device.id, &descriptor, None)
        }?;
        Ok(CoreTexture {
            context: self.clone(),
            id
        })
    }

    pub unsafe fn create_buffer_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_buffer: A::Buffer,
        device: &CoreDevice,
        desc: &BufferDescriptor<'_>,
    ) -> Result<CoreBuffer, CreateBufferError> {
        let id = unsafe {
            self.0.create_buffer_from_hal::<A>(
                hal_buffer,
                device.id,
                &desc.map_label(|l| l.map(Borrowed)),
                None,
            )?
        };
        Ok(CoreBuffer {
            context: self.clone(),
            id
        })
    }

    pub unsafe fn device_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        device: &CoreDevice,
        hal_device_callback: F,
    ) -> R {
        unsafe {
            self.0
                .device_as_hal::<A, F, R>(device.id, hal_device_callback)
        }
    }

    pub unsafe fn surface_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Surface>) -> R,
        R,
    >(
        &self,
        surface: &CoreSurface,
        hal_surface_callback: F,
    ) -> R {
        unsafe {
            self.0
                .surface_as_hal::<A, F, R>(surface.id, hal_surface_callback)
        }
    }

    pub unsafe fn texture_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Texture>) -> R,
        R,
    >(
        &self,
        texture: &CoreTexture,
        hal_texture_callback: F,
    ) -> R {
        unsafe {
            self.0
                .texture_as_hal::<A, F, R>(texture.id, hal_texture_callback)
        }
    }

    pub unsafe fn texture_view_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::TextureView>) -> R,
        R,
    >(
        &self,
        texture_view: &CoreTextureView,
        hal_texture_view_callback: F,
    ) -> R {
        unsafe {
            self.0
                .texture_view_as_hal::<A, F, R>(texture_view.id, hal_texture_view_callback)
        }
    }

    /// This method will start the wgpu_core level command recording.
    pub unsafe fn command_encoder_as_hal_mut<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        command_encoder: &CoreCommandEncoder,
        hal_command_encoder_callback: F,
    ) -> R {
        unsafe {
            self.0.command_encoder_as_hal_mut::<A, F, R>(
                command_encoder.id,
                hal_command_encoder_callback,
            )
        }
    }

    pub fn generate_report(&self) -> wgc::global::GlobalReport {
        self.0.generate_report()
    }
}

fn map_buffer_copy_view(view: crate::TexelCopyBufferInfo<'_>) -> wgc::command::TexelCopyBufferInfo {
    wgc::command::TexelCopyBufferInfo {
        buffer: view.buffer.inner.as_core().id,
        layout: view.layout,
    }
}

fn map_texture_copy_view(
    view: crate::TexelCopyTextureInfo<'_>,
) -> wgc::command::TexelCopyTextureInfo {
    wgc::command::TexelCopyTextureInfo {
        texture: view.texture.inner.as_core().id,
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), target_os = "emscripten"),
    expect(unused)
)]
fn map_texture_tagged_copy_view(
    view: wgt::CopyExternalImageDestInfo<&api::Texture>,
) -> wgc::command::CopyExternalImageDestInfo {
    wgc::command::CopyExternalImageDestInfo {
        texture: view.texture.inner.as_core().id,
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
        color_space: view.color_space,
        premultiplied_alpha: view.premultiplied_alpha,
    }
}

fn map_load_op<V: Copy>(load: &LoadOp<V>) -> LoadOp<Option<V>> {
    match load {
        LoadOp::Clear(clear_value) => LoadOp::Clear(Some(*clear_value)),
        LoadOp::Load => LoadOp::Load,
    }
}

fn map_pass_channel<V: Copy>(ops: Option<&Operations<V>>) -> wgc::command::PassChannel<Option<V>> {
    match ops {
        Some(&Operations { load, store }) => wgc::command::PassChannel {
            load_op: Some(map_load_op(&load)),
            store_op: Some(store),
            read_only: false,
        },
        None => wgc::command::PassChannel {
            load_op: None,
            store_op: None,
            read_only: true,
        },
    }
}

#[derive(Debug)]
pub struct CoreSurface {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::SurfaceId,
    /// Configured device is needed to know which backend
    /// code to execute when acquiring a new frame.
    configured_device: Mutex<Option<wgc::id::DeviceId>>,
}

#[derive(Debug)]
pub struct CoreAdapter {
    pub(crate) context: ContextWgpuCore,
    pub(crate) id: wgc::id::AdapterId,
}

#[derive(Debug)]
pub struct CoreDevice {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::DeviceId,
    features: Features,
}

#[derive(Debug)]
pub struct CoreBuffer {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::BufferId,
}

#[derive(Debug)]
pub struct CoreShaderModule {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::ShaderModuleId,
    compilation_info: CompilationInfo,
}

#[derive(Debug)]
pub struct CoreBindGroupLayout {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::BindGroupLayoutId,
}

#[derive(Debug)]
pub struct CoreBindGroup {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::BindGroupId,
}

#[derive(Debug)]
pub struct CoreTexture {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::TextureId,
}

#[derive(Debug)]
pub struct CoreTextureView {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::TextureViewId,
}

#[derive(Debug)]
pub struct CoreSampler {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::SamplerId,
}

#[derive(Debug)]
pub struct CoreQuerySet {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::QuerySetId,
}

#[derive(Debug)]
pub struct CorePipelineLayout {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::PipelineLayoutId,
}

#[derive(Debug)]
pub struct CorePipelineCache {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::PipelineCacheId,
}

#[derive(Debug)]
pub struct CoreCommandBuffer {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::CommandBufferId,
}

#[derive(Debug)]
pub struct CoreRenderBundleEncoder {
    pub(crate) context: ContextWgpuCore,
    encoder: wgc::command::RenderBundleEncoder,
    id: crate::cmp::Identifier,
}

#[derive(Debug)]
pub struct CoreRenderBundle {
    id: wgc::id::RenderBundleId,
}

#[derive(Debug)]
pub struct CoreQueue {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::QueueId,
}

#[derive(Debug)]
pub struct CoreComputePipeline {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::ComputePipelineId,
}

#[derive(Debug)]
pub struct CoreRenderPipeline {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::RenderPipelineId,
}

#[derive(Debug)]
pub struct CoreComputePass {
    pub(crate) context: ContextWgpuCore,
    pass: wgc::command::ComputePass,
    id: crate::cmp::Identifier,
}

#[derive(Debug)]
pub struct CoreRenderPass {
    pub(crate) context: ContextWgpuCore,
    pass: wgc::command::RenderPass,
    id: crate::cmp::Identifier,
}

#[derive(Debug)]
pub struct CoreCommandEncoder {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::CommandEncoderId,
    open: bool,
}

#[derive(Debug)]
pub struct CoreBlas {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::BlasId,
}

#[derive(Debug)]
pub struct CoreTlas {
    pub(crate) context: ContextWgpuCore,
    id: wgc::id::TlasId,
}

#[derive(Debug)]
pub struct CoreSurfaceOutputDetail {
    context: ContextWgpuCore,
    surface_id: wgc::id::SurfaceId,
}

impl From<CreateShaderModuleError> for CompilationInfo {
    fn from(value: CreateShaderModuleError) -> Self {
        match value {
            #[cfg(feature = "wgsl")]
            CreateShaderModuleError::Parsing(v) => v.into(),
            #[cfg(feature = "glsl")]
            CreateShaderModuleError::ParsingGlsl(v) => v.into(),
            #[cfg(feature = "spirv")]
            CreateShaderModuleError::ParsingSpirV(v) => v.into(),
            CreateShaderModuleError::Validation(v) => v.into(),
            // Device errors are reported through the error sink, and are not compilation errors.
            // Same goes for native shader module generation errors.
            CreateShaderModuleError::Device(_) | CreateShaderModuleError::Generation => {
                CompilationInfo {
                    messages: Vec::new(),
                }
            }
            // Everything else is an error message without location information.
            _ => CompilationInfo {
                messages: vec![CompilationMessage {
                    message: value.to_string(),
                    message_type: CompilationMessageType::Error,
                    location: None,
                }],
            },
        }
    }
}

#[derive(Debug)]
pub struct CoreQueueWriteBuffer {
    buffer_id: wgc::id::StagingBufferId,
    mapping: CoreBufferMappedRange,
}

#[derive(Debug)]
pub struct CoreBufferMappedRange {
    ptr: NonNull<u8>,
    size: usize,
}

#[cfg(send_sync)]
unsafe impl Send for CoreBufferMappedRange {}
#[cfg(send_sync)]
unsafe impl Sync for CoreBufferMappedRange {}

impl Drop for CoreBufferMappedRange {
    fn drop(&mut self) {
        // Intentionally left blank so that `BufferMappedRange` still
        // implements `Drop`, to match the web backend
    }
}

crate::cmp::impl_eq_ord_hash_arc_address!(ContextWgpuCore => .0);
crate::cmp::impl_eq_ord_hash_proxy!(CoreAdapter => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreDevice => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreQueue => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreShaderModule => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBindGroupLayout => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBindGroup => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreTextureView => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreSampler => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBuffer => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreTexture => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBlas => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreTlas => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreQuerySet => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CorePipelineLayout => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreRenderPipeline => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreComputePipeline => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CorePipelineCache => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreCommandEncoder => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreComputePass => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreRenderPass => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreCommandBuffer => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreRenderBundleEncoder => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreRenderBundle => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreSurface => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreSurfaceOutputDetail => .surface_id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreQueueWriteBuffer => .mapping.ptr);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBufferMappedRange => .ptr);

impl InterfaceTypes for ContextWgpuCore {
    type Instance = ContextWgpuCore;
    type Adapter = CoreAdapter;
    type Device = CoreDevice;
    type Queue = CoreQueue;
    type ShaderModule = CoreShaderModule;
    type BindGroupLayout = CoreBindGroupLayout;
    type BindGroup = CoreBindGroup;
    type TextureView = CoreTextureView;
    type Sampler = CoreSampler;
    type Buffer = CoreBuffer;
    type Texture = CoreTexture;
    type Blas = CoreBlas;
    type Tlas = CoreTlas;
    type QuerySet = CoreQuerySet;
    type PipelineLayout = CorePipelineLayout;
    type RenderPipeline = CoreRenderPipeline;
    type ComputePipeline = CoreComputePipeline;
    type PipelineCache = CorePipelineCache;
    type CommandEncoder = CoreCommandEncoder;
    type ComputePass = CoreComputePass;
    type RenderPass = CoreRenderPass;
    type CommandBuffer = CoreCommandBuffer;
    type RenderBundleEncoder = CoreRenderBundleEncoder;
    type RenderBundle = CoreRenderBundle;
    type Surface = CoreSurface;
    type SurfaceOutputDetail = CoreSurfaceOutputDetail;
    type QueueWriteBuffer = CoreQueueWriteBuffer;
    type BufferMappedRange = CoreBufferMappedRange;
}

impl dispatch::InstanceInterface for ContextWgpuCore {
    fn new(desc: &wgt::InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        Self(Arc::new(wgc::global::Global::new("wgpu", desc)))
    }

    unsafe fn create_surface(
        &self,
        target: crate::api::SurfaceTargetUnsafe,
    ) -> Result<dispatch::DispatchSurface, crate::CreateSurfaceError> {
        let id = match target {
            SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            } => unsafe {
                self.0
                    .instance_create_surface(raw_display_handle, raw_window_handle, None)
            },

            #[cfg(metal)]
            SurfaceTargetUnsafe::CoreAnimationLayer(layer) => unsafe {
                self.0.instance_create_surface_metal(layer, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::CompositionVisual(visual) => unsafe {
                self.0.instance_create_surface_from_visual(visual, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SurfaceHandle(surface_handle) => unsafe {
                self.0
                    .instance_create_surface_from_surface_handle(surface_handle, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SwapChainPanel(swap_chain_panel) => unsafe {
                self.0
                    .instance_create_surface_from_swap_chain_panel(swap_chain_panel, None)
            },
        }?;

        Ok(CoreSurface {
            context: self.clone(),
            id,
            configured_device: Mutex::default(),
        }
        .into())
    }

    fn request_adapter(
        &self,
        options: &crate::api::RequestAdapterOptions<'_, '_>,
    ) -> Pin<Box<dyn dispatch::RequestAdapterFuture>> {
        let id = self.0.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options
                    .compatible_surface
                    .map(|surface| surface.inner.as_core().id),
            },
            wgt::Backends::all(),
            None,
        );
        let adapter = id.map(|id| {
            let core = CoreAdapter {
                context: self.clone(),
                id,
            };
            let generic: dispatch::DispatchAdapter = core.into();
            generic
        });
        Box::pin(ready(adapter.ok()))
    }

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        match self.0.poll_all_devices(force_wait) {
            Ok(all_queue_empty) => all_queue_empty,
            Err(err) => panic!("Instance::poll_all_devices: {err}"),
        }
    }

    #[cfg(feature = "wgsl")]
    fn wgsl_language_features(&self) -> crate::WgslLanguageFeatures {
        wgc::naga::front::wgsl::ImplementedLanguageExtension::all()
            .iter()
            .copied()
            .fold(
                crate::WgslLanguageFeatures::empty(),
                #[expect(unreachable_code)]
                |acc, wle| acc | match wle {},
            )
    }
}

impl dispatch::AdapterInterface for CoreAdapter {
    fn request_device(
        &self,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn dispatch::RequestDeviceFuture>> {
        if trace_dir.is_some() {
            log::error!("Feature 'trace' has been removed temporarily, see https://github.com/gfx-rs/wgpu/issues/5974");
        }
        let res = self.context.0.adapter_request_device(
            self.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
            None,
            None,
        );
        let (device_id, queue_id) = match res {
            Ok(ids) => ids,
            Err(err) => {
                return Box::pin(ready(Err(err.into())));
            }
        };
        let device = CoreDevice {
            context: self.context.clone(),
            id: device_id,
            features: desc.required_features,
        };
        let queue = CoreQueue {
            context: self.context.clone(),
            id: queue_id
        };
        Box::pin(ready(Ok((device.into(), queue.into()))))
    }

    fn is_surface_supported(&self, surface: &dispatch::DispatchSurface) -> bool {
        let surface = surface.as_core();

        self.context
            .0
            .adapter_is_surface_supported(self.id, surface.id)
    }

    fn features(&self) -> crate::Features {
        self.context.0.adapter_features(self.id)
    }

    fn limits(&self) -> crate::Limits {
        self.context.0.adapter_limits(self.id)
    }

    fn downlevel_capabilities(&self) -> crate::DownlevelCapabilities {
        self.context.0.adapter_downlevel_capabilities(self.id)
    }

    fn get_info(&self) -> crate::AdapterInfo {
        self.context.0.adapter_get_info(self.id)
    }

    fn get_texture_format_features(
        &self,
        format: crate::TextureFormat,
    ) -> crate::TextureFormatFeatures {
        self.context
            .0
            .adapter_get_texture_format_features(self.id, format)
    }

    fn get_presentation_timestamp(&self) -> crate::PresentationTimestamp {
        self.context.0.adapter_get_presentation_timestamp(self.id)
    }
}

impl Drop for CoreAdapter {
    fn drop(&mut self) {
        self.context.0.adapter_drop(self.id)
    }
}

impl dispatch::DeviceInterface for CoreDevice {
    fn features(&self) -> crate::Features {
        self.context.0.device_features(self.id)
    }

    fn limits(&self) -> crate::Limits {
        self.context.0.device_limits(self.id)
    }

    // If we have no way to create a shader module, we can't return one, and so most of the function is unreachable.
    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga-ir"
        )),
        expect(unused)
    )]
    fn create_shader_module(
        &self,
        desc: crate::ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderRuntimeChecks,
    ) -> Result<dispatch::DispatchShaderModule, CreateShaderModuleError> {
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            runtime_checks: shader_bound_checks,
        };
        let source = match desc.source {
            #[cfg(feature = "spirv")]
            ShaderSource::SpirV(ref spv) => {
                // Parse the given shader code and store its representation.
                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false, // we require NDC_Y_UP feature
                    strict_capabilities: true,
                    block_ctx_dump_prefix: None,
                };
                wgc::pipeline::ShaderModuleSource::SpirV(Borrowed(spv), options)
            }
            #[cfg(feature = "glsl")]
            ShaderSource::Glsl {
                ref shader,
                stage,
                defines,
            } => {
                let options = naga::front::glsl::Options { stage, defines };
                wgc::pipeline::ShaderModuleSource::Glsl(Borrowed(shader), options)
            }
            #[cfg(feature = "wgsl")]
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
            #[cfg(feature = "naga-ir")]
            ShaderSource::Naga(module) => wgc::pipeline::ShaderModuleSource::Naga(module),
            ShaderSource::Dummy(_) => panic!("found `ShaderSource::Dummy`"),
        };
        let id =
            self.context
                .0
                .device_create_shader_module(self.id, &descriptor, source, None)?;
        let compilation_info = CompilationInfo { messages: vec![] };

        Ok(CoreShaderModule {
            context: self.context.clone(),
            id,
            compilation_info,
        }.into())
    }

    unsafe fn create_shader_module_spirv(
        &self,
        desc: &crate::ShaderModuleDescriptorSpirV<'_>,
    ) -> Result<dispatch::DispatchShaderModule, CreateShaderModuleError> {
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            // Doesn't matter the value since spirv shaders aren't mutated to include
            // runtime checks
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };
        let id = unsafe {
            self.context.0.device_create_shader_module_spirv(
                self.id,
                &descriptor,
                Borrowed(&desc.source),
                None,
            )?
        };
        let compilation_info = CompilationInfo { messages: vec![] };
        Ok(
        CoreShaderModule {
            context: self.context.clone(),
            id,
            compilation_info,
        }
        .into()
        )
    }

    fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor<'_>,
    ) -> Result<dispatch::DispatchBindGroupLayout, CreateBindGroupLayoutError> {
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let id =
            self.context
                .0
                .device_create_bind_group_layout(self.id, &descriptor, None)?;
        Ok(CoreBindGroupLayout {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<'_>,
    ) -> Result<dispatch::DispatchBindGroup, CreateBindGroupError> {
        use wgc::binding_model as bm;

        let mut arrayed_texture_views = Vec::new();
        let mut arrayed_samplers = Vec::new();
        if self.features.contains(Features::TEXTURE_BINDING_ARRAY) {
            // gather all the array view IDs first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views.extend(array.iter().map(|view| view.inner.as_core().id));
                }
                if let BindingResource::SamplerArray(array) = entry.resource {
                    arrayed_samplers.extend(array.iter().map(|sampler| sampler.inner.as_core().id));
                }
            }
        }
        let mut remaining_arrayed_texture_views = &arrayed_texture_views[..];
        let mut remaining_arrayed_samplers = &arrayed_samplers[..];

        let mut arrayed_buffer_bindings = Vec::new();
        if self.features.contains(Features::BUFFER_BINDING_ARRAY) {
            // gather all the buffers first
            for entry in desc.entries.iter() {
                if let BindingResource::BufferArray(array) = entry.resource {
                    arrayed_buffer_bindings.extend(array.iter().map(|binding| bm::BufferBinding {
                        buffer_id: binding.buffer.inner.as_core().id,
                        offset: binding.offset,
                        size: binding.size,
                    }));
                }
            }
        }
        let mut remaining_arrayed_buffer_bindings = &arrayed_buffer_bindings[..];

        let entries = desc
            .entries
            .iter()
            .map(|entry| bm::BindGroupEntry {
                binding: entry.binding,
                resource: match entry.resource {
                    BindingResource::Buffer(BufferBinding {
                        buffer,
                        offset,
                        size,
                    }) => bm::BindingResource::Buffer(bm::BufferBinding {
                        buffer_id: buffer.inner.as_core().id,
                        offset,
                        size,
                    }),
                    BindingResource::BufferArray(array) => {
                        let slice = &remaining_arrayed_buffer_bindings[..array.len()];
                        remaining_arrayed_buffer_bindings =
                            &remaining_arrayed_buffer_bindings[array.len()..];
                        bm::BindingResource::BufferArray(Borrowed(slice))
                    }
                    BindingResource::Sampler(sampler) => {
                        bm::BindingResource::Sampler(sampler.inner.as_core().id)
                    }
                    BindingResource::SamplerArray(array) => {
                        let slice = &remaining_arrayed_samplers[..array.len()];
                        remaining_arrayed_samplers = &remaining_arrayed_samplers[array.len()..];
                        bm::BindingResource::SamplerArray(Borrowed(slice))
                    }
                    BindingResource::TextureView(texture_view) => {
                        bm::BindingResource::TextureView(texture_view.inner.as_core().id)
                    }
                    BindingResource::TextureViewArray(array) => {
                        let slice = &remaining_arrayed_texture_views[..array.len()];
                        remaining_arrayed_texture_views =
                            &remaining_arrayed_texture_views[array.len()..];
                        bm::BindingResource::TextureViewArray(Borrowed(slice))
                    }
                    BindingResource::AccelerationStructure(acceleration_structure) => {
                        bm::BindingResource::AccelerationStructure(
                            acceleration_structure.shared.inner.as_core().id,
                        )
                    }
                },
            })
            .collect::<Vec<_>>();
        let descriptor = bm::BindGroupDescriptor {
            label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
            layout: desc.layout.inner.as_core().id,
            entries: Borrowed(&entries),
        };

        let id = self
            .context
            .0
            .device_create_bind_group(self.id, &descriptor, None)?;

        Ok(CoreBindGroup {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<'_>,
    ) -> Result<dispatch::DispatchPipelineLayout, CreatePipelineLayoutError> {
        // Limit is always less or equal to hal::MAX_BIND_GROUPS, so this is always right
        // Guards following ArrayVec
        assert!(
            desc.bind_group_layouts.len() <= wgc::MAX_BIND_GROUPS,
            "Bind group layout count {} exceeds device bind group limit {}",
            desc.bind_group_layouts.len(),
            wgc::MAX_BIND_GROUPS
        );

        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.inner.as_core().id)
            .collect::<ArrayVec<_, { wgc::MAX_BIND_GROUPS }>>();
        let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
            label: desc.label.map(Borrowed),
            bind_group_layouts: Borrowed(&temp_layouts),
            push_constant_ranges: Borrowed(desc.push_constant_ranges),
        };

        let id = self
            .context
            .0
            .device_create_pipeline_layout(self.id, &descriptor, None)?;
        Ok(CorePipelineLayout {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<'_>,
    ) -> Result<dispatch::DispatchRenderPipeline, CreateRenderPipelineError> {
        use wgc::pipeline as pipe;

        let vertex_buffers: ArrayVec<_, { wgc::MAX_VERTEX_BUFFERS }> = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| pipe::VertexBufferLayout {
                array_stride: vbuf.array_stride,
                step_mode: vbuf.step_mode,
                attributes: Borrowed(vbuf.attributes),
            })
            .collect();

        let descriptor = pipe::RenderPipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|layout| layout.inner.as_core().id),
            vertex: pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc.vertex.module.inner.as_core().id,
                    entry_point: desc.vertex.entry_point.map(Borrowed),
                    constants: Borrowed(desc.vertex.compilation_options.constants),
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
                buffers: Borrowed(&vertex_buffers),
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| pipe::FragmentState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: frag.module.inner.as_core().id,
                    entry_point: frag.entry_point.map(Borrowed),
                    constants: Borrowed(frag.compilation_options.constants),
                    zero_initialize_workgroup_memory: frag
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
                targets: Borrowed(frag.targets),
            }),
            multiview: desc.multiview,
            cache: desc.cache.map(|cache| cache.inner.as_core().id),
        };

        let id =
            match self.context
                .0
                .device_create_render_pipeline(self.id, &descriptor, None, None) {
                    Ok(id) => id,
                    Err(e) => {
                        if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = e {
                            log::error!("Shader translation error for stage {:?}: {}", stage, error);
                            log::error!("Please report it to https://github.com/gfx-rs/wgpu");
                        };

                        return Err(e)
                    }
                };

        Ok(CoreRenderPipeline {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<'_>,
    ) -> Result<dispatch::DispatchComputePipeline, CreateComputePipelineError> {
        use wgc::pipeline as pipe;

        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|pll| pll.inner.as_core().id),
            stage: pipe::ProgrammableStageDescriptor {
                module: desc.module.inner.as_core().id,
                entry_point: desc.entry_point.map(Borrowed),
                constants: Borrowed(desc.compilation_options.constants),
                zero_initialize_workgroup_memory: desc
                    .compilation_options
                    .zero_initialize_workgroup_memory,
            },
            cache: desc.cache.map(|cache| cache.inner.as_core().id),
        };

        let id =
            self.context
                .0
                .device_create_compute_pipeline(self.id, &descriptor, None, None)?;
        Ok(CoreComputePipeline {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    unsafe fn create_pipeline_cache(
        &self,
        desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> Result<dispatch::DispatchPipelineCache, pipeline::CreatePipelineCacheError> {
        use wgc::pipeline as pipe;

        let descriptor = pipe::PipelineCacheDescriptor {
            label: desc.label.map(Borrowed),
            data: desc.data.map(Borrowed),
            fallback: desc.fallback,
        };
        let id = unsafe {
            self.context
                .0
                .device_create_pipeline_cache(self.id, &descriptor, None)?
        };

        Ok(CorePipelineCache {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_buffer(&self, desc: &crate::BufferDescriptor<'_>) -> Result<dispatch::DispatchBuffer, CreateBufferError> {
        let id = self.context.0.device_create_buffer(
            self.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        )?;

        Ok(CoreBuffer {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_texture(&self, desc: &crate::TextureDescriptor<'_>) -> Result<dispatch::DispatchTexture, CreateTextureError> {
        let wgt_desc = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let id = self
            .context
            .0
            .device_create_texture(self.id, &wgt_desc, None)?;

        Ok(CoreTexture {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_blas(
        &self,
        desc: &crate::CreateBlasDescriptor<'_>,
        sizes: crate::BlasGeometrySizeDescriptors,
    ) -> Result<(u64, dispatch::DispatchBlas), CreateBlasError> {
        let global = &self.context.0;
        let (id, handle) =
            global.device_create_blas(self.id, &desc.map_label(|l| l.map(Borrowed)), sizes, None)?;
        Ok((
            handle,
            CoreBlas {
                context: self.context.clone(),
                id,
            }
            .into(),
        ))
    }

    fn create_tlas(&self, desc: &crate::CreateTlasDescriptor<'_>) -> Result<dispatch::DispatchTlas, CreateTlasError> {
        let global = &self.context.0;
        let id =
            global.device_create_tlas(self.id, &desc.map_label(|l| l.map(Borrowed)), None)?;
        Ok(CoreTlas {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_sampler(&self, desc: &crate::SamplerDescriptor<'_>) -> Result<dispatch::DispatchSampler, resource::CreateSamplerError> {
        let descriptor = wgc::resource::SamplerDescriptor {
            label: desc.label.map(Borrowed),
            address_modes: [
                desc.address_mode_u,
                desc.address_mode_v,
                desc.address_mode_w,
            ],
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare,
            anisotropy_clamp: desc.anisotropy_clamp,
            border_color: desc.border_color,
        };

        let id = self
            .context
            .0
            .device_create_sampler(self.id, &descriptor, None)?;
        Ok(CoreSampler {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_query_set(&self, desc: &crate::QuerySetDescriptor<'_>) -> Result<dispatch::DispatchQuerySet, resource::CreateQuerySetError> {
        let id = self.context.0.device_create_query_set(
            self.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        )?;

        Ok(CoreQuerySet {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<'_>,
    ) -> Result<dispatch::DispatchCommandEncoder, DeviceError> {
        let id = self.context.0.device_create_command_encoder(
            self.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        )?;

        Ok(CoreCommandEncoder {
            context: self.context.clone(),
            id,
            open: true,
        }
        .into())
    }

    fn create_render_bundle_encoder(
        &self,
        desc: &crate::RenderBundleEncoderDescriptor<'_>,
    ) -> dispatch::DispatchRenderBundleEncoder {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
            multiview: desc.multiview,
        };
        let encoder = match wgc::command::RenderBundleEncoder::new(&descriptor, self.id, None) {
            Ok(encoder) => encoder,
            Err(e) => panic!("Error in Device::create_render_bundle_encoder: {e}"),
        };

        CoreRenderBundleEncoder {
            context: self.context.clone(),
            encoder,
            id: crate::cmp::Identifier::create(),
        }
        .into()
    }

    fn set_device_lost_callback(&self, device_lost_callback: dispatch::BoxDeviceLostCallback) {
        self.context
            .0
            .device_set_device_lost_closure(self.id, device_lost_callback);
    }
    fn start_capture(&self) {
        self.context.0.device_start_capture(self.id);
    }

    fn stop_capture(&self) {
        self.context.0.device_stop_capture(self.id);
    }

    fn poll(&self, maintain: crate::Maintain) -> crate::MaintainResult {
        let maintain_inner = maintain.map_index(|i| i.index);
        match self.context.0.device_poll(self.id, maintain_inner) {
            Ok(done) => match done {
                true => wgt::MaintainResult::SubmissionQueueEmpty,
                false => wgt::MaintainResult::Ok,
            },
            Err(err) => panic!("{err}Device::poll"),
        }
    }

    fn get_internal_counters(&self) -> crate::InternalCounters {
        self.context.0.device_get_internal_counters(self.id)
    }

    fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport> {
        self.context.0.device_generate_allocator_report(self.id)
    }

    fn destroy(&self) {
        self.context.0.device_destroy(self.id);
    }
}

impl Drop for CoreDevice {
    fn drop(&mut self) {
        self.context.0.device_drop(self.id)
    }
}

impl dispatch::QueueInterface for CoreQueue {
    fn write_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        data: &[u8],
    ) -> Result<(), wgc::device::queue::QueueWriteError> {
        let buffer = buffer.as_core();

        self
            .context
            .0
            .queue_write_buffer(self.id, buffer.id, offset, data)
    }

    fn create_staging_buffer(
        &self,
        size: crate::BufferSize,
    ) -> Result<dispatch::DispatchQueueWriteBuffer, QueueWriteError> {
        match self
            .context
            .0
            .queue_create_staging_buffer(self.id, size, None)
        {
            Ok((buffer_id, ptr)) => Ok(
                CoreQueueWriteBuffer {
                    buffer_id,
                    mapping: CoreBufferMappedRange {
                        ptr,
                        size: size.get() as usize,
                    },
                }
                .into(),
            ),
            Err(err) => Err(err)
        }
    }

    fn validate_write_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Result<(), wgc::device::queue::QueueWriteError> {
        let buffer = buffer.as_core();

        self
            .context
            .0
            .queue_validate_write_buffer(self.id, buffer.id, offset, size)
    }

    fn write_staging_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        staging_buffer: &dispatch::DispatchQueueWriteBuffer,
    ) -> Result<(), wgc::device::queue::QueueWriteError> {
        let buffer = buffer.as_core();
        let staging_buffer = staging_buffer.as_core();

        self.context.0.queue_write_staging_buffer(
            self.id,
            buffer.id,
            offset,
            staging_buffer.buffer_id,
        )
    }

    fn write_texture(
        &self,
        texture: crate::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: crate::TexelCopyBufferLayout,
        size: crate::Extent3d,
    ) -> Result<(), wgc::device::queue::QueueWriteError> {
        self.context.0.queue_write_texture(
            self.id,
            &map_texture_copy_view(texture),
            data,
            &data_layout,
            &size,
        )
    }

    #[cfg(any(webgpu, webgl))]
    fn copy_external_image_to_texture(
        &self,
        source: &wgt::CopyExternalImageSourceInfo,
        dest: wgt::CopyExternalImageDestInfo<&crate::api::Texture>,
        size: crate::Extent3d,
    ) -> Result<(), QueueWriteError> {
        self.context.0.queue_copy_external_image_to_texture(
            self.id,
            source,
            map_texture_tagged_copy_view(dest),
            size,
        )
    }

    fn submit(
        &self,
        command_buffers: &mut dyn Iterator<Item = dispatch::DispatchCommandBuffer>,
    ) -> Result<u64, (u64, QueueSubmitError)> {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();
        let command_buffer_ids = temp_command_buffers
            .iter()
            .map(|cmdbuf| cmdbuf.as_core().id)
            .collect::<SmallVec<[_; 4]>>();

        let index = self.context.0.queue_submit(self.id, &command_buffer_ids)?;

        drop(temp_command_buffers);

        Ok(index)
    }

    fn get_timestamp_period(&self) -> f32 {
        self.context.0.queue_get_timestamp_period(self.id)
    }

    fn on_submitted_work_done(&self, callback: dispatch::BoxSubmittedWorkDoneCallback) {
        self.context
            .0
            .queue_on_submitted_work_done(self.id, callback);
    }
}

impl Drop for CoreQueue {
    fn drop(&mut self) {
        self.context.0.queue_drop(self.id)
    }
}

impl dispatch::ShaderModuleInterface for CoreShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn dispatch::ShaderCompilationInfoFuture>> {
        Box::pin(ready(self.compilation_info.clone()))
    }
}

impl Drop for CoreShaderModule {
    fn drop(&mut self) {
        self.context.0.shader_module_drop(self.id)
    }
}

impl dispatch::BindGroupLayoutInterface for CoreBindGroupLayout {}

impl Drop for CoreBindGroupLayout {
    fn drop(&mut self) {
        self.context.0.bind_group_layout_drop(self.id)
    }
}

impl dispatch::BindGroupInterface for CoreBindGroup {}

impl Drop for CoreBindGroup {
    fn drop(&mut self) {
        self.context.0.bind_group_drop(self.id)
    }
}

impl dispatch::TextureViewInterface for CoreTextureView {}

impl Drop for CoreTextureView {
    fn drop(&mut self) {
        // TODO: We don't use this error at all?
        let _ = self.context.0.texture_view_drop(self.id);
    }
}

impl dispatch::SamplerInterface for CoreSampler {}

impl Drop for CoreSampler {
    fn drop(&mut self) {
        self.context.0.sampler_drop(self.id)
    }
}

impl dispatch::BufferInterface for CoreBuffer {
    fn map_async(
        &self,
        mode: crate::MapMode,
        range: Range<crate::BufferAddress>,
        callback: dispatch::BufferMapCallback,
    ) -> Result<(), resource::BufferAccessError> {
        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: Some(Box::new(|status| {
                let res = status.map_err(|_| crate::BufferAsyncError);
                callback(res);
            })),
        };

        self.context.0.buffer_map_async(
            self.id,
            range.start,
            Some(range.end - range.start),
            operation,
        ).map(|_| ())
    }

    fn get_mapped_range(
        &self,
        sub_range: Range<crate::BufferAddress>,
    ) -> dispatch::DispatchBufferMappedRange {
        let size = sub_range.end - sub_range.start;
        match self
            .context
            .0
            .buffer_get_mapped_range(self.id, sub_range.start, Some(size))
        {
            Ok((ptr, size)) => CoreBufferMappedRange {
                ptr,
                size: size as usize,
            }.into(),
            Err(err) => panic!("{err}Buffer::get_mapped_range"),
        }
    }

    #[cfg(webgpu)]
    fn get_mapped_range_as_array_buffer(
        &self,
        _sub_range: Range<wgt::BufferAddress>,
    ) -> Option<js_sys::ArrayBuffer> {
        None
    }

    fn unmap(&self) -> Result<(), wgc::resource::BufferAccessError> {
        self.context.0.buffer_unmap(self.id)
    }

    fn destroy(&self) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = self.context.0.buffer_destroy(self.id);
    }
}

impl Drop for CoreBuffer {
    fn drop(&mut self) {
        self.context.0.buffer_drop(self.id)
    }
}

impl dispatch::TextureInterface for CoreTexture {
    fn create_view(
        &self,
        desc: &crate::TextureViewDescriptor<'_>,
    ) -> Result<dispatch::DispatchTextureView, CreateTextureViewError> {
        let descriptor = wgc::resource::TextureViewDescriptor {
            label: desc.label.map(Borrowed),
            format: desc.format,
            dimension: desc.dimension,
            usage: desc.usage,
            range: wgt::ImageSubresourceRange {
                aspect: desc.aspect,
                base_mip_level: desc.base_mip_level,
                mip_level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                array_layer_count: desc.array_layer_count,
            },
        };
        let id = self
            .context
            .0
            .texture_create_view(self.id, &descriptor, None)?;

        Ok(CoreTextureView {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn destroy(&self) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = self.context.0.texture_destroy(self.id);
    }
}

impl Drop for CoreTexture {
    fn drop(&mut self) {
        self.context.0.texture_drop(self.id)
    }
}

impl dispatch::BlasInterface for CoreBlas {}

impl Drop for CoreBlas {
    fn drop(&mut self) {
        self.context.0.blas_drop(self.id)
    }
}

impl dispatch::TlasInterface for CoreTlas {}

impl Drop for CoreTlas {
    fn drop(&mut self) {
        self.context.0.tlas_drop(self.id)
    }
}

impl dispatch::QuerySetInterface for CoreQuerySet {}

impl Drop for CoreQuerySet {
    fn drop(&mut self) {
        self.context.0.query_set_drop(self.id)
    }
}

impl dispatch::PipelineLayoutInterface for CorePipelineLayout {}

impl Drop for CorePipelineLayout {
    fn drop(&mut self) {
        self.context.0.pipeline_layout_drop(self.id)
    }
}

impl dispatch::RenderPipelineInterface for CoreRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> Result<dispatch::DispatchBindGroupLayout, binding_model::GetBindGroupLayoutError> {
        let id = self
            .context
            .0
            .render_pipeline_get_bind_group_layout(self.id, index, None)?;
        Ok(CoreBindGroupLayout {
            context: self.context.clone(),
            id,
        }
        .into())
    }
}

impl Drop for CoreRenderPipeline {
    fn drop(&mut self) {
        self.context.0.render_pipeline_drop(self.id)
    }
}

impl dispatch::ComputePipelineInterface for CoreComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> Result<dispatch::DispatchBindGroupLayout, binding_model::GetBindGroupLayoutError> {
        let id = self
            .context
            .0
            .compute_pipeline_get_bind_group_layout(self.id, index, None)?;

        Ok(CoreBindGroupLayout {
            context: self.context.clone(),
            id,
        }
        .into())
    }
}

impl Drop for CoreComputePipeline {
    fn drop(&mut self) {
        self.context.0.compute_pipeline_drop(self.id)
    }
}

impl dispatch::PipelineCacheInterface for CorePipelineCache {
    fn get_data(&self) -> Option<Vec<u8>> {
        self.context.0.pipeline_cache_get_data(self.id)
    }
}

impl Drop for CorePipelineCache {
    fn drop(&mut self) {
        self.context.0.pipeline_cache_drop(self.id)
    }
}

impl dispatch::CommandEncoderInterface for CoreCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &dispatch::DispatchBuffer,
        source_offset: crate::BufferAddress,
        destination: &dispatch::DispatchBuffer,
        destination_offset: crate::BufferAddress,
        copy_size: crate::BufferAddress,
    ) -> Result<(), wgc::command::CopyError> {
        let source = source.as_core();
        let destination = destination.as_core();

        self.context.0.command_encoder_copy_buffer_to_buffer(
            self.id,
            source.id,
            source_offset,
            destination.id,
            destination_offset,
            copy_size,
        )
    }

    fn copy_buffer_to_texture(
        &self,
        source: crate::TexelCopyBufferInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError> {
        self.context.0.command_encoder_copy_buffer_to_texture(
            self.id,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        )
    }

    fn copy_texture_to_buffer(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyBufferInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError> {
        self.context.0.command_encoder_copy_texture_to_buffer(
            self.id,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size,
        )
    }

    fn copy_texture_to_texture(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) -> Result<(), wgc::command::CopyError> {
        self.context.0.command_encoder_copy_texture_to_texture(
            self.id,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        )
    }

    fn begin_compute_pass(
        &self,
        desc: &crate::ComputePassDescriptor<'_>,
    ) -> Result<dispatch::DispatchComputePass, CommandEncoderError> {
        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.inner.as_core().id,
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let pass = self.context.0.command_encoder_create_compute_pass(
            self.id,
            &wgc::command::ComputePassDescriptor {
                label: desc.label.map(Borrowed),
                timestamp_writes: timestamp_writes.as_ref(),
            },
        )?;

        Ok(CoreComputePass {
            context: self.context.clone(),
            pass,
            id: crate::cmp::Identifier::create(),
        }
        .into())
    }

    fn begin_render_pass(
        &self,
        desc: &crate::RenderPassDescriptor<'_>,
    ) -> Result<dispatch::DispatchRenderPass, CommandEncoderError> {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| {
                ca.as_ref()
                    .map(|at| wgc::command::RenderPassColorAttachment {
                        view: at.view.inner.as_core().id,
                        resolve_target: at.resolve_target.map(|view| view.inner.as_core().id),
                        load_op: at.ops.load,
                        store_op: at.ops.store,
                    })
            })
            .collect::<Vec<_>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachment {
                view: dsa.view.inner.as_core().id,
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.inner.as_core().id,
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let pass = self.context.0.command_encoder_create_render_pass(
            self.id,
            &wgc::command::RenderPassDescriptor {
                label: desc.label.map(Borrowed),
                timestamp_writes: timestamp_writes.as_ref(),
                color_attachments: std::borrow::Cow::Borrowed(&colors),
                depth_stencil_attachment: depth_stencil.as_ref(),
                occlusion_query_set: desc.occlusion_query_set.map(|qs| qs.inner.as_core().id),
            },
        )?;

        Ok(CoreRenderPass {
            context: self.context.clone(),
            pass,
            id: crate::cmp::Identifier::create(),
        }
        .into())
    }

    fn finish(&mut self) -> Result<dispatch::DispatchCommandBuffer, CommandEncoderError> {
        let descriptor = wgt::CommandBufferDescriptor::default();
        self.open = false; // prevent the drop
        let id = self.context.0.command_encoder_finish(self.id, &descriptor)?;

        Ok(CoreCommandBuffer {
            context: self.context.clone(),
            id,
        }
        .into())
    }

    fn clear_texture(
        &self,
        texture: &dispatch::DispatchTexture,
        subresource_range: &crate::ImageSubresourceRange,
    ) -> Result<(), wgc::command::ClearError> {
        let texture = texture.as_core();

        self.context
            .0
            .command_encoder_clear_texture(self.id, texture.id, subresource_range)
    }

    fn clear_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferAddress>,
    ) -> Result<(), wgc::command::ClearError> {
        let buffer = buffer.as_core();

        self
            .context
            .0
            .command_encoder_clear_buffer(self.id, buffer.id, offset, size)
    }

    fn insert_debug_marker(&self, label: &str) -> Result<(), CommandEncoderError> {
        self
            .context
            .0
            .command_encoder_insert_debug_marker(self.id, label)
    }

    fn push_debug_group(&self, label: &str) -> Result<(), CommandEncoderError> {
        self
            .context
            .0
            .command_encoder_push_debug_group(self.id, label)
    }

    fn pop_debug_group(&self) -> Result<(), CommandEncoderError> {
        self.context.0.command_encoder_pop_debug_group(self.id)
    }

    fn write_timestamp(&self, query_set: &dispatch::DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::QueryError> {
        let query_set = query_set.as_core();

        self.context
            .0
            .command_encoder_write_timestamp(self.id, query_set.id, query_index)
    }

    fn resolve_query_set(
        &self,
        query_set: &dispatch::DispatchQuerySet,
        first_query: u32,
        query_count: u32,
        destination: &dispatch::DispatchBuffer,
        destination_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::QueryError> {
        let query_set = query_set.as_core();
        let destination = destination.as_core();

        self.context.0.command_encoder_resolve_query_set(
            self.id,
            query_set.id,
            first_query,
            query_count,
            destination.id,
            destination_offset,
        )
    }

    fn build_acceleration_structures_unsafe_tlas<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a crate::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a crate::TlasBuildEntry<'a>>,
    ) -> Result<(), wgc::ray_tracing::BuildAccelerationStructureError> {
        let blas = blas.map(|e: &crate::BlasBuildEntry<'_>| {
            let geometries = match e.geometry {
                crate::BlasGeometries::TriangleGeometries(ref triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg| {
                        wgc::ray_tracing::BlasTriangleGeometry {
                            vertex_buffer: tg.vertex_buffer.inner.as_core().id,
                            index_buffer: tg.index_buffer.map(|buf| buf.inner.as_core().id),
                            transform_buffer: tg.transform_buffer.map(|buf| buf.inner.as_core().id),
                            size: tg.size,
                            transform_buffer_offset: tg.transform_buffer_offset,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            first_index: tg.first_index,
                        }
                    });
                    wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            wgc::ray_tracing::BlasBuildEntry {
                blas_id: e.blas.inner.as_core().id,
                geometries,
            }
        });

        let tlas = tlas.into_iter().map(|e: &crate::TlasBuildEntry<'a>| {
            wgc::ray_tracing::TlasBuildEntry {
                tlas_id: e.tlas.shared.inner.as_core().id,
                instance_buffer_id: e.instance_buffer.inner.as_core().id,
                instance_count: e.instance_count,
            }
        });

        self
            .context
            .0
            .command_encoder_build_acceleration_structures_unsafe_tlas(self.id, blas, tlas)
    }

    fn build_acceleration_structures<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a crate::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a crate::TlasPackage>,
    ) -> Result<(), wgc::ray_tracing::BuildAccelerationStructureError> {
        let blas = blas.map(|e: &crate::BlasBuildEntry<'_>| {
            let geometries = match e.geometry {
                crate::BlasGeometries::TriangleGeometries(ref triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg| {
                        wgc::ray_tracing::BlasTriangleGeometry {
                            vertex_buffer: tg.vertex_buffer.inner.as_core().id,
                            index_buffer: tg.index_buffer.map(|buf| buf.inner.as_core().id),
                            transform_buffer: tg.transform_buffer.map(|buf| buf.inner.as_core().id),
                            size: tg.size,
                            transform_buffer_offset: tg.transform_buffer_offset,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            first_index: tg.first_index,
                        }
                    });
                    wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            wgc::ray_tracing::BlasBuildEntry {
                blas_id: e.blas.inner.as_core().id,
                geometries,
            }
        });

        let tlas = tlas.into_iter().map(|e| {
            let instances = e
                .instances
                .iter()
                .map(|instance: &Option<crate::TlasInstance>| {
                    instance
                        .as_ref()
                        .map(|instance| wgc::ray_tracing::TlasInstance {
                            blas_id: instance.blas.as_core().id,
                            transform: &instance.transform,
                            custom_index: instance.custom_index,
                            mask: instance.mask,
                        })
                });
            wgc::ray_tracing::TlasPackage {
                tlas_id: e.tlas.shared.inner.as_core().id,
                instances: Box::new(instances),
                lowest_unmodified: e.lowest_unmodified,
            }
        });

        self
            .context
            .0
            .command_encoder_build_acceleration_structures(self.id, blas, tlas)
    }
}

impl Drop for CoreCommandEncoder {
    fn drop(&mut self) {
        if self.open {
            self.context.0.command_encoder_drop(self.id)
        }
    }
}

impl dispatch::CommandBufferInterface for CoreCommandBuffer {}

impl Drop for CoreCommandBuffer {
    fn drop(&mut self) {
        self.context.0.command_buffer_drop(self.id)
    }
}

impl dispatch::ComputePassInterface for CoreComputePass {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchComputePipeline) -> Result<(), wgc::command::ComputePassError> {
        let pipeline = pipeline.as_core();

        self
            .context
            .0
            .compute_pass_set_pipeline(&mut self.pass, pipeline.id)
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) -> Result<(), wgc::command::ComputePassError> {
        let bg = bind_group.map(|bg| bg.as_core().id);

        self.context
            .0
            .compute_pass_set_bind_group(&mut self.pass, index, bg, offsets)
    }

    fn set_push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), wgc::command::ComputePassError> {
        self.context
            .0
            .compute_pass_set_push_constants(&mut self.pass, offset, data)
    }

    fn insert_debug_marker(&mut self, label: &str) -> Result<(), wgc::command::ComputePassError> {
        self.context
            .0
            .compute_pass_insert_debug_marker(&mut self.pass, label, 0)
    }

    fn push_debug_group(&mut self, group_label: &str) -> Result<(), wgc::command::ComputePassError> {
        self.context
            .0
            .compute_pass_push_debug_group(&mut self.pass, group_label, 0)
    }

    fn pop_debug_group(&mut self) -> Result<(), wgc::command::ComputePassError> {
        self.context.0.compute_pass_pop_debug_group(&mut self.pass)
    }

    fn write_timestamp(&mut self, query_set: &dispatch::DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::ComputePassError> {
        let query_set = query_set.as_core();

        self.context
            .0
            .compute_pass_write_timestamp(&mut self.pass, query_set.id, query_index)
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        query_set: &dispatch::DispatchQuerySet,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        let query_set = query_set.as_core();

        self.context.0.compute_pass_begin_pipeline_statistics_query(
            &mut self.pass,
            query_set.id,
            query_index,
        )
    }

    fn end_pipeline_statistics_query(&mut self) -> Result<(), ComputePassError> {
        self
            .context
            .0
            .compute_pass_end_pipeline_statistics_query(&mut self.pass)
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) -> Result<(), ComputePassError> {
        self
            .context
            .0
            .compute_pass_dispatch_workgroups(&mut self.pass, x, y, z)
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), ComputePassError> {
        let indirect_buffer = indirect_buffer.as_core();

        self.context.0.compute_pass_dispatch_workgroups_indirect(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
        )
    }

    fn end(&mut self) -> Result<(), ComputePassError> {
        self.context.0.compute_pass_end(&mut self.pass)
    }
}

impl Drop for CoreComputePass {
    fn drop(&mut self) {
        dispatch::ComputePassInterface::end(self).expect("Call ComputePass::end() to handle this error");
    }
}

impl dispatch::RenderPassInterface for CoreRenderPass {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchRenderPipeline) -> Result<(), wgc::command::RenderPassError> {
        let pipeline = pipeline.as_core();

        self
            .context
            .0
            .render_pass_set_pipeline(&mut self.pass, pipeline.id)
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) -> Result<(), wgc::command::RenderPassError> {
        let bg = bind_group.map(|bg| bg.as_core().id);

        self.context
            .0
            .render_pass_set_bind_group(&mut self.pass, index, bg, offsets)
    }

    fn set_index_buffer(
        &mut self,
        buffer: &dispatch::DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) -> Result<(), wgc::command::RenderPassError> {
        let buffer = buffer.as_core();

        self.context.0.render_pass_set_index_buffer(
            &mut self.pass,
            buffer.id,
            index_format,
            offset,
            size,
        )
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) -> Result<(), wgc::command::RenderPassError> {
        let buffer = buffer.as_core();

        self.context.0.render_pass_set_vertex_buffer(
            &mut self.pass,
            slot,
            buffer.id,
            offset,
            size,
        )
    }

    fn set_push_constants(&mut self, stages: crate::ShaderStages, offset: u32, data: &[u8]) -> Result<(), wgc::command::RenderPassError> {
        self.context
            .0
            .render_pass_set_push_constants(&mut self.pass, stages, offset, data)
    }

    fn set_blend_constant(&mut self, color: crate::Color) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_set_blend_constant(&mut self.pass, color)
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<(), wgc::command::RenderPassError> {
        self.context
            .0
            .render_pass_set_scissor_rect(&mut self.pass, x, y, width, height)
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), wgc::command::RenderPassError> {
        self.context.0.render_pass_set_viewport(
            &mut self.pass,
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        )
    }

    fn set_stencil_reference(&mut self, reference: u32) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_set_stencil_reference(&mut self.pass, reference)
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> Result<(), wgc::command::RenderPassError> {
        self.context.0.render_pass_draw(
            &mut self.pass,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        )
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) -> Result<(), wgc::command::RenderPassError> {
        self.context.0.render_pass_draw_indexed(
            &mut self.pass,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        )
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();

        self.context.0.render_pass_draw_indirect(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
        )
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();

        self.context.0.render_pass_draw_indexed_indirect(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
        )
    }

    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();

        self.context.0.render_pass_multi_draw_indirect(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
            count,
        )
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();

        self.context.0.render_pass_multi_draw_indexed_indirect(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
            count,
        )
    }

    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &dispatch::DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();
        let count_buffer = count_buffer.as_core();

        self.context.0.render_pass_multi_draw_indirect_count(
            &mut self.pass,
            indirect_buffer.id,
            indirect_offset,
            count_buffer.id,
            count_buffer_offset,
            max_count,
        )
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &dispatch::DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) -> Result<(), wgc::command::RenderPassError> {
        let indirect_buffer = indirect_buffer.as_core();
        let count_buffer = count_buffer.as_core();

        self
            .context
            .0
            .render_pass_multi_draw_indexed_indirect_count(
                &mut self.pass,
                indirect_buffer.id,
                indirect_offset,
                count_buffer.id,
                count_buffer_offset,
                max_count,
            )
    }

    fn insert_debug_marker(&mut self, label: &str) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_insert_debug_marker(&mut self.pass, label, 0)
    }

    fn push_debug_group(&mut self, group_label: &str) -> Result<(), wgc::command::RenderPassError> {
        self.context
            .0
            .render_pass_push_debug_group(&mut self.pass, group_label, 0)
    }

    fn pop_debug_group(&mut self) -> Result<(), wgc::command::RenderPassError> {
        self.context.0.render_pass_pop_debug_group(&mut self.pass)
    }

    fn write_timestamp(&mut self, query_set: &dispatch::DispatchQuerySet, query_index: u32) -> Result<(), wgc::command::RenderPassError> {
        let query_set = query_set.as_core();

        self.context
            .0
            .render_pass_write_timestamp(&mut self.pass, query_set.id, query_index)
    }

    fn begin_occlusion_query(&mut self, query_index: u32) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_begin_occlusion_query(&mut self.pass, query_index)
    }

    fn end_occlusion_query(&mut self) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_end_occlusion_query(&mut self.pass)
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        query_set: &dispatch::DispatchQuerySet,
        query_index: u32,
    ) -> Result<(), wgc::command::RenderPassError> {
        let query_set = query_set.as_core();

        self.context.0.render_pass_begin_pipeline_statistics_query(
            &mut self.pass,
            query_set.id,
            query_index,
        )
    }

    fn end_pipeline_statistics_query(&mut self) -> Result<(), wgc::command::RenderPassError> {
        self
            .context
            .0
            .render_pass_end_pipeline_statistics_query(&mut self.pass)
    }

    fn execute_bundles(
        &mut self,
        render_bundles: &mut dyn Iterator<Item = &dispatch::DispatchRenderBundle>,
    ) -> Result<(), wgc::command::RenderPassError> {
        let temp_render_bundles = render_bundles
            .map(|rb| rb.as_core().id)
            .collect::<SmallVec<[_; 4]>>();
        self
            .context
            .0
            .render_pass_execute_bundles(&mut self.pass, &temp_render_bundles)
    }

    fn end(&mut self) -> Result<(), wgc::command::RenderPassError> {
        self.context.0.render_pass_end(&mut self.pass)
    }
}

impl Drop for CoreRenderPass {
    fn drop(&mut self) {
        dispatch::RenderPassInterface::end(self).expect("Call CoreRenderPass::end() before dropping to handle this error");
    }
}

impl dispatch::RenderBundleEncoderInterface for CoreRenderBundleEncoder {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchRenderPipeline) {
        let pipeline = pipeline.as_core();

        wgpu_render_bundle_set_pipeline(&mut self.encoder, pipeline.id)
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) {
        let bg = bind_group.map(|bg| bg.as_core().id);

        unsafe {
            wgpu_render_bundle_set_bind_group(
                &mut self.encoder,
                index,
                bg,
                offsets.as_ptr(),
                offsets.len(),
            )
        }
    }

    fn set_index_buffer(
        &mut self,
        buffer: &dispatch::DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.as_core();

        self.encoder
            .set_index_buffer(buffer.id, index_format, offset, size)
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.as_core();

        wgpu_render_bundle_set_vertex_buffer(&mut self.encoder, slot, buffer.id, offset, size)
    }

    fn set_push_constants(&mut self, stages: crate::ShaderStages, offset: u32, data: &[u8]) {
        unsafe {
            wgpu_render_bundle_set_push_constants(
                &mut self.encoder,
                stages,
                offset,
                data.len().try_into().unwrap(),
                data.as_ptr(),
            )
        }
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        wgpu_render_bundle_draw(
            &mut self.encoder,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        )
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        wgpu_render_bundle_draw_indexed(
            &mut self.encoder,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        )
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        wgpu_render_bundle_draw_indirect(&mut self.encoder, indirect_buffer.id, indirect_offset)
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        wgpu_render_bundle_draw_indexed_indirect(
            &mut self.encoder,
            indirect_buffer.id,
            indirect_offset,
        )
    }

    fn finish(self, desc: &crate::RenderBundleDescriptor<'_>) -> Result<dispatch::DispatchRenderBundle, command::RenderBundleError>
    where
        Self: Sized,
    {
        let id = self.context.0.render_bundle_encoder_finish(
            self.encoder,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        )?;
        Ok(CoreRenderBundle { id }.into())
    }
}

impl dispatch::RenderBundleInterface for CoreRenderBundle {}

impl dispatch::SurfaceInterface for CoreSurface {
    fn get_capabilities(&self, adapter: &dispatch::DispatchAdapter) -> wgt::SurfaceCapabilities {
        let adapter = adapter.as_core();

        self.context
            .0
            .surface_get_capabilities(self.id, adapter.id)
            .unwrap_or_default()
    }

    fn configure(&self, device: &dispatch::DispatchDevice, config: &crate::SurfaceConfiguration) -> Result<(), present::ConfigureSurfaceError> {
        let device = device.as_core();

        self.context.0.surface_configure(self.id, device.id, config)?;

        *self.configured_device.lock() = Some(device.id);

        Ok(())
    }

    fn get_current_texture(
        &self,
    ) -> Result<(
        Option<dispatch::DispatchTexture>,
        crate::SurfaceStatus,
        dispatch::DispatchSurfaceOutputDetail,
    ), SurfaceError> {
        let output_detail = CoreSurfaceOutputDetail {
            context: self.context.clone(),
            surface_id: self.id,
        }
        .into();

        match self.context.0.surface_get_current_texture(self.id, None) {
            Ok(wgc::present::SurfaceOutput { status, texture_id }) => {
                let data = texture_id
                    .map(|id| CoreTexture {
                        context: self.context.clone(),
                        id,
                    })
                    .map(Into::into);

                Ok((data, status, output_detail))
            }
            Err(err) => Err(err)
        }
    }
}

impl Drop for CoreSurface {
    fn drop(&mut self) {
        self.context.0.surface_drop(self.id)
    }
}

impl dispatch::SurfaceOutputDetailInterface for CoreSurfaceOutputDetail {
    fn present(&self) {
        match self.context.0.surface_present(self.surface_id) {
            Ok(_status) => (),
            Err(err) => panic!("Surface::present: {err}"),
        }
    }

    fn texture_discard(&self) {
        match self.context.0.surface_texture_discard(self.surface_id) {
            Ok(_status) => (),
            Err(err) => panic!("Surface::discard_texture: {err}"),
        }
    }
}
impl Drop for CoreSurfaceOutputDetail {
    fn drop(&mut self) {
        // Discard gets called by the api struct

        // no-op
    }
}

impl dispatch::QueueWriteBufferInterface for CoreQueueWriteBuffer {
    fn slice(&self) -> &[u8] {
        panic!()
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        self.mapping.slice_mut()
    }
}
impl Drop for CoreQueueWriteBuffer {
    fn drop(&mut self) {
        // The api struct calls queue.write_staging_buffer

        // no-op
    }
}

impl dispatch::BufferMappedRangeInterface for CoreBufferMappedRange {
    #[inline]
    fn slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}
