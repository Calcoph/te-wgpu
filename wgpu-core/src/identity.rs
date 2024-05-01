use parking_lot::Mutex;
use wgt::Backend;

use crate::{
    id::{self, TypedId},
    Epoch, Index,
};
use std::{fmt::Debug, marker::PhantomData, sync::Arc};

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`free`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// Use `IdentityManager::default` to construct new instances.
///
/// `IdentityManager` returns `Id`s whose index values are suitable for use as
/// indices into a `Storage<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Each live id's index selects a
///   distinct element in the vector.
///
/// - `IdentityManager` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityManager` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// See the module-level documentation for an overview of how this
/// fits together.
///
/// [`Id`]: crate::id::Id
/// [`Backend`]: wgt::Backend;
/// [`alloc`]: IdentityManager::alloc
/// [`free`]: IdentityManager::free
#[derive(Debug)]
pub(super) struct IdentityValues {
    free: Vec<(Index, Epoch)>,
    next_index: Index,
    count: usize,
}

impl IdentityValues {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<I: TypedId>(&mut self, backend: Backend) -> I {
        self.count += 1;
        match self.free.pop() {
            Some((index, epoch)) => I::zip(index, epoch + 1, backend),
            None => {
                let index = self.next_index;
                self.next_index += 1;
                let epoch = 1;
                I::zip(index, epoch, backend)
            }
        }
    }

    pub fn mark_as_used<I: TypedId>(&mut self, id: I) -> I {
        self.count += 1;
        id
    }

    /// Free `id`. It will never be returned from `alloc` again.
    pub fn release<I: TypedId>(&mut self, id: I) {
        let (index, epoch, _backend) = id.unzip();
        self.free.push((index, epoch));
        self.count -= 1;
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug)]
pub struct IdentityManager<I: TypedId> {
    pub(super) values: Mutex<IdentityValues>,
    _phantom: PhantomData<I>,
}

impl<I: TypedId> IdentityManager<I> {
    pub fn process(&self, backend: Backend) -> I {
        self.values.lock().alloc(backend)
    }
    pub fn mark_as_used(&self, id: I) -> I {
        self.values.lock().mark_as_used(id)
    }
    pub fn free(&self, id: I) {
        self.values.lock().release(id)
    }
}

impl<I: TypedId> IdentityManager<I> {
    pub fn new() -> Self {
        Self {
            values: Mutex::new(IdentityValues {
                free: Vec::new(),
                next_index: 0,
                count: 0,
            }),
            _phantom: PhantomData,
        }
    }
}

/// A type that can produce [`IdentityManager`] filters for ids of type `I`.
///
/// See the module-level documentation for details.
pub trait IdentityHandlerFactory<I: TypedId> {
    type Input: Copy;
    /// Create an [`IdentityManager<I>`] implementation that can
    /// transform proto-ids into ids of type `I`.
    /// It can return None if ids are passed from outside
    /// and are not generated by wgpu
    ///
    /// [`IdentityManager<I>`]: IdentityManager
    fn spawn(&self) -> Arc<IdentityManager<I>> {
        Arc::new(IdentityManager::new())
    }
    fn autogenerate_ids() -> bool;
    fn input_to_id(id_in: Self::Input) -> I;
}

/// A global identity handler factory based on [`IdentityManager`].
///
/// Each of this type's `IdentityHandlerFactory<I>::spawn` methods
/// returns a `Mutex<IdentityManager<I>>`, which allocates fresh `I`
/// ids itself, and takes `()` as its proto-id type.
#[derive(Debug)]
pub struct IdentityManagerFactory;

impl<I: TypedId> IdentityHandlerFactory<I> for IdentityManagerFactory {
    type Input = ();
    fn autogenerate_ids() -> bool {
        true
    }

    fn input_to_id(_id_in: Self::Input) -> I {
        unreachable!("It should not be called")
    }
}

/// A factory that can build [`IdentityManager`]s for all resource
/// types.
pub trait GlobalIdentityHandlerFactory:
    IdentityHandlerFactory<id::AdapterId>
    + IdentityHandlerFactory<id::DeviceId>
    + IdentityHandlerFactory<id::PipelineLayoutId>
    + IdentityHandlerFactory<id::ShaderModuleId>
    + IdentityHandlerFactory<id::BindGroupLayoutId>
    + IdentityHandlerFactory<id::BindGroupId>
    + IdentityHandlerFactory<id::CommandBufferId>
    + IdentityHandlerFactory<id::RenderBundleId>
    + IdentityHandlerFactory<id::RenderPipelineId>
    + IdentityHandlerFactory<id::ComputePipelineId>
    + IdentityHandlerFactory<id::QuerySetId>
    + IdentityHandlerFactory<id::BufferId>
    + IdentityHandlerFactory<id::StagingBufferId>
    + IdentityHandlerFactory<id::TextureId>
    + IdentityHandlerFactory<id::TextureViewId>
    + IdentityHandlerFactory<id::SamplerId>
    + IdentityHandlerFactory<id::SurfaceId>
{
}

impl GlobalIdentityHandlerFactory for IdentityManagerFactory {}

pub type Input<G, I> = <G as IdentityHandlerFactory<I>>::Input;

#[test]
fn test_epoch_end_of_life() {
    use id::TypedId as _;
    let man = IdentityManager::<id::BufferId>::new();
    let forced_id = man.mark_as_used(id::BufferId::zip(0, 1, Backend::Empty));
    assert_eq!(forced_id.unzip().0, 0);
    let id1 = man.process(Backend::Empty);
    assert_eq!(id1.unzip(), (0, 1, Backend::Empty));
    man.free(id1);
    let id2 = man.process(Backend::Empty);
    // confirm that the epoch 1 is no longer re-used
    assert_eq!(id2.unzip(), (0, 2, Backend::Empty));
}
