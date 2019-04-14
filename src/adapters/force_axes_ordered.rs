use crate::{AxesMask, CanMerge, IntoAxesFor, NdProducer, NdReshape};
use ndarray::{Axis, IxDyn};

/// A wrapper that forces iteration to occur in order for each specified axis.
///
/// This struct is created by the `force_axes_ordered` method on `NdProducer`.
/// See its documentation for more.
pub struct ForceAxesOrdered<T>
where
    T: NdReshape,
{
    inner: T,
    /// Whether each axis must be iterated in order.
    force_ordered: AxesMask<T::Dim, IxDyn>,
}

impl<T> ForceAxesOrdered<T>
where
    T: NdProducer,
{
    pub(crate) fn new(inner: T, ordered_axes: impl IntoAxesFor<T::Dim>) -> Self {
        let force_ordered = ordered_axes
            .into_axes_mask(inner.ndim())
            .into_dyn_num_true();
        ForceAxesOrdered {
            inner,
            force_ordered,
        }
    }
}

impl<T> NdProducer for ForceAxesOrdered<T>
where
    T: NdProducer,
{
    type Item = T::Item;
    type Source = T::Source;
    fn into_source(self) -> Self::Source {
        self.inner.into_source()
    }
}

impl<T> NdReshape for ForceAxesOrdered<T>
where
    T: NdReshape,
{
    type Dim = T::Dim;

    fn shape(&self) -> Self::Dim {
        self.inner.shape()
    }

    fn approx_abs_strides(&self) -> Self::Dim {
        self.inner.approx_abs_strides()
    }

    fn is_axis_ordered(&self, axis: Axis) -> bool {
        self.inner.is_axis_ordered(axis) || self.force_ordered.read(axis)
    }

    fn invert_axis(&mut self, axis: Axis) {
        assert!(
            !self.force_ordered.read(axis),
            "Axis {} must be iterated in order",
            axis.index(),
        );
        self.inner.invert_axis(axis)
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        self.inner.can_merge_axes(take, into)
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        // Propagate ordered constraint to `into` axis because movement along
        // `take` axis becomes movement along the `into` axis after merging.
        if self.force_ordered.read(take) {
            self.force_ordered.write(into, true);
        }
        self.inner.merge_axes(take, into)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn len_of(&self, axis: Axis) -> usize {
        self.inner.len_of(axis)
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
}
