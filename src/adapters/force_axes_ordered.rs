use crate::{CanMerge, DimensionExt, IntoAxesFor, NdProducer, NdReshape};
use ndarray::{Axis, Dimension};

/// A wrapper that forces iteration to occur in order for each specified axis.
///
/// This struct is created by the `force_axes_ordered` method on `NdProducer`.
/// See its documentation for more.
pub struct ForceAxesOrdered<T>
where
    T: NdReshape,
{
    inner: T,
    /// Whether axes are forbidden to be inverted (0 means inverting is okay; 1
    /// means inverting is forbidden).
    force_ordered: T::Dim,
}

impl<T> ForceAxesOrdered<T>
where
    T: NdProducer,
{
    pub(crate) fn new(inner: T, ordered_axes: impl IntoAxesFor<T::Dim>) -> Self {
        let mut force_ordered = T::Dim::zeros(inner.ndim());
        let axes = ordered_axes.into_axes_for(inner.ndim()).into_inner();
        axes.visitv(|axis| force_ordered[axis] = 1);
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
        self.inner.is_axis_ordered(axis) || self.force_ordered[axis.index()] != 0
    }

    fn invert_axis(&mut self, axis: Axis) {
        assert!(
            self.force_ordered[axis.index()] == 0,
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
        if self.force_ordered[take.index()] != 0 {
            self.force_ordered[into.index()] = 1;
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
