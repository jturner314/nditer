use crate::{CanMerge, NdProducer, NdReshape};
use ndarray::{Axis, Dimension};

/// A wrapper that prevents optimization from inverting the specified axes of
/// the inner producer.
///
/// This struct is created by the `forbid_invert_axes` method on `NdProducer`.
/// See its documentation for more.
pub struct ForbidInvertAxes<T>
where
    T: NdReshape,
{
    inner: T,
    /// Whether axes are forbidden to be inverted (0 means inverting is okay; 1
    /// means inverting is forbidden).
    forbid_invert: T::Dim,
}

impl<T> ForbidInvertAxes<T>
where
    T: NdProducer,
{
    pub(crate) fn new(inner: T, forbid_invert_axes: impl IntoIterator<Item = usize>) -> Self {
        let mut forbid_invert = T::Dim::zeros(inner.ndim());
        for axis in forbid_invert_axes {
            forbid_invert[axis] = 1;
        }
        ForbidInvertAxes {
            inner,
            forbid_invert,
        }
    }
}

impl<T> NdProducer for ForbidInvertAxes<T>
where
    T: NdProducer,
{
    type Item = T::Item;
    type Source = T::Source;
    fn into_source(self) -> Self::Source {
        self.inner.into_source()
    }
}

impl<T> NdReshape for ForbidInvertAxes<T>
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

    fn can_invert_axis(&self, axis: Axis) -> bool {
        self.inner.can_invert_axis(axis) && self.forbid_invert[axis.index()] == 0
    }

    fn invert_axis(&mut self, axis: Axis) {
        assert!(
            self.forbid_invert[axis.index()] == 0,
            "Inverting axis {} is forbidden",
            axis.index(),
        );
        self.inner.invert_axis(axis)
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        self.inner.can_merge_axes(take, into)
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        // Propagate inversion constraint to `into` axis because movement along
        // `take` axis becomes movement along the `into` axis after merging.
        if self.forbid_invert[take.index()] != 0 {
            self.forbid_invert[into.index()] = 1;
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
