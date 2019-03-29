use crate::{CanMerge, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::{ArrayView1, Axis};
use std::cmp;

/// A producer that selects specific indices along an axis.
///
/// This struct is created by the `select_indices_axis` method on `NdProducer`.
/// See its documentation for more.
pub struct SelectIndicesAxis<'a, T> {
    /// The inner producer.
    inner: T,
    /// The axis the apply the indices to.
    axis: Axis,
    /// The indices selected along the axis.
    ///
    /// Note that it would be possible to use a producer instead, but that
    /// would require a bounds-check during iteration every time the pointer is
    /// offset along `self.axis`. (`NdSourceRepeat` doesn't guarantee that the
    /// same value is always returned for the same location, so it wouldn't be
    /// sufficient to pre-check that all the indices are in-bounds in
    /// `NdProducer::into_source`.)
    indices: ArrayView1<'a, usize>,
}

impl<'a, T> SelectIndicesAxis<'a, T> {
    pub(crate) fn new(inner: T, axis: Axis, indices: ArrayView1<'a, usize>) -> Self {
        SelectIndicesAxis {
            inner,
            axis,
            indices,
        }
    }
}

impl<'a, T> NdProducer for SelectIndicesAxis<'a, T>
where
    T: NdProducer,
    T::Source: NdSourceRepeat,
{
    type Item = T::Item;
    type Source = SelectIndicesAxis<'a, T::Source>;
    fn into_source(self) -> Self::Source {
        let inner = self.inner.into_source();
        let axis = self.axis;
        let indices = self.indices;
        // Check that all indices are in-bounds and can be cast to isize
        // without overflow.
        let axis_len = inner.len_of(axis);
        for &index in &self.indices {
            assert!(index < axis_len && index <= std::isize::MAX as usize);
        }
        SelectIndicesAxis {
            inner,
            axis,
            indices,
        }
    }
}

impl<'a, T> NdReshape for SelectIndicesAxis<'a, T>
where
    T: NdReshape,
{
    type Dim = T::Dim;

    fn shape(&self) -> Self::Dim {
        let mut shape = self.inner.shape();
        shape[self.axis.index()] = self.indices.len();
        shape
    }

    fn approx_abs_strides(&self) -> Self::Dim {
        let mut strides = self.inner.approx_abs_strides();
        // TODO: Experimentally determine a good heuristic.
        // Multiply indexed axis stride by average stride between indices,
        // assuming they're in-order and evenly distributed along the entire
        // axis length.
        strides[self.axis.index()] *=
            cmp::max(1, self.inner.len_of(self.axis) / self.indices.len());
        strides
    }

    fn can_invert_axis(&self, axis: Axis) -> bool {
        axis == self.axis || self.inner.can_invert_axis(axis)
    }

    fn invert_axis(&mut self, axis: Axis) {
        if axis == self.axis {
            self.indices.invert_axis(Axis(0))
        } else {
            self.inner.invert_axis(axis)
        }
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        if take == self.axis || into == self.axis {
            CanMerge::Never
        } else {
            self.inner.can_merge_axes(take, into)
        }
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        assert_ne!(self.axis, take);
        assert_ne!(self.axis, into);
        self.inner.merge_axes(take, into);
    }

    fn len_of(&self, axis: Axis) -> usize {
        if axis == self.axis {
            self.indices.len()
        } else {
            self.inner.len_of(axis)
        }
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.inner.is_empty()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
}

impl<'a, T> NdSource for SelectIndicesAxis<'a, T>
where
    T: NdSourceRepeat,
{
    type Item = T::Item;

    unsafe fn read_once_unchecked(&mut self, ptr: &Self::Ptr) -> T::Item {
        let (_, ref p) = ptr;
        // It's safe to call `read_once_unchecked` multiple times for the same
        // index of the inner producer because `T: NdSourceRepeat`. (This
        // property is necessary because `SelectIndicesAxis` doesn't check that
        // the selected indices are unique.)
        self.inner.read_once_unchecked(p)
    }
}

unsafe impl<'a, T> NdAccess for SelectIndicesAxis<'a, T>
where
    T: NdAccess,
{
    type Dim = T::Dim;

    /// Pointer into `self.indices` and pointer for inner source.
    type Ptr = (usize, T::Ptr);

    fn shape(&self) -> Self::Dim {
        let mut shape = self.inner.shape();
        shape[self.axis.index()] = self.indices.len();
        shape
    }

    fn first_ptr(&self) -> Option<(usize, T::Ptr)> {
        if self.indices.is_empty() {
            None
        } else {
            self.inner.first_ptr().map(|ptr| (0, ptr))
        }
    }

    unsafe fn ptr_offset_axis(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (ref mut i, ref mut p) = ptr;
        if axis == self.axis {
            let old_index = *self.indices.uget(*i);
            *i = (*i as isize + count) as usize;
            let new_index = *self.indices.uget(*i);
            let inner_count = new_index as isize - old_index as isize;
            self.inner.ptr_offset_axis(p, axis, inner_count);
        } else {
            self.inner.ptr_offset_axis(p, axis, count);
        }
    }

    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (_, ref mut p) = ptr;
        // We ensure in `in_axis_contiguous` that `axis != self.axis`.
        self.inner.ptr_offset_axis_contiguous(p, axis, count);
    }

    fn is_axis_contiguous(&self, axis: Axis) -> bool {
        axis != self.axis && self.inner.is_axis_contiguous(axis)
    }

    fn len_of(&self, axis: Axis) -> usize {
        if axis == self.axis {
            self.indices.len()
        } else {
            self.inner.len_of(axis)
        }
    }

    fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.inner.is_empty()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
}

unsafe impl<'a, T> NdSourceRepeat for SelectIndicesAxis<'a, T> where T: NdSourceRepeat {}
