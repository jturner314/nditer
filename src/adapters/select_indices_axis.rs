use crate::{CanMerge, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::{ArrayView1, Axis};
use std::cmp;

/// Wrapper for a view of indices that keeps track of the maximum index value.
///
/// This provides a clean way to pre-compute and store the maximum index so
/// that it doesn't need to be managed by `SelectIndicesAxis`. (It's helpful to
/// pre-compute the maximum index so that `SelectIndicesAxis::into_source()` is
/// fast in case the user wants to clone the `SelectIndicesAxis` producer and
/// consume it multiple times.)
#[derive(Clone, Debug)]
struct Indices<'a> {
    /// The array of indices.
    indices: ArrayView1<'a, usize>,
    /// The maximum value of `indices`, or `None` iff `indices` is empty.
    max_index: Option<usize>,
}

impl<'a> Indices<'a> {
    /// Creates a new instance of `Indices` containing the `indices`.
    pub fn new(indices: ArrayView1<'a, usize>) -> Indices<'a> {
        let max_index = if indices.is_empty() {
            None
        } else {
            Some(indices.fold(0, |acc, &x| acc.max(x)))
        };
        Indices { indices, max_index }
    }

    /// Returns `true` iff no index is `>= axis_len`.
    pub fn are_in_bounds(&self, axis_len: usize) -> bool {
        match self.max_index {
            None => true,
            Some(max) if max < axis_len => true,
            _ => false,
        }
    }

    /// Returns the number of indices.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` iff there aren't any indices.
    pub fn is_empty(&self) -> bool {
        self.max_index.is_none()
    }

    /// Inverts the order of the indices.
    pub fn invert_axis(&mut self) {
        self.indices.invert_axis(Axis(0))
    }

    /// Gets the index value located at `index` in the sequence of indices.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is less than `self.len()`.
    pub unsafe fn uget(&self, index: usize) -> &usize {
        self.indices.uget(index)
    }
}

/// A producer that selects specific indices along an axis.
///
/// This struct is created by the `select_indices_axis` method on `NdProducer`.
/// See its documentation for more.
#[derive(Clone, Debug)]
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
    indices: Indices<'a>,
}

impl<'a, T> SelectIndicesAxis<'a, T> {
    pub(crate) fn new(inner: T, axis: Axis, indices: ArrayView1<'a, usize>) -> Self {
        SelectIndicesAxis {
            inner,
            axis,
            indices: Indices::new(indices),
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
        assert!(indices.are_in_bounds(cmp::min(inner.len_of(axis), std::isize::MAX as usize + 1)));
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
            self.indices.invert_axis()
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
