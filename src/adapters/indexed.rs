use crate::{AxesMask, CanMerge, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::{Axis, Dimension, IxDyn};

/// A producer that yields the index along with the element.
///
/// This struct is created by the `indexed` method on `NdProducer`. See its
/// documentation for more.
pub struct IndexedProducer<T, D: Dimension> {
    inner: T,
    /// Which axes are inverted.
    is_inverted: AxesMask<D, IxDyn>,
}

/// A source that yields the index along with the element.
pub struct IndexedSource<T, D> {
    inner: T,
    index_strides: D,
    first_index: Option<D>,
}

impl<T, D> IndexedProducer<T, D>
where
    T: NdProducer<Dim = D>,
    D: Dimension,
{
    pub(crate) fn new(inner: T) -> Self {
        IndexedProducer {
            is_inverted: AxesMask::all_false(inner.ndim()).into_dyn_num_true(),
            inner,
        }
    }
}

impl<T, D> NdProducer for IndexedProducer<T, D>
where
    T: NdProducer<Dim = D>,
    D: Dimension,
{
    type Item = (D::Pattern, T::Item);
    type Source = IndexedSource<T::Source, D>;
    fn into_source(self) -> Self::Source {
        let inner = self.inner.into_source();
        let is_inverted = self.is_inverted;
        assert_eq!(inner.ndim(), is_inverted.for_ndim());
        let index_strides =
            is_inverted.mapv_to_dim(|is_inv| if is_inv { (-1isize) as usize } else { 1 });
        let mut first_index = D::first_index(&inner.shape());
        if let Some(ref mut idx) = first_index {
            is_inverted.indexed_visitv(|axis, is_inv| {
                if is_inv {
                    idx[axis.index()] = inner.len_of(axis) - 1;
                }
            });
        }
        IndexedSource {
            inner,
            index_strides,
            first_index,
        }
    }
}

// TODO: Prohibiting merging and permuting axes is one solution, but is it the
// best one?
impl<T, D> NdReshape for IndexedProducer<T, D>
where
    T: NdProducer<Dim = D>,
    D: Dimension,
{
    type Dim = D;
    fn shape(&self) -> Self::Dim {
        self.inner.shape()
    }
    fn approx_abs_strides(&self) -> Self::Dim {
        let mut strides = self.inner.approx_abs_strides();
        for s in strides.slice_mut() {
            // Add 1 for the cost of adjusting the index.
            *s += 1;
        }
        strides
    }
    fn is_axis_ordered(&self, axis: Axis) -> bool {
        self.inner.is_axis_ordered(axis)
    }
    fn invert_axis(&mut self, axis: Axis) {
        self.inner.invert_axis(axis);
        self.is_inverted.write(axis, !self.is_inverted.read(axis));
    }
    /// Always returns `CanMerge::Never`.
    fn can_merge_axes(&self, _take: Axis, _into: Axis) -> CanMerge {
        CanMerge::Never
    }
    /// **Always panics**
    fn merge_axes(&mut self, _take: Axis, _into: Axis) {
        panic!("Merging axes is not allowed for `IndexedProducer`");
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

impl<T, D> NdSource for IndexedSource<T, D>
where
    T: NdSource<Dim = D>,
    D: Dimension,
{
    type Item = (D::Pattern, T::Item);

    unsafe fn read_once_unchecked(&mut self, ptr: &(D, T::Ptr)) -> (D::Pattern, T::Item) {
        let (ref idx, ref ptr) = ptr;
        (
            idx.clone().into_pattern(),
            self.inner.read_once_unchecked(ptr),
        )
    }
}

unsafe impl<T, D> NdAccess for IndexedSource<T, D>
where
    T: NdAccess<Dim = D>,
    D: Dimension,
{
    type Dim = D;
    type Ptr = (D, T::Ptr);

    fn shape(&self) -> D {
        self.inner.shape()
    }

    fn first_ptr(&self) -> Option<(D, T::Ptr)> {
        match (self.first_index.clone(), self.inner.first_ptr()) {
            (Some(idx), Some(ptr)) => Some((idx, ptr)),
            (None, None) => None,
            _ => unreachable!(),
        }
    }

    unsafe fn ptr_offset_axis(&self, ptr: &mut (D, T::Ptr), axis: Axis, count: isize) {
        let (ref mut idx, ref mut ptr) = ptr;
        self.inner.ptr_offset_axis(ptr, axis, count);
        idx[axis.index()] = (idx[axis.index()] as isize
            + self.index_strides[axis.index()] as isize * count)
            as usize;
    }

    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut (D, T::Ptr), axis: Axis, count: isize) {
        let (ref mut idx, ref mut ptr) = ptr;
        self.inner.ptr_offset_axis_contiguous(ptr, axis, count);
        idx[axis.index()] = ((idx[axis.index()] as isize) + count) as usize;
    }

    fn is_axis_contiguous(&self, axis: Axis) -> bool {
        self.inner.is_axis_contiguous(axis) && self.index_strides[axis.index()] == 1
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

unsafe impl<T, D> NdSourceRepeat for IndexedSource<T, D>
where
    T: NdSourceRepeat<Dim = D>,
    D: Dimension,
{
}
