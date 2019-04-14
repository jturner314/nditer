use crate::{CanMerge, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use itertools::izip;
use ndarray::{Axis, Dimension};

/// A producer that yields the index along with the element.
///
/// This struct is created by the `indexed` method on `NdProducer`. See its
/// documentation for more.
pub struct IndexedProducer<T, D> {
    inner: T,
    /// Which axes are inverted (0 for "not inverted", and 1 for "is inverted").
    is_inverted: D,
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
            is_inverted: D::zeros(inner.ndim()),
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
        assert_eq!(inner.ndim(), is_inverted.ndim());
        let mut index_strides = D::zeros(is_inverted.ndim());
        for (stride, &is_inv) in izip!(index_strides.slice_mut(), is_inverted.slice()) {
            *stride = if is_inv != 0 { (-1isize) as usize } else { 1 };
        }
        let mut first_index = D::first_index(&inner.shape());
        if let Some(ref mut idx) = first_index {
            for (ax, &is_inverted) in is_inverted.slice().iter().enumerate() {
                if is_inverted != 0 {
                    idx[ax] = inner.len_of(Axis(ax)) - 1;
                }
            }
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
        self.is_inverted[axis.index()] = if self.is_inverted[axis.index()] != 0 {
            0
        } else {
            1
        };
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
