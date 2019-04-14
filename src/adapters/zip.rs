use crate::{CanMerge, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat};
use ndarray::Axis;

/// A producer that operates over two producers simultaneously.
///
/// This struct is created by the `zip` method on `NdProducer`. See its
/// documentation for more.
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Zip<A, B>
where
    A: NdReshape,
    B: NdReshape<Dim = A::Dim>,
{
    pub(crate) fn new(a: A, b: B) -> Self {
        // TODO: allow broadcasting
        assert_eq!(
            a.shape(),
            b.shape(),
            "Zipped producers must have the same shape."
        );
        Zip { a, b }
    }
}

impl<A, B> NdProducer for Zip<A, B>
where
    A: NdProducer,
    B: NdProducer<Dim = A::Dim>,
{
    type Item = (A::Item, B::Item);
    type Source = Zip<A::Source, B::Source>;
    fn into_source(self) -> Self::Source {
        let a = self.a.into_source();
        let b = self.b.into_source();
        assert_eq!(a.shape(), b.shape());
        Zip { a, b }
    }
}

impl<A, B> NdReshape for Zip<A, B>
where
    A: NdReshape,
    B: NdReshape<Dim = A::Dim>,
{
    type Dim = A::Dim;
    fn shape(&self) -> Self::Dim {
        debug_assert_eq!(self.a.shape(), self.b.shape());
        self.a.shape()
    }
    fn approx_abs_strides(&self) -> Self::Dim {
        let a = self.a.approx_abs_strides();
        let b = self.b.approx_abs_strides();
        a + b
    }
    fn is_axis_ordered(&self, axis: Axis) -> bool {
        self.a.is_axis_ordered(axis) || self.b.is_axis_ordered(axis)
    }
    fn invert_axis(&mut self, axis: Axis) {
        self.a.invert_axis(axis);
        self.b.invert_axis(axis);
    }
    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        self.a.can_merge_axes(take, into) & self.b.can_merge_axes(take, into)
    }
    fn merge_axes(&mut self, take: Axis, into: Axis) {
        self.a.merge_axes(take, into);
        self.b.merge_axes(take, into);
        debug_assert_eq!(self.a.shape(), self.b.shape());
    }
    fn len(&self) -> usize {
        debug_assert_eq!(self.a.len(), self.b.len());
        self.a.len()
    }
    fn len_of(&self, axis: Axis) -> usize {
        debug_assert_eq!(self.a.len_of(axis), self.b.len_of(axis));
        self.a.len_of(axis)
    }
    fn is_empty(&self) -> bool {
        debug_assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }
    fn ndim(&self) -> usize {
        debug_assert_eq!(self.a.ndim(), self.b.ndim());
        self.a.ndim()
    }
}

impl<A, B> NdSource for Zip<A, B>
where
    A: NdSource,
    B: NdSource<Dim = A::Dim>,
{
    type Item = (A::Item, B::Item);
    unsafe fn read_once_unchecked(&mut self, ptr: &Self::Ptr) -> Self::Item {
        let (ref ptr_a, ref ptr_b) = ptr;
        (
            self.a.read_once_unchecked(ptr_a),
            self.b.read_once_unchecked(ptr_b),
        )
    }
}

unsafe impl<A, B> NdAccess for Zip<A, B>
where
    A: NdAccess,
    B: NdAccess<Dim = A::Dim>,
{
    type Dim = A::Dim;
    type Ptr = (A::Ptr, B::Ptr);

    fn shape(&self) -> Self::Dim {
        debug_assert_eq!(self.a.shape(), self.b.shape());
        self.a.shape()
    }

    fn first_ptr(&self) -> Option<Self::Ptr> {
        match (self.a.first_ptr(), self.b.first_ptr()) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => unreachable!("One of the sources/sinks implementations' is broken."),
        }
    }

    unsafe fn ptr_offset_axis(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (ref mut ptr_a, ref mut ptr_b) = ptr;
        self.a.ptr_offset_axis(ptr_a, axis, count);
        self.b.ptr_offset_axis(ptr_b, axis, count);
    }

    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (ref mut ptr_a, ref mut ptr_b) = ptr;
        self.a.ptr_offset_axis_contiguous(ptr_a, axis, count);
        self.b.ptr_offset_axis_contiguous(ptr_b, axis, count);
    }

    fn is_axis_contiguous(&self, axis: Axis) -> bool {
        self.a.is_axis_contiguous(axis) && self.b.is_axis_contiguous(axis)
    }

    fn len_of(&self, axis: Axis) -> usize {
        debug_assert_eq!(self.a.len_of(axis), self.b.len_of(axis));
        self.a.len_of(axis)
    }

    fn is_empty(&self) -> bool {
        debug_assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }

    fn ndim(&self) -> usize {
        debug_assert_eq!(self.a.ndim(), self.b.ndim());
        self.a.ndim()
    }
}

unsafe impl<A, B> NdSourceRepeat for Zip<A, B>
where
    A: NdSourceRepeat,
    B: NdSourceRepeat<Dim = A::Dim>,
{
}
