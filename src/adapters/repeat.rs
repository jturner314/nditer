use crate::{
    CanMerge, IntoNdProducerWithShape, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat,
};
use ndarray::{Array, Axis, Dimension, IntoDimension};

/// Creates an instance that can be converted into a producer that clones
/// `value` repeatedly.
///
/// Note that the value is cloned in the order of iteration (which may be
/// arbitrary).
pub fn into_repeat<T>(value: T) -> IntoRepeat<T>
where
    T: Clone,
{
    IntoRepeat { value }
}

/// Creates a producer that clones `value` repeatedly.
///
/// Note that the value is cloned in the order of iteration (which may be
/// arbitrary).
pub fn repeat<T, E>(shape: E, value: T) -> Repeat<T, E::Dim>
where
    T: Clone,
    E: IntoDimension,
{
    Repeat {
        value,
        shape: shape.into_dimension(),
    }
}

/// An object that can be converted into a producer that clones a value
/// repeatedly.
pub struct IntoRepeat<T> {
    value: T,
}

impl<T, D> IntoNdProducerWithShape<D> for IntoRepeat<T>
where
    T: Clone,
    D: Dimension,
{
    type Item = T;
    type Producer = Repeat<T, D>;
    fn into_producer(self, shape: D) -> Self::Producer {
        Repeat {
            value: self.value,
            shape,
        }
    }
}

/// Creates a producer that clones a value repeatedly.
pub struct Repeat<T, D> {
    value: T,
    shape: D,
}

impl<T, D> Repeat<T, D> {
    /// Returns a reference to the value that this producer repeats.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Converts this producer into the value it repeats.
    pub fn into_value(self) -> T {
        self.value
    }
}

impl<T, D> NdProducer for Repeat<T, D>
where
    T: Clone,
    D: Dimension,
{
    type Item = T;
    type Source = Repeat<T, D>;

    fn into_source(self) -> Self::Source {
        self
    }

    fn collect_array(self) -> Array<T, D> {
        Array::from_elem(self.shape, self.value)
    }
}

impl<T, D> NdReshape for Repeat<T, D>
where
    D: Dimension,
{
    type Dim = D;

    fn shape(&self) -> D {
        self.shape.clone()
    }

    fn approx_abs_strides(&self) -> D {
        D::zeros(self.shape.ndim())
    }

    fn is_axis_ordered(&self, axis: Axis) -> bool {
        debug_assert!(axis.index() < self.shape.ndim());
        false
    }

    fn invert_axis(&mut self, axis: Axis) {
        assert!(axis.index() <= self.shape.ndim());
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        let t = take.index();
        let i = into.index();
        if t == i && self.shape[t] > 1 {
            CanMerge::Never
        } else {
            CanMerge::Always
        }
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        let t = take.index();
        let i = into.index();
        let prod = self.shape[t] * self.shape[i];
        self.shape[i] = prod;
        self.shape[t] = if prod == 0 { 0 } else { 1 };
    }

    fn len(&self) -> usize {
        self.shape.size()
    }

    fn len_of(&self, axis: Axis) -> usize {
        self.shape[axis.index()]
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}

impl<T, D> NdSource for Repeat<T, D>
where
    T: Clone,
    D: Dimension,
{
    type Item = T;
    unsafe fn read_once_unchecked(&mut self, _ptr: &()) -> T {
        self.value.clone()
    }
}

unsafe impl<T, D> NdAccess for Repeat<T, D>
where
    D: Dimension,
{
    type Dim = D;
    type Ptr = ();

    fn shape(&self) -> Self::Dim {
        self.shape.clone()
    }

    fn first_ptr(&self) -> Option<()> {
        if self.shape.size_checked().unwrap() == 0 {
            None
        } else {
            Some(())
        }
    }

    unsafe fn ptr_offset_axis(&self, _ptr: &mut Self::Ptr, _axis: Axis, _count: isize) {}

    unsafe fn ptr_offset_axis_contiguous(&self, _ptr: &mut Self::Ptr, _axis: Axis, _count: isize) {}

    fn is_axis_contiguous(&self, _axis: Axis) -> bool {
        true
    }

    fn len_of(&self, axis: Axis) -> usize {
        self.shape[axis.index()]
    }

    fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}

unsafe impl<T, D> NdSourceRepeat for Repeat<T, D>
where
    T: Clone,
    D: Dimension,
{
}
