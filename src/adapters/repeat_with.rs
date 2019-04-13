use crate::{
    CanMerge, IntoNdProducerWithShape, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat,
};
use ndarray::{Axis, Dimension, IntoDimension};

/// Creates an instance that can be converted into a producer that calls
/// `repeater` to compute each element.
///
/// **Note**:
///
/// * The closure is called in the order of iteration (which may be arbitrary).
///
/// * If the producer is split for parallelization, the closure will be cloned.
///   So, if you're generating random numbers in the closure, you probably want
///   to wrap your RNG in `ReseedingRng`.
///
/// * The `NdSourceRepeat` implementation for the producer requires that `F:
///   Fn` but assumes that it will always return the same value. In the rare
///   case that `F: Fn` but does not always return the same value (e.g. if it's
///   mutating a `Cell` or `RefCell`), you should avoid using the
///   `NdSourceRepeat` implementation (e.g. for broadcasting).
pub fn into_repeat_with<A, F>(repeater: F) -> IntoRepeatWith<F>
where
    F: FnMut() -> A,
{
    IntoRepeatWith { repeater }
}

/// Creates a producer that calls `repeater` to compute each element.
///
/// **Note**:
///
/// * The closure is called in the order of iteration (which may be arbitrary).
///
/// * If the producer instance is split for parallelization, the closure will
///   be cloned. So, if you're generating random numbers in the closure, you
///   probably want to wrap your RNG in `ReseedingRng`.
///
/// * The `NdSourceRepeat` implementation for the producer requires that `F:
///   Fn` but assumes that it will always return the same value. In the rare
///   case that `F: Fn` but does not always return the same value (e.g. if it's
///   mutating a `Cell` or `RefCell`), you should avoid using the
///   `NdSourceRepeat` implementation (e.g. for broadcasting).
pub fn repeat_with<A, F, E>(shape: E, repeater: F) -> RepeatWith<F, E::Dim>
where
    F: FnMut() -> A,
    E: IntoDimension,
{
    RepeatWith {
        repeater,
        shape: shape.into_dimension(),
    }
}

/// An object that can be converted into a producer that calls a closure to
/// compute each element.
pub struct IntoRepeatWith<F> {
    repeater: F,
}

impl<F, A, D> IntoNdProducerWithShape<D> for IntoRepeatWith<F>
where
    F: FnMut() -> A,
    D: Dimension,
{
    type Item = A;
    type Producer = RepeatWith<F, D>;
    fn into_producer(self, shape: D) -> Self::Producer {
        RepeatWith {
            repeater: self.repeater,
            shape,
        }
    }
}

/// A producer that generates elements by calling a closure.
pub struct RepeatWith<F, D> {
    repeater: F,
    shape: D,
}

impl<F, A, D> NdProducer for RepeatWith<F, D>
where
    F: FnMut() -> A,
    D: Dimension,
{
    type Item = A;
    type Source = RepeatWith<F, D>;
    fn into_source(self) -> Self::Source {
        self
    }
}

impl<F, D> NdReshape for RepeatWith<F, D>
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

    fn can_invert_axis(&self, axis: Axis) -> bool {
        debug_assert!(axis.index() < self.shape.ndim());
        true
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

impl<F, A, D> NdSource for RepeatWith<F, D>
where
    F: FnMut() -> A,
    D: Dimension,
{
    type Item = A;
    unsafe fn read_once_unchecked(&mut self, _ptr: &()) -> A {
        (self.repeater)()
    }
}

unsafe impl<F, D> NdAccess for RepeatWith<F, D>
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

unsafe impl<F, A, D> NdSourceRepeat for RepeatWith<F, D>
where
    F: Fn() -> A,
    D: Dimension,
{
}
