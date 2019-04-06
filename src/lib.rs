//! Experimental, unstable crate for high-performance iteration over
//! n-dimensional arrays. [`nditer::NdProducer`](trait.NdProducer.html) is
//! intended to be a replacement for `ndarray::NdProducer` that provides
//!
//! * more sophisticated optimizations for better performance,
//!
//! * more adapters other than just zipping arrays together, and
//!
//! * the ability to collect the results into an array instead of having to
//!   always manually pre-allocate the result array.

#![deny(missing_docs)]

pub use self::dim_traits::SubDim;
pub use self::impl_ndarray::ArrayBaseExt;
pub use adapters::{into_repeat_with, repeat_with};

use self::adapters::{
    BroadcastProducer, Cloned, FoldAxesProducer, ForbidInvertAxes, IndexedProducer, Inspect, Map,
    SelectIndicesAxis, Zip,
};
use itertools::izip;
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, IntoDimension, Ix1, ShapeBuilder};

/// Conversion into an `NdProducer`.
pub trait IntoNdProducer {
    /// The dimension type of the shape of the resulting producer.
    type Dim: Dimension;
    /// The type of elements being produced.
    type Item;
    /// The type of the resulting producer.
    type Producer: NdProducer<Dim = Self::Dim, Item = Self::Item>;
    /// Converts `self` into a producer.
    fn into_producer(self) -> Self::Producer;
}

/// Conversion into an `NdProducer` with the given shape.
pub trait IntoNdProducerWithShape<D: Dimension> {
    /// The element type of the resulting producer.
    type Item;
    /// The type of the resulting producer.
    type Producer: NdProducer<Dim = D, Item = Self::Item>;
    /// Converts `self` into a producer of the given shape.
    ///
    /// **Panics** if the given shape cannot be used.
    // TODO: don't panic on shape issues; return Option/Result
    fn into_producer(self, shape: D) -> Self::Producer;
}

impl<P> IntoNdProducer for P
where
    P: NdProducer,
{
    type Dim = P::Dim;
    type Item = P::Item;
    type Producer = Self;
    fn into_producer(self) -> Self {
        self
    }
}

impl<P> IntoNdProducerWithShape<P::Dim> for P
where
    P: NdProducer,
{
    type Item = <Self as NdProducer>::Item;
    type Producer = Self;
    /// Performs the conversion.
    ///
    /// **Panics** if `self.shape() != shape`.
    fn into_producer(self, shape: P::Dim) -> Self {
        assert_eq!(
            self.shape(),
            shape,
            "The `shape` given to `into_producer` must match the shape of the producer.",
        );
        self
    }
}

/// A producer of n-dimensional sources.
///
/// This trait provides methods for creating new producers, consuming
/// producers, and converting into iterators. It also provides the necessary
/// methods to optimize the producer for iteration.
///
/// Unlike the standard library's `Iterator` trait, iteration is not performed
/// directly on `NdProducer` but rather by converting the producer into a
/// source with `.into_source()`, getting a pointer with `.first_ptr()`, moving
/// the pointer with `.ptr_offset_axis()`/`.ptr_offset_axis_contiguous()`, and
/// getting items with `.get_once_unchecked()`. The producer does not contain
/// any iteration state; it just contains the configuration for creating the
/// source.
///
/// Usually, you will not need to interact with the source or pointer directly.
/// Instead, it's simpler to convert the producer to an iterator with
/// `.into_iter()` or `.into_iter_any_ord()` and then operate on the iterator,
/// or use one of the consumer methods directly (e.g. `.fold()`, `.for_each()`,
/// etc.).
pub trait NdProducer: NdReshape + Sized {
    /// The type of elements being produced.
    type Item;
    /// The type of the resulting source.
    type Source: NdSource<Item = Self::Item, Dim = Self::Dim>;

    /// Converts the producer into a source.
    #[must_use = "A source must be consumed for it to do anything useful."]
    fn into_source(self) -> Self::Source;

    /// Converts the producer into an iterator in standard (row-major) order.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::{array, Array2, ShapeBuilder};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let mut arr = Array2::zeros((2, 2).f());
    /// arr.assign(&array![[1, 2], [3, 4]]);
    /// let elems: Vec<_> = arr.producer().into_iter().cloned().collect();
    /// assert_eq!(elems, vec![1, 2, 3, 4]);
    /// # }
    /// ```
    fn into_iter(mut self) -> Iter<Self> {
        let axes = optimize::optimize_same_ord(&mut self);
        Iter::new(self, axes)
    }

    /// Converts the producer into an iterator, where iteration order doesn't
    /// matter.
    ///
    /// For example, this method may merge, reorder, and invert axes to improve
    /// performance. Prooducers zipped together using `.zip()` will still have
    /// their elements paired correctly, but the iteration order may be
    /// arbitrary.
    fn into_iter_any_ord(mut self) -> Iter<Self> {
        let axes = optimize::optimize_any_ord(&mut self);
        Iter::new(self, axes)
    }

    /// Prevents any methods from inverting the specified axes.
    ///
    /// This is useful when you want iteration to move in order along some axes
    /// (you want to prevent them from being inverted), but you don't care
    /// about iteration order across axes or iteration order along any other
    /// axes.
    ///
    /// # Example
    ///
    /// The optimizer usually inverts axes when doing so allows axes to be
    /// merged, so for demonstration, let's construct an array such that
    /// inverting an axis will allow the axes to be merged.
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::{array, Axis};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let mut a = array![[3, 4], [1, 2]];
    /// a.invert_axis(Axis(0));
    /// assert_eq!(a, array![[1, 2], [3, 4]]);
    /// assert_eq!(a.strides(), &[-2, 1]);
    /// # }
    /// ```
    ///
    /// Without any constraints, the optimizer inverts axis 0 in this case so
    /// that the axes can be merged and iteration can proceed in memory order.
    /// (Note that this iteration order is not guaranteed.) Observe that since
    /// the optimizer has inverted axis 0, element `3` occurs before element
    /// `1`, and element `4` occurs before element `2`.
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// # use ndarray::{array, Axis};
    /// # use nditer::{ArrayBaseExt, NdProducer};
    /// #
    /// # fn main() {
    /// # let mut a = array![[3, 4], [1, 2]];
    /// # a.invert_axis(Axis(0));
    /// # assert_eq!(a, array![[1, 2], [3, 4]]);
    /// #
    /// let allow_invert: Vec<_> = a
    ///     .producer()
    ///     .cloned()
    ///     .into_iter_any_ord()
    ///     .collect();
    /// assert_eq!(allow_invert, vec![3, 4, 1, 2]);
    /// # }
    /// ```
    ///
    /// Let's forbid axis 0 from being inverted. In this case, the optimizer
    /// finds that while it can't invert axis 0, inverting axis 1 is beneficial
    /// because that allows the axes to be merged and iteration can proceed in
    /// (reverse) memory order. (Note that other than not inverting axis, the
    /// iteration order is not guaranteed.) Observe that since we've forbidden
    /// inversion of axis 0, element `1` is guaranteed to occur before element
    /// `3`, and element `2` is guaranteed to occur before element `4`.
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// # use ndarray::{array, Axis};
    /// # use nditer::{ArrayBaseExt, NdProducer};
    /// #
    /// # fn main() {
    /// # let mut a = array![[3, 4], [1, 2]];
    /// # a.invert_axis(Axis(0));
    /// # assert_eq!(a, array![[1, 2], [3, 4]]);
    /// #
    /// let forbid_invert: Vec<_> = a
    ///     .producer()
    ///     .cloned()
    ///     .forbid_invert_axes(0)
    ///     .into_iter_any_ord()
    ///     .collect();
    /// assert_eq!(forbid_invert, vec![2, 1, 4, 3]);
    /// # }
    /// ```
    fn forbid_invert_axes<E: IntoDimension>(self, axes: E) -> ForbidInvertAxes<Self> {
        ForbidInvertAxes::new(self, axes.into_dimension().slice().iter().cloned())
    }

    /// Zips up two producers into a single producer of pairs.
    ///
    /// **Panics** if `self` and `other` have different shapes.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let a = array![[3, 7], [9, 10]];
    /// let b = array![[1, 2], [3, 4]];
    /// let difference = a.producer().zip(&b).map(|(a, b)| a - b).collect_array();
    /// assert_eq!(difference, a - b);
    /// # }
    /// ```
    fn zip<U>(self, other: U) -> Zip<Self, U::Producer>
    where
        U: IntoNdProducer<Dim = Self::Dim>,
    {
        Zip::new(self, other.into_producer())
    }

    /// Broadcasts the producer to a larger shape.
    ///
    /// The axes of `self` producer are mapped onto new axes as specified. In
    /// other words, axis `i` of the `self` becomes axis `axes_mapping[i]` in
    /// the result.
    ///
    /// Returns `None` if `self` cannot be broadcast to the specified shape.
    ///
    /// **Panics** if the axes in `axes_mapping` are not unique or if any of
    /// the axes are too large for `shape.ndim()`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::{array, Ix2, Ix3};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let a = array![[1], [2]];
    /// let b = a.producer().broadcast((2, 1), (3, 4, 2))
    ///     .expect("Broadcast shape must be compatible")
    ///     .cloned()
    ///     .collect_array();
    /// assert_eq!(
    ///     b,
    ///     array![
    ///         [[1, 2], [1, 2], [1, 2], [1, 2]],
    ///         [[1, 2], [1, 2], [1, 2], [1, 2]],
    ///         [[1, 2], [1, 2], [1, 2], [1, 2]],
    ///     ],
    /// );
    /// # }
    /// ```
    fn broadcast<E: IntoDimension>(
        self,
        axes_mapping: impl IntoDimension<Dim = Self::Dim>,
        shape: E,
    ) -> Option<BroadcastProducer<Self, E::Dim>> {
        BroadcastProducer::try_new(self, axes_mapping.into_dimension(), shape.into_dimension())
    }

    /// Takes a closure and creates a producer which calls that closure on each
    /// element.
    ///
    /// **Note**:
    ///
    /// * The closure is called in the order of iteration (which may be
    ///   arbitrary).
    ///
    /// * If the `Map` instance is split for parallelization, the closure will
    ///   be cloned. So, if you're generating random numbers in the closure,
    ///   you probably want to wrap your RNG in `ReseedingRng`.
    ///
    /// * The `NdSourceRepeat` implementation for `Map` requires that `F: Fn`
    ///   but assumes that given the same input, it will return the same
    ///   output. In the rare case that `F: Fn` but does not return the same
    ///   output for the same input (e.g. if it's mutating a `Cell` or
    ///   `RefCell`), you should avoid using the `NdSourceRepeat`
    ///   implementation (e.g. for broadcasting).
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[1, 2], [3, 4]];
    /// let squared = arr
    ///     .producer()
    ///     .map(|&x| x * x)
    ///     .collect_array();
    /// assert_eq!(squared, array![[1, 4], [9, 16]]);
    /// # }
    /// ```
    #[must_use = "`Map` must be consumed for the mapping to be applied."]
    fn map<B, F>(self, f: F) -> Map<Self, F>
    where
        F: FnMut(Self::Item) -> B,
    {
        Map::new(self, f)
    }

    /// Calls a closure on each element of the producer.
    ///
    /// **Note**: This method visits the elements in arbitrary order.
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let mut arr = array![[1, 2], [3, 4]];
    /// arr.producer_mut().for_each(|x| *x += 1);
    /// assert_eq!(arr, array![[2, 3], [4, 5]]);
    /// # }
    /// ```
    fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Self::Item),
    {
        self.fold((), move |(), item| f(item));
    }

    /// Creates a producer which gives the current index as well as the element
    /// value.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[0, 1], [1, 2]];
    /// arr.producer()
    ///     .indexed()
    ///     .for_each(|((row, col), &x)| assert_eq!(row + col, x));
    /// # }
    /// ```
    fn indexed(self) -> IndexedProducer<Self, Self::Dim> {
        IndexedProducer::new(self)
    }

    /// Do something on each element of a producer, passing the value on.
    ///
    /// **Note**:
    ///
    /// * The closure is called in the order of iteration (which may be
    ///   arbitrary).
    ///
    /// * If the `Inspect` instance is split for parallelization, the closure
    ///   will be cloned. So, if you're generating random numbers in the
    ///   closure, you probably want to wrap your RNG in `ReseedingRng`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let mut arr = array![[0, 1], [1, 2]];
    /// arr.producer_mut()
    ///     .inspect(|x| println!("Adding 1 to {}", x))
    ///     .for_each(|x| *x += 1);
    /// # }
    /// ```
    #[must_use = "`Inspect` must be consumed for iteration to occur."]
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        F: FnMut(&Self::Item),
    {
        Inspect::new(self, f)
    }

    /// A producer method that applies a function, producing a single, final
    /// value.
    ///
    /// **Note**: This method visits the elements in arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[0, 1], [2, 3]];
    /// let sum_sq = arr.producer().fold(0, |acc, &x| acc + x * x);
    /// assert_eq!(sum_sq, arr.fold(0, |acc, &x| acc + x * x));
    /// # }
    /// ```
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.into_iter_any_ord().fold(init, f)
    }

    /// A producer that folds over the given axes.
    ///
    /// **Note**: The folder visits the elements in arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer, into_repeat_with};
    ///
    /// # fn main() {
    /// let arr = array![
    ///     [[1, 2, 3], [4, 5, 6]],
    ///     [[7, 8, 9], [10, 11, 12]],
    /// ];
    /// let zeros = into_repeat_with(|| 0);
    /// let sum = arr
    ///     .producer()
    ///     .fold_axes((0, 2), zeros, |acc, &elem| acc + elem)
    ///     .collect_array();
    /// assert_eq!(sum, array![1 + 2 + 3 + 7 + 8 + 9, 4 + 5 + 6 + 10 + 11 + 12]);
    /// # }
    /// ```
    #[must_use = "`FoldAxesProducer` must be consumed for folding to occur."]
    fn fold_axes<T, I, F>(
        self,
        fold_axes: T,
        init: I,
        f: F,
    ) -> FoldAxesProducer<Self, T::Dim, I::Producer, F>
    where
        T: IntoDimension,
        I: IntoNdProducerWithShape<<Self::Dim as SubDim<T::Dim>>::Out>,
        F: FnMut(
            <I::Producer as NdProducer>::Item,
            Self::Item,
        ) -> <I::Producer as NdProducer>::Item,
        Self::Dim: SubDim<T::Dim>,
    {
        FoldAxesProducer::new(self, fold_axes.into_dimension(), init, f)
    }

    /// Creates a producer which `clone`s all of its elements.
    ///
    /// Note that the resulting `Cloned` instance implements `NdSourceRepeat`
    /// if the inner producer implements `NdSourceRepeat`. For correctness, if
    /// you use the `NdSourceRepeat` implementation (e.g. when broadcasting),
    /// you should ensure that `elem.clone()` always returns the same thing for
    /// the same element value `elem`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[1, 2], [3, 4]];
    /// let cloned = arr.producer().cloned().collect_array();
    /// assert_eq!(cloned, arr);
    /// # }
    /// ```
    fn cloned<'a, T>(self) -> Cloned<Self>
    where
        Self: NdProducer<Item = &'a T>,
        T: 'a + Clone,
    {
        Cloned::new(self)
    }

    /// Creates a wrapper that selects specific indices along an axis of the
    /// inner producer.
    ///
    /// **Panics** when converting to a source if any of the indices are
    /// out-of-bounds.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::{array, Axis};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]];
    /// let indices = array![0, 1, 3, 5];
    /// let selected_squared = arr
    ///     .producer()
    ///     .select_indices_axis(Axis(1), &indices)
    ///     .map(|x| x * x)
    ///     .collect_array();
    /// assert_eq!(selected_squared, array![[1, 4, 16, 36], [49, 64, 100, 144]]);
    /// # }
    /// ```
    fn select_indices_axis<'a, S>(
        self,
        axis: Axis,
        indices: &'a ArrayBase<S, Ix1>,
    ) -> SelectIndicesAxis<'a, Self>
    where
        S: Data<Elem = usize>,
        Self::Source: NdSourceRepeat,
    {
        SelectIndicesAxis::new(self, axis, indices.view())
    }

    /// Collects the producer into an `Array`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate ndarray;
    /// # extern crate nditer;
    /// #
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// # fn main() {
    /// let arr = array![[1, 2], [3, 4]];
    /// let collected = arr.producer().collect_array();
    /// assert_eq!(collected, array![[&1, &2], [&3, &4]]);
    /// # }
    /// ```
    fn collect_array(mut self) -> Array<Self::Item, Self::Dim> {
        let (iter_axes, Layout { shape, strides, .. }) =
            optimize::optimize_any_ord_with_layout(&mut self);
        let iter = Iter::new(self, iter_axes);
        // TODO: See how much of an impact implementing TrustedLen has. Also
        // consider using ndarray's from_iter.
        let data: Vec<Self::Item> = iter.collect();

        // TODO: Add negative stride support to `Array::from_shape_vec` and
        // clean this up.
        let mut need_invert = Self::Dim::zeros(strides.ndim());
        let mut abs_strides = Self::Dim::zeros(strides.ndim());
        if shape.size() > 0 {
            for (need_inv, abs_stride, &stride) in izip!(
                need_invert.slice_mut(),
                abs_strides.slice_mut(),
                strides.slice()
            ) {
                let stride = stride as isize;
                *need_inv = if stride < 0 { 1 } else { 0 };
                *abs_stride = stride.abs() as usize;
            }
        }
        let mut out = Array::from_shape_vec(shape.strides(abs_strides), data).unwrap();
        for (ax, &need_inv) in need_invert.slice().iter().enumerate() {
            if need_inv != 0 {
                out.invert_axis(Axis(ax));
            }
        }
        out
    }
}

/// An object that has an n-dimensional shape and supports various
/// shape-changing operations.
pub trait NdReshape {
    /// The dimension type of the shape.
    type Dim: Dimension;

    /// Returns the shape of the producer.
    fn shape(&self) -> Self::Dim;

    /// Returns the absolute values of the approximate strides of the producer.
    ///
    /// The two most important ways this is used are:
    ///
    /// 1. To determine which axes to try merging when iteration order doesn't
    ///    matter. (Axes are attempted to be merged in order of descending
    ///    absolute stride.)
    ///
    /// 2. To determine the best iteration order (when order doesn't matter).
    ///    (The relative cost of various orders is estimated based on the shape
    ///    and approximate strides, and the best order is chosen.)
    fn approx_abs_strides(&self) -> Self::Dim;

    /// Returns whether the axis can be inverted.
    ///
    /// **May panic** if `axis` is out-of-bounds.
    fn can_invert_axis(&self, axis: Axis) -> bool;

    /// Reverses the direction of the axis.
    ///
    /// **Panics** if the axis is out-of-bounds or if inverting it is not
    /// possible.
    fn invert_axis(&mut self, axis: Axis);

    /// Returns whether it's possible to merge the axis `take` to `into`.
    ///
    /// **May panic** if an axis is out-of-bounds.
    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge;

    /// Merges the axis `take` to `into`.
    ///
    /// The implementation of `merge_axes` must follow these constraints:
    ///
    /// 1. After merging, the `into` axis is the merged axis.
    ///
    /// 2. Movement along the two original axes (moving fastest along the
    ///    `into` axis) is equivalent to movement along the one (merged) axis.
    ///
    /// 3. The new length of the `into` axis is the product of the original
    ///    lengths of the two axes.
    ///
    /// 4. The new length of the `take` axis is 0 if the product of the
    ///    original lengths of the two axes is 0, and 1 otherwise.
    ///
    /// Note a couple of implications of these constraints:
    ///
    /// * In the special case that `take` and `into` are the same axis, merging
    ///   is only possible if the length of the axis is ≤ 1.
    ///
    /// * Since order must be preserved (constraint 2), if it's possible to
    ///   merge `take` into `into`, it's usually not possible to merge `into`
    ///   into `take`.
    ///
    /// **Panics** if an axis is out-of-bounds or if merging is not possible.
    fn merge_axes(&mut self, take: Axis, into: Axis);

    /// Returns the total number of elements in the producer.
    fn len(&self) -> usize {
        self.shape().size()
    }

    /// Returns the length of `axis`.
    ///
    /// **Panics** if the axis is out-of-bounds.
    fn len_of(&self, axis: Axis) -> usize {
        self.shape()[axis.index()]
    }

    /// Returns whether the array is empty (has no elements).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of axes in the producer.
    fn ndim(&self) -> usize {
        self.shape().ndim()
    }
}

/// An n-dimensional source of items.
///
/// # For implementers
///
/// You must be very careful when implementing this trait, because implementing
/// it incorrectly can easily cause undefined behavior. Implementing `NdSource`
/// correctly is especially tricky for sources whose `Item` contains mutable
/// references since we have to avoid mutable aliasing in Rust. The constraints
/// that the caller must follow are described by the documentation for the
/// individual methods, but they still leave quite a bit of flexibility. In
/// particular, note that:
///
/// * The caller can create arbitrarily many pointers with `.first_ptr()`.
///
/// * Items returned by `.get_once_unchecked()` are of the associated type
///   `Item`, so it's impossible for them to have a lifetime shorter than the
///   lifetime of `Self`. This can be quite problematic if you're not careful.
///
/// For example, consider the type `ArrayViewMut<'a, A, D>` from the `ndarray`
/// crate. It represents a mutable view of some data, and it has accessors for
/// getting data from the view. For example, it implements the `IndexMut` trait
/// which provides the method `fn index_mut(&mut self, index: I) -> &mut
/// S::Elem` which makes it possible to get a mutable reference to an element
/// that lives as long as the borrow of `self`.
///
/// If we were to implement `NdSource` directly on `ArrayViewMut<'a, A, D>`
/// with type `Item = &'a mut A`, then we would be fine *given only the methods
/// in the `NdSource` trait*. Callers could not obtain aliasing references to
/// elements by calling `.get_once_unchecked()` if they followed the required
/// constraints. However, since `ArrayViewMut` also implements `IndexMut`,
/// callers could, without violating any constraints, (1) get a mutable
/// reference to an element using `.get_once_unchecked()` and then (2) get a
/// mutable reference to the same element using `.index_mut()`.
///
/// In conclusion, if the source's `Item` contains mutable references, the
/// source must not provide any way to access elements other than
/// `.get_once_unchecked()` to prevent mutable aliasing. For example, for the
/// `ArrayViewMut` case, `ArrayViewMut` must not implement `NdSource`; instead,
/// another type must be used.
///
/// One way that may be helpful to think about converting an `ArrayViewMut`
/// into the corresponding source is that the "single borrow" of the view is
/// being decomposed into "borrows of all of the individual elements"
/// accessible with `.get_once_unchecked()`. This helps show that (1) the
/// source must take ownership of the view or have items that have the lifetime
/// of a mutable borrow of the view and that (2) the source must provide no way
/// to access elements other than `.get_once_unchecked()`.
///
/// Once Rust has support for generic associated types ([RFC
/// 1598](https://github.com/rust-lang/rfcs/blob/master/text/1598-generic_associated_types.md)),
/// it will be possible to implement `NdSource` directly on `ArrayViewMut` and
/// add support for things like mutable sliding window producers (which produce
/// overlapping mutable borrows). Until then, we have to be very careful to
/// follow the constraints on the unsafe methods provided by this trait to
/// avoid undefined behavior.
pub trait NdSource: NdAccess {
    /// The type of each element in the source.
    type Item;

    /// Reads the item at the pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the following:
    ///
    /// * The pointer is at a valid location. This can be ensured by creating
    ///   the first pointer with `.first_ptr()` and then following the
    ///   constraints of the
    ///   `.ptr_offset_axis()`/`.ptr_offset_axis_contiguous()` methods.
    ///
    /// * This method is called no more than once for any single index over the
    ///   entire life of the source. (This is necessary because
    ///   `.read_once_unchecked()` can return a mutable reference, and we must
    ///   not alias mutable references.)
    unsafe fn read_once_unchecked(&mut self, ptr: &Self::Ptr) -> Self::Item;
}

/// A source that allows getting items repeatedly at the same index.
///
/// Implementing this trait guarantees that it's safe to call
/// `NdSource::read_once_unchecked` repeatedly for the same location. (The
/// caller must still ensure that the pointer is at a valid location.)
///
/// Calling `NdSource::read_once_unchecked` multiple times for the same
/// location *should* return the same value, but callers must not rely on this
/// behavior for safety.
pub unsafe trait NdSourceRepeat: NdSource {}

/// Provides pointer-based access to elements of the source.
///
/// # For implementers
///
/// For safety, you must guarantee that:
///
/// * if `first_ptr` returns `Some`, it is safe to offset the pointer to every
///   location within the shape using `ptr_offset_axis` and/or (for offsets
///   along contiguous axes) `ptr_offset_axis_contiguous`
///
/// * the return value of `shape` is correct and consistent with `len_of`,
///   `is_empty`, and `ndim`
///
/// * if `is_axis_contiguous` returns `true` for an axis,
///   `ptr_offset_axis_contiguous` can safely be used for that axis
///
/// Note, in particular, that while you can rely on `NdAccess` to be
/// implemented correctly, you cannot rely on the correctness of any
/// `NdProducer`/`NdReshape` implementations for safety purposes since they are
/// safe to implement.
pub unsafe trait NdAccess {
    /// The dimension type of the shape.
    type Dim: Dimension;

    /// A type that specifies a location of an element in the volume.
    ///
    /// This is a cross between an index and a pointer. As with an index, an
    /// item should only be accessed by passing the `Ptr` to a method on the
    /// volume (instead of directly dereferencing a pointer). However, like a
    /// pointer, a `Ptr` to a specific location should not be created directly,
    /// but rather by getting an initial pointer with `.first_pointer()` and
    /// then offsetting it.
    type Ptr: Clone;

    /// Returns the shape of the volume.
    fn shape(&self) -> Self::Dim;

    /// Creates a pointer to the first element in the volume (at index
    /// `(0,0,0,...)`).
    ///
    /// Returns `Some(ptr)` if the volume is not empty, or `None` if the volume
    /// is empty.
    fn first_ptr(&self) -> Option<Self::Ptr>;

    /// Moves the pointer `count` indices along `axis`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new location of the pointer is still
    /// within the shape of the volume.
    unsafe fn ptr_offset_axis(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize);

    /// Moves the pointer `count` indices along `axis`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the following:
    ///
    /// * The new location of the pointer is still within the shape of the
    ///   volume.
    ///
    /// * The axis is contiguous (i.e. `self.is_axis_contiguous(axis) ==
    ///   true`).
    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize);

    /// Returns `true` iff the axis is contiguous.
    ///
    /// An axis is contiguous if a pointer can be offset along the axis using
    /// `ptr_offset_axis_contiguous`.
    fn is_axis_contiguous(&self, axis: Axis) -> bool;

    /// Returns the length of `axis`.
    ///
    /// **Panics** if the axis is out-of-bounds.
    fn len_of(&self, axis: Axis) -> usize {
        self.shape()[axis.index()]
    }

    /// Returns whether the array is empty (has no elements).
    fn is_empty(&self) -> bool {
        self.shape()
            .foldv(false, |is_empty, len| is_empty | (len == 0))
    }

    /// Returns the number of axes in the producer.
    fn ndim(&self) -> usize {
        self.shape().ndim()
    }
}

/// Flat iterator over an n-dimensional source.
pub struct Iter<P: NdProducer> {
    /// The source to iterate over.
    source: P::Source,
    /// Pointer and index to next item. `None` if iteration is complete.
    ///
    /// Note that the order of the elements in the index matches the order of
    /// `axes`. For example, `index[3]` is the index for axis `Axis(axes[3])`,
    /// not `Axis(3)`.
    ptr_idx: Option<(<P::Source as NdAccess>::Ptr, P::Dim)>,
    /// Axes to iterate over (outermost axis first).
    axes: P::Dim,
    /// Lengths of the axes (in order of `axes`).
    ///
    /// All axis lengths must be `<= isize::MAX` so that indices can be safely
    /// cast to `isize` without overflow. Additionally, all axis lengths must
    /// be no larger than the corresponding axis lengths of `source`.
    axis_lens: P::Dim,
}

impl<P: NdProducer> Iter<P> {
    /// Creates a new `Iter` from the producer that iterates over the axes in
    /// the specified order.
    ///
    /// `axes` must be a subset of the axes of the producer.
    ///
    /// **Panics** if any of the axes in `axes` are out of bounds, if an axis
    /// is repeated more than once, or if any axis length overflows `isize`.
    fn new(producer: P, axes: P::Dim) -> Self {
        let source = producer.into_source();
        assert_valid_unique_axes::<P::Dim>(source.ndim(), axes.slice());
        let axis_lens = axes.mapv(|axis| {
            let axis_len = source.len_of(Axis(axis));
            assert!(axis_len <= std::isize::MAX as usize);
            axis_len
        });
        let ptr_idx = match (source.first_ptr(), P::Dim::first_index(&axis_lens)) {
            (Some(ptr), Some(idx)) => Some((ptr, idx)),
            _ => None,
        };
        Iter {
            source,
            ptr_idx,
            axes,
            axis_lens,
        }
    }
}

/// Flat iterator over a borrowed n-dimensional source.
struct IterBorrowed<'a, S: 'a + NdSource, D: 'a + Dimension> {
    source: &'a mut S,
    /// Pointer and index to next item. `None` if iteration is complete.
    ///
    /// Note that the order of the elements in the index matches the order of
    /// `axes`. For example, `index[3]` is the index for axis `Axis(axes[3])`,
    /// not `Axis(3)`.
    ptr_idx: Option<(S::Ptr, D)>,
    /// Axes to iterate over (outermost axis first).
    axes: &'a D,
    /// Lengths of the axes (in order of `axes`).
    ///
    /// All axis lengths must be `<= isize::MAX` so that indices can be safely
    /// cast to `isize` without overflow. Additionally, all axis lengths must
    /// be no larger than the corresponding axis lengths of `source`.
    axis_lens: &'a D,
}

impl<'a, S: 'a + NdSource, D: 'a + Dimension> IterBorrowed<'a, S, D> {
    /// Creates an `IterBorrowed` from its raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure the following:
    ///
    /// * The new iterator can never read the same location from the source as
    ///   anything else (to make sure the constraints are satisfied for
    ///   `read_once_unchecked`).
    ///
    /// * The index is a valid index within the axis lengths.
    ///
    /// * The pointer is at a valid location within the source, and all
    ///   locations reachable by offsetting the pointer to all remaining
    ///   indices within the `axis_lens` are also valid locations.
    ///
    /// * The `axes` are unique, in-bounds axes for `source`.
    ///
    /// * The `axis_lens` are no larger than the axis lengths of the `source`
    ///   for the given `axes`, and all axis lengths are `<= isize::MAX`.
    unsafe fn from_raw_parts(
        source: &'a mut S,
        ptr_idx: Option<(S::Ptr, D)>,
        axes: &'a D,
        axis_lens: &'a D,
    ) -> Self {
        // A few sanity checks.
        if cfg!(debug_assertions) {
            assert_valid_unique_axes::<D>(source.ndim(), axes.slice());
            for (&ax, &axis_len) in izip!(axes.slice(), axis_lens.slice()) {
                debug_assert!(axis_len <= source.len_of(Axis(ax)));
                debug_assert!(axis_len <= std::isize::MAX as usize);
            }
            if let Some((_, idx)) = &ptr_idx {
                for (&i, &axis_len) in izip!(idx.slice(), axis_lens.slice()) {
                    debug_assert!(i < axis_len);
                }
            }
        }
        IterBorrowed {
            source,
            ptr_idx,
            axes,
            axis_lens,
        }
    }
}

macro_rules! impl_iter {
    (($($generics:tt)*), $self:ty, $item:ty) => {
        impl<$($generics)*> $self {
            /// Moves to the next index and updates the pointer.
            fn move_to_next(&mut self) {
                self.ptr_idx = self.ptr_idx.take().and_then(|(mut ptr, mut idx)| {
                    for (i, &axis, &len) in izip!(idx.slice_mut(), self.axes.slice(), self.axis_lens.slice()).rev() {
                        *i += 1;
                        // This is safe because the pointer always stays within the shape.
                        unsafe {
                            if *i < len {
                                self.source.ptr_offset_axis(&mut ptr, Axis(axis), 1);
                                return Some((ptr, idx));
                            } else {
                                self.source.ptr_offset_axis(&mut ptr, Axis(axis), -((*i - 1) as isize));
                                *i = 0;
                            }
                        }
                    }
                    None
                });
            }
        }

        impl<$($generics)*> ExactSizeIterator for $self {
            fn len(&self) -> usize {
                match &self.ptr_idx {
                    None => 0,
                    Some((_, idx)) => {
                        // Number of elements that have been consumed so far.
                        let mut consumed = 0;
                        // Product of the lengths of axes that are inner of the current one.
                        let mut inner_len = 1;
                        for (i, axis_len) in izip!(idx.slice(), self.axis_lens.slice()).rev() {
                            consumed += i * inner_len;
                            inner_len *= axis_len;
                        }
                        self.axis_lens.size() - consumed
                    }
                }
            }
        }

        impl<$($generics)*> Iterator for $self {
            type Item = $item;

            fn next(&mut self) -> Option<Self::Item> {
                let item = match self.ptr_idx {
                    None => None,
                    Some((ref ptr, _)) => Some(unsafe { self.source.read_once_unchecked(ptr) }),
                };
                self.move_to_next();
                item
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }

            #[inline]
            fn fold<B, F>(mut self, init: B, mut f: F) -> B
            where
                F: FnMut(B, Self::Item) -> B,
            {
                debug_assert_eq!(self.axes.ndim(), self.axis_lens.ndim());
                if self.axes.ndim() == 0 {
                    let elem = self.next().unwrap();
                    debug_assert!(self.next().is_none());
                    return f(init, elem);
                }
                let mut acc = init;
                let inner_unroll = self.axes.ndim() - 1;
                let inner_unroll_axis = Axis(self.axes[inner_unroll]);
                let inner_unroll_len = self.axis_lens[inner_unroll];
                if self.source.is_axis_contiguous(inner_unroll_axis) {
                    if self.axes.ndim() >= 2 {
                        let outer_unroll = self.axes.ndim() - 2;
                        let outer_unroll_axis = Axis(self.axes[outer_unroll]);
                        let outer_unroll_len = self.axis_lens[outer_unroll];
                        loop {
                            if let Some((ref mut ptr, ref mut idx)) = self.ptr_idx {
                                // "unroll" the loop over the two innermost axes
                                unsafe {
                                    debug_assert!(outer_unroll_len > 0);
                                    debug_assert!(inner_unroll_len > 0);
                                    acc = f(acc, self.source.read_once_unchecked(ptr));
                                    for _ in 1..inner_unroll_len {
                                        self.source
                                            .ptr_offset_axis_contiguous(ptr, inner_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                    }
                                    for _ in 1..outer_unroll_len {
                                        self.source.ptr_offset_axis_contiguous(
                                            ptr,
                                            inner_unroll_axis,
                                            -(inner_unroll_len as isize - 1),
                                        );
                                        self.source.ptr_offset_axis(ptr, outer_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                        for _ in 1..inner_unroll_len {
                                            self.source.ptr_offset_axis_contiguous(
                                                ptr,
                                                inner_unroll_axis,
                                                1,
                                            );
                                            acc = f(acc, self.source.read_once_unchecked(ptr));
                                        }
                                    }
                                    idx[inner_unroll] = inner_unroll_len - 1;
                                    idx[outer_unroll] = outer_unroll_len - 1;
                                }
                            } else {
                                break;
                            }
                            self.move_to_next();
                        }
                    } else {
                        loop {
                            if let Some((ref mut ptr, ref mut idx)) = self.ptr_idx {
                                // "unroll" the loop over the innermost axis
                                unsafe {
                                    debug_assert!(inner_unroll_len > 0);
                                    acc = f(acc, self.source.read_once_unchecked(ptr));
                                    for _ in 1..inner_unroll_len {
                                        self.source
                                            .ptr_offset_axis_contiguous(ptr, inner_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                    }
                                    idx[inner_unroll] = inner_unroll_len - 1;
                                }
                            } else {
                                break;
                            }
                            self.move_to_next();
                        }
                    }
                } else {
                    if self.axes.ndim() >= 2 {
                        let outer_unroll = self.axes.ndim() - 2;
                        let outer_unroll_axis = Axis(self.axes[outer_unroll]);
                        let outer_unroll_len = self.axis_lens[outer_unroll];
                        loop {
                            if let Some((ref mut ptr, ref mut idx)) = self.ptr_idx {
                                // "unroll" the loop over the two innermost axes
                                unsafe {
                                    debug_assert!(outer_unroll_len > 0);
                                    debug_assert!(inner_unroll_len > 0);
                                    acc = f(acc, self.source.read_once_unchecked(ptr));
                                    for _ in 1..inner_unroll_len {
                                        self.source.ptr_offset_axis(ptr, inner_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                    }
                                    for _ in 1..outer_unroll_len {
                                        self.source.ptr_offset_axis(
                                            ptr,
                                            inner_unroll_axis,
                                            -(inner_unroll_len as isize - 1),
                                        );
                                        self.source.ptr_offset_axis(ptr, outer_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                        for _ in 1..inner_unroll_len {
                                            self.source.ptr_offset_axis(ptr, inner_unroll_axis, 1);
                                            acc = f(acc, self.source.read_once_unchecked(ptr));
                                        }
                                    }
                                    idx[inner_unroll] = inner_unroll_len - 1;
                                    idx[outer_unroll] = outer_unroll_len - 1;
                                }
                            } else {
                                break;
                            }
                            self.move_to_next();
                        }
                    } else {
                        loop {
                            if let Some((ref mut ptr, ref mut idx)) = self.ptr_idx {
                                // "unroll" the loop over the innermost axis
                                unsafe {
                                    debug_assert!(inner_unroll_len > 0);
                                    acc = f(acc, self.source.read_once_unchecked(ptr));
                                    for _ in 1..inner_unroll_len {
                                        self.source.ptr_offset_axis(ptr, inner_unroll_axis, 1);
                                        acc = f(acc, self.source.read_once_unchecked(ptr));
                                    }
                                    idx[inner_unroll] = inner_unroll_len - 1;
                                }
                            } else {
                                break;
                            }
                            self.move_to_next();
                        }
                    }
                }
                acc
            }
        }
    };
}

impl_iter!((P: NdProducer), Iter<P>, P::Item);
impl_iter!(('a, S: NdSource, D: Dimension), IterBorrowed<'a, S, D>, S::Item);

/// Indicates whether two axes can be merged.
///
/// This is the type of the return value of `NdProducer::can_merge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CanMerge {
    /// The axes cannot be merged.
    Never = 0b00,
    /// The axes can be merged if they are unchanged.
    IfUnchanged = 0b01,
    /// The axes can be merged after inverting one of them (assuming it's
    /// possible to invert one of them).
    ///
    /// This does not imply that it's possible to invert one of the axes; you
    /// need to check `NdReshape::can_invert_axis` to determine if an axis can
    /// be inverted.
    IfInverted = 0b10,
    /// The axes can be merged after inverting one of them (assuming it's
    /// possible to invert one of them) or if they are unchanged.
    ///
    /// This does not imply that it's possible to invert one of the axes; you
    /// need to check `NdReshape::can_invert_axis` to determine if an axis can
    /// be inverted.
    IfEither = 0b11,
}

impl ::std::ops::BitAnd for CanMerge {
    type Output = Self;
    fn bitand(self, rhs: CanMerge) -> CanMerge {
        let and = (self as u8) & (rhs as u8);
        let out = match and {
            0b00 => CanMerge::Never,
            0b01 => CanMerge::IfUnchanged,
            0b10 => CanMerge::IfInverted,
            0b11 => CanMerge::IfEither,
            _ => unreachable!(),
        };
        debug_assert_eq!(and, out as u8);
        out
    }
}

/// The layout of the items (iteration order) for use in collecting as an array.
///
/// If the items are collected into a `Vec` in order, the shape and strides are
/// given, and the pointer to the first element is
/// `vec.as_ptr().offset(offset)`.
pub struct Layout<D: Dimension> {
    /// Original shape of the producer before optimization. (Consumers of the
    /// producer should be this shape.)
    pub shape: D,
    /// Contiguous strides for interpreting the result of iteration. (The
    /// elements of `strides` should be interpreted as `isize`.)
    pub strides: D,
    /// Offset from first element yielded by the iterator to the first element
    /// of the array. (Will always be nonnegative.)
    pub offset: isize,
}

/// Asserts that all axes are valid for dimension `D` with `ndim` dimensions
/// and that none are repeated. (Note that this also implies that the
/// `axes.len()` is ensured to be <= `ndim`.)
///
/// **Panics** if this is not the case.
pub(crate) fn assert_valid_unique_axes<D: Dimension>(ndim: usize, axes: &[usize]) {
    let mut usage_counts = D::zeros(ndim);
    for &axis in axes {
        assert_eq!(
            usage_counts[axis], 0,
            "Each axis must be listed no more than once."
        );
        usage_counts[axis] = 1;
    }
}

/// Extension methods for `Dimension` types.
pub(crate) trait DimensionExt {
    /// Applies the fold to the values and returns the result.
    fn foldv<F, B>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, usize) -> B;

    /// Applies `f` to each element by value and creates a new instance with
    /// the results.
    fn mapv<F>(&self, f: F) -> Self
    where
        F: FnMut(usize) -> usize;

    /// Calls `f` for each element by value.
    fn visitv<F>(&self, f: F)
    where
        F: FnMut(usize);
}

impl<D: Dimension> DimensionExt for D {
    fn foldv<F, B>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, usize) -> B,
    {
        self.slice().iter().cloned().fold(init, f)
    }

    fn mapv<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize) -> usize,
    {
        let mut out = Self::zeros(self.ndim());
        for (o, &i) in izip!(out.slice_mut(), self.slice()) {
            *o = f(i);
        }
        out
    }

    fn visitv<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.slice().iter().cloned().for_each(f)
    }
}

mod adapters;
mod dim_traits;
mod impl_ndarray;
mod optimize;
