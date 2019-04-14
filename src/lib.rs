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

pub use self::adapters::{into_repeat_with, repeat_with};
pub use self::axes::{axes, axes_all, axes_none, axes_except, IntoAxesFor};
pub use self::dim_traits::SubDim;
pub use self::impl_ndarray::ArrayBaseExt;
pub use self::iter::Iter;

use self::adapters::{
    BroadcastProducer, Cloned, FoldAxesProducer, ForceAxesOrdered, IndexedProducer, Inspect, Map,
    SelectIndicesAxis, Zip,
};
use self::axes::AxesFor;
use self::errors::{BroadcastError, OrderedAxisError};
use self::pairwise_sum::pairwise_sum;
use itertools::izip;
use ndarray::{Array, ArrayBase, Axis, Data, DataMut, Dimension, IntoDimension, Ix1, ShapeBuilder};
use num_traits::Zero;

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
    /// use ndarray::{array, Array2, ShapeBuilder};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let mut arr = Array2::zeros((2, 2).f());
    /// arr.assign(&array![[1, 2], [3, 4]]);
    /// let elems: Vec<_> = arr.producer().into_iter().cloned().collect();
    /// assert_eq!(elems, vec![1, 2, 3, 4]);
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

    /// Forces iteration to occur in order for each specified axis.
    ///
    /// This is useful when you want iteration to move in order along some axes
    /// (not in reverse or random order), but you don't care about iteration
    /// order across axes or iteration order along any other axes.
    ///
    /// # Example
    ///
    /// The optimizer usually inverts axes when doing so allows axes to be
    /// merged, so for demonstration, let's construct an array such that
    /// inverting an axis will allow the axes to be merged.
    ///
    /// ```
    /// use ndarray::{array, Axis};
    /// use nditer::{axes, ArrayBaseExt, NdProducer};
    ///
    /// let mut a = array![[3, 4], [1, 2]];
    /// a.invert_axis(Axis(0));
    /// assert_eq!(a, array![[1, 2], [3, 4]]);
    /// assert_eq!(a.strides(), &[-2, 1]);
    /// ```
    ///
    /// Without any constraints, the optimizer inverts axis 0 in this case so
    /// that the axes can be merged and iteration can proceed in memory order.
    /// (Note that this iteration order is not guaranteed.) Observe that since
    /// the optimizer has inverted axis 0, element `3` occurs before element
    /// `1`, and element `4` occurs before element `2`.
    ///
    /// ```
    /// # use ndarray::{array, Axis};
    /// # use nditer::{axes, ArrayBaseExt, NdProducer};
    /// #
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
    /// ```
    ///
    /// Let's force axis 0 to be iterated in order. In this case, the optimizer
    /// finds that while it can't invert axis 0, inverting axis 1 is beneficial
    /// because that allows the axes to be merged and iteration can proceed in
    /// (reverse) memory order. (Note that other than iterating over axis 0 in
    /// order, the iteration order is not guaranteed.) Observe that since we've
    /// required axis 0 to be iterated in order, element `1` is guaranteed to
    /// occur before element `3`, and element `2` is guaranteed to occur before
    /// element `4`.
    ///
    /// ```
    /// # use ndarray::{array, Axis};
    /// # use nditer::{axes, ArrayBaseExt, NdProducer};
    /// #
    /// # let mut a = array![[3, 4], [1, 2]];
    /// # a.invert_axis(Axis(0));
    /// # assert_eq!(a, array![[1, 2], [3, 4]]);
    /// #
    /// let forbid_invert: Vec<_> = a
    ///     .producer()
    ///     .cloned()
    ///     .force_axes_ordered(axes(0))
    ///     .into_iter_any_ord()
    ///     .collect();
    /// assert_eq!(forbid_invert, vec![2, 1, 4, 3]);
    /// ```
    fn force_axes_ordered(self, axes: impl IntoAxesFor<Self::Dim>) -> ForceAxesOrdered<Self> {
        ForceAxesOrdered::new(self, axes)
    }

    /// Zips up two producers into a single producer of pairs.
    ///
    /// **Panics** if `self` and `other` have different shapes.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let a = array![[3, 7], [9, 10]];
    /// let b = array![[1, 2], [3, 4]];
    /// let difference = a.producer().zip(&b).map(|(a, b)| a - b).collect_array();
    /// assert_eq!(difference, a - b);
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
    /// Returns `Err` if `self` cannot be broadcast to the specified shape.
    ///
    /// **Panics** if the axes in `axes_mapping` are not unique or if any of
    /// the axes are too large for `shape.ndim()`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{array, Ix2, Ix3};
    /// use nditer::{axes, ArrayBaseExt, NdProducer};
    ///
    /// let a = array![[1], [2]];
    /// let b = a.producer().broadcast(axes((2, 1)), (3, 4, 2))
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
    /// ```
    fn broadcast<E: IntoDimension>(
        self,
        axes_mapping: impl IntoAxesFor<E::Dim, Axes = Self::Dim>,
        shape: E,
    ) -> Result<BroadcastProducer<Self, E::Dim>, BroadcastError> {
        BroadcastProducer::try_new(self, axes_mapping, shape.into_dimension())
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[1, 2], [3, 4]];
    /// let squared = arr
    ///     .producer()
    ///     .map(|&x| x * x)
    ///     .collect_array();
    /// assert_eq!(squared, array![[1, 4], [9, 16]]);
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let mut arr = array![[1, 2], [3, 4]];
    /// arr.producer_mut().for_each(|x| *x += 1);
    /// assert_eq!(arr, array![[2, 3], [4, 5]]);
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[0, 1], [1, 2]];
    /// arr.producer()
    ///     .indexed()
    ///     .for_each(|((row, col), &x)| assert_eq!(row + col, x));
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let mut arr = array![[0, 1], [1, 2]];
    /// arr.producer_mut()
    ///     .inspect(|x| println!("Adding 1 to {}", x))
    ///     .for_each(|x| *x += 1);
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[0, 1], [2, 3]];
    /// let sum_sq = arr.producer().fold(0, |acc, &x| acc + x * x);
    /// assert_eq!(sum_sq, arr.fold(0, |acc, &x| acc + x * x));
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
    /// use ndarray::array;
    /// use nditer::{axes, ArrayBaseExt, NdProducer, into_repeat_with};
    ///
    /// let arr = array![
    ///     [[1, 2, 3], [4, 5, 6]],
    ///     [[7, 8, 9], [10, 11, 12]],
    /// ];
    /// let zeros = into_repeat_with(|| 0);
    /// let sum = arr
    ///     .producer()
    ///     .fold_axes(axes((0, 2)), zeros, |acc, &elem| acc + elem)
    ///     .collect_array();
    /// assert_eq!(sum, array![1 + 2 + 3 + 7 + 8 + 9, 4 + 5 + 6 + 10 + 11 + 12]);
    /// ```
    #[must_use = "`FoldAxesProducer` must be consumed for folding to occur."]
    fn fold_axes<T, I, F>(
        self,
        fold_axes: T,
        init: I,
        f: F,
    ) -> FoldAxesProducer<Self, T::Axes, I::Producer, F>
    where
        T: IntoAxesFor<Self::Dim>,
        I: IntoNdProducerWithShape<<T::IntoOthers as IntoAxesFor<Self::Dim>>::Axes>,
        F: FnMut(
            <I::Producer as NdProducer>::Item,
            Self::Item,
        ) -> <I::Producer as NdProducer>::Item,
    {
        FoldAxesProducer::new(self, fold_axes, init, f)
    }

    /// A producer that folds over the given axes, modifying the accumulator
    /// in-place.
    ///
    /// **Note**: The folder visits the elements in arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{array, Array1, Array2, Axis};
    /// use nditer::{axes, axes_except, ArrayBaseExt, NdProducer};
    ///
    /// let data = array![
    ///     [[1, 2, 3], [4, 5, 6]],
    ///     [[7, 8, 9], [10, 11, 12]],
    /// ];
    /// assert_eq!(data.shape(), &[2, 2, 3]);
    ///
    /// let mut a = Array1::zeros(data.len_of(Axis(1)));
    /// assert_eq!(a.shape(), &[2]);
    /// data.producer().fold_inplace_axes(axes((0, 2)), &mut a, |acc, &elem| *acc += elem);
    /// assert_eq!(a, array![1 + 2 + 3 + 7 + 8 + 9, 4 + 5 + 6 + 10 + 11 + 12]);
    ///
    /// let mut b = Array2::zeros((data.len_of(Axis(2)), data.len_of(Axis(0))));
    /// assert_eq!(b.shape(), &[3, 2]);
    /// data.producer().fold_inplace_axes(axes_except((2, 0)), &mut b, |acc, &elem| *acc += elem);
    /// assert_eq!(b, array![
    ///     [1 + 4, 7 + 10],
    ///     [2 + 5, 8 + 11],
    ///     [3 + 6, 9 + 12],
    /// ]);
    /// ```
    fn fold_inplace_axes<T, S, F>(
        self,
        fold_axes: T,
        accumulator: &mut ArrayBase<S, <T::IntoOthers as IntoAxesFor<Self::Dim>>::Axes>,
        mut f: F,
    ) -> Result<(), BroadcastError>
    where
        T: IntoAxesFor<Self::Dim>,
        S: DataMut,
        F: FnMut(&mut S::Elem, Self::Item),
    {
        let remaining_axes = fold_axes.into_others();
        Ok(accumulator
            .raw_producer_mut()
            .broadcast(remaining_axes, self.shape())?
            .zip(self)
            .for_each(move |(acc_ptr, item)| {
                // We can safely access mutable references to elements in the
                // accumulator because we have a mutable borrow of it and `S`
                // implements `DataMut`. It is safe to broadcast the accumulator
                // and dereference elements because the lifetimes of the borrows
                // of the elements are shorter than the borrow of the
                // accumulator and the lifetimes of the borrows of the elements
                // don't overlap each other. (The lifetime of each borrow lasts
                // only for the duration of the call to the closure.)
                let acc_ref = unsafe { &mut *acc_ptr };
                f(acc_ref, item)
            }))
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
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[1, 2], [3, 4]];
    /// let cloned = arr.producer().cloned().collect_array();
    /// assert_eq!(cloned, arr);
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
    /// Returns `Err` if the inner producer requires iteration over `axis` to
    /// occur in order.
    ///
    /// **Panics** when converting to a source if any of the indices are
    /// out-of-bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{array, Axis};
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]];
    /// let indices = array![0, 1, 3, 5];
    /// let selected_squared = arr
    ///     .producer()
    ///     .select_indices_axis(Axis(1), &indices)?
    ///     .map(|x| x * x)
    ///     .collect_array();
    /// assert_eq!(selected_squared, array![[1, 4, 16, 36], [49, 64, 100, 144]]);
    /// # Ok::<(), nditer::errors::OrderedAxisError>(())
    /// ```
    fn select_indices_axis<'a, S>(
        self,
        axis: Axis,
        indices: &'a ArrayBase<S, Ix1>,
    ) -> Result<SelectIndicesAxis<'a, Self>, OrderedAxisError>
    where
        S: Data<Elem = usize>,
        Self::Source: NdSourceRepeat,
    {
        SelectIndicesAxis::try_new(self, axis, indices.view())
    }

    /// Collects the producer into an `Array`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[1, 2], [3, 4]];
    /// let collected = arr.producer().collect_array();
    /// assert_eq!(collected, array![[&1, &2], [&3, &4]]);
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

    /// Computes the sum of the elements in a pairwise fashion.
    ///
    /// Note that this isn't strictly a pairwise sum; at the lowest level of
    /// the tree, at most `MAX_SEQUENTIAL` elements are added in sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use nditer::{ArrayBaseExt, NdProducer};
    ///
    /// let arr = array![[1., 2.], [3., 4.]];
    /// let sum = arr.producer().cloned().pairwise_sum();
    /// assert_eq!(sum, 10.);
    /// ```
    // Implementation note: This is explicitly not a method on `Iter` because
    // it relies on splitting over the source, and the splitting procedure
    // assumes that the iterator has not been partially iterated over.
    fn pairwise_sum(self) -> Self::Item
    where
        Self::Item: Zero + Clone + std::ops::Add<Output = Self::Item>,
    {
        pairwise_sum(self)
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

    /// Returns whether iteration along the axis must proceed in order.
    ///
    /// If the return value is `false`, iteration along the axis may be done in
    /// any order (e.g. in reverse order or in random order). If the return
    /// value is `true`, iteration along the axis must be done in order.
    ///
    /// Implementation requirements:
    ///
    /// * If a producer returns `true` for an axis, any producers that wrap it
    ///   must ensure that iteration proceeds in order for that axis (which
    ///   typically means the wrapper would have to return `true` as well).
    ///
    /// * If a producer returns `true` for an axis, it may rely on the
    ///   iteration going in order along that axis for correctness, but not for
    ///   safety, since `NdReshape` is safe to implement.
    ///
    /// * If the return value is `false`, calling `.invert_axis()` for that
    ///   axis must not panic.
    ///
    /// **May panic** if `axis` is out-of-bounds.
    fn is_axis_ordered(&self, axis: Axis) -> bool;

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
/// * if `first_ptr` returns `Some`, the volume is not empty, and it's safe to
///   offset the pointer to every location within the shape using
///   `ptr_offset_axis` and/or (for offsets along contiguous axes)
///   `ptr_offset_axis_contiguous`
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

/// Indicates whether two axes can be merged.
///
/// This is the type of the return value of `NdProducer::can_merge`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CanMerge {
    /// The axes cannot be merged.
    Never = 0b00,
    /// The axes can be merged if they are left unchanged, and they can be
    /// merged after inverting both of them (assuming it's possible to invert
    /// both of them).
    ///
    /// This does not imply that it's possible to invert the axes; you need to
    /// check `NdReshape::is_axis_ordered` to determine if an axis can be
    /// inverted.
    IfUnchangedOrBothInverted = 0b01,
    /// The axes can be merged after inverting either one of them (assuming
    /// it's possible to invert one of them).
    ///
    /// This does not imply that it's possible to invert one of the axes; you
    /// need to check `NdReshape::is_axis_ordered` to determine if an axis can
    /// be inverted.
    IfOneInverted = 0b10,
    /// The axes can be merged after inverting one of them (assuming it's
    /// possible to invert one of them), they can be merged if they are
    /// unchanged, and they can be merged after inverting both of them
    /// (assuming it's possible to invert both of them).
    ///
    /// This does not imply that it's possible to invert either of the axes;
    /// you need to check `NdReshape::is_axis_ordered` to determine if an axis
    /// can be inverted.
    Always = 0b11,
}

impl ::std::ops::BitAnd for CanMerge {
    type Output = Self;
    fn bitand(self, rhs: CanMerge) -> CanMerge {
        let and = (self as u8) & (rhs as u8);
        let out = match and {
            0b00 => CanMerge::Never,
            0b01 => CanMerge::IfUnchangedOrBothInverted,
            0b10 => CanMerge::IfOneInverted,
            0b11 => CanMerge::Always,
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

    /// Applies `f` to each element by mutable reference.
    fn map_inplace<F>(&mut self, f: F)
    where
        F: FnMut(&mut usize);

    /// Calls `f` for each element by value.
    fn visitv<F>(&self, f: F)
    where
        F: FnMut(usize);

    /// Calls `f` for each axis index and element value.
    fn indexed_visitv<F>(&self, f: F)
    where
        F: FnMut(Axis, usize);
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

    fn map_inplace<F>(&mut self, f: F)
    where
        F: FnMut(&mut usize),
    {
        self.slice_mut().iter_mut().for_each(f)
    }

    fn visitv<F>(&self, f: F)
    where
        F: FnMut(usize),
    {
        self.slice().iter().cloned().for_each(f)
    }

    fn indexed_visitv<F>(&self, mut f: F)
    where
        F: FnMut(Axis, usize),
    {
        self.slice()
            .iter()
            .enumerate()
            .for_each(move |(ax, &elem)| f(Axis(ax), elem))
    }
}

mod adapters;
mod axes;
mod dim_traits;
pub mod errors;
mod impl_ndarray;
mod iter;
mod optimize;
mod pairwise_sum;
