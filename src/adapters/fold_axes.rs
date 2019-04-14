// TODO: check that behavior is correct with zero-length axes

use crate::{
    iter::IterBorrowed, optimize::optimize_any_ord_axes, AxesFor, CanMerge, DimensionExt,
    IntoAxesFor, IntoNdProducerWithShape, NdAccess, NdProducer, NdReshape, NdSource,
};
use itertools::izip;
use ndarray::{Axis, Dimension};

/// A producer that folds over a subset of axes of an inner producer.
pub struct FoldAxesProducer<P, Df, I, F>
where
    P: NdReshape,
    Df: Dimension,
    I: NdReshape,
{
    inner: P,
    /// Axes (of the `inner` source) that are being folded over.
    fold_axes: AxesFor<P::Dim, Df>,
    /// Mapping of "outer" axis to corresponding axis of `inner`.
    outer_to_inner: AxesFor<P::Dim, I::Dim>,
    /// Source of init values for fold.
    init: I,
    f: F,
}

impl<P, Df, I, F> FoldAxesProducer<P, Df, I, F>
where
    P: NdProducer,
    Df: Dimension,
    I: NdProducer,
    F: FnMut(I::Item, P::Item) -> I::Item,
{
    /// Creates a new `FoldAxesProducer`.
    ///
    /// **Panics** if any of `fold_axes` are out of bounds or if an axis is
    /// repeated more than once.
    pub fn new<T>(
        inner: P,
        fold_axes: T,
        init: impl IntoNdProducerWithShape<I::Dim, Item = I::Item, Producer = I>,
        f: F,
    ) -> Self
    where
        T: IntoAxesFor<P::Dim, Axes = Df>,
        T::IntoOthers: IntoAxesFor<P::Dim, Axes = I::Dim>,
    {
        let (fold_axes, outer_to_inner) = fold_axes.into_these_and_other_axes_for(inner.ndim());
        let ndim = outer_to_inner.num_axes();
        let shape = {
            let mut shape = I::Dim::zeros(ndim);
            for (outer_axis, len) in shape.slice_mut().iter_mut().enumerate() {
                *len = inner.len_of(Axis(outer_to_inner[outer_axis]));
            }
            shape
        };
        let init = init.into_producer(shape.clone());
        assert_eq!(init.shape(), shape);
        FoldAxesProducer {
            inner,
            fold_axes,
            outer_to_inner,
            init,
            f,
        }
    }
}

impl<P, Df, I, F> NdProducer for FoldAxesProducer<P, Df, I, F>
where
    P: NdProducer,
    Df: Dimension,
    I: NdProducer,
    F: FnMut(I::Item, P::Item) -> I::Item,
{
    type Item = I::Item;
    type Source = FoldAxesSource<P::Source, Df, I::Source, F>;

    fn into_source(mut self) -> Self::Source {
        let mut fold_axes = self.fold_axes.into_inner();
        optimize_any_ord_axes(&mut self.inner, &mut fold_axes);
        FoldAxesSource::new(
            self.inner.into_source(),
            fold_axes,
            self.outer_to_inner.into_inner(),
            self.init.into_source(),
            self.f,
        )
    }
}

impl<P, Df, I, F> NdReshape for FoldAxesProducer<P, Df, I, F>
where
    P: NdReshape,
    Df: Dimension,
    I: NdReshape,
{
    type Dim = I::Dim;

    fn shape(&self) -> Self::Dim {
        if cfg!(debug_assertions) {
            let shape = self.init.shape();
            let inner_shape = self.inner.shape();
            for (i, &len) in shape.slice().iter().enumerate() {
                debug_assert_eq!(len, inner_shape[self.outer_to_inner[i]]);
            }
        }
        self.init.shape()
    }

    fn approx_abs_strides(&self) -> Self::Dim {
        let mut strides = self.init.approx_abs_strides();
        let inner_strides = self.inner.approx_abs_strides();
        for (ax, s) in strides.slice_mut().iter_mut().enumerate() {
            *s += inner_strides[self.outer_to_inner[ax]];
        }
        strides
    }

    fn is_axis_ordered(&self, axis: Axis) -> bool {
        let inner_axis = Axis(self.outer_to_inner[axis.index()]);
        self.init.is_axis_ordered(axis) || self.inner.is_axis_ordered(inner_axis)
    }

    fn invert_axis(&mut self, axis: Axis) {
        self.init.invert_axis(axis);
        let inner_axis = Axis(self.outer_to_inner[axis.index()]);
        self.inner.invert_axis(inner_axis);
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        let init = self.init.can_merge_axes(take, into);
        let inner_take = Axis(self.outer_to_inner[take.index()]);
        let inner_into = Axis(self.outer_to_inner[into.index()]);
        let inner = self.inner.can_merge_axes(inner_take, inner_into);
        init & inner
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        self.init.merge_axes(take, into);
        let inner_take = Axis(self.outer_to_inner[take.index()]);
        let inner_into = Axis(self.outer_to_inner[into.index()]);
        self.inner.merge_axes(inner_take, inner_into);
        debug_assert_eq!(self.init.len_of(take), self.inner.len_of(inner_take));
        debug_assert_eq!(self.init.len_of(into), self.inner.len_of(inner_into));
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.init.len(), self.shape().size());
        self.init.len()
    }

    fn len_of(&self, axis: Axis) -> usize {
        debug_assert_eq!(
            self.init.len_of(axis),
            self.inner.len_of(Axis(self.outer_to_inner[axis.index()])),
        );
        self.init.len_of(axis)
    }

    fn ndim(&self) -> usize {
        self.init.ndim()
    }
}

/// A source that folds over a subset of axes of an inner producer.
pub struct FoldAxesSource<S, Df, I, F>
where
    S: NdSource,
    Df: Dimension,
    I: NdSource,
{
    inner: S,
    /// Axes (of the `inner` source) that are being folded over.
    fold_axes: Df,
    /// Lengths of the axes that are being folded over (in the same order as `fold_axes`).
    fold_axis_lens: Df,
    /// First index for the fold.
    fold_first_index: Option<Df>,
    /// Mapping of "outer" axis to corresponding axis of `inner`.
    outer_to_inner: I::Dim,
    /// Source of init values for fold.
    init: I,
    f: F,
}

impl<S, Df, I, F> FoldAxesSource<S, Df, I, F>
where
    S: NdSource,
    Df: Dimension,
    I: NdSource,
{
    /// Creates a new `FoldAxesSource` instance. This is the only safe way to
    /// create a `FoldAxesSource` instance.
    fn new(inner: S, fold_axes: Df, outer_to_inner: I::Dim, init: I, f: F) -> Self {
        // Check that the ndims are consistent.
        assert!(inner.ndim() >= init.ndim());
        assert_eq!(init.ndim(), outer_to_inner.ndim());
        assert_eq!(inner.ndim() - fold_axes.ndim(), outer_to_inner.ndim());
        // Check that the lengths of the axes common to `inner` and `init` are
        // consistent.
        for (outer_ax, &inner_ax) in outer_to_inner.slice().iter().enumerate() {
            assert_eq!(init.len_of(Axis(outer_ax)), inner.len_of(Axis(inner_ax)));
        }
        // Check that each axis appears exactly once in `fold_axes` or
        // `outer_to_inner`.
        {
            let mut axes_used = S::Dim::zeros(inner.ndim());
            fold_axes.visitv(|ax| axes_used[ax] += 1);
            outer_to_inner.visitv(|ax| axes_used[ax] += 1);
            axes_used.visitv(|usage_count| assert_eq!(usage_count, 1));
        }
        // Check that the lengths of all fold axes fit in `isize`. (See the
        // constraints of `IterBorrowed::from_raw_parts`, which is called in
        // `read_once_unchecked`.)
        fold_axes.visitv(|ax| assert!(inner.len_of(Axis(ax)) <= std::isize::MAX as usize));
        unsafe { FoldAxesSource::new_unchecked(inner, fold_axes, outer_to_inner, init, f) }
    }

    unsafe fn new_unchecked(
        inner: S,
        fold_axes: Df,
        outer_to_inner: I::Dim,
        init: I,
        f: F,
    ) -> Self {
        let mut fold_axis_lens = Df::zeros(fold_axes.ndim());
        for (len, &inner_ax) in izip!(fold_axis_lens.slice_mut(), fold_axes.slice()) {
            *len = inner.len_of(Axis(inner_ax));
        }
        FoldAxesSource {
            inner,
            fold_axes,
            fold_first_index: Df::first_index(&fold_axis_lens),
            fold_axis_lens,
            outer_to_inner,
            init,
            f,
        }
    }
}

impl<S, Df, I, F> NdSource for FoldAxesSource<S, Df, I, F>
where
    S: NdSource,
    Df: Dimension,
    I: NdSource,
    F: FnMut(I::Item, S::Item) -> I::Item,
{
    type Item = I::Item;

    unsafe fn read_once_unchecked(&mut self, ptr: &Self::Ptr) -> Self::Item {
        let (ref ptr_inner, ref ptr_init) = ptr;
        let init = self.init.read_once_unchecked(ptr_init);
        let ptr_idx = self
            .fold_first_index
            .clone()
            .map(|idx| (ptr_inner.clone(), idx));
        IterBorrowed::from_raw_parts(
            &mut self.inner,
            ptr_idx,
            &self.fold_axes,
            &self.fold_axis_lens,
        )
        .fold(init, &mut self.f)
    }
}

unsafe impl<S, Df, I, F> NdAccess for FoldAxesSource<S, Df, I, F>
where
    S: NdSource,
    Df: Dimension,
    I: NdSource,
{
    type Dim = I::Dim;
    type Ptr = (S::Ptr, I::Ptr);

    fn shape(&self) -> I::Dim {
        let init_shape = self.init.shape();
        if cfg!(debug_assertions) {
            let inner_shape = self.inner.shape();
            for outer_axis in 0..self.outer_to_inner.ndim() {
                let inner_axis = self.outer_to_inner[outer_axis];
                debug_assert_eq!(inner_shape[inner_axis], init_shape[outer_axis]);
            }
        }
        init_shape
    }

    fn first_ptr(&self) -> Option<Self::Ptr> {
        match (self.inner.first_ptr(), self.init.first_ptr()) {
            (Some(inner), Some(init)) => Some((inner, init)),
            (None, None) => None,
            _ => unreachable!(),
        }
    }

    unsafe fn ptr_offset_axis(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (ref mut ptr_inner, ref mut ptr_init) = ptr;
        let inner_axis = Axis(self.outer_to_inner[axis.index()]);
        self.inner.ptr_offset_axis(ptr_inner, inner_axis, count);
        self.init.ptr_offset_axis(ptr_init, axis, count);
    }

    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        let (ref mut ptr_inner, ref mut ptr_init) = ptr;
        let inner_axis = Axis(self.outer_to_inner[axis.index()]);
        self.inner
            .ptr_offset_axis_contiguous(ptr_inner, inner_axis, count);
        self.init.ptr_offset_axis_contiguous(ptr_init, axis, count);
    }

    fn is_axis_contiguous(&self, axis: Axis) -> bool {
        let inner_axis = Axis(self.outer_to_inner[axis.index()]);
        self.inner.is_axis_contiguous(inner_axis) && self.init.is_axis_contiguous(axis)
    }

    fn len_of(&self, axis: Axis) -> usize {
        let init_len = self.init.len_of(axis);
        if cfg!(debug_assertions) {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            let inner_len = self.inner.len_of(inner_axis);
            debug_assert_eq!(inner_len, init_len);
        }
        init_len
    }

    fn ndim(&self) -> usize {
        debug_assert_eq!(self.init.ndim(), self.outer_to_inner.ndim());
        self.outer_to_inner.ndim()
    }
}
