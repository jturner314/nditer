use crate::{
    errors::BroadcastError, CanMerge, IntoAxesFor, NdAccess, NdProducer, NdReshape, NdSource,
    NdSourceRepeat,
};
use ndarray::{Axis, Dimension};

/// A wrapper that broadcasts a producer to a larger shape.
///
/// This struct is created by the `broadcast` method on `NdProducer`. See its
/// documentation for more.
#[derive(Clone, Debug)]
pub struct BroadcastProducer<T, D>
where
    T: NdReshape,
{
    /// The inner producer that is being broadcasted.
    inner: T,
    /// The outer shape, i.e. the shape that the producer is broadcasted to.
    shape: D,
    /// Indicates which axes are not broadcasted (are passed through to the
    /// inner producer).
    ///
    /// The value of `pass_through[outer_axis]` is `1` if the axis is passed
    /// through (not broadcasted) or `0` if it is not passed through (is
    /// broadcasted).
    pass_through: D,
    /// Mapping of outer axes to inner axes (`inner_axis = outer_to_inner[outer_axis]`).
    ///
    /// Note that the values are only correct for axes that are passed-through
    /// (`pass_through[outer_axis] != 0`).
    outer_to_inner: D,
    /// Mapping of inner axes to outer axes (`outer_axis = inner_to_outer[inner_axis]`).
    inner_to_outer: T::Dim,
}

/// A source that broadcasts a source to a larger shape.
pub struct BroadcastSource<T, D>
where
    T: NdAccess,
{
    inner: T,
    shape: D,
    pass_through: D,
    outer_to_inner: D,
}

impl<T, D> BroadcastProducer<T, D>
where
    T: NdReshape,
    D: Dimension,
{
    /// Creates a new producer that broadcasts `inner` to a larger shape.
    pub(crate) fn try_new(
        inner: T,
        axes_mapping: impl IntoAxesFor<D, Axes = T::Dim>,
        shape: D,
    ) -> Result<Self, BroadcastError> {
        let axes_mapping = axes_mapping.into_axes_for(shape.ndim());
        assert_eq!(inner.ndim(), axes_mapping.num_axes());
        // Compute `outer_to_inner` and `pass_through`, and check that the
        // lengths of passed-through axes match `shape`.
        let mut outer_to_inner = D::zeros(shape.ndim());
        let mut pass_through = D::zeros(shape.ndim());
        for (inner_ax, &outer_ax) in axes_mapping.slice().iter().enumerate() {
            outer_to_inner[outer_ax] = inner_ax;
            let inner_len = inner.len_of(Axis(inner_ax));
            if inner_len != 1 {
                if shape[outer_ax] != inner_len {
                    return Err(BroadcastError::new(&inner.shape(), &shape, &axes_mapping));
                }
                pass_through[outer_ax] = 1;
            }
        }
        Ok(BroadcastProducer {
            inner,
            inner_to_outer: axes_mapping.into_inner(),
            outer_to_inner,
            pass_through,
            shape,
        })
    }
}

impl<T, D> NdProducer for BroadcastProducer<T, D>
where
    T: NdProducer,
    T::Source: NdSourceRepeat,
    D: Dimension,
{
    type Item = T::Item;
    type Source = BroadcastSource<T::Source, D>;
    fn into_source(self) -> Self::Source {
        let inner = self.inner.into_source();
        let shape = self.shape;
        let pass_through = self.pass_through;
        let outer_to_inner = self.outer_to_inner;
        {
            let mut usage_counts = T::Dim::zeros(inner.ndim());
            // Check that the passed-through axes are mapped to in-bounds axes
            // of `inner.ndim()`and that their lengths are consistent.
            for (outer_axis, &pass_through) in pass_through.slice().iter().enumerate() {
                if pass_through != 0 {
                    let inner_axis = outer_to_inner[outer_axis];
                    usage_counts[inner_axis] += 1;
                    assert_eq!(shape[outer_axis], inner.len_of(Axis(inner_axis)));
                }
            }
            // Check that the passed-through axes are mapped to unique axes of
            // `inner.ndim()`, and check that all axes of `inner` that aren't
            // passed-into have length 1.
            for (inner_axis, &usage_count) in usage_counts.slice().iter().enumerate() {
                match usage_count {
                    0 => assert!(inner.len_of(Axis(inner_axis)) == 1),
                    1 => {}
                    _ => panic!("There should not be duplicate passed-through axes."),
                }
            }
        }
        BroadcastSource {
            inner,
            shape,
            pass_through,
            outer_to_inner,
        }
    }
}

impl<T, D> NdReshape for BroadcastProducer<T, D>
where
    T: NdReshape,
    D: Dimension,
{
    type Dim = D;

    fn shape(&self) -> Self::Dim {
        self.shape.clone()
    }

    fn approx_abs_strides(&self) -> Self::Dim {
        let inner_strides = self.inner.approx_abs_strides();
        // Zero stride for all broadcasted axes.
        let mut outer_strides = D::zeros(self.shape.ndim());
        for (inner_axis, &inner_stride) in inner_strides.slice().iter().enumerate() {
            outer_strides[self.inner_to_outer[inner_axis]] = inner_stride;
        }
        outer_strides
    }

    fn is_axis_ordered(&self, axis: Axis) -> bool {
        if self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            self.inner.is_axis_ordered(inner_axis)
        } else {
            false
        }
    }

    fn invert_axis(&mut self, axis: Axis) {
        if self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            self.inner.invert_axis(inner_axis);
        }
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        let pass_take = self.pass_through[take.index()];
        let pass_into = self.pass_through[into.index()];
        match (pass_take, pass_into) {
            (1, 1) => {
                let inner_take = Axis(self.outer_to_inner[take.index()]);
                let inner_into = Axis(self.outer_to_inner[into.index()]);
                self.inner.can_merge_axes(inner_take, inner_into)
            }
            (0, 0) if take != into || self.len_of(take) <= 1 => CanMerge::Always,
            _ => CanMerge::Never,
        }
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        assert!(take != into || self.len_of(take) <= 1);

        // Update outer shape.
        let prod = self.shape[take.index()] * self.shape[into.index()];
        self.shape[into.index()] = prod;
        self.shape[take.index()] = if prod == 0 { 0 } else { 1 };

        // Pass through to inner producer if necessary.
        let pass_take = self.pass_through[take.index()];
        let pass_into = self.pass_through[into.index()];
        match (pass_take, pass_into) {
            (1, 1) => {
                let inner_take = Axis(self.outer_to_inner[take.index()]);
                let inner_into = Axis(self.outer_to_inner[into.index()]);
                self.inner.merge_axes(inner_take, inner_into);
                debug_assert_eq!(self.shape[take.index()], self.inner.len_of(inner_take));
                debug_assert_eq!(self.shape[into.index()], self.inner.len_of(inner_into));
            }
            (0, 0) => {}
            _ => panic!("Invalid attempt to merge broadcasted and non-broadcasted axes."),
        }
    }

    fn len(&self) -> usize {
        self.shape.size()
    }

    fn len_of(&self, axis: Axis) -> usize {
        let len_of = self.shape[axis.index()];
        if cfg!(debug_assertions) && self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            assert_eq!(len_of, self.inner.len_of(inner_axis));
        }
        len_of
    }

    fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}

impl<T, D> NdSource for BroadcastSource<T, D>
where
    T: NdSourceRepeat,
    D: Dimension,
{
    type Item = T::Item;
    unsafe fn read_once_unchecked(&mut self, ptr: &T::Ptr) -> T::Item {
        // It's safe to call `read_once_unchecked` multiple times for the same
        // index of the inner producer because `T: NdSourceRepeat`. (This
        // property is necessary because broadcasting repeats elements.)
        self.inner.read_once_unchecked(ptr)
    }
}

unsafe impl<T, D> NdAccess for BroadcastSource<T, D>
where
    T: NdAccess,
    D: Dimension,
{
    type Dim = D;
    type Ptr = T::Ptr;

    fn shape(&self) -> Self::Dim {
        self.shape.clone()
    }

    fn first_ptr(&self) -> Option<Self::Ptr> {
        if self.is_empty() {
            None
        } else {
            self.inner.first_ptr()
        }
    }

    unsafe fn ptr_offset_axis(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        if self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            self.inner.ptr_offset_axis(ptr, inner_axis, count)
        }
    }

    unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut Self::Ptr, axis: Axis, count: isize) {
        if self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            self.inner
                .ptr_offset_axis_contiguous(ptr, inner_axis, count)
        }
    }

    fn is_axis_contiguous(&self, axis: Axis) -> bool {
        if self.pass_through[axis.index()] != 0 {
            let inner_axis = Axis(self.outer_to_inner[axis.index()]);
            self.inner.is_axis_contiguous(inner_axis)
        } else {
            true
        }
    }

    fn len_of(&self, axis: Axis) -> usize {
        self.shape[axis.index()]
    }

    fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}

unsafe impl<T, D> NdSourceRepeat for BroadcastSource<T, D>
where
    T: NdSourceRepeat,
    D: Dimension,
{
}
