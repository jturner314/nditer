use crate::{assert_valid_unique_axes, DimensionExt, NdAccess, NdProducer, NdSource};
use itertools::izip;
use ndarray::{Axis, Dimension};

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
    pub(crate) fn new(producer: P, axes: P::Dim) -> Self {
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
pub(crate) struct IterBorrowed<'a, S: 'a + NdSource, D: 'a + Dimension> {
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
    pub(crate) unsafe fn from_raw_parts(
        source: &'a mut S,
        ptr_idx: Option<(S::Ptr, D)>,
        axes: &'a D,
        axis_lens: &'a D,
    ) -> Self {
        // A few sanity checks.
        if cfg!(debug_assertions) {
            assert_valid_unique_axes::<S::Dim>(source.ndim(), axes.slice());
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
