use crate::axes::{Axes, AxesExcept, AxesMask, IntoAxesFor};
use crate::{DimensionExt, SubDim};
use ndarray::{Axis, Dimension};
use std::marker::PhantomData;
use std::ops::Index;

/// A list of unique axes for an object of dimension `D`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxesFor<D: Dimension, X: Dimension> {
    /// The list of axes. They're guaranteed to be unique and in-bounds of
    /// `for_ndim`.
    axes: Axes<X>,
    /// The number of dimensions of the object these axes are for. This is
    /// guaranteed to be consistent with `for_dim`.
    for_ndim: usize,
    /// The dimensionality of the object these axes are for.
    for_dim: PhantomData<D>,
}

/// Used in `AxesFor` method implementations.
///
/// **Panics** if `for_ndim` is inconsistent with `D` or if any of `axes` are
/// out-of-bounds of `for_ndim`.
fn check_axes<D, X>(axes: &Axes<X>, for_ndim: usize)
where
    D: Dimension,
    X: Dimension,
{
    // Check that `for_ndim` is consistent with `D`.
    D::NDIM.map(|n| assert_eq!(n, for_ndim));

    // Check that the axes are in-bounds. (Uniqueness is checked in the `Axes`
    // constructor.)
    axes.0.visitv(|ax| assert!(ax < for_ndim));
}

impl<D, X> AxesFor<D, X>
where
    D: Dimension,
    X: Dimension,
{
    /// Creates a new `AxesFor` instance for the given axes and `for_ndim`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `for_ndim` is consistent with `D` and that
    /// all `axes` are in-bounds of `for_ndim`.
    pub(crate) unsafe fn from_axes_unchecked(axes: Axes<X>, for_ndim: usize) -> AxesFor<D, X> {
        AxesFor {
            axes,
            for_ndim,
            for_dim: PhantomData,
        }
    }

    /// Computes all the other axes for `for_ndim` (i.e. all axes except for `axes`).
    ///
    /// **Panics** if `O` is incorrect.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `for_ndim` is consistent with `D` and that
    /// all `axes` are in-bounds of `for_ndim`.
    pub(crate) unsafe fn others_from_axes_unchecked<O: Dimension>(
        axes: &Axes<X>,
        for_ndim: usize,
    ) -> AxesFor<D, O> {
        // Determine the other axes. (This also checks that `O` is correct.)
        let mut others = O::zeros(for_ndim - axes.0.ndim());
        let mut done = 0;
        for ax_for in 0..for_ndim {
            if !axes.0.slice().contains(&ax_for) {
                others[done] = ax_for;
                done += 1;
            }
        }
        assert_eq!(done, others.ndim());

        AxesFor::from_axes_unchecked(Axes(others), for_ndim)
    }

    /// Creates a new `AxesFor` instance for the given axes and `for_ndim`.
    ///
    /// **Panics** if `for_ndim` is inconsistent with `D` or if any of `axes`
    /// are out-of-bounds of `for_ndim`.
    pub(crate) fn from_axes(axes: Axes<X>, for_ndim: usize) -> AxesFor<D, X> {
        check_axes::<D, X>(&axes, for_ndim);
        unsafe { AxesFor::from_axes_unchecked(axes, for_ndim) }
    }

    /// Computes all the other axes for `for_ndim` (i.e. all axes except for `axes`).
    ///
    /// **Panics** if `for_ndim` is inconsistent with `D`, if any of `axes` are
    /// out-of-bounds of `for_ndim`, or if `O` is incorrect.
    pub(crate) fn others_from_axes<O: Dimension>(axes: &Axes<X>, for_ndim: usize) -> AxesFor<D, O> {
        check_axes::<D, X>(&axes, for_ndim);
        unsafe { AxesFor::others_from_axes_unchecked(axes, for_ndim) }
    }

    /// Creates a new `AxesFor` instance for the given axes and `for_ndim`, and
    /// simultaneously computes all the other axes for `for_ndim` (i.e. all
    /// axes except for `axes`).
    ///
    /// **Panics** if `for_ndim` is inconsistent with `D`, if any of `axes` are
    /// out-of-bounds of `for_ndim`, or if `O` is incorrect.
    pub(crate) fn these_and_others_from_axes<O: Dimension>(
        axes: Axes<X>,
        for_ndim: usize,
    ) -> (AxesFor<D, X>, AxesFor<D, O>) {
        check_axes::<D, X>(&axes, for_ndim);
        let others = unsafe { AxesFor::others_from_axes_unchecked(&axes, for_ndim) };
        let these = unsafe { AxesFor::from_axes_unchecked(axes, for_ndim) };
        (these, others)
    }

    /// Returns the number of dimensions these axes are for.
    #[inline]
    pub fn for_ndim(&self) -> usize {
        if let Some(for_ndim) = D::NDIM {
            debug_assert_eq!(for_ndim, self.for_ndim);
            for_ndim
        } else {
            self.for_ndim
        }
    }

    /// Returns the number of axes.
    #[inline]
    pub fn num_axes(&self) -> usize {
        self.axes.0.ndim()
    }

    /// Swaps the axes at the given indices.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.axes.0.slice_mut().swap(a, b)
    }

    /// Rolls the axes at the given indices by the given shift.
    pub(crate) fn roll<I>(&mut self, indices: I, shift: isize)
    where
        I: std::slice::SliceIndex<[usize], Output = [usize]>,
    {
        roll(&mut self.axes.0.slice_mut()[indices], shift)
    }

    /// Sorts the axes.
    pub fn sort_unstable(&mut self) {
        self.axes.0.slice_mut().sort_unstable()
    }

    /// Sorts the axes with a comparator function.
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(Axis, Axis) -> std::cmp::Ordering,
    {
        self.axes
            .0
            .slice_mut()
            .sort_unstable_by(move |&a, &b| compare(Axis(a), Axis(b)))
    }

    /// Returns a slice of the axes.
    #[inline]
    pub(crate) fn slice(&self) -> &[usize] {
        self.axes.0.slice()
    }

    /// Returns the inner representation of the axes.
    #[inline]
    pub(crate) fn into_inner(self) -> X {
        self.axes.0
    }

    /// Calls `f` for each of the axes.
    pub fn visitv<F>(&self, mut f: F)
    where
        F: FnMut(Axis),
    {
        self.axes.0.visitv(move |ax| f(Axis(ax)))
    }

    /// Applies the mapping to the axes.
    pub fn mapv_to_dim<F>(&self, mut f: F) -> X
    where
        F: FnMut(Axis) -> usize,
    {
        self.axes.0.mapv(move |ax| f(Axis(ax)))
    }
}

impl<D, X> Index<usize> for AxesFor<D, X>
where
    D: Dimension,
    X: Dimension,
{
    // TODO: Change this to `Axis`.
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize {
        &self.axes.0[index]
    }
}

impl<D, X, E> IntoAxesFor<D> for AxesFor<D, X>
where
    D: Dimension + SubDim<X, Out = E>,
    X: Dimension,
    E: Dimension,
{
    type Axes = X;
    // TODO: Don't lose information (D validation) here.
    type IntoOthers = AxesExcept<X>;

    fn into_others(self) -> AxesExcept<X> {
        IntoAxesFor::<D>::into_others(self.axes)
    }

    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, Self::Axes> {
        assert_eq!(for_ndim, self.for_ndim);
        self
    }

    fn into_these_and_other_axes_for(self, for_ndim: usize) -> (AxesFor<D, X>, AxesFor<D, E>) {
        assert_eq!(for_ndim, self.for_ndim);
        let others = unsafe { AxesFor::others_from_axes_unchecked(&self.axes, for_ndim) };
        (self, others)
    }

    fn into_axes_mask(self, for_ndim: usize) -> AxesMask<D, X> {
        assert_eq!(for_ndim, self.for_ndim);
        self.axes.into_axes_mask(for_ndim)
    }
}

/// Rolls the slice by the given shift.
///
/// Rolling is like a shift, except that elements shifted off the end are moved
/// to the other end. Rolling is performed in the direction of `shift`
/// (positive for right, negative for left).
fn roll<T>(slice: &mut [T], mut shift: isize) {
    let len = slice.len();
    if len <= 1 {
        return;
    }

    // Minimize the absolute shift.
    shift = shift % len as isize;
    if shift > len as isize / 2 {
        shift -= len as isize;
    } else if shift < -(len as isize) / 2 {
        shift += len as isize;
    }

    // Perform the roll.
    if shift >= 0 {
        for _ in 0..shift {
            for i in 0..(len - 1) {
                slice.swap(i, len - 1);
            }
        }
    } else {
        for _ in 0..(-shift) {
            for i in (1..len).rev() {
                slice.swap(i, 0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::roll;

    #[test]
    fn roll_empty() {
        for shift in -2..2 {
            let mut data: [i32; 0] = [];
            roll(&mut data, shift);
            assert_eq!(data, []);
        }
    }

    #[test]
    fn roll_one_element() {
        for shift in -2..2 {
            let mut data = [1];
            roll(&mut data, shift);
            assert_eq!(data, [1]);
        }
    }

    #[test]
    fn roll_two_elements() {
        for shift in -3..3 {
            let mut data = [1, 2];
            roll(&mut data, shift);
            if (shift % 2).abs() == 0 {
                assert_eq!(data, [1, 2]);
            } else {
                assert_eq!(data, [2, 1]);
            }
        }
    }

    #[test]
    fn roll_five_elements() {
        fn check_roll(orig: &mut [i32], shift: isize, rolled: &[i32]) {
            roll(orig, shift);
            println!("shift = {}", shift);
            assert_eq!(orig, rolled);
        }
        check_roll(&mut [1, 2, 3, 4, 5], -7, &[3, 4, 5, 1, 2]);
        check_roll(&mut [1, 2, 3, 4, 5], -6, &[2, 3, 4, 5, 1]);
        check_roll(&mut [1, 2, 3, 4, 5], -5, &[1, 2, 3, 4, 5]);
        check_roll(&mut [1, 2, 3, 4, 5], -4, &[5, 1, 2, 3, 4]);
        check_roll(&mut [1, 2, 3, 4, 5], -3, &[4, 5, 1, 2, 3]);
        check_roll(&mut [1, 2, 3, 4, 5], -2, &[3, 4, 5, 1, 2]);
        check_roll(&mut [1, 2, 3, 4, 5], -1, &[2, 3, 4, 5, 1]);
        check_roll(&mut [1, 2, 3, 4, 5], 0, &[1, 2, 3, 4, 5]);
        check_roll(&mut [1, 2, 3, 4, 5], 1, &[5, 1, 2, 3, 4]);
        check_roll(&mut [1, 2, 3, 4, 5], 2, &[4, 5, 1, 2, 3]);
        check_roll(&mut [1, 2, 3, 4, 5], 3, &[3, 4, 5, 1, 2]);
        check_roll(&mut [1, 2, 3, 4, 5], 4, &[2, 3, 4, 5, 1]);
        check_roll(&mut [1, 2, 3, 4, 5], 5, &[1, 2, 3, 4, 5]);
        check_roll(&mut [1, 2, 3, 4, 5], 6, &[5, 1, 2, 3, 4]);
        check_roll(&mut [1, 2, 3, 4, 5], 7, &[4, 5, 1, 2, 3]);
    }
}
