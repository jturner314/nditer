use crate::axes::Axes;
use crate::DimensionExt;
use ndarray::Dimension;
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
