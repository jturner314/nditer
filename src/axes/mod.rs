use crate::SubDim;
use ndarray::{Dimension, IntoDimension, Ix0};

pub use axes_for::AxesFor;

/// An object that can be converted into a list of unique axes for a given
/// dimensionality.
pub trait IntoAxesFor<D: Dimension> {
    /// The type of the specified list of axes.
    type Axes: Dimension;
    /// A type that can be converted into all the other axes.
    type IntoOthers: IntoAxesFor<D>;

    /// Returns an object that can be converted into the unique axes not
    /// included in `self`.
    fn into_others(self) -> Self::IntoOthers;

    /// Converts `self` into unique axes for an object that has dimensionality
    /// `D` and `for_ndim` dimensions.
    ///
    /// **Panics** if `for_ndim` is inconsistent with `D` or if any axes are
    /// out-of-bounds for `for_ndim`.
    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, Self::Axes>;

    /// Converts `self` into unique axes for an object that has dimensionality
    /// `D` and `for_ndim` dimensions, and simultaneously returns the unique
    /// axes not included in `self`.
    ///
    /// **Panics** if `for_ndim` is inconsistent with `D` or if any axes are
    /// out-of-bounds for `for_ndim`.
    fn into_these_and_other_axes_for(
        self,
        for_ndim: usize,
    ) -> (
        AxesFor<D, Self::Axes>,
        AxesFor<D, <Self::IntoOthers as IntoAxesFor<D>>::Axes>,
    );
}

/// A list of unique axes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Axes<X: Dimension>(X);

/// Creates a new list of unique axes.
///
/// **Panics** if the axes are not unique.
pub fn axes<T>(axes: T) -> Axes<T::Dim>
where
    T: IntoDimension,
{
    let axes = axes.into_dimension();

    // Check for uniqueness.
    let mut tmp = axes.clone();
    tmp.slice_mut().sort_unstable();
    for i in 0..(tmp.ndim() - 1) {
        assert_ne!(tmp[i], tmp[i + 1], "Axes must be unique.");
    }

    Axes(axes)
}

impl<D, X, E> IntoAxesFor<D> for Axes<X>
where
    D: Dimension + SubDim<X, Out = E>,
    X: Dimension,
    E: Dimension,
{
    type Axes = X;
    type IntoOthers = AxesExcept<X>;

    /// The axes that result from calling `.into_axes_for()` on the return
    /// value are guaranteed to be in order.
    fn into_others(self) -> AxesExcept<X> {
        AxesExcept(self.0)
    }

    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, X> {
        AxesFor::from_axes(self, for_ndim)
    }

    /// The "other" axes are guaranteed to be in order.
    fn into_these_and_other_axes_for(self, for_ndim: usize) -> (AxesFor<D, X>, AxesFor<D, E>) {
        AxesFor::these_and_others_from_axes(self, for_ndim)
    }
}

/// A list of unique axes that should be excluded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxesExcept<E: Dimension>(E);

/// Creates a new list of unique axes that should be excluded.
///
/// **Panics** if the axes are not unique.
pub fn axes_except<T>(axes: T) -> AxesExcept<T::Dim>
where
    T: IntoDimension,
{
    let axes = axes.into_dimension();

    // Check for uniqueness.
    let mut tmp = axes.clone();
    tmp.slice_mut().sort_unstable();
    for i in 0..(tmp.ndim() - 1) {
        assert_ne!(tmp[i], tmp[i + 1], "Axes must be unique.");
    }

    AxesExcept(axes)
}

impl<D, X, E> IntoAxesFor<D> for AxesExcept<E>
where
    D: Dimension + SubDim<E, Out = X>,
    X: Dimension,
    E: Dimension,
{
    type Axes = X;
    type IntoOthers = Axes<E>;

    fn into_others(self) -> Axes<E> {
        Axes(self.0)
    }

    /// The resulting axes are guaranteed to be in order.
    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, X> {
        AxesFor::others_from_axes(&Axes(self.0), for_ndim)
    }

    /// The "these" axes are guaranteed to be in order.
    fn into_these_and_other_axes_for(self, for_ndim: usize) -> (AxesFor<D, X>, AxesFor<D, E>) {
        let (others, these) = AxesFor::these_and_others_from_axes(Axes(self.0), for_ndim);
        (these, others)
    }
}

/// Represents all axes for an object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxesAll;

/// Creates a representation of all the axes for an object.
pub fn axes_all() -> AxesAll {
    AxesAll
}

impl<D> IntoAxesFor<D> for AxesAll
where
    D: Dimension,
{
    type Axes = D;
    type IntoOthers = AxesNone;

    fn into_others(self) -> AxesNone {
        AxesNone
    }

    /// The resulting axes are guaranteed to be in order.
    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, D> {
        // This line ensures that `for_ndim` is consistent with `D`.
        let mut axes = D::zeros(for_ndim);
        for (i, ax) in axes.slice_mut().iter_mut().enumerate() {
            *ax = i;
        }
        unsafe { AxesFor::from_axes_unchecked(Axes(axes), for_ndim) }
    }

    /// The axes are guaranteed to be in order.
    fn into_these_and_other_axes_for(self, for_ndim: usize) -> (AxesFor<D, D>, AxesFor<D, Ix0>) {
        let these = self.into_axes_for(for_ndim);
        let others = AxesNone.into_axes_for(for_ndim);
        (these, others)
    }
}

/// Represents none of the axes for an object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxesNone;

/// Creates a representation of none of the axes for an object.
pub fn axes_none() -> AxesNone {
    AxesNone
}

impl<D> IntoAxesFor<D> for AxesNone
where
    D: Dimension,
{
    type Axes = Ix0;
    type IntoOthers = AxesAll;

    fn into_others(self) -> AxesAll {
        AxesAll
    }

    fn into_axes_for(self, for_ndim: usize) -> AxesFor<D, Ix0> {
        AxesFor::from_axes(Axes(Ix0()), for_ndim)
    }

    /// The axes are guaranteed to be in order.
    fn into_these_and_other_axes_for(self, for_ndim: usize) -> (AxesFor<D, Ix0>, AxesFor<D, D>) {
        let these = self.into_axes_for(for_ndim);
        let others = AxesAll.into_axes_for(for_ndim);
        (these, others)
    }
}

mod axes_for;

#[cfg(test)]
mod tests {
    use super::{axes, axes_all, axes_except, axes_none, IntoAxesFor};
    use ndarray::prelude::*;

    #[test]
    #[should_panic]
    fn axes_checks_uniqueness() {
        let _ = axes((2, 3, 3, 4));
    }

    #[test]
    #[should_panic]
    fn axes_into_axes_for_checks_ndim() {
        let _ = IntoAxesFor::<Ix4>::into_axes_for(axes((0, 1, 2)), 5);
    }

    #[test]
    #[should_panic]
    fn axes_into_axes_for_checks_in_bounds() {
        let _ = IntoAxesFor::<Ix4>::into_axes_for(axes((2, 3, 4)), 4);
    }

    #[test]
    fn axes_is_correct() {
        let this = IntoAxesFor::<Ix4>::into_axes_for(axes((2, 3, 1)), 4);
        assert_eq!(this.for_ndim(), 4);
        assert_eq!(this.num_axes(), 3);
        assert_eq!(this.into_inner(), Ix3(2, 3, 1));

        let this = IntoAxesFor::<IxDyn>::into_axes_for(axes((2, 3, 1)), 4);
        assert_eq!(this.for_ndim(), 4);
        assert_eq!(this.num_axes(), 3);
        assert_eq!(this.into_inner(), Ix3(2, 3, 1));

        let other = IntoAxesFor::<Ix6>::into_axes_for(
            IntoAxesFor::<Ix6>::into_others(axes((2, 5, 1))),
            6,
        );
        assert_eq!(other.for_ndim(), 6);
        assert_eq!(other.num_axes(), 3);
        assert_eq!(other.into_inner(), Ix3(0, 3, 4));

        let other = IntoAxesFor::<IxDyn>::into_axes_for(
            IntoAxesFor::<IxDyn>::into_others(axes((2, 5, 1))),
            6,
        );
        assert_eq!(other.for_ndim(), 6);
        assert_eq!(other.num_axes(), 3);
        assert_eq!(other.into_inner(), IxDyn(&[0, 3, 4]));
    }

    #[test]
    #[should_panic]
    fn axes_except_checks_uniqueness() {
        let _ = axes_except((2, 3, 3, 4));
    }

    #[test]
    #[should_panic]
    fn axes_except_into_axes_for_checks_ndim() {
        let _ = IntoAxesFor::<Ix4>::into_axes_for(axes_except((0, 1, 2)), 5);
    }

    #[test]
    #[should_panic]
    fn axes_except_into_axes_for_checks_in_bounds() {
        let _ = IntoAxesFor::<Ix4>::into_axes_for(axes_except((2, 3, 4)), 4);
    }

    #[test]
    fn axes_except_is_correct() {
        let remaining = IntoAxesFor::<Ix5>::into_axes_for(axes_except((2, 4, 1)), 5);
        assert_eq!(remaining.for_ndim(), 5);
        assert_eq!(remaining.num_axes(), 2);
        assert_eq!(remaining.into_inner(), Ix2(0, 3));

        let remaining = IntoAxesFor::<IxDyn>::into_axes_for(axes_except((2, 4, 1)), 5);
        assert_eq!(remaining.for_ndim(), 5);
        assert_eq!(remaining.num_axes(), 2);
        assert_eq!(remaining.into_inner(), IxDyn(&[0, 3]));

        let except = IntoAxesFor::<Ix5>::into_axes_for(
            IntoAxesFor::<Ix5>::into_others(axes_except((2, 4, 1))),
            5,
        );
        assert_eq!(except.for_ndim(), 5);
        assert_eq!(except.num_axes(), 3);
        assert_eq!(except.into_inner(), Ix3(2, 4, 1));

        let except = IntoAxesFor::<IxDyn>::into_axes_for(
            IntoAxesFor::<IxDyn>::into_others(axes_except((2, 4, 1))),
            5,
        );
        assert_eq!(except.for_ndim(), 5);
        assert_eq!(except.num_axes(), 3);
        assert_eq!(except.into_inner(), Ix3(2, 4, 1));
    }

    #[test]
    fn axes_all_is_correct() {
        let all = IntoAxesFor::<Ix5>::into_axes_for(axes_all(), 5);
        assert_eq!(all.for_ndim(), 5);
        assert_eq!(all.num_axes(), 5);
        assert_eq!(all.into_inner(), Ix5(0, 1, 2, 3, 4));

        let all = IntoAxesFor::<IxDyn>::into_axes_for(axes_all(), 5);
        assert_eq!(all.for_ndim(), 5);
        assert_eq!(all.num_axes(), 5);
        assert_eq!(all.into_inner(), IxDyn(&[0, 1, 2, 3, 4]));

        let other =
            IntoAxesFor::<Ix5>::into_axes_for(IntoAxesFor::<Ix5>::into_others(axes_all()), 5);
        assert_eq!(other.for_ndim(), 5);
        assert_eq!(other.num_axes(), 0);
        assert_eq!(other.into_inner(), Ix0());

        let other = IntoAxesFor::<IxDyn>::into_axes_for(
            IntoAxesFor::<IxDyn>::into_others(axes_all()),
            5,
        );
        assert_eq!(other.for_ndim(), 5);
        assert_eq!(other.num_axes(), 0);
        assert_eq!(other.into_inner(), Ix0());
    }

    #[test]
    fn axes_none_is_correct() {
        let all = IntoAxesFor::<Ix5>::into_axes_for(axes_none(), 5);
        assert_eq!(all.for_ndim(), 5);
        assert_eq!(all.num_axes(), 0);
        assert_eq!(all.into_inner(), Ix0());

        let all = IntoAxesFor::<IxDyn>::into_axes_for(axes_none(), 5);
        assert_eq!(all.for_ndim(), 5);
        assert_eq!(all.num_axes(), 0);
        assert_eq!(all.into_inner(), Ix0());

        let other =
            IntoAxesFor::<Ix5>::into_axes_for(IntoAxesFor::<Ix5>::into_others(axes_none()), 5);
        assert_eq!(other.for_ndim(), 5);
        assert_eq!(other.num_axes(), 5);
        assert_eq!(other.into_inner(), Ix5(0, 1, 2, 3, 4));

        let other = IntoAxesFor::<IxDyn>::into_axes_for(
            IntoAxesFor::<IxDyn>::into_others(axes_none()),
            5,
        );
        assert_eq!(other.for_ndim(), 5);
        assert_eq!(other.num_axes(), 5);
        assert_eq!(other.into_inner(), IxDyn(&[0, 1, 2, 3, 4]));
    }
}
