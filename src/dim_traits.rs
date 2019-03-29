use ndarray::{Dim, Dimension, IxDyn};

/// Defines subtraction of dimension sizes.
pub trait SubDim<RHS: Dimension>: Dimension {
    /// The type-level equivalent of subtraction (`Self - RHS`).
    type Out: Dimension;
}

impl<D: Dimension> SubDim<IxDyn> for D {
    type Out = IxDyn;
}

macro_rules! impl_subdim_const {
    ($left:expr, $right:expr) => {
        impl SubDim<Dim<[usize; $right]>> for Dim<[usize; $left]> {
            type Out = Dim<[usize; $left - $right]>;
        }
    };
}

impl_subdim_const!(0, 0);

impl_subdim_const!(1, 0);
impl_subdim_const!(1, 1);

impl_subdim_const!(2, 0);
impl_subdim_const!(2, 1);
impl_subdim_const!(2, 2);

impl_subdim_const!(3, 0);
impl_subdim_const!(3, 1);
impl_subdim_const!(3, 2);
impl_subdim_const!(3, 3);

impl_subdim_const!(4, 0);
impl_subdim_const!(4, 1);
impl_subdim_const!(4, 2);
impl_subdim_const!(4, 3);
impl_subdim_const!(4, 4);

impl_subdim_const!(5, 0);
impl_subdim_const!(5, 1);
impl_subdim_const!(5, 2);
impl_subdim_const!(5, 3);
impl_subdim_const!(5, 4);
impl_subdim_const!(5, 5);

impl_subdim_const!(6, 0);
impl_subdim_const!(6, 1);
impl_subdim_const!(6, 2);
impl_subdim_const!(6, 3);
impl_subdim_const!(6, 4);
impl_subdim_const!(6, 5);
impl_subdim_const!(6, 6);
