//! Adapters to perform various operations on producers.
//!
//! Most adapters are created using the convenience methods on `NdProducer`,
//! but some (e.g. `RepeatWith`) can be created manually.

pub use self::broadcast::{BroadcastProducer, BroadcastSource};
pub use self::cloned::Cloned;
pub use self::fold_axes::{FoldAxesProducer, FoldAxesSource};
pub use self::force_axes_ordered::ForceAxesOrdered;
pub use self::indexed::{IndexedProducer, IndexedSource};
pub use self::inspect::Inspect;
pub use self::map::Map;
pub use self::repeat::{into_repeat, repeat};
pub use self::repeat_with::{into_repeat_with, repeat_with};
pub use self::select_indices_axis::SelectIndicesAxis;
pub use self::zip::Zip;

/// Implements all required (and a couple more) `NdReshape` methods.
macro_rules! impl_ndreshape_methods_for_wrapper {
    ($inner:ident) => {
        fn shape(&self) -> Self::Dim {
            $crate::NdReshape::shape(&self.$inner)
        }
        fn approx_abs_strides(&self) -> Self::Dim {
            $crate::NdReshape::approx_abs_strides(&self.$inner)
        }
        fn is_axis_ordered(&self, axis: Axis) -> bool {
            $crate::NdReshape::is_axis_ordered(&self.$inner, axis)
        }
        fn invert_axis(&mut self, axis: Axis) {
            $crate::NdReshape::invert_axis(&mut self.$inner, axis)
        }
        fn can_merge_axes(&self, take: Axis, into: Axis) -> $crate::CanMerge {
            $crate::NdReshape::can_merge_axes(&self.$inner, take, into)
        }
        fn merge_axes(&mut self, take: Axis, into: Axis) {
            $crate::NdReshape::merge_axes(&mut self.$inner, take, into)
        }
        fn len(&self) -> usize {
            $crate::NdReshape::len(&self.$inner)
        }
        fn len_of(&self, axis: Axis) -> usize {
            $crate::NdReshape::len_of(&self.$inner, axis)
        }
        fn is_empty(&self) -> bool {
            $crate::NdReshape::is_empty(&self.$inner)
        }
        fn ndim(&self) -> usize {
            $crate::NdReshape::ndim(&self.$inner)
        }
    };
}

/// Implements all required `NdAccess` methods.
macro_rules! impl_ndaccess_methods_for_wrapper {
    ($inner:ident) => {
        fn shape(&self) -> Self::Dim {
            self.$inner.shape()
        }
        fn first_ptr(&self) -> Option<T::Ptr> {
            self.$inner.first_ptr()
        }
        unsafe fn ptr_offset_axis(&self, ptr: &mut T::Ptr, axis: Axis, count: isize) {
            self.$inner.ptr_offset_axis(ptr, axis, count)
        }
        unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut T::Ptr, axis: Axis, count: isize) {
            self.$inner.ptr_offset_axis_contiguous(ptr, axis, count)
        }
        fn is_axis_contiguous(&self, axis: Axis) -> bool {
            self.$inner.is_axis_contiguous(axis)
        }
        fn len_of(&self, axis: Axis) -> usize {
            self.$inner.len_of(axis)
        }
        fn is_empty(&self) -> bool {
            self.$inner.is_empty()
        }
        fn ndim(&self) -> usize {
            self.$inner.ndim()
        }
    }
}

mod broadcast;
mod cloned;
mod fold_axes;
mod force_axes_ordered;
mod indexed;
mod inspect;
mod map;
mod repeat;
mod repeat_with;
mod select_indices_axis;
mod zip;
