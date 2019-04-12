use crate::{
    axes, CanMerge, IntoNdProducer, NdAccess, NdProducer, NdReshape, NdSource, NdSourceRepeat,
    SubDim,
};
use itertools::izip;
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RawData, RawDataClone, RawDataMut, RawViewRepr, Slice, ViewRepr};
use std::marker::PhantomData;

/// Extension trait for `ndarray::ArrayBase`.
///
/// This trait provides methods for creating producers and various useful
/// methods implemented in terms of producers.
pub trait ArrayBaseExt<A, S, D>
where
    S: RawData<Elem = A>,
{
    /// Creates a producer of references (`&A`) to the elements in the array.
    fn producer(&self) -> ArrayBaseProducer<ViewRepr<&A>, D>
    where
        S: Data;

    /// Creates a producer of mutable references (`&mut A`) to the elements in
    /// the array.
    fn producer_mut(&mut self) -> ArrayBaseProducer<ViewRepr<&mut A>, D>
    where
        S: DataMut;

    /// Creates a producer of raw pointers (`*const A`) to the elements in the
    /// array.
    fn raw_producer(&self) -> ArrayBaseProducer<RawViewRepr<*const A>, D>;

    /// Creates a producer of mutable raw pointers (`*mut A`) to the elements
    /// in the array.
    fn raw_producer_mut(&mut self) -> ArrayBaseProducer<RawViewRepr<*mut A>, D>
    where
        S: RawDataMut;

    /// Iterates over pairs of consecutive elements along the axis.
    ///
    /// The first argument to the closure is an element, and the second
    /// argument is the next element along the axis. Iteration is guaranteed to
    /// proceed in order along the specified axis, but in all other respects
    /// the iteration order is unspecified.
    ///
    /// # Example
    ///
    /// For example, this can be used to compute the cumulative sum along an
    /// axis:
    ///
    /// ```
    /// use ndarray::{array, Axis};
    /// use nditer::ArrayBaseExt;
    ///
    /// let mut arr = array![
    ///     [[1, 2], [3, 4], [5, 6]],
    ///     [[7, 8], [9, 10], [11, 12]],
    /// ];
    /// arr.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    /// assert_eq!(
    ///     arr,
    ///     array![
    ///         [[1, 2], [4, 6], [9, 12]],
    ///         [[7, 8], [16, 18], [27, 30]],
    ///     ],
    /// );
    /// ```
    fn accumulate_axis_inplace<F>(&mut self, axis: Axis, f: F)
    where
        F: FnMut(&A, &mut A),
        S: DataMut,
        D: SubDim<Ix1>;
}

impl<A, S, D> ArrayBaseExt<A, S, D> for ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn producer(&self) -> ArrayBaseProducer<ViewRepr<&A>, D>
    where
        S: Data,
    {
        ArrayBaseProducer { arr: self.view() }
    }

    fn producer_mut(&mut self) -> ArrayBaseProducer<ViewRepr<&mut A>, D>
    where
        S: DataMut,
    {
        ArrayBaseProducer {
            arr: self.view_mut(),
        }
    }

    fn raw_producer(&self) -> ArrayBaseProducer<RawViewRepr<*const A>, D> {
        ArrayBaseProducer {
            arr: self.raw_view(),
        }
    }

    fn raw_producer_mut(&mut self) -> ArrayBaseProducer<RawViewRepr<*mut A>, D>
    where
        S: RawDataMut,
    {
        ArrayBaseProducer {
            arr: self.raw_view_mut(),
        }
    }

    fn accumulate_axis_inplace<F>(&mut self, axis: Axis, mut f: F)
    where
        F: FnMut(&A, &mut A),
        S: DataMut,
        D: SubDim<Ix1>,
    {
        if self.len_of(axis) <= 1 {
            return;
        }
        let mut prev = self.raw_view();
        prev.slice_axis_inplace(axis, Slice::from(..-1));
        let mut curr = self.raw_view_mut();
        curr.slice_axis_inplace(axis, Slice::from(1..));
        prev.into_producer()
            .zip(curr)
            .forbid_invert_axes(axes(axis.index()))
            .for_each(|(prev, curr)| unsafe {
                // These pointer dereferences and borrows are safe because:
                //
                // 1. They're pointers to elements in the array.
                //
                // 2. `S: DataMut` guarantees that elements are safe to borrow
                //    mutably and that they don't alias.
                //
                // 3. The lifetimes of the borrows last only for the duration
                //    of the call to `f`, so aliasing across calls to `f`
                //    cannot occur.
                f(&*prev, &mut *curr)
            });
    }
}

impl<'a, A: 'a, S, D> IntoNdProducer for &'a ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    type Dim = D;
    type Item = &'a A;
    type Producer = ArrayBaseProducer<ViewRepr<&'a A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<ViewRepr<&'a A>, D> {
        ArrayBaseProducer { arr: self.view() }
    }
}

impl<'a, A: 'a, S, D> IntoNdProducer for &'a mut ArrayBase<S, D>
where
    S: DataMut<Elem = A>,
    D: Dimension,
{
    type Dim = D;
    type Item = &'a mut A;
    type Producer = ArrayBaseProducer<ViewRepr<&'a mut A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<ViewRepr<&'a mut A>, D> {
        ArrayBaseProducer {
            arr: self.view_mut(),
        }
    }
}

impl<'a, A, D: Dimension> IntoNdProducer for ArrayView<'a, A, D> {
    type Dim = D;
    type Item = &'a A;
    type Producer = ArrayBaseProducer<ViewRepr<&'a A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<ViewRepr<&'a A>, D> {
        ArrayBaseProducer { arr: self }
    }
}

impl<'a, A, D: Dimension> IntoNdProducer for ArrayViewMut<'a, A, D> {
    type Dim = D;
    type Item = &'a mut A;
    type Producer = ArrayBaseProducer<ViewRepr<&'a mut A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<ViewRepr<&'a mut A>, D> {
        ArrayBaseProducer { arr: self }
    }
}

impl<A, D: Dimension> IntoNdProducer for RawArrayView<A, D> {
    type Dim = D;
    type Item = *const A;
    type Producer = ArrayBaseProducer<RawViewRepr<*const A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<RawViewRepr<*const A>, D> {
        ArrayBaseProducer { arr: self }
    }
}

impl<A, D: Dimension> IntoNdProducer for RawArrayViewMut<A, D> {
    type Dim = D;
    type Item = *mut A;
    type Producer = ArrayBaseProducer<RawViewRepr<*mut A>, D>;
    fn into_producer(self) -> ArrayBaseProducer<RawViewRepr<*mut A>, D> {
        ArrayBaseProducer { arr: self }
    }
}

pub struct ArrayBaseProducer<S, D>
where
    S: RawData,
{
    arr: ArrayBase<S, D>,
}

pub struct ArrayBaseSource<S, D>
where
    S: RawData,
{
    data: PhantomData<S>,
    shape: D,
    strides: D,
    first_ptr: Option<*mut S::Elem>,
}

impl<S, D> Into<ArrayBase<S, D>> for ArrayBaseProducer<S, D>
where
    S: RawData,
{
    fn into(self) -> ArrayBase<S, D> {
        self.arr
    }
}

impl<S, D> Clone for ArrayBaseProducer<S, D>
where
    S: RawDataClone,
    D: Clone,
{
    fn clone(&self) -> ArrayBaseProducer<S, D> {
        ArrayBaseProducer {
            arr: self.arr.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.arr.clone_from(&other.arr)
    }
}

impl<S, D> Copy for ArrayBaseProducer<S, D>
where
    S: RawDataClone + Copy,
    D: Copy,
{
}

impl<S, D> NdReshape for ArrayBaseProducer<S, D>
where
    S: RawData,
    D: Dimension,
{
    type Dim = D;
    fn shape(&self) -> D {
        self.arr.raw_dim()
    }
    fn approx_abs_strides(&self) -> D {
        let mut abs_strides = D::zeros(self.ndim());
        for (s_abs, &s) in izip!(abs_strides.slice_mut(), self.arr.strides()) {
            *s_abs = s.abs() as usize;
        }
        abs_strides
    }
    fn can_invert_axis(&self, axis: Axis) -> bool {
        debug_assert!(axis.index() < self.ndim());
        true
    }
    fn invert_axis(&mut self, axis: Axis) {
        self.arr.invert_axis(axis);
    }
    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        let take_len = self.arr.len_of(take);
        let take_stride = self.arr.stride_of(take);
        let into_len = self.arr.len_of(into);
        let into_stride = self.arr.stride_of(into);
        let outer_stride = into_len as isize * into_stride;
        if take_len <= 1 || into_len <= 1 {
            CanMerge::IfEither
        } else if outer_stride == take_stride {
            CanMerge::IfUnchanged
        } else if outer_stride == -take_stride {
            CanMerge::IfInverted
        } else {
            CanMerge::Never
        }
    }
    fn merge_axes(&mut self, take: Axis, into: Axis) {
        assert!(self.arr.merge_axes(take, into));
    }
    fn len(&self) -> usize {
        self.arr.len()
    }
    fn len_of(&self, axis: Axis) -> usize {
        self.arr.len_of(axis)
    }
    fn is_empty(&self) -> bool {
        self.arr.is_empty()
    }
    fn ndim(&self) -> usize {
        self.arr.ndim()
    }
}

macro_rules! impl_ndproducer {
    ([$($generic:tt)*], $storage:ty, $item:ty) => {
        impl<$($generic)*, D: Dimension> NdProducer for ArrayBaseProducer<$storage, D> {
            type Item = $item;
            type Source = ArrayBaseSource<$storage, D>;
            fn into_source(self) -> ArrayBaseSource<$storage, D> {
                let first_ptr = if self.is_empty() {
                    None
                } else {
                    Some(self.arr.as_ptr() as *mut _)
                };
                let mut strides = D::zeros(self.ndim());
                for (s_copy, &s) in izip!(strides.slice_mut(), self.arr.strides()) {
                    *s_copy = s as usize;
                }
                ArrayBaseSource {
                    data: PhantomData,
                    shape: self.arr.raw_dim(),
                    strides,
                    first_ptr,
                }
            }
        }
    };
}
impl_ndproducer!(['a, A], ViewRepr<&'a A>, &'a A);
impl_ndproducer!(['a, A], ViewRepr<&'a mut A>, &'a mut A);
impl_ndproducer!([A], RawViewRepr<*const A>, *const A);
impl_ndproducer!([A], RawViewRepr<*mut A>, *mut A);

macro_rules! impl_ndaccess {
    ([$($generic:tt)*], $storage:ty, $item:ty, $ptr:ty) => {
        unsafe impl<$($generic)*, D: Dimension> NdAccess for ArrayBaseSource<$storage, D> {
            type Dim = D;
            type Ptr = $ptr;

            fn shape(&self) -> D {
                self.shape.clone()
            }

            fn first_ptr(&self) -> Option<$ptr> {
                self.first_ptr.map(|p| p as $ptr)
            }

            unsafe fn ptr_offset_axis(&self, ptr: &mut $ptr, axis: Axis, count: isize) {
                *ptr = ptr.offset(self.strides[axis.index()] as isize * count);
            }

            #[inline]
            unsafe fn ptr_offset_axis_contiguous(&self, ptr: &mut $ptr, _axis: Axis, count: isize) {
                *ptr = ptr.offset(count);
            }

            fn is_axis_contiguous(&self, axis: Axis) -> bool {
                self.strides[axis.index()] as isize == 1
            }

            fn len_of(&self, axis: Axis) -> usize {
                self.shape[axis.index()]
            }

            fn ndim(&self) -> usize {
                self.shape.ndim()
            }
        }
    };
}
impl_ndaccess!(['a, A: 'a], ViewRepr<&'a A>, &'a A, *const A);
impl_ndaccess!(['a, A: 'a], ViewRepr<&'a mut A>, &'a mut A, *mut A);
impl_ndaccess!([A], RawViewRepr<*const A>, *const A, *const A);
impl_ndaccess!([A], RawViewRepr<*mut A>, *mut A, *mut A);

impl<'a, A: 'a, D: Dimension> NdSource for ArrayBaseSource<ViewRepr<&'a A>, D> {
    type Item = &'a A;

    #[inline]
    unsafe fn read_once_unchecked(&mut self, ptr: &*const A) -> &'a A {
        &**ptr
    }
}

impl<'a, A: 'a, D: Dimension> NdSource for ArrayBaseSource<ViewRepr<&'a mut A>, D> {
    type Item = &'a mut A;

    #[inline]
    unsafe fn read_once_unchecked(&mut self, ptr: &*mut A) -> &'a mut A {
        &mut **ptr
    }
}

impl<A, D: Dimension> NdSource for ArrayBaseSource<RawViewRepr<*const A>, D> {
    type Item = *const A;

    #[inline]
    unsafe fn read_once_unchecked(&mut self, ptr: &*const A) -> *const A {
        *ptr
    }
}

impl<A, D: Dimension> NdSource for ArrayBaseSource<RawViewRepr<*mut A>, D> {
    type Item = *mut A;

    #[inline]
    unsafe fn read_once_unchecked(&mut self, ptr: &*mut A) -> *mut A {
        *ptr
    }
}

unsafe impl<'a, A, D: Dimension> NdSourceRepeat for ArrayBaseSource<ViewRepr<&'a A>, D> {}
unsafe impl<A, D: Dimension> NdSourceRepeat for ArrayBaseSource<RawViewRepr<*const A>, D> {}
unsafe impl<A, D: Dimension> NdSourceRepeat for ArrayBaseSource<RawViewRepr<*mut A>, D> {}
