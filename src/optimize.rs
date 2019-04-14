use itertools::izip;
use ndarray::{Axis, Dimension, IxDyn};
use crate::{assert_valid_unique_axes, AxesMask, CanMerge, Layout, NdReshape};

/// Optimizes the producer, preserving order, and returns the order for
/// iterating over axes (assuming the last index moves the fastest).
///
/// This method may change the shape of the producer but not the order of
/// iteration.
pub(crate) fn optimize_same_ord<T>(producer: &mut T) -> T::Dim
where
    T: NdReshape + ?Sized,
{
    let ndim = producer.ndim();
    if ndim <= 1 {
        return T::Dim::zeros(ndim);
    }
    for axis in 0..(ndim - 1) {
        match producer.can_merge_axes(Axis(axis), Axis(axis + 1)) {
            CanMerge::IfUnchangedOrBothInverted | CanMerge::Always => {
                producer.merge_axes(Axis(axis), Axis(axis + 1))
            }
            CanMerge::IfOneInverted | CanMerge::Never => {}
        }
    }
    let mut axes = T::Dim::zeros(ndim);
    for (i, a) in axes.slice_mut().iter_mut().enumerate() {
        *a = i;
    }
    axes
}

/// Executes `optimize_any_ord_axes` for all axes and returns the suggested
/// iteration order.
pub(crate) fn optimize_any_ord<T>(producer: &mut T) -> T::Dim
where
    T: NdReshape + ?Sized,
{
    let mut axes = T::Dim::zeros(producer.ndim());
    for (i, ax) in axes.slice_mut().iter_mut().enumerate() {
        *ax = i;
    }
    unsafe { optimize_any_ord_axes_unchecked(producer, &mut axes) };
    axes
}

/// Optimizes the producer, possibly changing the order, and adjusts `axes`
/// into good iteration order (assuming the last index moves the fastest).
///
/// This function may change the shape of the producer and the order of
/// iteration. Optimization is performed only on the given `axes`; all other
/// axes are left unchanged.
///
/// When choosing axes to attempt merging, it only tries merging axes when the
/// absolute stride of the `take` axes is >= the absolute stride of the `into`
/// axis.
///
/// The suggested iteration order is in order of descending absolute stride
/// (except for axes of length <= 1, which are positioned as outer axes). This
/// isn't necessarily the optimal iteration order, but it should be a
/// reasonable heuristic in most cases.
///
/// **Panics** if any of the axes in `axes` are out of bounds or if an axis is
/// repeated more than once.
pub(crate) fn optimize_any_ord_axes<T, D>(producer: &mut T, axes: &mut D)
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    assert_valid_unique_axes::<T::Dim>(producer.ndim(), axes.slice());
    unsafe { optimize_any_ord_axes_unchecked(producer, axes) }
}

/// `unsafe` because `axes` are not checked to ensure that they're in-bounds
/// and not repeated.
unsafe fn optimize_any_ord_axes_unchecked<T, D>(producer: &mut T, axes: &mut D)
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    // TODO: Should there be a minimum producer size for the more advanced (and
    // costly) optimizations?

    // TODO: specialize for ndim == 3?

    let ndim = axes.ndim();
    if ndim <= 1 {
        return;
    } else if ndim == 2 {
        // Reorder axes according to shape and strides.
        let abs_strides = producer.approx_abs_strides();
        if abs_strides[axes[0]] < abs_strides[axes[1]] && producer.len_of(Axis(axes[0])) > 1 {
            axes.slice_mut().swap(0, 1);
        }

        // Try merging axes.
        let take = Axis(axes[0]);
        let into = Axis(axes[1]);
        match producer.can_merge_axes(take, into) {
            CanMerge::IfUnchangedOrBothInverted | CanMerge::Always => {
                producer.merge_axes(take, into);
            }
            CanMerge::IfOneInverted if !producer.is_axis_ordered(take) => {
                producer.invert_axis(take);
                producer.merge_axes(take, into);
            }
            CanMerge::IfOneInverted if !producer.is_axis_ordered(into) => {
                producer.invert_axis(into);
                producer.merge_axes(take, into);
            }
            CanMerge::IfOneInverted | CanMerge::Never => {}
        }

        return;
    }

    // Determine initial order of axes. Sort axes by descending absolute stride
    // (except for axes with length <= 1, which are moved to the left).
    {
        let shape = producer.shape();
        let abs_strides = producer.approx_abs_strides();
        axes.slice_mut().sort_unstable_by(|&a, &b| {
            if shape[a] <= 1 || shape[b] <= 1 {
                shape[a].cmp(&shape[b])
            } else {
                abs_strides[b].cmp(&abs_strides[a])
            }
        });
    }

    // Merge as many axes with lengths > 1 as possible and move `take` axes
    // (which now have length <= 1) to the left.
    if let Some((mut rest, _)) = axes
        .slice()
        .iter()
        .enumerate()
        .find(|(_, &ax)| producer.len_of(Axis(ax)) > 1)
    {
        for i in (rest + 1)..ndim {
            let mut t = rest;
            while t < i {
                let take = Axis(axes[t]);
                let into = Axis(axes[i]);
                let can_merge = producer.can_merge_axes(take, into);
                match can_merge {
                    CanMerge::IfUnchangedOrBothInverted | CanMerge::Always => {
                        producer.merge_axes(take, into);
                        roll(&mut axes.slice_mut()[rest..=t], 1);
                        rest += 1;
                        t = rest;
                    }
                    CanMerge::IfOneInverted if !producer.is_axis_ordered(take) => {
                        producer.invert_axis(take);
                        producer.merge_axes(take, into);
                        roll(&mut axes.slice_mut()[rest..=t], 1);
                        rest += 1;
                        t = rest;
                    }
                    CanMerge::IfOneInverted if !producer.is_axis_ordered(into) => {
                        producer.invert_axis(into);
                        producer.merge_axes(take, into);
                        roll(&mut axes.slice_mut()[rest..=t], 1);
                        rest += 1;
                        t = rest;
                    }
                    CanMerge::IfOneInverted | CanMerge::Never => {
                        t += 1;
                    }
                }
            }
        }
    }
}

pub(crate) fn optimize_any_ord_with_layout<T>(producer: &mut T) -> (T::Dim, Layout<T::Dim>)
where
    T: NdReshape + ?Sized,
{
    let mut axes = T::Dim::zeros(producer.ndim());
    for (i, ax) in axes.slice_mut().iter_mut().enumerate() {
        *ax = i;
    }
    let layout = optimize_any_ord_axes_with_layout(producer, &mut axes);
    (axes, layout)
}

/// Optimizes the producer and order of `axes` and returns the layout of an
/// array for collecting the items in order.
///
/// See `Layout` for more information about the return value. Note that the
/// shape of `Layout` matches the original order of `axes`.
pub(crate) fn optimize_any_ord_axes_with_layout<T, D>(producer: &mut T, axes: &mut D) -> Layout<D>
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    let mut recorder = OptimRecorder::new(producer, axes.clone());
    optimize_any_ord_axes(&mut recorder, axes);
    recorder.into_layout(&axes)
}

struct OptimRecorder<'a, T, D>
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    inner: &'a mut T,
    /// Iteration axes before optimization.
    orig_axes: D,
    /// Shape of the producer before optimization.
    orig_shape: T::Dim,
    /// Whether the axes of the producer have been inverted.
    inverted: AxesMask<T::Dim, IxDyn>,
    /// Each element of `merged` is the list of axes merged into the axis.
    merged: Vec<Vec<usize>>,
}

impl<'a, T, D> OptimRecorder<'a, T, D>
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    fn new(inner: &'a mut T, orig_axes: D) -> OptimRecorder<'a, T, D> {
        OptimRecorder {
            orig_axes,
            orig_shape: inner.shape(),
            inverted: AxesMask::all_false(inner.ndim()).into_dyn_num_true(),
            merged: (0..inner.ndim()).map(|i| vec![i]).collect(),
            inner,
        }
    }

    /// `optim_axes` is the list of axes after optimization (in iteration order).
    fn into_layout(self, optim_axes: &D) -> Layout<D> {
        if cfg!(debug_assertions) {
            let mut orig_axes = self.orig_axes.clone();
            orig_axes.slice_mut().sort();
            let mut optim_axes = optim_axes.clone();
            optim_axes.slice_mut().sort();
            debug_assert_eq!(orig_axes, optim_axes);
        }

        // Determine the order the original axes of the producer will be
        // iterated over. (This is usually not the same as `optim_axes` because
        // axes have been merged together in the producer.) Also observe that
        // `iter_axes` may contain fewer axes than the underlying producer. (It
        // only contains those axes that will be iterated over.)
        let mut iter_axes = D::zeros(optim_axes.ndim());
        iter_axes
            .slice_mut()
            .iter_mut()
            .rev()
            .zip(
                optim_axes
                    .slice()
                    .iter()
                    .rev()
                    .flat_map(|&ax| &self.merged[ax]),
            )
            .for_each(|(dst, &src)| *dst = src);

        let mut shape = D::zeros(self.orig_axes.ndim());
        for (&ax, len) in izip!(self.orig_axes.slice(), shape.slice_mut()) {
            *len = self.orig_shape[ax];
        }

        let mut strides = D::zeros(shape.ndim());
        let mut offset = 0;
        if shape.size() != 0 {
            let mut cum_prod: isize = 1;
            for &ax in iter_axes.slice().iter().rev() {
                let len = self.orig_shape[ax];
                if self.inverted.read(Axis(ax)) {
                    offset += (len - 1) as isize * cum_prod;
                    strides[ax] = (-cum_prod) as usize;
                } else {
                    strides[ax] = cum_prod as usize;
                }
                cum_prod *= len as isize;
            }
        }

        Layout {
            shape,
            strides,
            offset,
        }
    }
}

impl<'a, T, D> NdReshape for OptimRecorder<'a, T, D>
where
    T: NdReshape + ?Sized,
    D: Dimension,
{
    type Dim = T::Dim;

    fn shape(&self) -> T::Dim {
        self.inner.shape()
    }

    fn approx_abs_strides(&self) -> T::Dim {
        self.inner.approx_abs_strides()
    }

    fn is_axis_ordered(&self, axis: Axis) -> bool {
        self.inner.is_axis_ordered(axis)
    }

    fn invert_axis(&mut self, axis: Axis) {
        for &ax in &self.merged[axis.index()] {
            let axis = Axis(ax);
            self.inverted.write(axis, !self.inverted.read(axis));
        }
        self.inner.invert_axis(axis)
    }

    fn can_merge_axes(&self, take: Axis, into: Axis) -> CanMerge {
        self.inner.can_merge_axes(take, into)
    }

    fn merge_axes(&mut self, take: Axis, into: Axis) {
        let t = take.index();
        let i = into.index();
        if i < t {
            let (left, right) = self.merged.split_at_mut(t);
            let into = &mut left[i];
            let take = &mut right[0];
            into.extend_from_slice(take);
            take.clear();
        } else if t < i {
            let (left, right) = self.merged.split_at_mut(i);
            let take = &mut left[t];
            let into = &mut right[0];
            into.extend_from_slice(take);
            take.clear();
        }
        self.inner.merge_axes(take, into)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn len_of(&self, axis: Axis) -> usize {
        self.inner.len_of(axis)
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
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
    use super::*;
    use ndarray::prelude::*;
    use crate::{ArrayBaseExt, IntoNdProducer};

    #[test]
    fn optimize_any_ord_ix4_c() {
        // Can merge axis 0 into 1, 1 into 2, and 2 into 3.
        let a = Array4::<u8>::zeros((2, 5, 4, 3));
        let mut p = a.producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[1, 1, 1, 120]);
        assert_eq!(v.strides()[3], 1);
        assert_eq!(axes_order[3], 3);
    }

    #[test]
    fn optimize_any_ord_ix4_f() {
        // Can merge axis 3 into 2, 2 into 1, and 1 into 0.
        let a = Array4::<u8>::zeros((2, 5, 4, 3).f());
        let mut p = a.producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[120, 1, 1, 1]);
        assert_eq!(v.strides()[0], 1);
        assert_eq!(axes_order[3], 0);
    }

    #[test]
    fn optimize_any_ord_ix4_discont0_c() {
        // Can merge axis 1 into 2 and 2 into 3.
        let a = Array4::<u8>::zeros((3, 5, 4, 3));
        let mut p = a.slice(s![..;2, .., .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[2, 1, 1, 5 * 4 * 3]);
        assert_eq!(v.strides()[0], 2 * (5 * 4 * 3));
        assert_eq!(v.strides()[3], 1);
        assert_eq!(axes_order[3], 3);
        assert_eq!(axes_order[2], 0);
    }

    #[test]
    fn optimize_any_ord_ix4_discont1odd_c() {
        // Can only merge axis 2 into 3.
        let a = Array4::<u8>::zeros((2, 5, 4, 3));
        let mut p = a.slice(s![.., ..;2, .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[2, 3, 1, 4 * 3]);
        assert_eq!(v.strides()[0], 5 * 4 * 3);
        assert_eq!(v.strides()[1], 2 * (4 * 3));
        assert_eq!(v.strides()[3], 1);
        assert_eq!(axes_order.slice(), &[2, 0, 1, 3]);
    }

    #[test]
    fn optimize_any_ord_ix4_discont1even_c() {
        // Can merge axis 0 into 1 and 2 into 3.
        let a = Array4::<u8>::zeros((2, 4, 4, 3));
        let mut p = a.slice(s![.., ..;2, .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[1, 2 * 2, 1, 4 * 3]);
        assert_eq!(v.strides()[1], 2 * (4 * 3));
        assert_eq!(v.strides()[3], 1);
        assert_eq!(axes_order[3], 3);
        assert_eq!(axes_order[2], 1);
    }

    #[test]
    fn optimize_any_ord_ix4_discont0odd_f() {
        // Can merge axis 3 into 2 and 2 into 1.
        let a = Array4::<u8>::zeros((3, 5, 4, 3).f());
        let mut p = a.slice(s![..;2, .., .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[2, 5 * 4 * 3, 1, 1]);
        assert_eq!(v.strides()[0], 2);
        assert_eq!(v.strides()[1], 3);
        assert_eq!(axes_order[3], 0);
        assert_eq!(axes_order[2], 1);
    }

    #[test]
    fn optimize_any_ord_ix4_discont0even_f() {
        // Can merge axis 3 into 2, 2 into 1, and 1 into 0.
        let a = Array4::<u8>::zeros((4, 5, 4, 3).f());
        let mut p = a.slice(s![..;2, .., .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[2 * 5 * 4 * 3, 1, 1, 1]);
        assert_eq!(v.strides()[0], 2);
        assert_eq!(axes_order[3], 0);
    }

    #[test]
    fn optimize_any_ord_ix4_discont1odd_f() {
        // Can only merge axis 3 into 2.
        let a = Array4::<u8>::zeros((4, 5, 4, 3).f());
        let mut p = a.slice(s![.., ..;2, .., ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[4, 3, 4 * 3, 1]);
        assert_eq!(v.strides()[0], 1);
        assert_eq!(v.strides()[1], 2 * 4);
        assert_eq!(v.strides()[2], 4 * 5);
        assert_eq!(axes_order.slice(), &[3, 2, 1, 0]);
    }

    #[test]
    fn optimize_any_ord_ix4_discont2even_f() {
        // Can merge axis 3 into 2 and 1 into 0.
        let a = Array4::<u8>::zeros((4, 5, 4, 3).f());
        let mut p = a.slice(s![.., .., ..;2, ..]).into_producer();
        let axes_order = optimize_any_ord(&mut p);
        let v: ArrayView4<u8> = p.into();
        assert_eq!(v.shape(), &[4 * 5, 1, 2 * 3, 1]);
        assert_eq!(v.strides()[0], 1);
        assert_eq!(v.strides()[2], 2 * (4 * 5));
        assert_eq!(axes_order[3], 0);
        assert_eq!(axes_order[2], 2);
    }

    // TODO: Test for `rec_axes_order` with axes not in order of descending
    // absolute stride.

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
