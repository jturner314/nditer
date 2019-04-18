use crate::{optimize, AxesFor, NdProducer, NdSource};
use itertools::izip;
use ndarray::{Axis, Dimension};
use num_traits::Zero;

/// Size of unrolled chunks.
pub const UNROLL_LEN: usize = 8;

/// Maximum number of elements that can be added in a non-pairwise fashion.
const MAX_SEQUENTIAL: usize = 64;

/// Computes the pairwise sum of the producer's items.
///
/// Note that this isn't strictly a pairwise sum; at the lowest level
/// of the tree, at most `MAX_SEQUENTIAL` elements are added in
/// sequence.
pub fn pairwise_sum<P>(producer: P) -> P::Item
where
    P: NdProducer,
    P::Item: Zero + Clone + std::ops::Add<Output = P::Item>,
{
    SourceSubset::map_producer(producer, |view| view.pairwise_sum())
}

/// A borrowed subset of a source.
struct SourceSubset<'a, S: NdSource, D: Dimension> {
    /// The source to operate on.
    source: &'a mut S,
    /// Pointer to the first element in the subset, or `None` if the subset is
    /// empty.
    ptr: Option<S::Ptr>,
    /// The axes to iterate over, in order.
    axes: &'a AxesFor<S::Dim, D>,
    /// The lengths of the `axes`, all of which must be `<= isize::MAX`.
    // Implementation note: It would be nice to make this owned instead of a
    // mutable reference, but doing so requires it to be cloned when splitting
    // the subset in `pairwise_sum_recur`, and this has a substantial
    // performance impact in some cases (primarily where ndim >= 3 and much of
    // the iteration is not confined to the last axis).
    axis_lens: &'a mut D,
}

impl<'a, S: NdSource> SourceSubset<'a, S, S::Dim> {
    /// Optimizes the producer, converts it into a source, and calls `f` with a
    /// `SourceSubset` over the entire source.
    pub fn map_producer<P, F, T>(mut producer: P, f: F) -> T
    where
        P: NdProducer<Item = S::Item, Source = S, Dim = S::Dim>,
        F: FnOnce(SourceSubset<S, S::Dim>) -> T,
    {
        let axes = optimize::optimize_any_ord(&mut producer);
        let mut source = producer.into_source();
        let mut axis_lens = axes.mapv_to_dim(|axis| {
            let axis_len = source.len_of(axis);
            assert!(axis_len <= std::isize::MAX as usize);
            axis_len
        });
        f(SourceSubset {
            axes: &axes,
            ptr: source.first_ptr(),
            source: &mut source,
            axis_lens: &mut axis_lens,
        })
    }
}

impl<'a, S: NdSource, D: Dimension> SourceSubset<'a, S, D> {
    /// Creates a `SourceSubset` from its raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure the following:
    ///
    /// * The new iterator can never read the same location from the source as
    ///   anything else (to make sure the constraints are satisfied for
    ///   `read_once_unchecked`).
    ///
    /// * The pointer is at a valid location within the source, and all
    ///   locations reachable by offsetting the pointer to all indices within
    ///   the `axis_lens` are also valid locations.
    ///
    /// * The `axes` are unique, in-bounds axes for `source`.
    ///
    /// * The `axis_lens` are no larger than the axis lengths of the `source`
    ///   for the given `axes`, and all axis lengths are `<= isize::MAX`.
    pub unsafe fn from_raw_parts(
        source: &'a mut S,
        ptr: Option<S::Ptr>,
        axes: &'a AxesFor<S::Dim, D>,
        axis_lens: &'a mut D,
    ) -> Self {
        // A few sanity checks.
        if cfg!(debug_assertions) {
            debug_assert_eq!(source.ndim(), axes.for_ndim());
            for (&ax, &axis_len) in izip!(axes.slice(), axis_lens.slice()) {
                debug_assert!(axis_len <= source.len_of(Axis(ax)));
                debug_assert!(axis_len <= std::isize::MAX as usize);
            }
        }
        SourceSubset {
            source,
            ptr,
            axes,
            axis_lens,
        }
    }

    /// Computes the pairwise sum of the source subset.
    pub fn pairwise_sum(self) -> S::Item
    where
        S::Item: Zero + Clone + std::ops::Add<Output = S::Item>,
    {
        if self.ptr.is_none() {
            S::Item::zero()
        } else if self.source.ndim() == 0 {
            unsafe { self.source.read_once_unchecked(&self.ptr.unwrap()) }
        } else {
            self.pairwise_sum_recur(0)
        }
    }

    /// Recursive portion of `pairwise_sum`. This is an implementation detail
    /// of `pairwise_sum`.
    ///
    /// `min_split_axis_index` should be the lowest index into `self.axes` to
    /// try splitting. If too large a value is provided, not all axes will be
    /// fully split, and an incorrect result will be returned.
    ///
    /// **Panics** if `min_split_axis_index` is out-of-bounds or if `self.ptr`
    /// is `None` (i.e. if the subset is empty).
    fn pairwise_sum_recur(self, min_split_axis_index: usize) -> S::Item
    where
        S::Item: Zero + Clone + std::ops::Add<Output = S::Item>,
    {
        let source = self.source;
        let mut ptr = self.ptr.unwrap();
        let axes = self.axes;
        let axis_lens = self.axis_lens;

        // Skip over possible split axes that have length <= 1.
        let mut axis_index = min_split_axis_index;
        while axis_lens[axis_index] <= 1 && axis_index < axes.num_axes() - 1 {
            axis_index += 1;
        }

        let axis = Axis(axes[axis_index]);
        let axis_len = axis_lens[axis_index];
        if axis_index == self.axes.num_axes() - 1 {
            unsafe { pairwise_sum_axis(source, ptr, axis, axis_len) }
        } else {
            debug_assert!(axis_len >= 2);
            let left_len = axis_len / 2;

            // Compute sum of left half.
            let left = unsafe {
                axis_lens[axis_index] = left_len;
                SourceSubset::from_raw_parts(source, Some(ptr.clone()), axes, axis_lens)
                    .pairwise_sum_recur(axis_index)
            };

            // Compute sum of right half.
            let right = unsafe {
                axis_lens[axis_index] = axis_len - left_len;
                source.ptr_offset_axis(&mut ptr, axis, left_len as isize);
                SourceSubset::from_raw_parts(source, Some(ptr), axes, axis_lens)
                    .pairwise_sum_recur(axis_index)
            };

            // Restore the original value to `axis_lens` (since we only have a
            // temporary reference to it, and higher levels reuse `axis_lens`).
            axis_lens[axis_index] = axis_len;

            left + right
        }
    }
}

/// Computes the pairwise sum of `axis_lens` items in `source` along `axis`,
/// starting at `ptr`.
///
/// # Safety
///
/// The caller must ensure the following:
///
/// * The source is not empty.
///
/// * The pointer is at a valid location within the source, and all locations
///   reachable by offsetting the pointer along `axis` to all indices within
///   the `axis_len` are also valid locations.
///
/// * All of those locations will not be read by anything else for the lifetime
///   of the source (to make sure the constraints are satisfied for
///   `read_once_unchecked`).
///
/// * The `axis` is in-bounds for `source`.
///
/// * The `axis_len` is no larger than the corresponding axis length of the
///   `source` and `axis_len <= isize::MAX`.
unsafe fn pairwise_sum_axis<S>(source: &mut S, ptr: S::Ptr, axis: Axis, axis_len: usize) -> S::Item
where
    S: NdSource,
    S::Item: Zero + Clone + std::ops::Add<Output = S::Item>,
{
    if axis_len > UNROLL_LEN * MAX_SEQUENTIAL {
        let left_len = axis_len / 2;
        let right_len = axis_len - left_len;
        let left_ptr = ptr.clone();
        let mut right_ptr = ptr;
        source.ptr_offset_axis(&mut right_ptr, axis, left_len as isize);
        pairwise_sum_axis(source, left_ptr, axis, left_len)
            + pairwise_sum_axis(source, right_ptr, axis, right_len)
    } else {
        sum_axis(source, ptr, axis, axis_len)
    }
}

/// Sums over the specified `axis` of the `source` for the specified
/// `axis_len`.
///
/// The axis is split into chunks of size `UNROLL_LEN` to encourage the
/// compiler to use SIMD instructions. The accumulator vector is converted into
/// a scalar in a pairwise fashion so that this function can be used to compute
/// a pairwise sum.
///
/// # Safety
///
/// The caller must ensure the following:
///
/// * The source is not empty.
///
/// * The pointer is at a valid location within the source, and all locations
///   reachable by offsetting the pointer along `axis` to all indices within
///   the `axis_len` are also valid locations.
///
/// * All of those locations will not be read by anything else for the lifetime
///   of the source (to make sure the constraints are satisfied for
///   `read_once_unchecked`).
///
/// * The `axis` is in-bounds for `source`.
///
/// * The `axis_len` is no larger than the corresponding axis length of the
///   `source` and `axis_len <= isize::MAX`.
unsafe fn sum_axis<S>(source: &mut S, mut ptr: S::Ptr, axis: Axis, axis_len: usize) -> S::Item
where
    S: NdSource,
    S::Item: Zero + Clone + std::ops::Add<Output = S::Item>,
{
    let num_chunks = axis_len / UNROLL_LEN;
    let rest = axis_len % UNROLL_LEN;
    let mut sum = [
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
        S::Item::zero(),
    ];
    if source.is_axis_contiguous(axis) {
        if num_chunks > 0 {
            sum = read_vector_contiguous_once_unchecked(source, ptr.clone(), axis);
            for _ in 1..num_chunks {
                source.ptr_offset_axis_contiguous(&mut ptr, axis, UNROLL_LEN as isize);
                let [c0, c1, c2, c3, c4, c5, c6, c7] =
                    read_vector_contiguous_once_unchecked(source, ptr.clone(), axis);
                let [s0, s1, s2, s3, s4, s5, s6, s7] = sum;
                sum = [
                    s0 + c0,
                    s1 + c1,
                    s2 + c2,
                    s3 + c3,
                    s4 + c4,
                    s5 + c5,
                    s6 + c6,
                    s7 + c7,
                ];
            }
            if rest > 0 {
                source.ptr_offset_axis_contiguous(&mut ptr, axis, UNROLL_LEN as isize);
            }
        }
        if rest > 0 {
            sum[0] = sum[0].clone() + source.read_once_unchecked(&ptr);
            for i in 1..rest {
                assert!(i < UNROLL_LEN);
                source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
                sum[i] = sum[i].clone() + source.read_once_unchecked(&ptr);
            }
        }
    } else {
        if num_chunks > 0 {
            sum = read_vector_once_unchecked(source, ptr.clone(), axis);
            for _ in 1..num_chunks {
                source.ptr_offset_axis(&mut ptr, axis, UNROLL_LEN as isize);
                let [c0, c1, c2, c3, c4, c5, c6, c7] =
                    read_vector_once_unchecked(source, ptr.clone(), axis);
                let [s0, s1, s2, s3, s4, s5, s6, s7] = sum;
                sum = [
                    s0 + c0,
                    s1 + c1,
                    s2 + c2,
                    s3 + c3,
                    s4 + c4,
                    s5 + c5,
                    s6 + c6,
                    s7 + c7,
                ];
            }
            if rest > 0 {
                source.ptr_offset_axis(&mut ptr, axis, UNROLL_LEN as isize);
            }
        }
        if rest > 0 {
            sum[0] = sum[0].clone() + source.read_once_unchecked(&ptr);
            for i in 1..rest {
                assert!(i < UNROLL_LEN);
                source.ptr_offset_axis(&mut ptr, axis, 1);
                sum[i] = sum[i].clone() + source.read_once_unchecked(&ptr);
            }
        }
    }
    let [s0, s1, s2, s3, s4, s5, s6, s7] = sum;
    let [t0, t1, t2, t3] = [s0 + s4, s1 + s5, s2 + s6, s3 + s7];
    (t0 + t2) + (t1 + t3)
}

/// Reads `UNROLL_LEN` items at the pointer along `axis`.
///
/// # Safety
///
/// The caller must ensure the following:
///
/// * The current location of the pointer and the following `UNROLL_LEN - 1`
///   locations along `axis` are valid.
///
/// * No single location is read more than once over the entire life of the
///   source.
unsafe fn read_vector_once_unchecked<S>(
    source: &mut S,
    mut ptr: S::Ptr,
    axis: Axis,
) -> [S::Item; UNROLL_LEN]
where
    S: NdSource,
{
    let x0 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x1 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x2 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x3 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x4 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x5 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x6 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis(&mut ptr, axis, 1);
    let x7 = source.read_once_unchecked(&ptr);
    [x0, x1, x2, x3, x4, x5, x6, x7]
}

/// Reads `UNROLL_LEN` items at the pointer along the contiguous  `axis`.
///
/// # Safety
///
/// The caller must ensure the following:
///
/// * `source.is_axis_contiguous(axis)` is `true`.
///
/// * The current location of the pointer and the following `UNROLL_LEN - 1`
///   locations along `axis` are valid.
///
/// * No single location is read more than once over the entire life of the
///   source.
unsafe fn read_vector_contiguous_once_unchecked<S>(
    source: &mut S,
    mut ptr: S::Ptr,
    axis: Axis,
) -> [S::Item; UNROLL_LEN]
where
    S: NdSource,
{
    let x0 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x1 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x2 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x3 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x4 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x5 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x6 = source.read_once_unchecked(&ptr);
    source.ptr_offset_axis_contiguous(&mut ptr, axis, 1);
    let x7 = source.read_once_unchecked(&ptr);
    [x0, x1, x2, x3, x4, x5, x6, x7]
}
