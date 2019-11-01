use crate::{axes_all, AxesFor, AxesMask, DimensionExt, IntoAxesFor};
use approx::ulps_eq;
use itertools::{izip, Itertools};
use ndarray::{Array, ArrayView, Dimension, IxDyn, ShapeBuilder};
use num_traits::ToPrimitive;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use rand::{
    distributions::{Bernoulli, Beta, Distribution},
    seq::SliceRandom,
    Rng,
};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::ops::Range;

/// Returns `n!`.
///
/// **Panics** if `n!` would overflow `u64`.
fn factorial(n: u64) -> u64 {
    assert!(n <= 20, "n must be <= 20 to avoid overflow.");
    (2..=n).product()
}

/// Returns the n-dimensional volume of an n-dimensional corner,
///
/// ```text
/// s  s−xₙ₋₁ s−xₙ₋₁−xₙ₋₂   s−xₙ₋₁−⋯−x₂ s−xₙ₋₁−⋯−x₁
/// ⌠    ⌠        ⌠              ⌠           ⌠
/// ⎮    ⎮        ⎮       ⋯      ⎮           ⎮      dx₀ dx₁ … dxₙ₋₃ dxₙ₋₂ dxₙ₋₁ = sⁿ / n!
/// ⌡    ⌡        ⌡              ⌡           ⌡
/// 0    0        0              0           0
/// ```
///
/// where `n` is the number of dimensions and `s` is the side length. (For
/// `ndim == 0`, we define the volume as `1.`.)
///
/// In other words, this is the n-dimensional volume of
///
/// ```text
/// ⎧                                                            ⎛n−1      ⎞ ⎫
/// ⎨ (x₀, x₁, …, xₙ₋₁) | (x₀ ≥ 0) ∧ (x₁ ≥ 0) ∧ … ∧ (xₙ₋₁ ≥ 0) ∧ ⎜ ∑ xᵢ < s⎟ ⎬
/// ⎩                                                            ⎝i=0      ⎠ ⎭
/// ```
///
/// Note that `s` must be nonnegative for the result to make sense as a volume.
///
/// **Panics** if the factorial of `ndim` would overflow `u64`.
///
/// # Proof
///
/// A proof (by induction) of the formula follows.
///
/// Define
///
/// ```text
///          ⎧ 1                                                                             for n = 0
///          ⎪
///          ⎪ s  s−xₙ₋₁ s−xₙ₋₁−xₙ₋₂   s−xₙ₋₁−⋯−x₂ s−xₙ₋₁−⋯−x₁
/// f(n,s) = ⎨ ⌠    ⌠        ⌠              ⌠           ⌠
///          ⎪ ⎮    ⎮        ⎮       ⋯      ⎮           ⎮      dx₀ dx₁ … dxₙ₋₃ dxₙ₋₂ dxₙ₋₁   for n = 1,2,…
///          ⎪ ⌡    ⌡        ⌡              ⌡           ⌡
///          ⎩ 0    0        0              0           0
/// ```
///
/// We wish to show that `f(n,s) = sⁿ / n! for (n ∈ {0,1,2,…}) ∧ (s > 0)`.
///
/// First, it holds for `n = 0` and `n = 1`:
///
/// ```text
/// f(0,s) = 1            and    s⁰ / 0! = 1,    so f(0,s) = s⁰ / 0!
///
///          s
/// f(1,s) = ∫ dx₀ = s    and    s¹ / 1! = s,    so f(1,s) = s¹ / 1!
///          0
/// ```
///
/// Next, we show that if it holds for `n ∈ {1,2,…}`, it also holds for `n+1 ∈ {2,3,…}`:
///
/// ```text
///             s  s−xₙ s−xₙ−xₙ₋₁   s−xₙ−⋯−x₂ s−xₙ−⋯−x₁
///             ⌠   ⌠      ⌠            ⌠          ⌠
///  f(n+1,s) = ⎮   ⎮      ⎮      ⋯     ⎮          ⎮      dx₀ dx₁ … dxₙ₋₂ dxₙ₋₁ dxₙ    for n ∈ {1,2,…}, by definition of f
///             ⌡   ⌡      ⌡            ⌡          ⌡
///             0   0      0            0          0
///
///             s
///           = ∫ f(n,s−xₙ) dxₙ                                                        for n ∈ {1,2,…} by definition of f
///             0
///
///             s
///           = ∫ (s-xₙ)ⁿ / n! dxₙ                                                     by the assumption that it holds for n ∈ {1,2,…}
///             0
///
///           = sⁿ⁺¹ / (n+1)!                                                          for (n ∈ {0,1,2,…}) ∧ (s > 0)
/// ```
fn volume_of_corner(ndim: u8, side_length: f64) -> f64 {
    side_length.powi(ndim as i32) / factorial(ndim as u64) as f64
}

/// Returns the generalized volume of n-dimensional points `x` that satisfy both
///
/// ```text
/// min[i] <= x[i] < max[i], i = 0,1,…,n-1
/// ```
///
/// and
///
/// ```text
/// ⎛ n-1      ⎞
/// ⎜  ∑  x[i] ⎟ < max_sum
/// ⎝ i=0      ⎠
/// ```
///
/// **Panics** if `min.len() != max.len()`, if `min.len()` doesn't fit in `u8`,
/// or if the factorial of `min.len()` would overflow `u64`.
fn volume_within_bounds(min: &[f64], max: &[f64], max_sum: f64) -> f64 {
    let ndim = min.len();
    assert_eq!(
        max.len(),
        ndim,
        "`min` and `max` must have the same length.",
    );
    let ndim = u8::try_from(ndim).expect("`min.len()` must fit in `u8`.");
    let mut volume = 0.;
    let mut sign = 1.;
    for n_maxes in (0..=ndim).rev() {
        for chosen_maxes in (0..ndim).combinations(n_maxes as usize) {
            let mut choose_max = vec![false; ndim as usize];
            for &i in &chosen_maxes {
                choose_max[i as usize] = true;
            }
            let lengths: Vec<f64> = choose_max
                .iter()
                .enumerate()
                .map(|(i, &choose_max)| if choose_max { max[i] } else { min[i] })
                .collect();
            let product = lengths.iter().product::<f64>();
            let excess_sum = lengths.iter().sum::<f64>() - max_sum;
            let this_vol = if excess_sum > 0. {
                product - volume_of_corner(ndim, excess_sum)
            } else {
                product
            };
            volume += sign * this_vol;
        }
        sign = -sign;
    }
    volume
}

/// Searches for `x` in `domain` such that `f(x)` close to `desired_output`,
/// assuming that `f` is monotonically increasing.
///
/// Shrinks the domain containing `x` until the start and end of the domain are
/// within `max_ulps` of each other. This approach has good behavior in edge
/// cases. Note that this does not necessarily guarantee that `f(x)` is within
/// `max_ulps` of `desired_output`.
///
/// **Panics** if either edge of `domain` is not finite.
fn binary_search<F>(mut f: F, domain: Range<f64>, desired_output: f64, max_ulps: u32) -> f64
where
    F: FnMut(f64) -> f64,
{
    assert!(domain.start.is_finite() && domain.end.is_finite());
    let mid = 0.5 * (domain.start + domain.end);
    let out = f(mid);
    if ulps_eq!(domain.start, domain.end, max_ulps = max_ulps) {
        assert!(f(domain.start) <= desired_output && desired_output <= f(domain.end));
        mid
    } else if out < desired_output {
        binary_search(f, mid..domain.end, desired_output, max_ulps)
    } else {
        binary_search(f, domain.start..mid, desired_output, max_ulps)
    }
}

fn uniform_sample_in_bounds<R: Rng + ?Sized>(
    mut min: &[f64],
    mut max: &[f64],
    mut max_sum: f64,
    rng: &mut R,
) -> Vec<f64> {
    fn uniform_sample_in_bounds_first_dim<R: Rng + ?Sized>(
        min: &[f64],
        max: &[f64],
        max_sum: f64,
        rng: &mut R,
    ) -> f64 {
        let full_volume: f64 = volume_within_bounds(min, max, max_sum);
        assert!(full_volume.is_finite());
        let frac: f64 = rng.gen();
        let mut max_owned = max.to_owned();
        binary_search(
            |len| {
                max_owned[0] = len;
                volume_within_bounds(min, &max_owned, max_sum)
            },
            min[0]..max[0],
            frac * full_volume,
            10,
        )
    }
    let ndim = min.len();
    assert_eq!(ndim, max.len());
    let mut sample = vec![0.; ndim];
    for i in 0..ndim {
        sample[i] = uniform_sample_in_bounds_first_dim(min, max, max_sum, rng);
        min = &min[1..];
        max = &max[1..];
        max_sum -= sample[i];
    }
    sample
}

/// Returns a random sample of a shape within the given bounds.
///
/// `low` are inclusive bounds, while `high` and `high_product` are exclusive
/// bounds.
///
/// **Panics** if `low.len() != high.len()`, any element of `low` is >= the
/// corresponding element of `high`, or the product of the elements of `low` is
/// >= `high_product`.
pub fn sample_shape<R: Rng + ?Sized>(
    low: &[usize],
    high: &[usize],
    high_product: usize,
    rng: &mut R,
) -> Vec<usize> {
    let ndim = low.len();
    assert_eq!(
        ndim,
        high.len(),
        "`low` and `high` must have the same length."
    );
    assert!(
        izip!(low, high).any(|(&low, &high)| low < high),
        "Each element of `low` must be less than the corresponding element of `high`.",
    );
    assert!(
        low.iter().product::<usize>() < high_product,
        "The product of `low` must be less than `high_product`.",
    );

    /// Generate a nonzero-size shape. This implementation assumes that `ndim
    /// != 0`, that the bounds have been sanity-checked, and that it's possible
    /// to generate a non-zero-size shape with the specified bounds.
    fn sample_nonzero_size_shape<R: Rng + ?Sized>(
        low: &[usize],
        high: &[usize],
        high_product: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        let high_ln_sum: f64 = (high_product as f64).ln();
        let low_ln: Vec<f64> = low
            .iter()
            .map(|&low| f64::max((low as f64).ln(), 0.))
            .collect();
        let high_ln: Vec<f64> = high
            .iter()
            .map(|&high| f64::min((high as f64).ln(), high_ln_sum))
            .collect();
        let ln_sample: Vec<f64> = uniform_sample_in_bounds(&low_ln, &high_ln, high_ln_sum, rng);
        let sample: Vec<usize> = izip!(low, high, &ln_sample)
            .map(|(&low, &high, &ln_sample)| {
                let sample = ln_sample
                    .exp()
                    .to_usize()
                    .expect("Error converting axis length to `usize`.");
                // Sanity check.
                assert!(
                    sample >= low && sample < high,
                    "Failed to meet bounds. This is a bug; please report it.",
                );
                sample
            })
            .collect();
        // Sanity check.
        assert!(
            sample.iter().product::<usize>() < high_product,
            "Failed to meet bounds. This is a bug; please report it.",
        );
        sample
    }

    let can_be_empty = low.iter().any(|&low| low == 0);
    let must_be_empty = high_product == 1 || high.iter().any(|&high| high == 1);

    if ndim == 0 {
        Vec::new()
    } else if can_be_empty && (must_be_empty || rng.gen_bool(0.02)) {
        // Generate a zero-size shape.

        // Determine which axis lengths to set to zero.
        let mut zero_axes = vec![false; ndim];

        // Randomly select at least one of the axes that can be zero.
        let axes_can_zero: Vec<usize> = low
            .iter()
            .enumerate()
            .filter(|(_, &low)| low == 0)
            .map(|(ax, _)| ax)
            .collect();
        let num_to_zero = rng.gen_range(1, axes_can_zero.len() + 1);
        for &ax in axes_can_zero.choose_multiple(rng, num_to_zero) {
            zero_axes[ax] = true;
        }

        // Also, write which axis lengths *must* be zero.
        for (ax, &high) in high.iter().enumerate() {
            if high == 1 {
                zero_axes[ax] = true;
            }
        }

        // Get bounds of non-zero-length axes.
        let (nonzero_low, nonzero_high): (Vec<usize>, Vec<usize>) = izip!(&zero_axes, low, high)
            .filter_map(|(&zero, &low, &high)| if zero { None } else { Some((low, high)) })
            .unzip();

        // Sample the non-zero-length axes such that their product does not
        // exceed `isize::MAX` (required by `ArrayBase`).
        let nonzero_sample =
            sample_nonzero_size_shape(&nonzero_low, &nonzero_high, std::isize::MAX as usize, rng);

        // Join the zero-length axes and non-zero-length axes together.
        let mut sample = Vec::with_capacity(ndim);
        let mut nonzero_sample_iter = nonzero_sample.iter().copied();
        for ax in 0..ndim {
            if zero_axes[ax] {
                sample.push(0);
            } else {
                sample.push(nonzero_sample_iter.next().unwrap());
            }
        }
        debug_assert!(nonzero_sample_iter.next().is_none());
        sample
    } else {
        // Generate a nonzero-size shape.
        sample_nonzero_size_shape(low, high, high_product, rng)
    }
}

/// Sample positive strides such that
///
/// * They are valid for an array of the specified shape.
///
/// * If `must_be_mergeable[i]` is `true`, then it must be possible to merge
///   axis `i` into the next inner axis. (If `must_be_mergeable` is `true` for
///   the innermost axis, then its stride must be 1.)
///
/// * The sizes of the strides are such that axis `axes_order[0]` has the
///   largest stride, axis `axes_order[1]` has the next largest stride, etc.
///
/// * The length of the array in memory does not exceed `high_mem_len`.
///
/// The sampling strategy is as follows:
///
/// 1. Compute the ratio of `high_mem_len` to the memory length of a
///    contiguous array with the specified shape.
///
/// 2. Randomly generate values that represent ratios with respect to the
///    contiguous strides.
///
/// 3. Actually multiply those ratios by the minimum possible strides instead
///    of the contiguous strides. (This may skew the distribution somewhat, and
///    makes it impossible to reach `high_mem_len` in some cases, but it
///    keeps the implementation simple.)
pub fn sample_positive_strides<D, R>(
    shape: D,
    require_abs_step_one: AxesMask<D, IxDyn>,
    axes_order: AxesFor<D, D>,
    high_mem_len: usize,
    rng: &mut R,
) -> D
where
    D: Dimension,
    R: Rng + ?Sized,
{
    let ndim = shape.ndim();
    assert_eq!(ndim, require_abs_step_one.for_ndim());
    assert_eq!(ndim, axes_order.num_axes());

    // FIXME: What about nonzero strides with zero-sized shape?
    if shape.size() == 0 {
        return D::zeros(ndim);
    }

    let max_sum_ln_ratios = (high_mem_len as f64 / shape.size() as f64).ln();
    let num_random = require_abs_step_one.num_false();
    let random_ln_ratios = uniform_sample_in_bounds(
        &vec![0.; num_random],
        &vec![max_sum_ln_ratios; num_random],
        max_sum_ln_ratios,
        rng,
    );
    let mut random_ln_ratios_iter = random_ln_ratios.iter();

    let mut rev_axes_order = axes_order;
    rev_axes_order.reverse();

    let mut strides = D::zeros(ndim);
    // Maximum offset reachable by moving along axes that are "inner" with
    // respect to the current one.
    let mut max_inner_offset = 0;
    let mut abs_step_one_stride = 1;
    rev_axes_order.visitv(|axis| {
        let ax = axis.index();
        let len = shape[ax];
        let stride = if require_abs_step_one.read(axis) {
            abs_step_one_stride
        } else {
            let random_ratio = random_ln_ratios_iter.next().unwrap().exp();
            ((max_inner_offset + 1) as f64 * random_ratio)
                .to_usize()
                .unwrap()
        };
        strides[ax] = stride;

        max_inner_offset += (len - 1) * stride;
        abs_step_one_stride = len * stride;
    });

    strides
}

/// Returns the absolute difference in units of the element type between least
/// and greatest address accessible by moving along all axes.
fn max_abs_offset<D: Dimension>(dim: &D, strides: &D) -> usize {
    izip!(dim.slice(), strides.slice()).fold(0usize, |acc, (&d, &s)| {
        let s = s as isize;
        // Calculate maximum possible absolute movement along this axis.
        let off = d.saturating_sub(1) * (s.abs() as usize);
        acc + off
    })
}

pub trait AxesConstraints: Debug {
    type Dim: Dimension;

    fn sample_ndim<R>(&self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        Self::Dim::NDIM.unwrap_or_else(|| rng.gen_range(0, 8))
    }

    fn require_abs_step_one(&self, ndim: usize) -> AxesMask<Self::Dim, IxDyn>;

    fn forbid_invert(&self, ndim: usize) -> AxesMask<Self::Dim, IxDyn>;

    fn axis_lens_low(&self, ndim: usize) -> Self::Dim;

    fn axis_lens_high(&self, ndim: usize) -> Self::Dim;
}

#[derive(Clone, Debug)]
pub struct AxesConstraintsConstNdim<D: Dimension> {
    pub require_abs_step_one: AxesMask<D, IxDyn>,
    pub forbid_invert: AxesMask<D, IxDyn>,
    /// The allowable lengths of the axes.
    pub axis_lens: Vec<Range<usize>>,
}

impl<D> AxesConstraints for AxesConstraintsConstNdim<D>
where
    D: Dimension,
{
    type Dim = D;

    fn require_abs_step_one(&self, ndim: usize) -> AxesMask<D, IxDyn> {
        assert_eq!(ndim, self.require_abs_step_one.for_ndim());
        self.require_abs_step_one.clone()
    }

    fn forbid_invert(&self, ndim: usize) -> AxesMask<D, IxDyn> {
        assert_eq!(ndim, self.forbid_invert.for_ndim());
        self.forbid_invert.clone()
    }

    fn axis_lens_low(&self, ndim: usize) -> D {
        assert_eq!(ndim, self.axis_lens.len());
        let mut axis_lens_low = D::zeros(ndim);
        axis_lens_low.indexed_map_inplace(|axis, low| *low = self.axis_lens[axis.index()].start);
        axis_lens_low
    }

    fn axis_lens_high(&self, ndim: usize) -> D {
        assert_eq!(ndim, self.axis_lens.len());
        let mut axis_lens_high = D::zeros(ndim);
        axis_lens_high.indexed_map_inplace(|axis, high| *high = self.axis_lens[axis.index()].end);
        axis_lens_high
    }
}

impl<D> Default for AxesConstraintsConstNdim<D>
where
    D: Dimension,
{
    fn default() -> AxesConstraintsConstNdim<D> {
        let ndim = D::NDIM
            .expect("AxesConstraintsConstNdim must only be used for dimensions with a const ndim.");
        AxesConstraintsConstNdim {
            require_abs_step_one: AxesMask::all_false(ndim).into_dyn_num_true(),
            forbid_invert: AxesMask::all_false(ndim).into_dyn_num_true(),
            axis_lens: vec![0..std::isize::MAX as usize; ndim],
        }
    }
}

#[derive(Clone, Debug)]
pub struct AxesConstraintsDynNdim {
    /// The range of possible ndim.
    pub ndim: Range<usize>,
    pub require_abs_step_one: bool,
    pub forbid_invert: bool,
    /// The allowable length of each axis.
    pub axis_len: Range<usize>,
}

impl AxesConstraints for AxesConstraintsDynNdim {
    type Dim = IxDyn;

    fn sample_ndim<R>(&self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        rng.gen_range(self.ndim.start, self.ndim.end)
    }

    fn require_abs_step_one(&self, ndim: usize) -> AxesMask<IxDyn, IxDyn> {
        if self.require_abs_step_one {
            AxesMask::all_true(ndim).into_dyn_num_true()
        } else {
            AxesMask::all_false(ndim).into_dyn_num_true()
        }
    }

    fn forbid_invert(&self, ndim: usize) -> AxesMask<IxDyn, IxDyn> {
        if self.forbid_invert {
            AxesMask::all_true(ndim).into_dyn_num_true()
        } else {
            AxesMask::all_false(ndim).into_dyn_num_true()
        }
    }

    fn axis_lens_low(&self, ndim: usize) -> IxDyn {
        IxDyn::from_elem(ndim, self.axis_len.start)
    }

    fn axis_lens_high(&self, ndim: usize) -> IxDyn {
        IxDyn::from_elem(ndim, self.axis_len.end)
    }
}

impl Default for AxesConstraintsDynNdim {
    fn default() -> AxesConstraintsDynNdim {
        AxesConstraintsDynNdim {
            ndim: 0..8,
            require_abs_step_one: false,
            forbid_invert: false,
            axis_len: 0..std::isize::MAX as usize,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayoutStrategy<C>
where
    C: AxesConstraints,
{
    pub permute_axes: bool,
    pub axes_constraints: C,
    pub high_len: usize,
    pub high_memory_len: usize,
}

impl<C> Default for LayoutStrategy<C>
where
    C: AxesConstraints + Default,
{
    fn default() -> LayoutStrategy<C> {
        LayoutStrategy {
            permute_axes: true,
            axes_constraints: C::default(),
            high_len: 10_000,
            high_memory_len: 10_000,
        }
    }
}

impl<C> LayoutStrategy<C>
where
    C: AxesConstraints,
{
    pub fn new_tree(&self, runner: &mut TestRunner) -> LayoutValueTree<C::Dim> {
        let rng = runner.rng();

        let ndim = self.axes_constraints.sample_ndim(rng);
        let require_abs_step_one = self.axes_constraints.require_abs_step_one(ndim);
        let forbid_invert = self.axes_constraints.forbid_invert(ndim);
        let axis_lens_low = self.axes_constraints.axis_lens_low(ndim);
        let axis_lens_high = self.axes_constraints.axis_lens_high(ndim);

        let shape: C::Dim = {
            let sample = sample_shape(
                axis_lens_low.slice(),
                axis_lens_high.slice(),
                std::cmp::min(self.high_len, self.high_memory_len),
                rng,
            );
            let mut shape = C::Dim::zeros(ndim);
            shape.slice_mut().copy_from_slice(&sample);
            shape
        };

        let sum_borders = ((self.high_memory_len - shape.size() - 1) as f64
            * Beta::new(1., 10.).sample(rng))
        .to_usize()
        .unwrap();
        let start_border = rng.gen_range(0, sum_borders + 1);

        let mut axis_order = axes_all().into_axes_for(ndim);
        if self.permute_axes {
            axis_order.shuffle(rng);
        }

        let abs_strides = sample_positive_strides(
            shape.clone(),
            require_abs_step_one,
            axis_order,
            self.high_memory_len - sum_borders,
            rng,
        );

        let mut invert_axes = AxesMask::all_false(ndim).into_dyn_num_true();
        invert_axes.indexed_mapv_inplace(|axis, _| {
            if forbid_invert.read(axis) {
                false
            } else {
                rng.gen_bool(0.3)
            }
        });

        LayoutValueTree {
            data_len: sum_borders + max_abs_offset(&shape, &abs_strides) + 1,
            offset: start_border,
            shape,
            abs_strides,
            // axis_order,
            invert_axes,

            force_standard: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayoutValueTree<D: Dimension> {
    /// Length of the `Vec` containing the data.
    data_len: usize,
    /// Offset from start of data to first element, before applying axis inversions.
    offset: usize,
    /// Shape of the array.
    shape: D,
    /// Strides before applying axis inversions.
    abs_strides: D,
    // /// Axes sorted by descending absolute stride.
    // axis_order: D,
    /// Which axes to invert.
    invert_axes: AxesMask<D, IxDyn>,

    /// Force standard layout. This is the state of the value tree.
    force_standard: bool,
}

impl<D> LayoutValueTree<D>
where
    D: Dimension,
{
    // pub fn ndim(&self) -> usize {
    //     let ndim = self.shape.ndim();
    //     assert_eq!(ndim, self.abs_strides.ndim());
    //     // assert_eq!(ndim, self.axis_order.ndim());
    //     assert_eq!(ndim, self.invert_axes.for_ndim());
    //     ndim
    // }

    /// Returns the length the `Vec` containing all of the element value trees
    /// must be.
    pub fn all_trees_len(&self) -> usize {
        self.data_len
    }

    // fn abs_contig_strides(&self) -> D {
    //     let ndim = self.ndim();
    //     let mut abs_contig_strides = D::zeros(ndim);
    //     let mut abs_stride = 1;
    //     for i in (0..ndim).rev() {
    //         let ax = self.axis_order[i];
    //         abs_contig_strides[ax] = abs_stride;
    //         abs_stride *= self.shape[ax];
    //     }
    //     abs_contig_strides
    // }

    /// Creates a view of the visible value trees.
    fn view_trees<A>(&self, all_trees: &[A]) -> ArrayView<'_, A, D> {
        assert!(self.offset < all_trees.len());
        let mut view = unsafe {
            let ptr = all_trees.as_ptr().add(self.offset);
            ArrayView::from_shape_ptr(self.shape.clone().strides(self.abs_strides.clone()), ptr)
        };
        self.invert_axes.indexed_visitv(|axis, invert| {
            if invert {
                view.invert_axis(axis)
            }
        });
        view
    }

    /// Creates an array of values, given a slice of all of the value trees.
    ///
    /// The resulting array will be constructed according to the current
    /// layout. In the base case, the resulting array will contain values
    /// corresponding to all the trees, but only a portion will be visible (due
    /// to non-contiguous layout). When the layout is simplified to standard
    /// layout, the hidden values (those between the elements of the
    /// non-contiguous layout) will no longer be present.
    pub fn current<A>(&self, all_trees: &[A]) -> Array<A::Value, D>
    where
        A: ValueTree,
    {
        assert_eq!(all_trees.len(), self.data_len);
        if self.force_standard {
            let values: Vec<A::Value> = self
                .view_trees(all_trees)
                .iter()
                .map(|tree| tree.current())
                .collect();
            Array::from_shape_vec(self.shape.clone(), values)
                .expect("Error creating array with standard layout.")
        } else {
            let all_values: Vec<A::Value> = all_trees.iter().map(|tree| tree.current()).collect();

            // FIXME: Handle offset.

            let mut arr = Array::from_shape_vec(
                self.shape.clone().strides(self.abs_strides.clone()),
                all_values,
            )
            .expect("Error creating array.");
            self.invert_axes.indexed_visitv(|axis, invert| {
                if invert {
                    arr.invert_axis(axis)
                }
            });
            arr
        }
    }

    pub fn simplify(&mut self) -> bool {
        if self.force_standard {
            false
        } else {
            self.force_standard = true;
            true
        }
    }

    pub fn complicate(&mut self) -> bool {
        if self.force_standard {
            self.force_standard = false;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use super::volume_within_bounds;
    use itertools::izip;
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng, Rng,
    };

    /// Approximate the volume within bounds and its uncertainty, using Monte Carlo integration.
    fn monte_carlo_volume_within_bounds<R: Rng>(
        min: &[f64],
        max: &[f64],
        max_sum: f64,
        rng: &mut R,
        n_samples: usize,
    ) -> (f64, f64) {
        assert_eq!(min.len(), max.len());
        let sample_volume: f64 = izip!(min, max).map(|(&min, &max)| max - min).product();
        let distributions: Vec<_> = izip!(min, max)
            .map(|(&min, &max)| Uniform::new(min, max))
            .collect();
        let mut n_in_bounds: usize = 0;
        for _ in 0..n_samples {
            let point: Vec<f64> = distributions
                .iter()
                .map(|distro| distro.sample(rng))
                .collect();
            let is_in_bounds: bool = izip!(min, max, &point)
                .all(|(&min, &max, &p)| p >= min && p < max)
                && point.iter().sum::<f64>() < max_sum;
            if is_in_bounds {
                n_in_bounds += 1;
            }
        }
        let n_in_bounds = n_in_bounds as f64;
        let n_samples = n_samples as f64;
        let frac_in_bounds = n_in_bounds / n_samples;
        let estimate = sample_volume * n_in_bounds / n_samples;
        let variance = (n_in_bounds * (1. - frac_in_bounds).powi(2)
            + (n_samples - n_in_bounds) * (0. - frac_in_bounds).powi(2))
            / (n_samples - 1.);
        let uncertainty = sample_volume * variance.sqrt() / n_samples.sqrt();
        (estimate, uncertainty)
    }

    fn test_volume_within_bounds(min: &[f64], max: &[f64], max_sum: f64) {
        let n_samples = 100_000;
        let (estimate, uncertainty) =
            monte_carlo_volume_within_bounds(min, max, max_sum, &mut thread_rng(), n_samples);
        let calc = volume_within_bounds(min, max, max_sum);
        assert!((calc - estimate).abs() < 3. * uncertainty);
    }

    #[test]
    fn volume_within_bounds_0d() {
        assert_eq!(0., volume_within_bounds(&[], &[], 1.));
    }

    #[test]
    fn volume_within_bounds_1d() {
        test_volume_within_bounds(&[1.], &[4.], 3.);
    }

    #[test]
    fn volume_within_bounds_2d() {
        test_volume_within_bounds(&[1., 5.], &[4., 7.], 8.);
    }

    #[test]
    fn volume_within_bounds_3d() {
        test_volume_within_bounds(&[1., 5., 2.], &[4., 7., 9.], 10.);
    }

    #[test]
    fn volume_within_bounds_4d() {
        test_volume_within_bounds(&[1., 5., 2., 3.], &[4., 7., 9., 10.], 20.);
    }

    #[test]
    fn volume_within_bounds_5d() {
        test_volume_within_bounds(&[1., 5., 2., 3., 4.], &[4., 7., 9., 10., 11.], 20.);
    }
}
