use crate::{Array, Axis, Dimension};
use itertools::{izip, Itertools};
use num_traits::ToPrimitive;
use proptest::{Strategy, ValueTree};
use rand::{distributions, seq::SliceRandom, Rng};
use std::marker::PhantomData;
use std::ops::Range;

/// Randomly generates `n` numbers that have the given `sum`.
fn gen_partition<R: Rng>(rng: &mut R, sum: f64, n: usize) -> impl Iterator<Item = f64> {
    let mut splits: Vec<f64> = vec![0.];
    splits.extend(
        rng.sample_iter::<f64, _>(&distributions::Uniform::new_inclusive(0., sum))
            .take(n - 1),
    );
    splits.push(sum);
    splits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    splits
        .into_iter()
        .tuple_windows()
        .map(|(left, right)| right - left)
}

/// Randomly generates a shape with size in the given range.
fn gen_shape<D, R>(rng: &mut R, mut size_range: Range<usize>) -> D
where
    D: Dimension,
    R: Rng,
{
    let ndim = D::NDIM.unwrap_or_else(|| rng.gen_range(0, 8));
    if ndim == 0 {
        return D::zeros(0);
    }
    let mut shape = D::zeros(ndim);

    if size_range.start == 0 {
        // The 0.02 threshold is chosen such that there is <1% probability of
        // generating 256 cases (default number for proptest) for which none
        // meet this condition, and such that the expected number of times this
        // condition is met out of 256 cases is about 5.
        if rng.gen::<f64>() < 0.02 {
            // Fill all but first element (since at least one axis length must
            // be zero).
            izip!(
                &mut shape.slice_mut()[1..],
                rng.sample_iter(&distributions::Uniform::new(0, size_range.end)),
            )
            .for_each(|(s, axis_len)| *s = axis_len);
            // Shuffle to move the zero axis length to a random position.
            shape.slice_mut().shuffle(rng);
            return shape;
        }
        size_range.start = 1;
    }
    debug_assert!(size_range.start >= 1);

    let mut remaining_size = rng.gen_range(size_range.start, size_range.end);
    for (i, ln_axis_len) in
        gen_partition(rng, remaining_size.to_f64().unwrap().ln(), ndim - 1).enumerate()
    {
        let axis_len = ln_axis_len.exp().round().to_usize().unwrap();
        shape[i] = axis_len;
        remaining_size /= axis_len;
    }
    shape[ndim - 1] = remaining_size;

    shape
}

#[derive(Clone, Copy, Debug)]
enum Shrink {
    DeleteElement(usize),
    ShrinkElement(usize),
}

#[derive(Clone, Copy, Debug)]
enum SimplifyLayout {
    UninvertAxis(Axis),
    RemoveStep(Axis),
    RemoveBorders(Axis),
    PartiallySortAxes,
}

#[derive(Clone, Copy, Debug)]
enum ShrinkState {
    SimplifyLayout(SimplifyLayout),
    BisectProportional,
    BisectAxis(Axis),
    ShrinkElement(usize),
}

struct AxisMasks<D: Dimension> {
    /// Each element of `masks` is a mask for the corresponding axis.
    masks: Vec<Vec<bool>>,
    /// Number of axes.
    dim: PhantomData<D>,
}

impl<D: Dimension> AxisMasks<D> {
    /// Returns a new `AxisMasks` instance of all `true` values for an array of
    /// the given `shape`.
    pub fn saturated(shape: D) -> AxisMasks<D> {
        AxisMasks {
            masks: shape
                .slice()
                .iter()
                .map(|&axis_len| vec![true; axis_len])
                .collect(),
            dim: PhantomData,
        }
    }

    pub fn set(&mut self, axis: Axis, index: usize) {
        self.masks[axis.index()][index] = true;
    }

    pub fn clear(&mut self, axis: Axis, index: usize) {
        self.masks[axis.index()][index] = false;
    }
}

struct ShapeLayout<D> {
    /// Shape of final array.
    shape: D,
    steps: D,
    lower_borders: D,
    upper_borders: D,
    axis_permutation: D,
}

/// `ValueTree` corresponding to `ArrayStrategy`.
#[derive(Clone, Debug)]
pub struct ArrayValueTree<A: ValueTree, D: Dimension> {
    elements: Array<A, D>,
    /// Masks of included elements.
    axis_masks: AxisMasks<D>,
    min_size: usize,
    shrink: Shrink,
    prev_shrink: Option<Shrink>,
}

impl<A: ValueTree, D: Dimension> ValueTree for ArrayValueTree<A, D> {
    type Value = Array<A::Value, D>;

    fn current(&self) -> Vec<A::Value> {
        self.elements
            .iter()
            .enumerate()
            .filter(|&(ix, _)| self.included_elements.test(ix))
            .map(|(_, element)| element.current())
            .collect()
    }

    fn simplify(&mut self) -> bool {
        // The overall strategy here is to iteratively delete elements from the
        // list until we can do so no further, then to shrink each remaining
        // element in sequence.
        //
        // For `complicate()`, we simply undo the last shrink operation, if
        // there was any.
        if let Shrink::DeleteElement(ix) = self.shrink {
            // Can't delete an element if beyond the end of the vec or if it
            // would put us under the minimum length.
            if ix >= self.elements.len() || self.included_elements.count() == self.min_size {
                self.shrink = Shrink::ShrinkElement(0);
            } else {
                self.included_elements.clear(ix);
                self.prev_shrink = Some(self.shrink);
                self.shrink = Shrink::DeleteElement(ix + 1);
                return true;
            }
        }

        while let Shrink::ShrinkElement(ix) = self.shrink {
            if ix >= self.elements.len() {
                // Nothing more we can do
                return false;
            }

            if !self.included_elements.test(ix) {
                // No use shrinking something we're not including.
                self.shrink = Shrink::ShrinkElement(ix + 1);
                continue;
            }

            if !self.elements[ix].simplify() {
                // Move on to the next element
                self.shrink = Shrink::ShrinkElement(ix + 1);
            } else {
                self.prev_shrink = Some(self.shrink);
                return true;
            }
        }

        panic!("Unexpected shrink state");
    }

    fn complicate(&mut self) -> bool {
        match self.prev_shrink {
            None => false,
            Some(Shrink::DeleteElement(ix)) => {
                // Undo the last item we deleted. Can't complicate any further,
                // so unset prev_shrink.
                self.included_elements.set(ix);
                self.prev_shrink = None;
                true
            }
            Some(Shrink::ShrinkElement(ix)) => {
                if self.elements[ix].complicate() {
                    // Don't unset prev_shrink; we may be able to complicate
                    // again.
                    true
                } else {
                    // Can't complicate the last element any further.
                    self.prev_shrink = None;
                    false
                }
            }
        }
    }
}
