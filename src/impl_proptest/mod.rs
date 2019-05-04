use self::state::{BordersSteps, ChunkInfo, MemoryOrder};
use itertools::{izip, Itertools};
use ndarray::{Array, ArrayViewMut, Axis, Dimension};
use num_traits::ToPrimitive;
use proptest::strategy::{NewTree, Strategy, ValueTree};
use proptest::test_runner::TestRunner;
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
///
/// This implementation will take a very long time on average if there are few
/// sizes within `size_range` or if the sizes within `size_range` have few
/// factors on average.
///
/// **Panics** if `size_range.start >= size_range.end || size_range.end > isize::MAX as usize`.
fn gen_shape<D, R>(rng: &mut R, mut size_range: Range<usize>) -> D
where
    D: Dimension,
    R: Rng,
{
    assert!(size_range.start < size_range.end);
    assert!(size_range.end <= std::isize::MAX as usize);

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

    loop {
        let mut remaining_size = rng.gen_range(size_range.start, size_range.end);
        for (i, ln_axis_len) in
            gen_partition(rng, remaining_size.to_f64().unwrap().ln(), ndim - 1).enumerate()
        {
            let axis_len = ln_axis_len.exp().round().to_usize().unwrap();
            shape[i] = axis_len;
            remaining_size /= axis_len;
        }
        shape[ndim - 1] = remaining_size;

        // This can fail due to axis lengths not dividing evenly into the size.
        if shape.size() >= size_range.start {
            return shape;
        }
    }
}

// TODO: How to generate arrays with stride 0?
#[derive(Clone, Debug)]
pub struct ArrayStrategy<T, D> {
    pub elem: T,
    // TODO: Change this to size_with_hidden
    pub visible_size: Range<usize>,
    pub max_step: usize,
    pub max_lower_border: usize,
    pub max_upper_border: usize,
    pub invert_probability: f64,
    pub permute_axes: bool,
    pub dim_type: PhantomData<D>,
}

impl<T, D> ArrayStrategy<T, D> {
    pub fn default_with_elem(elem: T) -> ArrayStrategy<T, D> {
        ArrayStrategy {
            elem,
            visible_size: 0..1000,
            max_step: 4,
            max_lower_border: 20,
            max_upper_border: 20,
            invert_probability: 0.5,
            permute_axes: true,
            dim_type: PhantomData,
        }
    }
}

impl<T: Default, D> Default for ArrayStrategy<T, D> {
    fn default() -> ArrayStrategy<T, D> {
        ArrayStrategy::default_with_elem(T::default())
    }
}

impl<T, D> Strategy for ArrayStrategy<T, D>
where
    T: Strategy,
    D: Dimension,
{
    type Tree = ArrayValueTree<T::Tree, D>;
    type Value = Array<T::Value, D>;

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let visible_shape: D = gen_shape(runner.rng(), self.visible_size.clone());
        let ndim = visible_shape.ndim();
        let borders_steps = BordersSteps::gen_random(
            runner.rng(),
            ndim,
            self.max_lower_border,
            self.max_upper_border,
            self.max_step,
        );
        let memory_order = MemoryOrder::gen_random(
            runner.rng(),
            ndim,
            self.invert_probability,
            self.permute_axes,
        );
        ArrayValueTree::new(
            &self.elem,
            runner,
            visible_shape,
            borders_steps,
            memory_order,
        )
    }
}

/// A shrink action for an `ArrayValueTree`.
#[derive(Clone, Debug)]
enum ShrinkAction<D: Dimension> {
    RemoveBordersStep(Axis),
    ForbidInvertAxis(Axis),
    SortAxes,
    SelectSubchunk(Option<D>),
    /// Shrink the element at the given index in the current array.
    ///
    /// Note that the index is for `.all_current_trees()`, not `all_base_trees`.
    ShrinkElement(Option<D>),
}

/// `ValueTree` corresponding to `ArrayStrategy`.
#[derive(Clone, Debug)]
pub struct ArrayValueTree<A, D: Dimension> {
    all_trees: Array<A, D>,
    borders_steps: BordersSteps<D>,
    memory_order: MemoryOrder<D>,
    parent_chunk: Option<ChunkInfo<D>>,
    current_chunk: ChunkInfo<D>,
    // min_shape: D,
    /// Action to perform on next `simplify` call.
    next_action: Option<ShrinkAction<D>>,
    /// Action performed on most recent `simplify` call.
    last_action: Option<ShrinkAction<D>>,
}

impl<A: ValueTree, D: Dimension> ArrayValueTree<A, D> {
    pub fn new<T>(
        elem_strategy: &T,
        runner: &mut TestRunner,
        visible_shape: D,
        borders_steps: BordersSteps<D>,
        memory_order: MemoryOrder<D>,
    ) -> Result<ArrayValueTree<A, D>, proptest::test_runner::Reason>
    where
        T: Strategy<Tree = A, Value = A::Value>,
    {
        let ndim = visible_shape.ndim();
        assert_eq!(ndim, borders_steps.ndim());
        assert_eq!(ndim, memory_order.ndim());
        let current_chunk = ChunkInfo::first_chunk(visible_shape, &borders_steps);
        let shape_all = current_chunk.shape_with_hidden(&borders_steps);
        let size_all = shape_all.size();
        let all_trees = Array::from_shape_vec(
            shape_all,
            std::iter::repeat_with(|| elem_strategy.new_tree(runner))
                .take(size_all)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .unwrap();
        let next_action = Some(if ndim == 0 {
            ShrinkAction::ShrinkElement(Some(D::zeros(ndim)))
        } else {
            ShrinkAction::RemoveBordersStep(Axis(0))
        });
        Ok(ArrayValueTree {
            all_trees,
            borders_steps,
            memory_order,
            parent_chunk: None,
            current_chunk,
            next_action,
            last_action: None,
        })
    }

    /// Returns the number of dimensions of arrays produced by
    /// `self.current()`.
    pub fn ndim(&self) -> usize {
        self.current_chunk.ndim()
    }

    /// Returns the shape of `self.current()` (without actually calling `self.current()`).
    pub fn shape(&self) -> D {
        self.current_chunk.shape()
    }

    pub fn view_with_hidden_mut(&mut self) -> ArrayViewMut<'_, A, D> {
        let mut trees = self.all_trees.view_mut();
        self.current_chunk
            .slice_with_hidden(&mut trees, &self.borders_steps);
        trees
    }
}

impl<A: ValueTree, D: Dimension> ValueTree for ArrayValueTree<A, D> {
    type Value = Array<A::Value, D>;

    fn current(&self) -> Array<A::Value, D> {
        self.current_chunk.map(
            &self.all_trees,
            &self.borders_steps,
            &self.memory_order,
            |elem| elem.current(),
        )
    }

    fn simplify(&mut self) -> bool {
        let ndim = self.ndim();
        if let Some(action) = self.next_action.take() {
            match &action {
                &ShrinkAction::RemoveBordersStep(axis) => {
                    self.borders_steps.remove_borders_step(axis);
                    self.next_action = {
                        let next_axis = Axis(axis.index() + 1);
                        if next_axis.index() < ndim {
                            Some(ShrinkAction::RemoveBordersStep(next_axis))
                        } else {
                            Some(ShrinkAction::ForbidInvertAxis(Axis(0)))
                        }
                    };
                }
                &ShrinkAction::ForbidInvertAxis(axis) => {
                    self.memory_order.forbid_invert(axis);
                    self.next_action = {
                        let next_axis = Axis(axis.index() + 1);
                        if next_axis.index() < ndim {
                            Some(ShrinkAction::ForbidInvertAxis(next_axis))
                        } else {
                            Some(ShrinkAction::SortAxes)
                        }
                    };
                }
                &ShrinkAction::SortAxes => {
                    self.memory_order.sort_axes();
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.current_chunk.first_subchunk_index(),
                    ));
                }
                &ShrinkAction::SelectSubchunk(Some(ref subchunk_index)) => {
                    let subchunk = self.current_chunk.get_subchunk(subchunk_index.clone());
                    let old_current_chunk = std::mem::replace(&mut self.current_chunk, subchunk);
                    self.parent_chunk = Some(old_current_chunk);
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.current_chunk.first_subchunk_index(),
                    ));
                }
                &ShrinkAction::SelectSubchunk(None) => {
                    // FIXME: `first_index` is an undocumented method from `ndarray`.
                    self.next_action = Some(ShrinkAction::ShrinkElement(
                        self.current_chunk
                            .shape_with_hidden(&self.borders_steps)
                            .first_index(),
                    ));
                }
                &ShrinkAction::ShrinkElement(Some(ref index)) => {
                    let mut trees_with_hidden = self.view_with_hidden_mut();
                    trees_with_hidden[index.clone()].simplify();
                    // FIXME: `next_for` is an undocumented method from `ndarray`.
                    self.next_action = Some(ShrinkAction::ShrinkElement(
                        trees_with_hidden.raw_dim().next_for(index.clone()),
                    ));
                }
                &ShrinkAction::ShrinkElement(None) => {
                    self.next_action = None;
                }
            }
            self.last_action = Some(action);
            true
        } else {
            false
        }
    }

    fn complicate(&mut self) -> bool {
        if let Some(action) = self.last_action.take() {
            match action {
                ShrinkAction::RemoveBordersStep(axis) => {
                    self.borders_steps.restore_borders_step(axis);
                }
                ShrinkAction::ForbidInvertAxis(axis) => {
                    self.memory_order.allow_invert(axis);
                }
                ShrinkAction::SortAxes => {
                    self.memory_order.unsort_axes();
                }
                ShrinkAction::SelectSubchunk(Some(subchunk_index)) => {
                    let parent_chunk = self.parent_chunk.take();
                    self.current_chunk = parent_chunk.unwrap();
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.current_chunk.next_subchunk_index(subchunk_index),
                    ));
                }
                ShrinkAction::SelectSubchunk(None) => {}
                ShrinkAction::ShrinkElement(Some(index)) => {
                    let mut trees_with_hidden = self.view_with_hidden_mut();
                    if trees_with_hidden[index.clone()].complicate() {
                        // `.complicate()` only attempts to *partially* undo the last
                        // simplification, so it may be possible to complicate this element further.
                        self.last_action = Some(ShrinkAction::ShrinkElement(Some(index)));
                    }
                }
                ShrinkAction::ShrinkElement(None) => {}
            }
            true
        } else {
            false
        }
    }
}

mod state;

#[cfg(test)]
mod tests {
    use super::ArrayStrategy;
    use ndarray::prelude::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn example(arr in ArrayStrategy::<_, Ix3>::default_with_elem(0..10)) {
            prop_assert_ne!(arr.sum() % 3, 5);
        }
    }
}
