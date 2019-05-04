use self::state::{BordersSteps, ChunkInfo, MemoryOrder};
use crate::{AxesFor, AxesMask, DimensionExt};
use itertools::{izip, Itertools};
use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension, IxDyn, Slice};
use num_traits::ToPrimitive;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use rand::{distributions, seq::SliceRandom, Rng};
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

#[derive(Clone, Debug)]
struct LayoutTree<D: Dimension> {
    /// Lower borders to cut off when slicing.
    lower_borders: D,
    /// Upper borders to cut off when slicing.
    upper_borders: D,
    /// Positive steps for use in slicing.
    steps: D,
    /// Which axes are in reverse order in memory.
    inverted: AxesMask<D, IxDyn>,
    /// Iteration order over axes such that iteration occurs in memory order
    /// (ignoring inversions).
    iter_order: AxesFor<D, D>,
}

impl<D: Dimension> LayoutTree<D> {
    /// Returns the number of axes.
    pub fn ndim(&self) -> usize {
        let ndim = self.lower_borders.ndim();
        debug_assert_eq!(ndim, self.upper_borders.ndim());
        debug_assert_eq!(ndim, self.steps.ndim());
        debug_assert_eq!(ndim, self.inverted.for_ndim());
        debug_assert_eq!(ndim, self.iter_order.for_ndim());
        debug_assert_eq!(ndim, self.iter_order.num_axes());
        ndim
    }

    /// Returns the shape after calling `self.apply_borders_steps_invert` on an
    /// array of shape `before`.
    pub fn shape_after_apply(&self, before: D) -> D {
        before.indexed_mapv(|axis, len| {
            let ax = axis.index();
            let without_borders = len - self.lower_borders[ax] - self.upper_borders[ax];
            let after_step = without_borders / self.steps[ax]
                + if without_borders % self.steps[ax] != 0 {
                    1
                } else {
                    0
                };
            after_step
        })
    }

    /// Given the underlying array of elements, removes the borders and applies
    /// the steps such that the resulting array matches the layout specified by
    /// `self` (ignoring `iter_order` and `invert`).
    ///
    /// If you want the resulting array to have a memory layout matching
    /// `self.iter_order` and `self.invert`, `arr` should have that
    /// `iter_order`.
    pub fn apply_borders_steps<S: Data>(&self, arr: &mut ArrayBase<S, D>) {
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            arr.slice_axis_inplace(
                axis,
                Slice::new(
                    self.lower_borders[ax] as isize,
                    Some(self.upper_borders[ax] as isize),
                    self.steps[ax] as isize,
                ),
            );
        }
    }
}

// struct AxesPermutation<D: Dimension>(AxesFor<D, D>);

// impl<D: Dimension> AxesPermutation<D> {
//     /// Applies the permutation to the array.
//     pub fn apply<S: Data, D>(&self, arr: ArrayBase<S, D>) -> ArrayBase<S, D> {
//         arr.permuted_axes(self.into_inner())
//     }

//     /// Returns the permutation such that `returned.apply(self.apply(arr)) ==
//     /// arr`.
//     pub fn inverse(&self) -> AxesFor<D, D> {
//         unimplemented!()
//     }

//     /// Returns the permutation such that `composed.apply(arr) ==
//     /// self.apply(inner.apply(arr))`.
//     pub fn compose(&self, inner: &AxesPermutation<D>) -> AxesPermutation<D> {
//         unimplemented!()
//     }

//     /// Returns the permutation such that `self.apply(arr) ==
//     /// decomposed.apply(inner.apply(arr))`.
//     pub fn decompose(&self, inner: &AxesPermutation<D>) -> AxesPermutation<D> {
//         unimplemented!()
//     }
// }

// struct foo {
//     chunk_index: D,
//     step_bw_chunks: D,
//     chunk_visible_shape: D,
// }

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

// impl<D: Dimension> ShrinkAction<D> {
//     /// Returns the next shrink action for an array of the given shape.
//     pub fn next(&self, shape: &D) -> Option<ShrinkAction<D>> {
//         let ndim = shape.ndim();
//         match self {
//             &ShrinkAction::RemoveBordersStep(axis) => {
//                 let next = Axis(axis.index() + 1);
//                 if next.index() < ndim {
//                     Some(ShrinkAction::RemoveBordersStep(next))
//                 } else {
//                     Some(ShrinkAction::ForbidInvertAxis(Axis(0)))
//                 }
//             }
//             &ShrinkAction::ForbidInvertAxis(axis) => {
//                 let next = Axis(axis.index() + 1);
//                 if next.index() < ndim {
//                     Some(ShrinkAction::ForbidInvertAxis(next))
//                 } else {
//                     Some(ShrinkAction::SortAxes)
//                 }
//             }
//             &ShrinkAction::SortAxes => Some(ShrinkAction::ReduceSize),
//             &ShrinkAction::SelectS => Some(ShrinkAction::ReduceSize), // FIXME
//             &ShrinkAction::ShrinkElement(ref index) => {
//                 // FIXME: `next_for` is an undocumented method from `ndarray`.
//                 Some(ShrinkAction::ShrinkElement(shape.next_for(index.clone())?))
//             }
//         }
//     }
// }

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
        let all_trees = Array::from_shape_vec(
            shape_all,
            std::iter::repeat_with(|| elem_strategy.new_tree(runner))
                .take(shape_all.size())
                .collect()?,
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

// let mut shape_plus_borders =
//     self.layout_state
//         .remove_borders
//         .indexed_mapv_to_dim(|axis, remove| {
//             if remove {
//                 self.base_layout.shape[axis.index()]
//             } else {
//                 self.base_layout.shape[axis.index()]
//                     + self.base_layout.lower_borders[axis.index()]
//                     + self.base_layout.upper_borders[axis.index()]
//             }
//         });
// let mut elements_with_borders = all_base_trees;
// self.layout_state
//     .remove_borders
//     .indexed_visitv(|axis, remove| {
//         if remove {
//             elements_with_borders.slice_axis_inplace(
//                 axis,
//                 (self.base_layout.lower_borders[axis.index()] as isize)
//                     ..-(self.base_layout.upper_borders[axis.index()] as isize),
//             );
//         }
//     });
// self.layout_state
//     .uninvert_axes
//     .indexed_visitv(|axis, uninvert| {
//         unimplemented!()
//     });
// let shape_with_borders = elements_with_borders.raw_dim();
// let data: Vec<_> = elements_with_borders.permuted_axes().iter().cloned().collect();
// let with_borders = Array::from_shape_strides(shape_with_borders.strides(strides_without_steps)).unwrap();

mod state;
