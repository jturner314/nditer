pub use self::shape::{AnyShape, FixedShape};

use self::state::{BordersSteps, BordersStepsConfig, ChunkInfo, MemoryOrder, MemoryOrderConfig};
use itertools::izip;
use ndarray::{Array, ArrayViewMut, Axis, Dimension};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Reason, TestRunner};
use rand::{distributions::Distribution, Rng};
use std::cmp;
use std::fmt::Debug;
use std::ops::RangeInclusive;

const MAX_ITER: usize = 1000;

/// A strategy for generating shapes.
///
/// This is used to control the behavior of shape reduction in `ArrayStrategy`.
pub trait ShapeRange {
    type Dim: Dimension;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Self::Dim, Reason>;

    /// Returns the minimum shape size.
    fn min_size(&self) -> usize;

    fn max_size(&self) -> usize;

    fn min_shape(&self) -> Option<Self::Dim>;

    fn max_shape(&self) -> Option<Self::Dim>;
}

#[derive(Clone, Debug)]
pub struct ArrayStrategyConfig<T> {
    shape: T,
    borders_steps: BordersStepsConfig,
    memory_order: MemoryOrderConfig,
    max_size_with_hidden: usize,
}

impl<T: ShapeRange> ArrayStrategyConfig<T> {
    pub fn new(
        shape: T,
        borders_steps: BordersStepsConfig,
        memory_order: MemoryOrderConfig,
        max_size_with_hidden: usize,
    ) -> Self {
        ArrayStrategyConfig {
            shape,
            borders_steps,
            memory_order,
            max_size_with_hidden,
        }
    }

    pub fn from_shape(shape: T) -> Self {
        let max_size_with_hidden = cmp::max(
            10000,
            10 * cmp::max(
                shape.max_size(),
                shape.max_shape().map(|shape| shape.size()).unwrap_or(0),
            ),
        );
        ArrayStrategyConfig {
            shape,
            borders_steps: BordersStepsConfig {
                max_lower_border: 10,
                max_upper_border: 10,
                max_step: 4,
            },
            memory_order: MemoryOrderConfig {
                invert_probability: 0.5,
                permute_axes: true,
            },
            max_size_with_hidden,
        }
    }
}

impl<T> Default for ArrayStrategyConfig<T>
where
    T: Default + ShapeRange,
{
    fn default() -> ArrayStrategyConfig<T> {
        let shape = T::default();
        ArrayStrategyConfig {
            borders_steps: BordersStepsConfig {
                max_lower_border: 10,
                max_upper_border: 10,
                max_step: 4,
            },
            memory_order: MemoryOrderConfig {
                invert_probability: 0.5,
                permute_axes: true,
            },
            max_size_with_hidden: 10000,
            shape,
        }
    }
}

impl<T> Distribution<Result<(T::Dim, BordersSteps<T::Dim>, MemoryOrder<T::Dim>), Reason>>
    for ArrayStrategyConfig<T>
where
    T: ShapeRange,
{
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<(T::Dim, BordersSteps<T::Dim>, MemoryOrder<T::Dim>), Reason> {
        let visible_shape = dbg!(self.shape.sample(rng)?);
        let ndim = visible_shape.ndim();
        let borders_steps = self.borders_steps.sample(ndim, rng);
        let memory_order = self.memory_order.sample(ndim, rng);
        Ok((visible_shape, borders_steps, memory_order))
    }
}

// TODO: How to generate arrays with stride 0?
#[derive(Clone, Debug)]
pub struct ArrayStrategy<T, U> {
    pub elem: T,
    pub config: ArrayStrategyConfig<U>,
}

impl<T, U> Default for ArrayStrategy<T, U>
where
    T: Default,
    U: Default + ShapeRange,
{
    fn default() -> ArrayStrategy<T, U> {
        ArrayStrategy {
            elem: T::default(),
            config: ArrayStrategyConfig::default(),
        }
    }
}

impl<T, U> Strategy for ArrayStrategy<T, U>
where
    T: Strategy,
    U: Debug + ShapeRange,
{
    type Tree = ArrayValueTree<T::Tree, U::Dim>;
    type Value = Array<T::Value, U::Dim>;

    fn new_tree(&self, runner: &mut TestRunner) -> Result<ArrayValueTree<T::Tree, U::Dim>, Reason> {
        for _ in 0..MAX_ITER {
            let (visible_shape, borders_steps, memory_order) = self.config.sample(runner.rng())?;
            let ndim = visible_shape.ndim();
            let min_visible_size = self.config.shape.min_size();
            let min_visible_shape = self
                .config
                .shape
                .min_shape()
                .unwrap_or_else(|| U::Dim::zeros(ndim));

            // Sanity checks for `visible_shape`.
            debug_assert!(visible_shape.size() >= min_visible_size);
            debug_assert!(visible_shape.size() <= self.config.shape.max_size());
            debug_assert!(izip!(visible_shape.slice(), min_visible_shape.slice())
                .all(|(&axis_len, &min_axis_len)| axis_len >= min_axis_len));
            if let Some(max_shape) = self.config.shape.max_shape() {
                debug_assert!(izip!(visible_shape.slice(), max_shape.slice())
                    .all(|(&axis_len, &max_axis_len)| axis_len <= max_axis_len));
            }

            let current_chunk = ChunkInfo::first_chunk(dbg!(visible_shape), &borders_steps);
            let shape_all = dbg!(current_chunk.shape_with_hidden(&borders_steps));
            let size_all = dbg!(shape_all.size());
            if size_all <= self.config.max_size_with_hidden {
                let all_trees = Array::from_shape_vec(
                    shape_all,
                    std::iter::repeat_with(|| self.elem.new_tree(runner))
                        .take(size_all)
                        .collect::<Result<Vec<_>, _>>()?,
                )
                .unwrap();
                let next_action = Some(if ndim == 0 {
                    ShrinkAction::ShrinkElement(Some(U::Dim::zeros(ndim)))
                } else {
                    ShrinkAction::RemoveBordersStep(Axis(0))
                });
                return Ok(ArrayValueTree {
                    all_trees,
                    borders_steps,
                    memory_order,
                    parent_chunk: None,
                    current_chunk,
                    min_visible_size,
                    min_visible_shape,
                    next_action,
                    last_action: None,
                });
            }
        }
        Err(Reason::from(format!(
            "Exceeded {} iterations while trying to generate a shape and layout",
            MAX_ITER,
        )))
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
    min_visible_size: usize,
    min_visible_shape: D,
    /// Action to perform on next `simplify` call.
    next_action: Option<ShrinkAction<D>>,
    /// Action performed on most recent `simplify` call.
    last_action: Option<ShrinkAction<D>>,
}

impl<A: ValueTree, D: Dimension> ArrayValueTree<A, D> {
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

    fn is_shape_within_bounds(&self, shape: &D) -> bool {
        let size = shape.size();
        size >= self.min_visible_size
            && izip!(shape.slice(), self.min_visible_shape.slice())
                .all(|(&axis_len, &min_axis_len)| axis_len >= min_axis_len)
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
            dbg!(&action);
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
                    if self.is_shape_within_bounds(&subchunk.shape()) {
                        let old_current_chunk =
                            std::mem::replace(&mut self.current_chunk, subchunk);
                        self.parent_chunk = Some(old_current_chunk);
                        self.next_action = Some(ShrinkAction::SelectSubchunk(
                            self.current_chunk.first_subchunk_index(),
                        ));
                    } else {
                        self.next_action = Some(ShrinkAction::ShrinkElement(
                            self.current_chunk
                                .shape_with_hidden(&self.borders_steps)
                                .first_index(),
                        ));
                    }
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

mod random;
mod shape;
mod state;

#[cfg(test)]
mod tests {
    use super::{AnyShape, ArrayStrategy, ArrayStrategyConfig, FixedShape};
    use ndarray::prelude::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn example2(
            (arr1, arr2) in AnyShape::<Ix3>::default()
                .prop_flat_map(|shape| {
                    (
                        dbg!(ArrayStrategy {
                            elem: 0..10,
                            config: ArrayStrategyConfig::from_shape(FixedShape(shape)),
                        }),
                        dbg!(ArrayStrategy {
                            elem: 10..20,
                            config: ArrayStrategyConfig::from_shape(FixedShape(shape)),
                        }),
                    )
                })
        ) {
            dbg!((arr1.shape(), arr2.shape()));
            prop_assert_eq!(arr1.shape(), arr2.shape());
            prop_assert_ne!(arr1, arr2);
        }
    }

    proptest! {
        #[test]
        fn example(arr in ArrayStrategy { elem: 0..10, config: ArrayStrategyConfig::<AnyShape<Ix3>>::default() }) {
            prop_assert_ne!(arr.sum() % 5, 4);
        }
    }
}
