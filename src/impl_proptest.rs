use crate::{axes_all, AxesFor, AxesMask, DimensionExt, IntoAxesFor, SubDim};
use itertools::{izip, Itertools};
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension, IxDyn, OwnedRepr,
    ShapeBuilder, Slice, ViewRepr,
};
use num_traits::ToPrimitive;
use proptest::strategy::{Strategy, ValueTree};
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

/// Applies `mapping` to `orig`, returning an array with memory layout matching
/// `inverted` and `iter_order`.
fn map_with_memory_order<'a, A, B, D, F>(
    orig: ArrayView<'a, A, D>,
    inverted: &AxesMask<D, IxDyn>,
    iter_order: &AxesFor<D, D>,
    mapping: F,
) -> Array<B, D>
where
    D: Dimension,
    F: FnMut(&'a A) -> B,
{
    let ndim = orig.ndim();
    debug_assert_eq!(ndim, inverted.for_ndim());
    debug_assert_eq!(ndim, iter_order.for_ndim());
    let shape = orig.raw_dim();
    let mut orig_permuted = orig.permuted_axes(iter_order.clone().into_inner());
    inverted.indexed_visitv(|axis, inv| {
        if inv {
            orig_permuted.invert_axis(axis)
        }
    });

    let new_flat: Vec<B> = orig_permuted.iter().map(mapping).collect();
    let mut new_strides = D::zeros(ndim);
    if !orig_permuted.is_empty() {
        let mut cum_prod: isize = 1;
        for &ax in iter_order.slice().iter().rev() {
            let len = shape[ax];
            new_strides[ax] = cum_prod as usize;
            cum_prod *= len as isize;
        }
    }
    let mut new = Array::from_shape_vec(shape.strides(new_strides), new_flat).unwrap();
    inverted.indexed_visitv(|axis, inv| {
        if inv {
            new.invert_axis(axis)
        }
    });
    new
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

#[derive(Clone, Debug)]
struct ChunkInfo<S: Data, D: Dimension> {
    /// Array of all the trees that could be included in the underlying
    /// representation of chunks.
    ///
    /// The memory layout of this array doesn't matter.
    all_trees: ArrayBase<S, D>,

    first_visible_index: D,
    visible_shape: D,

    base_lower_borders: D,
    base_upper_borders: D,
    base_steps: D,
    remove_borders_steps: AxesMask<D, IxDyn>,

    base_invert: AxesMask<D, IxDyn>,
    allow_invert: AxesMask<D, IxDyn>,

    base_axis_order: AxesFor<D, D>,
    sort_axes: bool,
}

impl<A, S, D> ChunkInfo<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    pub fn ndim(&self) -> usize {
        // TODO: debug assert?
        self.all_trees.ndim()
    }

    pub fn shape(&self) -> D {
        self.visible_shape.clone()
    }

    // ///
    // pub fn lower_border(&self, axis: Axis) -> usize {
    //     if self.remove_borders_steps.read(axis) {
    //         0
    //     } else {
    //         self.base_lower_borders[axis.index()]
    //     }
    // }

    /// Returns a view of all the trees that could be included in the
    /// underlying representation of chunks.
    pub fn view_all(&self) -> ArrayView<'_, A, D> {
        self.all_trees.view()
    }

    /// Returns a view of the visible portion of the current chunk.
    pub fn view_current(&self) -> ArrayView<'_, A, D> {
        let mut view = self.all_trees.view();
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            let start = self.first_visible_index[ax] as isize;
            let step = self.base_steps[ax] as isize;
            let end = start + step * self.visible_shape[ax] as isize;
            view.slice_axis_inplace(axis, Slice::new(start, Some(end), step));
        }
        view
    }

    /// Returns a view of the underlying representation of an owned copy of the
    /// current chunk.
    pub fn view_all_current(&self) -> ArrayView<'_, A, D> {
        let mut view = self.all_trees.view();
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            let (start, end, step) = if self.remove_borders_steps.read(axis) {
                let start = self.first_visible_index[ax];
                let step = self.base_steps[ax];
                let end = start + step * self.visible_shape[ax];
                (start as isize, end as isize, step as isize)
            } else {
                let start = self.first_visible_index[ax] - self.base_lower_borders[ax];
                let step = 1;
                let end = start + step * self.visible_shape[ax] + self.base_upper_borders[ax];
                (start as isize, end as isize, step as isize)
            };
            view.slice_axis_inplace(axis, Slice::new(start, Some(end), step));
        }
        view
    }

    /// Applies the mapping to the underlying representation of the current
    /// chunk, returning an array sliced to show only the visible portion.
    pub fn map_current<'a, F, B>(&'a self, mapping: F) -> Array<B, D>
    where
        A: 'a,
        F: FnMut(&'a A) -> B,
    {
        let mut current = self.map_all_current(mapping);
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            if !self.remove_borders_steps.read(axis) {
                current.slice_axis_inplace(
                    axis,
                    Slice {
                        start: self.base_lower_borders[ax] as isize,
                        end: Some(-(self.base_upper_borders[ax] as isize)),
                        step: self.base_steps[ax] as isize,
                    },
                );
            }
        }
        current
    }

    /// Applies the mapping to the underlying representation of the current
    /// chunk, returning the underlying representation.
    pub fn map_all_current<'a, F, B>(&'a self, mapping: F) -> Array<B, D>
    where
        A: 'a,
        F: FnMut(&'a A) -> B,
    {
        let current_inverted = (&self.allow_invert) & (&self.base_invert);
        map_with_memory_order(
            self.view_all_current(),
            &current_inverted,
            &if self.sort_axes {
                axes_all().into_axes_for(self.ndim())
            } else {
                self.base_axis_order
            },
            mapping,
        )
    }

    /// Removes the borders and step for the owned representation of the
    /// current chunk.
    pub fn remove_borders_step(&mut self, axis: Axis) {
        self.remove_borders_steps.write(axis, true);
    }

    /// Restores the borders and step for the owned representation of the
    /// current chunk.
    pub fn restore_borders_step(&mut self, axis: Axis) {
        self.remove_borders_steps.write(axis, false);
    }

    /// Forbids the given axis from having a negative stride in the owned
    /// representation of the current chunk.
    pub fn forbid_invert(&mut self, axis: Axis) {
        self.allow_invert.write(axis, false);
    }

    /// Allows the given axis to have a negative stride in the owned
    /// representation of the current chunk.
    pub fn allow_invert(&mut self, axis: Axis) {
        self.allow_invert.write(axis, true);
    }

    /// Forces the axes to be in order (ignoring inversions) in the owned
    /// representation of the current chunk.
    pub fn sort_axes(&mut self) {
        self.sort_axes = true;
    }

    /// Allows the axes to be out-of-order in the owned representation of the
    /// current chunk.
    pub fn unsort_axes(&mut self) {
        self.sort_axes = false;
    }

    pub fn first_subchunk_index(&self) -> Option<D> {
        let can_subdivide = self.visible_shape.foldv(false, |acc, len| acc & (len >= 2));
        if can_subdivide {
            Some(D::zeros(self.ndim()))
        } else {
            None
        }
    }

    pub fn next_subchunk_index(&self, mut index: D) -> Option<D> {
        let ndim = self.ndim();
        assert_eq!(index.ndim(), ndim);
        for ax in (0..ndim).rev() {
            if self.visible_shape[ax] >= 2 {
                index[ax] += 1;
                if index[ax] < 2 {
                    return Some(index);
                } else {
                    index[ax] = 0;
                }
            } else {
                assert_eq!(index[ax], 0);
            }
        }
        None
    }

    pub fn get_subchunk(&self, index: D) -> ChunkInfo<ViewRepr<&'_ A>, D> {
        // TODO: this is unnecessarily expensive.
        let mut chunk = ChunkInfo {
            all_trees: self.all_trees.view(),

            first_visible_index: self.first_visible_index.clone(),
            visible_shape: self.visible_shape.clone(),

            base_lower_borders: self.base_lower_borders.clone(),
            base_upper_borders: self.base_upper_borders.clone(),
            base_steps: self.base_steps.clone(),
            remove_borders_steps: self.remove_borders_steps.clone(),

            base_invert: self.base_invert.clone(),
            allow_invert: self.allow_invert.clone(),

            base_axis_order: self.base_axis_order.clone(),
            sort_axes: self.sort_axes.clone(),
        };
        chunk.narrow_to_subchunk(index);
        chunk
    }

    pub fn narrow_to_subchunk(&mut self, index: D) {
        let ndim = self.ndim();
        assert_eq!(index.ndim(), ndim);
        self.first_visible_index
            .indexed_map_inplace(|axis, vis_ind| {
                let ax = axis.index();
                *vis_ind += match index[ax] {
                    0 => 0,
                    1 => self.visible_shape[ax] / 2,
                    _ => panic!("Index out of bounds for axis {}", ax),
                };
            });
        self.visible_shape
            .map_inplace(|len| *len = *len / 2 + (*len % 2 != 0) as usize);
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
    chunk: ChunkInfo<OwnedRepr<A>, D>,
    current_subchunk: Option<D>,
    // min_shape: D,
    /// Action to perform on next `simplify` call.
    next_action: Option<ShrinkAction<D>>,
    /// Action performed on most recent `simplify` call.
    last_action: Option<ShrinkAction<D>>,
}

impl<A: ValueTree, D: Dimension> ArrayValueTree<A, D> {
    /// Returns the number of dimensions of arrays produced by
    /// `self.current()`.
    pub fn ndim(&self) -> usize {
        self.chunk.ndim()
    }

    /// Returns the shape of `self.current()` (without actually calling `self.current()`).
    pub fn shape(&self) -> D {
        self.chunk.shape()
    }
}

impl<A: ValueTree, D: Dimension> ValueTree for ArrayValueTree<A, D> {
    type Value = Array<A::Value, D>;

    fn current(&self) -> Array<A::Value, D> {
        if let Some(subchunk_index) = self.current_subchunk {
            self.chunk
                .get_subchunk(subchunk_index)
                .map_current(|elem| elem.current())
        } else {
            self.chunk.map_current(|elem| elem.current())
        }
    }

    fn simplify(&mut self) -> bool {
        let ndim = self.ndim();
        if let Some(action) = self.next_action.take() {
            self.next_action = action.next(&self.shape());
            match &action {
                &ShrinkAction::RemoveBordersStep(axis) => {
                    self.chunk.remove_borders_step(axis);
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
                    self.chunk.forbid_invert(axis);
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
                    self.chunk.sort_axes();
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.chunk.first_subchunk_index(),
                    ));
                }
                &ShrinkAction::SelectSubchunk(Some(ref subchunk_index)) => {
                    if let Some(subchunk_index) = self.current_subchunk {
                        self.chunk.narrow_to_subchunk(subchunk_index);
                    }
                    self.current_subchunk = Some(subchunk_index.clone());
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.chunk
                            .get_subchunk(subchunk_index.clone())
                            .first_subchunk_index(),
                    ));
                }
                &ShrinkAction::SelectSubchunk(None) => {
                    self.next_action = Some(ShrinkAction::ShrinkElement(unimplemented!()));
                }
                &ShrinkAction::ShrinkElement(Some(ref index)) => {
                    self.chunk.view_all_current_mut()[index.clone()].simplify(); // TODO: subchunk?
                                                                                 // FIXME: `next_for` is an undocumented method from `ndarray`.
                    self.next_action =
                        Some(ShrinkAction::ShrinkElement(shape.next_for(index.clone())));
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
                    self.chunk.restore_borders_step(axis);
                }
                ShrinkAction::ForbidInvertAxis(axis) => {
                    self.chunk.allow_invert(axis);
                }
                ShrinkAction::SortAxes => {
                    self.chunk.unsort_axes();
                }
                ShrinkAction::SelectSubchunk(Some(subchunk_index)) => {
                    self.current_subchunk = None;
                    self.next_action = Some(ShrinkAction::SelectSubchunk(
                        self.chunk.next_subchunk_index(subchunk_index),
                    ));
                }
                ShrinkAction::SelectSubchunk(None) => {}
                ShrinkAction::ShrinkElement(Some(index)) => {
                    if self.all_current_trees_mut()[index.clone()].complicate() {
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
