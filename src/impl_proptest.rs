use crate::{AxesFor, AxesMask, DimensionExt};
use itertools::{izip, Itertools};
use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, Dimension, IxDyn, ShapeBuilder, Slice,
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

/// A shrink action for an `ArrayValueTree`.
#[derive(Clone, Debug)]
enum ShrinkAction<D: Dimension> {
    UninvertAxis(Axis),
    RemoveStep(Axis),
    RemoveBorders(Axis),
    PartiallySortAxes, // FIXME
    // BisectProportional,
    // BisectAxis(Axis),
    /// Shrink the element at the given index in the current array.
    ///
    /// Note that the index is for `.current()`, not `all_elements`.
    ShrinkElement(D),
}

impl<D: Dimension> ShrinkAction<D> {
    /// Returns the next shrink action for an array of the given shape.
    pub fn next(&self, shape: &D) -> Option<ShrinkAction<D>> {
        let ndim = shape.ndim();
        match self {
            &ShrinkAction::UninvertAxis(axis) => {
                let next = Axis(axis.index() + 1);
                if next.index() < ndim {
                    Some(ShrinkAction::UninvertAxis(next))
                } else {
                    Some(ShrinkAction::RemoveStep(Axis(0)))
                }
            }
            &ShrinkAction::RemoveStep(axis) => {
                let next = Axis(axis.index() + 1);
                if next.index() < ndim {
                    Some(ShrinkAction::RemoveStep(next))
                } else {
                    Some(ShrinkAction::RemoveBorders(Axis(0)))
                }
            }
            &ShrinkAction::RemoveBorders(axis) => {
                let next = Axis(axis.index() + 1);
                if next.index() < ndim {
                    Some(ShrinkAction::RemoveBorders(next))
                } else {
                    Some(ShrinkAction::PartiallySortAxes)
                }
            }
            &ShrinkAction::PartiallySortAxes => Some(ShrinkAction::PartiallySortAxes),
            &ShrinkAction::ShrinkElement(ref index) => {
                // FIXME: `next_for` is an undocumented method from `ndarray`.
                Some(ShrinkAction::ShrinkElement(shape.next_for(index.clone())?))
            }
        }
    }

    // /// Returns the previous shrink action for an array of the given shape.
    // pub fn prev(action: &Option<ShrinkAction<D>>, shape: &D) -> Option<ShrinkAction<D>> {
    //     let ndim = shape.ndim();
    //     match action {
    //         &Some(ShrinkAction::UninvertAxis(axis)) => {
    //             let next = Axis(axis.index() + 1);
    //             if next.index() < ndim {
    //                 Some(ShrinkAction::UninvertAxis(next))
    //             } else {
    //                 Some(ShrinkAction::RemoveStep(Axis(0)))
    //             }
    //         }
    //         &Some(ShrinkAction::RemoveStep(axis)) => {
    //             let next = Axis(axis.index() + 1);
    //             if next.index() < ndim {
    //                 Some(ShrinkAction::RemoveStep(next))
    //             } else {
    //                 Some(ShrinkAction::RemoveBorders(Axis(0)))
    //             }
    //         }
    //         &Some(ShrinkAction::RemoveBorders(axis)) => {
    //             let next = Axis(axis.index() + 1);
    //             if next.index() < ndim {
    //                 Some(ShrinkAction::RemoveBorders(next))
    //             } else {
    //                 Some(ShrinkAction::PartiallySortAxes)
    //             }
    //         }
    //         &Some(ShrinkAction::PartiallySortAxes) => Some(ShrinkAction::PartiallySortAxes),
    //         &Some(ShrinkAction::ShrinkElement(ref index)) => {
    //             // FIXME: `next_for` is an undocumented method from `ndarray`.
    //             Some(ShrinkAction::ShrinkElement(shape.next_for(index.clone())?))
    //         }
    //         &None => {
    //             // The previous action is to shrink the last element of the
    //             // array, if one exists.
    //             if shape.size() == 0 {
    //                 None
    //             } else {
    //                 Some(ShrinkAction::ShrinkElement(shape.mapv(|len| len - 1)))
    //             }
    //         }
    //     }
    // }
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
    /// the steps and axis inversions such that the resulting array matches the
    /// layout specified by `self` (ignoring `iter_order`).
    ///
    /// If you want the resulting array to have an `iter_order` matching
    /// `self`, `arr` should have that `iter_order`.
    pub fn apply_borders_steps_invert<S: Data>(&self, arr: &mut ArrayBase<S, D>) {
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            if self.inverted.read(axis) {
                arr.invert_axis(axis);
            }
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

/// Computes an array of values by applying mapping.
///
/// The resulting array will have the layout described by `desired_layout` and
/// contain the values computed applying `mapping` to
/// `all_elems.apply_borders_steps_invert(&mut all_elems)`.
fn compute_values_layout<A, B, D, F>(
    all_elems: ArrayView<'_, A, D>,
    base_layout: &LayoutTree<D>,
    desired_layout: &LayoutTree<D>,
    mapping: F,
) -> Array<B, D>
where
    D: Dimension,
    F: FnMut(&A) -> B,
{
    let ndim = all_elems.ndim();
    debug_assert_eq!(ndim, base_layout.ndim());
    debug_assert_eq!(ndim, desired_layout.ndim());

    // Compute `elems_with_borders` such that
    // `base_layout.apply_borders_steps_invert(&mut all_elems)` would contain
    // the same elements as `desired_layout.apply_borders_steps_invert(&mut
    // elems_with_borders)`.
    let mut elems_with_borders = all_elems;
    for ax in 0..ndim {
        let axis = Axis(ax);
        let start = (base_layout.lower_borders[ax] - desired_layout.lower_borders[ax]) as isize;
        let end = -((base_layout.upper_borders[ax] - desired_layout.upper_borders[ax]) as isize);
        let step = (base_layout.steps[ax] / desired_layout.steps[ax]) as isize;
        elems_with_borders.slice_axis_inplace(axis, Slice::new(start, Some(end), step));
        if desired_layout.inverted.read(axis) != base_layout.inverted.read(axis) {
            elems_with_borders.invert_axis(axis);
        }
    }

    // Get an array of the current values with the correct axis order.
    let values_with_borders = {
        let shape_with_borders = elems_with_borders.raw_dim();
        let elems_with_borders =
            elems_with_borders.permuted_axes(desired_layout.iter_order.clone().into_inner());
        let values_flat: Vec<B> = elems_with_borders.iter().map(mapping).collect();
        let mut strides_with_borders = D::zeros(ndim);
        if !elems_with_borders.is_empty() {
            let mut cum_prod: isize = 1;
            for &ax in desired_layout.iter_order.slice().iter().rev() {
                let len = shape_with_borders[ax];
                strides_with_borders[ax] = cum_prod as usize;
                cum_prod *= len as isize;
            }
        }
        Array::from_shape_vec(
            shape_with_borders.strides(strides_with_borders),
            values_flat,
        )
        .unwrap()
    };

    // Apply the axis inversions and slicing.
    let mut values = values_with_borders;
    desired_layout.apply_borders_steps_invert(&mut values);
    values
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

/// `ValueTree` corresponding to `ArrayStrategy`.
#[derive(Clone, Debug)]
pub struct ArrayValueTree<A, D: Dimension> {
    /// All elements that may be used to produce the values in the underlying
    /// representation of `.current()`.
    all_elements: Array<A, D>,
    /// The layout of `.current()` from when the `ArrayValueTree` was initially
    /// constructed (before any simplification).
    base_layout: LayoutTree<D>,
    /// The layout of `.current()`.
    current_layout: LayoutTree<D>,
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
        let ndim = self.all_elements.ndim();
        debug_assert_eq!(ndim, self.base_layout.ndim());
        debug_assert_eq!(ndim, self.current_layout.ndim());
        // TODO: more debug assert
        ndim
    }

    /// Returns the shape of `self.current()` (without actually calling `self.current()`).
    pub fn shape(&self) -> D {
        // TODO: Update this after adding support for shrinking the size of the array.
        self.base_layout
            .shape_after_apply(self.all_elements.raw_dim())
    }

    /// Returns a mutable array view of value trees corresponding to
    /// `.current()`.
    pub fn sliced_elements(&mut self) -> ArrayViewMut<'_, A, D> {
        let mut v = self.all_elements.view_mut();
        self.current_layout.apply_borders_steps_invert(&mut v);
        v
    }
}

impl<A: ValueTree, D: Dimension> ValueTree for ArrayValueTree<A, D> {
    type Value = Array<A::Value, D>;

    fn current(&self) -> Array<A::Value, D> {
        compute_values_layout(
            self.all_elements.view(),
            &self.base_layout,
            &self.current_layout,
            |elem| elem.current(),
        )
    }

    fn simplify(&mut self) -> bool {
        if let Some(action) = self.next_action.take() {
            self.next_action = action.next(&self.shape());
            match &action {
                &ShrinkAction::UninvertAxis(axis) => {
                    self.current_layout.inverted.write(axis, false);
                }
                &ShrinkAction::RemoveStep(axis) => {
                    self.current_layout.steps[axis.index()] = 1;
                }
                &ShrinkAction::RemoveBorders(axis) => {
                    self.current_layout.lower_borders[axis.index()] = 0;
                    self.current_layout.upper_borders[axis.index()] = 0;
                }
                ShrinkAction::PartiallySortAxes => unimplemented!(),
                ShrinkAction::ShrinkElement(index) => {
                    self.sliced_elements()[index.clone()].simplify();
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
                ShrinkAction::UninvertAxis(axis) => {
                    self.current_layout
                        .inverted
                        .write(axis, self.base_layout.inverted.read(axis));
                }
                ShrinkAction::RemoveStep(axis) => {
                    self.current_layout.steps[axis.index()] = self.base_layout.steps[axis.index()];
                }
                ShrinkAction::RemoveBorders(axis) => {
                    self.current_layout.lower_borders[axis.index()] =
                        self.base_layout.lower_borders[axis.index()];
                    self.current_layout.upper_borders[axis.index()] =
                        self.base_layout.upper_borders[axis.index()];
                }
                ShrinkAction::PartiallySortAxes => unimplemented!(),
                ShrinkAction::ShrinkElement(index) => {
                    if self.sliced_elements()[index.clone()].complicate() {
                        // `.complicate()` only attempts to *partially* undo the last
                        // simplification, so it may be possible to complicate this element further.
                        self.last_action = Some(ShrinkAction::ShrinkElement(index));
                    }
                }
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
// let mut elements_with_borders = all_elements;
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
