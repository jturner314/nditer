use crate::{axes_all, AxesFor, AxesMask, DimensionExt, IntoAxesFor};
use ndarray::{
    Array, ArrayBase, ArrayView, Axis, Data, Dimension, IxDyn, RawData, ShapeBuilder, Slice,
};

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

    let new_flat: Vec<B> = orig_permuted.into_iter().map(mapping).collect();
    let mut new_strides = D::zeros(ndim);
    if !new_flat.is_empty() {
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

#[derive(Clone, Debug)]
pub struct ChunkInfo<D: Dimension> {
    first_visible_index: D,
    visible_shape: D,
}

#[derive(Clone, Debug)]
pub struct BordersSteps<D: Dimension> {
    base_lower_borders: D,
    base_upper_borders: D,
    base_steps: D,
    remove_borders_steps: AxesMask<D, IxDyn>,
}

#[derive(Clone, Debug)]
pub struct MemoryOrder<D: Dimension> {
    base_invert: AxesMask<D, IxDyn>,
    allow_invert: AxesMask<D, IxDyn>,

    base_axis_order: AxesFor<D, D>,
    sort_axes: bool,
}

impl<D: Dimension> ChunkInfo<D> {
    /// Creates a `ChunkInfo` describing the first chunk in an array with the
    /// given visible shape, borders, and steps.
    ///
    /// **Panics** if the ndim is not consistent between the parameters.
    pub fn first_chunk(visible_shape: D, borders_steps: &BordersSteps<D>) -> ChunkInfo<D> {
        assert_eq!(visible_shape.ndim(), borders_steps.ndim());
        ChunkInfo {
            first_visible_index: borders_steps.lower_borders(),
            visible_shape,
        }
    }

    pub fn ndim(&self) -> usize {
        // TODO: debug assert?
        self.visible_shape.ndim()
    }

    pub fn shape(&self) -> D {
        self.visible_shape.clone()
    }

    pub fn shape_with_hidden(&self, borders_steps: &BordersSteps<D>) -> D {
        let ndim = self.ndim();
        let mut shape = D::zeros(ndim);
        for ax in 0..ndim {
            let axis = Axis(ax);
            if borders_steps.remove_borders_steps.read(axis) {
                shape[ax] = self.visible_shape[ax];
            } else {
                shape[ax] = borders_steps.base_lower_borders[ax]
                    + borders_steps.base_steps[ax] * self.visible_shape[ax]
                    + borders_steps.base_upper_borders[ax];
            };
        }
        shape
    }

    /// Returns a view of the visible portion of the current chunk.
    pub fn slice<S>(&self, all_trees: &mut ArrayBase<S, D>, borders_steps: &BordersSteps<D>)
    where
        S: RawData,
    {
        let trees = all_trees;
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            let start = self.first_visible_index[ax] as isize;
            let step = borders_steps.base_steps[ax] as isize;
            let end = start + step * self.visible_shape[ax] as isize;
            trees.slice_axis_inplace(axis, Slice::new(start, Some(end), step));
        }
    }

    /// Returns a view of the underlying representation of an owned copy of the
    /// current chunk.
    pub fn slice_with_hidden<S>(
        &self,
        all_trees: &mut ArrayBase<S, D>,
        borders_steps: &BordersSteps<D>,
    ) where
        S: RawData,
    {
        let trees = all_trees;
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            let (start, end, step) = if borders_steps.remove_borders_steps.read(axis) {
                let start = self.first_visible_index[ax];
                let step = borders_steps.base_steps[ax];
                let end = start + step * self.visible_shape[ax];
                (start as isize, end as isize, step as isize)
            } else {
                let start = self.first_visible_index[ax] - borders_steps.base_lower_borders[ax];
                let step = 1;
                let end =
                    start + step * self.visible_shape[ax] + borders_steps.base_upper_borders[ax];
                (start as isize, end as isize, step as isize)
            };
            trees.slice_axis_inplace(axis, Slice::new(start, Some(end), step));
        }
    }

    /// Applies the mapping to the underlying representation of the current
    /// chunk, returning an array sliced to show only the visible portion.
    pub fn map<S, F, B>(
        &self,
        all_trees: &ArrayBase<S, D>,
        borders_steps: &BordersSteps<D>,
        memory_order: &MemoryOrder<D>,
        mapping: F,
    ) -> Array<B, D>
    where
        S: Data,
        F: FnMut(&S::Elem) -> B,
    {
        // TODO: check consistentency ndim
        let mut with_hidden = self.map_with_hidden(all_trees, borders_steps, memory_order, mapping);
        for ax in 0..self.ndim() {
            let axis = Axis(ax);
            if !borders_steps.remove_borders_steps.read(axis) {
                with_hidden.slice_axis_inplace(
                    axis,
                    Slice {
                        start: borders_steps.base_lower_borders[ax] as isize,
                        end: Some(-(borders_steps.base_upper_borders[ax] as isize)),
                        step: borders_steps.base_steps[ax] as isize,
                    },
                );
            }
        }
        with_hidden
    }

    /// Applies the mapping to the underlying representation of the current
    /// chunk, returning the underlying representation.
    pub fn map_with_hidden<S, F, B>(
        &self,
        all_trees: &ArrayBase<S, D>,
        borders_steps: &BordersSteps<D>,
        memory_order: &MemoryOrder<D>,
        mapping: F,
    ) -> Array<B, D>
    where
        S: Data,
        F: FnMut(&S::Elem) -> B,
    {
        let mut trees = all_trees.view();
        self.slice_with_hidden(&mut trees, borders_steps);
        let current_invert = (&memory_order.allow_invert) & (&memory_order.base_invert);
        if memory_order.sort_axes {
            map_with_memory_order(
                trees,
                &current_invert,
                &axes_all().into_axes_for(self.ndim()),
                mapping,
            )
        } else {
            map_with_memory_order(
                trees,
                &current_invert,
                &memory_order.base_axis_order,
                mapping,
            )
        }
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

    pub fn get_subchunk(&self, index: D) -> ChunkInfo<D> {
        let mut subchunk = self.clone();
        subchunk.narrow_to_subchunk(index);
        subchunk
    }

    pub fn narrow_to_subchunk(&mut self, index: D) {
        let ndim = self.ndim();
        assert_eq!(index.ndim(), ndim);
        let ChunkInfo {
            first_visible_index,
            visible_shape,
        } = self;
        first_visible_index.indexed_map_inplace(|axis, vis_ind| {
            let ax = axis.index();
            *vis_ind += match index[ax] {
                0 => 0,
                1 => visible_shape[ax] / 2,
                _ => panic!("Index out of bounds for axis {}", ax),
            };
        });
        visible_shape.map_inplace(|len| *len = *len / 2 + (*len % 2 != 0) as usize);
    }
}

impl<D: Dimension> BordersSteps<D> {
    /// Creates a new `BordersSteps` instance for the given borders and steps.
    ///
    /// **Panics** if the ndim is not consistent between the parameters.
    pub fn new(&self, lower_borders: D, upper_borders: D, steps: D) -> BordersSteps<D> {
        let ndim = lower_borders.ndim();
        assert_eq!(upper_borders.ndim(), ndim);
        assert_eq!(steps.ndim(), ndim);
        BordersSteps {
            base_lower_borders: lower_borders,
            base_upper_borders: upper_borders,
            base_steps: steps,
            remove_borders_steps: AxesMask::all_false(ndim).into_dyn_num_true(),
        }
    }

    /// Returns the `ndim` this `BordersSteps` is for.
    pub fn ndim(&self) -> usize {
        self.base_lower_borders.ndim()
    }

    /// Returns the current values of the lower borders.
    pub fn lower_borders(&self) -> D {
        self.base_lower_borders.indexed_mapv(|axis, border| {
            if self.remove_borders_steps.read(axis) {
                0
            } else {
                border
            }
        })
    }

    /// Returns the current values of the upper borders.
    pub fn upper_borders(&self) -> D {
        self.base_upper_borders.indexed_mapv(|axis, border| {
            if self.remove_borders_steps.read(axis) {
                0
            } else {
                border
            }
        })
    }

    /// Returns the current values of the steps.
    pub fn steps(&self) -> D {
        self.base_steps.indexed_mapv(|axis, step| {
            if self.remove_borders_steps.read(axis) {
                0
            } else {
                step
            }
        })
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
}

impl<D: Dimension> MemoryOrder<D> {
    /// Creates a `MemoryOrder` instance with the given axis inversions and
    /// axis order.
    ///
    /// **Panics** if the `invert.for_ndim()`, `axis_order.num_axes()`, or
    /// `axes_order.for_ndim()` are inconsistent with each other.
    pub fn new(invert: AxesMask<D, IxDyn>, axis_order: AxesFor<D, D>) -> MemoryOrder<D> {
        let ndim = invert.for_ndim();
        assert_eq!(axis_order.num_axes(), ndim);
        assert_eq!(axis_order.for_ndim(), ndim);
        MemoryOrder {
            base_invert: invert,
            allow_invert: AxesMask::all_true(ndim).into_dyn_num_true(),
            base_axis_order: axis_order,
            sort_axes: false,
        }
    }

    /// Returns the number of dimensions this `MemoryOrder` is for.
    pub fn ndim(&self) -> usize {
        self.base_invert.for_ndim()
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
}
