use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use nditer::{ArrayBaseExt, NdProducer};
use quickcheck_macros::quickcheck;
use rand::distributions::Uniform;

#[test]
fn fold() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    assert_eq!(arr.producer().fold(0, |acc, &elem| acc + elem), arr.sum(),);
}

#[test]
fn pairwise_sum() {
    let arr = Array3::<i32>::random((31, 5, 515), Uniform::new(-10, 10));
    let view = arr.slice(s![..;2, .., ..]);
    assert_eq!(view.sum(), view.producer().cloned().pairwise_sum());
}

#[test]
fn collect() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    assert_eq!(arr.producer().cloned().collect_array(), arr);
}

#[quickcheck]
fn collect_permuted_inverted(
    shape_order: (usize, usize, usize, usize, usize, usize, usize, usize),
    layout_order: (usize, usize, usize, usize, usize, usize, usize, usize),
    inverted: (bool, bool, bool, bool, bool, bool, bool, bool),
) -> bool {
    let shape_order = [
        shape_order.0,
        shape_order.1,
        shape_order.2,
        shape_order.3,
        shape_order.4,
        shape_order.5,
        shape_order.6,
        shape_order.7,
    ];
    let layout_order = [
        layout_order.0,
        layout_order.1,
        layout_order.2,
        layout_order.3,
        layout_order.4,
        layout_order.5,
        layout_order.6,
        layout_order.7,
    ];
    let inverted = [
        inverted.0, inverted.1, inverted.2, inverted.3, inverted.4, inverted.5, inverted.6,
        inverted.7,
    ];

    // Determine shape of array before permutation.
    let mut shape = [1, 2, 3, 4, 5, 6, 7, 8];
    shape.sort_by_key(|&len| shape_order[len - 1]);

    // Generate initial array.
    let arr = Array::from_shape_fn(&shape[..], |idx| {
        let mut acc = 0;
        let mut stride = 1;
        for (&i, &len) in idx.slice().iter().zip(&shape) {
            acc += i * stride;
            stride *= len;
        }
        acc
    });

    // Apply axes permutation.
    let mut axes = [0, 1, 2, 3, 4, 5, 6, 7];
    axes.sort_by_key(|&ax| layout_order[ax]);
    let mut arr = arr.permuted_axes(&axes[..]);

    // Apply axes inversions.
    for (ax, &inv) in inverted.iter().enumerate() {
        if inv {
            arr.invert_axis(Axis(ax));
        }
    }

    arr.producer().cloned().collect_array() == arr
}

#[quickcheck]
fn pairwise_sum_permuted_inverted(
    shape_order: (usize, usize, usize, usize),
    layout_order: (usize, usize, usize, usize),
    inverted: (bool, bool, bool, bool),
) -> bool {
    let shape_order = [shape_order.0, shape_order.1, shape_order.2, shape_order.3];
    let layout_order = [
        layout_order.0,
        layout_order.1,
        layout_order.2,
        layout_order.3,
    ];
    let inverted = [inverted.0, inverted.1, inverted.2, inverted.3];

    // Determine shape of array before permutation.
    let shape = {
        let default_shape = [21, 3, 12, 33];
        let mut indices = [0, 1, 2, 3];
        indices.sort_by_key(|&i| shape_order[i]);
        [
            default_shape[indices[0]],
            default_shape[indices[1]],
            default_shape[indices[2]],
            default_shape[indices[3]],
        ]
    };

    // Generate initial array.
    let arr = Array::from_shape_fn(&shape[..], |idx| {
        let mut acc = 0;
        let mut stride = 1;
        for (&i, &len) in idx.slice().iter().zip(&shape) {
            acc += i * stride;
            stride *= len;
        }
        acc
    });

    // Apply axes permutation.
    let mut axes = [0, 1, 2, 3];
    axes.sort_by_key(|&ax| layout_order[ax]);
    let mut arr = arr.permuted_axes(&axes[..]);

    // Apply axes inversions.
    for (ax, &inv) in inverted.iter().enumerate() {
        if inv {
            arr.invert_axis(Axis(ax));
        }
    }

    arr.producer().cloned().pairwise_sum() == arr.sum()
}
