use ndarray::prelude::*;
use nditer::{axes, into_repeat, ArrayBaseExt, IntoNdProducer, NdProducer};

#[test]
fn zip() {
    let a = array![[1, 2, 3], [4, 5, 6]];
    let b = array![[7, 8, 9], [10, 11, 12]];
    let collected = a
        .t()
        .into_producer()
        .zip(b.t())
        .map(|(&a, &b)| (a, b))
        .collect_array();
    assert_eq!(
        collected,
        array![[(1, 7), (2, 8), (3, 9)], [(4, 10), (5, 11), (6, 12)]].t(),
    );
}

#[test]
fn map() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let collected = arr.producer().map(|&val| val + 1).collect_array();
    assert_eq!(collected, array![[2, 3, 4], [5, 6, 7]]);
}

#[test]
fn inspect() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let flat = arr.iter().collect::<Vec<_>>();
    let mut inspected = Vec::with_capacity(arr.len());
    arr.producer()
        .inspect(|elem| inspected.push(elem.clone()))
        .into_iter()
        .for_each(|_| {});
    assert_eq!(flat, inspected);
}

#[test]
fn indexed() {
    let arr = Array4::<u8>::zeros((2, 3, 4, 5)).permuted_axes((0, 3, 1, 2));
    assert_eq!(
        Array4::from_shape_fn(arr.raw_dim(), |idx| idx),
        arr.producer().indexed().map(|(idx, _)| idx).collect_array()
    );
}

#[test]
fn fold_axes() {
    let shape = [3, 2, 4, 5, 2];
    let arr = Array5::from_shape_fn(shape, |(i, j, k, l, m)| {
        (i * 5 * 5 * 5 * 5 + j * 5 * 5 * 5 + k * 5 * 5 + l * 5 + m) as u64
    })
    .permuted_axes((1, 2, 0, 4, 3));
    assert_eq!(
        arr.sum_axis(Axis(2)).sum_axis(Axis(0)),
        arr.producer()
            .fold_axes(axes((2, 0)), into_repeat(0), |acc, &elem| acc + elem)
            .collect_array()
    );
}
