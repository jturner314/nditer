use ndarray::prelude::*;
use nditer::{ArrayBaseExt, IntoNdProducer, NdProducer};

#[test]
fn iter_fold_ix0() {
    let arr = arr0(5);
    assert_eq!(arr.producer().into_iter().fold(1, |acc, x| acc + x), 6);
}

#[test]
fn iter_map() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let iter = arr.producer().into_iter();
    assert_eq!(
        iter.map(|&val| val).collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6],
    );
}

#[test]
fn iter_map_discontiguous() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let iter = arr.slice(s![.., ..;2]).into_producer().into_iter();
    assert_eq!(iter.map(|&val| val).collect::<Vec<_>>(), vec![1, 3, 4, 6],);
}

#[test]
fn iter_for_each() {
    let arr = array![[1, 2, 3], [4, 5, 6]];
    let iter = arr.producer().into_iter();
    let mut out = Vec::new();
    iter.for_each(|&val| out.push(val));
    assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn iter_discontiguous() {
    let a = Array3::from_shape_fn((5, 6, 3), |i| i);
    let v = a.slice(s![.., ..;2, ..]);
    let mut coll = Vec::new();
    v.into_producer()
        .into_iter_any_ord()
        .for_each(|i| coll.push(i));
    assert_eq!(v.iter().collect::<Vec<_>>(), coll);
}

#[test]
fn iter_fortran() {
    let a = Array3::from_shape_fn((5, 6, 3).f(), |i| i);
    let mut coll = Vec::new();
    a.producer().into_iter_any_ord().for_each(|i| coll.push(i));
    assert_eq!(a.t().iter().collect::<Vec<_>>(), coll);
}
