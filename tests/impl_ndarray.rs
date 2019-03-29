use ndarray::prelude::*;
use nditer::ArrayBaseExt;

#[test]
fn test_accumulate_axis_inplace_noop() {
    let mut a = Array2::<u8>::zeros((0, 3));
    a.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a, Array2::zeros((0, 3)));

    let mut a = Array2::<u8>::zeros((3, 1));
    a.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    assert_eq!(a, Array2::zeros((3, 1)));
}

#[test]
fn test_accumulate_axis_inplace_nonstandard_layout() {
    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10,11,12]]);

    let mut a_t = a.clone().reversed_axes();
    a_t.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a_t, aview2(&[[1, 4, 7, 10],
                             [3, 9, 15, 21],
                             [6, 15, 24, 33]]));

    let mut a0 = a.clone();
    a0.invert_axis(Axis(0));
    a0.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    assert_eq!(a0, aview2(&[[10, 11, 12],
                            [17, 19, 21],
                            [21, 24, 27],
                            [22, 26, 30]]));

    let mut a1 = a.clone();
    a1.invert_axis(Axis(1));
    a1.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
    assert_eq!(a1, aview2(&[[3, 5, 6],
                            [6, 11, 15],
                            [9, 17, 24],
                            [12, 23, 33]]));
}
