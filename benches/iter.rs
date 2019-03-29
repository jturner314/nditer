use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use ndarray::prelude::*;
use ndarray::Zip;
use nditer::{ArrayBaseExt, IntoNdProducer, NdProducer};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

fn rand_shape_ix3(approx_size: usize) -> [usize; 3] {
    let unnorm_lens: Vec<_> = thread_rng()
        .sample_iter(&Uniform::new(10., 100.))
        .take(3)
        .collect();
    let norm = (unnorm_lens.iter().product::<f64>() / approx_size as f64).powf(1. / 3.);
    let mut shape = [0; 3];
    for (u, s) in unnorm_lens.into_iter().zip(&mut shape) {
        *s = (u / norm).round() as usize
    }
    shape
}

fn equal_lengths_row_major_ix2(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20, 40, 80, 160];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<i32>::zeros([axis_len; 2]);
            bencher.iter(|| {
                Zip::from(&arr).apply(|x| {
                    black_box(x);
                })
            })
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<i32>::zeros([axis_len; 2]);
        bencher.iter(|| {
            arr.producer().into_iter_any_ord().for_each(|x| {
                black_box(x);
            });
        })
    });
    c.bench("equal_lengths_row_major_ix2", benchmark);
}

fn equal_lengths_row_major_ix4(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array4::<i32>::zeros([axis_len; 4]);
            bencher.iter(|| {
                Zip::from(&arr).apply(|x| {
                    black_box(x);
                })
            })
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array4::<i32>::zeros([axis_len; 4]);
        bencher.iter(|| {
            arr.producer().into_iter_any_ord().for_each(|x| {
                black_box(x);
            });
        })
    });
    c.bench("equal_lengths_row_major_ix4", benchmark);
}

fn equal_lengths_col_major_ix2(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20, 40, 80, 160];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<i32>::zeros([axis_len; 2].f());
            bencher.iter(|| {
                Zip::from(&arr).apply(|x| {
                    black_box(x);
                })
            })
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<i32>::zeros([axis_len; 2].f());
        bencher.iter(|| {
            arr.producer().into_iter_any_ord().for_each(|x| {
                black_box(x);
            });
        })
    });
    c.bench("equal_lengths_col_major_ix2", benchmark);
}

fn equal_lengths_col_major_ix4(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array4::<i32>::zeros([axis_len; 4].f());
            bencher.iter(|| {
                Zip::from(&arr).apply(|x| {
                    black_box(x);
                })
            })
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array4::<i32>::zeros([axis_len; 4].f());
        bencher.iter(|| {
            arr.producer().into_iter_any_ord().for_each(|x| {
                black_box(x);
            });
        })
    });
    c.bench("equal_lengths_col_major_ix4", benchmark);
}

fn equal_lengths_discontiguous_ix2(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20, 40, 80, 160];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<i32>::zeros([axis_len, axis_len * 2]);
            bencher.iter_with_setup(
                || arr.slice(s![.., ..;2]),
                |view| {
                    Zip::from(view).apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<i32>::zeros([axis_len, axis_len * 2]);
        bencher.iter_with_setup(
            || arr.slice(s![.., ..;2]),
            |view| {
                view.into_producer().into_iter_any_ord().for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("equal_lengths_discontiguous_ix2", benchmark);
}

fn equal_lengths_discontiguous0_ix3(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20, 40, 80];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array3::<i32>::zeros([axis_len * 2, axis_len, axis_len]);
            bencher.iter_with_setup(
                || arr.slice(s![..;2, .., ..]),
                |view| {
                    Zip::from(view).apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array3::<i32>::zeros([axis_len * 2, axis_len, axis_len]);
        bencher.iter_with_setup(
            || arr.slice(s![..;2, .., ..]),
            |view| {
                view.into_producer().into_iter_any_ord().for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("equal_lengths_discontiguous0_ix3", benchmark);
}

fn equal_lengths_discontiguous1_ix3(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20, 40, 80];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array3::<i32>::zeros([axis_len, axis_len * 2, axis_len]);
            bencher.iter_with_setup(
                || arr.slice(s![.., ..;2, ..]),
                |view| {
                    Zip::from(view).apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array3::<i32>::zeros([axis_len, axis_len * 2, axis_len]);
        bencher.iter_with_setup(
            || arr.slice(s![.., ..;2, ..]),
            |view| {
                view.into_producer().into_iter_any_ord().for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("equal_lengths_discontiguous1_ix3", benchmark);
}

fn equal_lengths_permuted_ix4(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array4::<i32>::zeros([axis_len; 4]);
            bencher.iter_with_setup(
                || {
                    let mut order = [0, 1, 2, 3];
                    thread_rng().shuffle(&mut order);
                    arr.view().permuted_axes(order)
                },
                |view| {
                    Zip::from(view).apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array4::<i32>::zeros([axis_len; 4]);
        bencher.iter_with_setup(
            || {
                let mut order = [0, 1, 2, 3];
                thread_rng().shuffle(&mut order);
                arr.view().permuted_axes(order)
            },
            |view| {
                view.into_producer().into_iter_any_ord().for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("equal_lengths_permuted_ix4", benchmark);
}

fn equal_lengths_row_major_ixdyn(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = ArrayD::<i32>::zeros(vec![axis_len; 5]);
            bencher.iter(|| {
                Zip::from(&arr).apply(|x| {
                    black_box(x);
                })
            })
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = ArrayD::<i32>::zeros(vec![axis_len; 5]);
        bencher.iter(|| {
            arr.producer().into_iter_any_ord().for_each(|x| {
                black_box(x);
            });
        })
    });
    c.bench("equal_lengths_row_major_ixdyn", benchmark);
}

fn equal_lengths_discontiguous_ixdyn(_c: &mut Criterion) {
    unimplemented!()
}

fn equal_lengths_permuted_ixdyn(c: &mut Criterion) {
    let axis_lens = vec![1, 3, 5, 10, 20];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = ArrayD::<i32>::zeros(vec![axis_len; 5]);
            bencher.iter_with_setup(
                || {
                    let mut order = vec![0, 1, 2, 3, 4];
                    thread_rng().shuffle(&mut order);
                    arr.view().permuted_axes(order)
                },
                |view| {
                    Zip::from(view).apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = ArrayD::<i32>::zeros(vec![axis_len; 5]);
        bencher.iter_with_setup(
            || {
                let mut order = vec![0, 1, 2, 3, 4];
                thread_rng().shuffle(&mut order);
                arr.view().permuted_axes(order)
            },
            |view| {
                view.into_producer().into_iter_any_ord().for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("equal_lengths_permuted_ixdyn", benchmark);
}

fn equal_lengths_indexed_ixdyn(_c: &mut Criterion) {
    unimplemented!()
}

fn unequal_lengths_discontiguous1_ix3(c: &mut Criterion) {
    let sizes = vec![125, 1000, 2000, 4000, 8000];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &size| {
            bencher.iter_with_setup(
                || {
                    let shape = rand_shape_ix3(size);
                    let larger = [shape[0], shape[1] * 2, shape[2]];
                    Array3::<i32>::zeros(larger)
                },
                |arr| {
                    let zip = Zip::from(arr.slice(s![.., ..;2, ..]));
                    zip.apply(|x| {
                        black_box(x);
                    })
                },
            )
        },
        sizes,
    )
    .with_function("nditer", |bencher, &size| {
        bencher.iter_with_setup(
            || {
                let shape = rand_shape_ix3(size);
                let larger = [shape[0], shape[1] * 2, shape[2]];
                Array3::<i32>::zeros(larger)
            },
            |arr| {
                let iter = arr
                    .slice(s![.., ..;2, ..])
                    .into_producer()
                    .into_iter_any_ord();
                iter.for_each(|x| {
                    black_box(x);
                });
            },
        )
    });
    c.bench("unequal_lengths_discontiguous1_ix3", benchmark);
}

fn unequal_lengths_permuted_ix4(_c: &mut Criterion) {
    unimplemented!()
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = equal_lengths_row_major_ix2, equal_lengths_row_major_ix4, equal_lengths_col_major_ix2, equal_lengths_col_major_ix4, equal_lengths_discontiguous_ix2, equal_lengths_discontiguous0_ix3, equal_lengths_discontiguous1_ix3, equal_lengths_permuted_ix4, equal_lengths_row_major_ixdyn, equal_lengths_permuted_ixdyn, unequal_lengths_discontiguous1_ix3
}
criterion_main!(benches);
