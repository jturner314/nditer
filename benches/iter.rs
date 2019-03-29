use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use ndarray::prelude::*;
use ndarray::Zip;
use nditer::{ArrayBaseExt, BroadcastProducer, IntoNdProducer, NdBroadcast, NdProducer};
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

fn broadcast(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80];
    let benchmark = ParameterizedBenchmark::new(
        "broadcast_inplace",
        |bencher, &axis_len| {
            let arr = Array2::<i32>::zeros([axis_len; 2]);
            bencher.iter(|| {
                BroadcastProducer::new(
                    arr.producer(),
                    Ix2(2, 0),
                    Ix4(axis_len, 80, axis_len, 80),
                )
                .unwrap()
                .for_each(|x| {
                    black_box(x);
                });
            })
        },
        axis_lens,
    )
    .with_function("broadcast_producer", |bencher, &axis_len| {
        let arr = Array2::<i32>::zeros([axis_len; 2]);
        bencher.iter(|| {
            arr.producer()
                .broadcast((2, 0), (axis_len, 80, axis_len, 80))
                .unwrap()
                .for_each(|x| {
                    black_box(x);
                });
        })
    });
    c.bench("broadcast", benchmark);
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

// fn fibonaccis(c: &mut Criterion) {
//     let fib_slow = Fun::new("Recursive", |b, i| b.iter(|| fibonacci_slow(*i)));
//     let fib_fast = Fun::new("Iterative", |b, i| b.iter(|| fibonacci_fast(*i)));
//     let functions = vec![fib_slow, fib_fast];
//     c.bench_functions("Fibonacci", functions, 20);
// }

// fn format(c: &mut Criterion) {
//     let parameters = vec![5, 10];
//     let mut benchmark =
//         ParameterizedBenchmark::new("print", |b, i| b.iter(|| print!("{}", i)), parameters)
//             .with_function("format", |b, i| b.iter(|| format!("{}", i)));
//     c.bench("test_bench_param", benchmark);
// }

fn cursor_vs_pointer(c: &mut Criterion) {
    let lens = vec![1, 3, 5, 10, 20, 40, 80];
    let benchmark = ParameterizedBenchmark::new(
        "cursor",
        |bencher, &len| {
            let v: Vec<i32> = vec![0; len];
            let stride = black_box(1);
            bencher.iter(|| {
                let mut ptr = &v[0] as *const i32;
                unsafe {
                    for _ in 0..len {
                        black_box(&*ptr);
                        ptr = ptr.offset(stride);
                    }
                }
            })
        },
        lens,
    )
    .with_function("pointer", |bencher, &len| {
        let v: Vec<i32> = vec![0; len];
        let stride = black_box(1);
        bencher.iter(|| {
            let ptr = &v[0] as *const i32;
            for i in 0..len {
                unsafe {
                    let ptr_i = ptr.offset(i as isize * stride);
                    black_box(&*ptr_i);
                }
            }
        })
    });
    c.bench("cursor_vs_pointer", benchmark);
}

// criterion_group!(benches, cursor_vs_pointer);
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = broadcast, equal_lengths_row_major_ix2, equal_lengths_row_major_ix4, equal_lengths_col_major_ix2, equal_lengths_col_major_ix4, equal_lengths_discontiguous_ix2, equal_lengths_discontiguous0_ix3, equal_lengths_discontiguous1_ix3, equal_lengths_permuted_ix4, equal_lengths_row_major_ixdyn, equal_lengths_permuted_ixdyn, unequal_lengths_discontiguous1_ix3
}
criterion_main!(benches);
