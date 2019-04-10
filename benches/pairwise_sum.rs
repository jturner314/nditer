use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use nditer::{ArrayBaseExt, NdProducer};
use rand::distributions::Uniform;

fn pairwise_sum_equal_lengths_ix2(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 320, 640, 1280];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<f32>::random([axis_len; 2], Uniform::new(-10., 10.));
            bencher.iter(|| black_box(arr.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<f32>::random([axis_len; 2], Uniform::new(-10., 10.));
        bencher.iter(|| black_box(arr.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_ix2", benchmark);
}

fn pairwise_sum_equal_lengths_ix2_f(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 320, 640, 1280];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<f32>::random([axis_len; 2].f(), Uniform::new(-10., 10.));
            bencher.iter(|| black_box(arr.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<f32>::random([axis_len; 2].f(), Uniform::new(-10., 10.));
        bencher.iter(|| black_box(arr.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_ix2_f", benchmark);
}

fn pairwise_sum_equal_lengths_discontiguous1_ix2(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 320, 640, 1280];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array2::<f32>::random([axis_len, axis_len * 2], Uniform::new(-10., 10.));
            let view = arr.slice(s![.., ..;2]);
            bencher.iter(|| black_box(view.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array2::<f32>::random([axis_len, axis_len * 2], Uniform::new(-10., 10.));
        let view = arr.slice(s![.., ..;2]);
        bencher.iter(|| black_box(view.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_discontiguous1_ix2", benchmark);
}

fn pairwise_sum_equal_lengths_discontiguous0_ix3(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 320];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr =
                Array3::<f32>::random([axis_len * 2, axis_len, axis_len], Uniform::new(-10., 10.));
            let view = arr.slice(s![..;2, .., ..]);
            bencher.iter(|| black_box(view.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr =
            Array3::<f32>::random([axis_len * 2, axis_len, axis_len], Uniform::new(-10., 10.));
        let view = arr.slice(s![..;2, .., ..]);
        bencher.iter(|| black_box(view.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_discontiguous0_ix3", benchmark);
}

fn pairwise_sum_equal_lengths_discontiguous1_ix3(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 320];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr =
                Array3::<f32>::random([axis_len, axis_len * 2, axis_len], Uniform::new(-10., 10.));
            let view = arr.slice(s![.., ..;2, ..]);
            bencher.iter(|| black_box(view.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr =
            Array3::<f32>::random([axis_len, axis_len * 2, axis_len], Uniform::new(-10., 10.));
        let view = arr.slice(s![.., ..;2, ..]);
        bencher.iter(|| black_box(view.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_discontiguous1_ix3", benchmark);
}

fn pairwise_sum_equal_lengths_discontiguous0_ix3_f(c: &mut Criterion) {
    let axis_lens = vec![1, 5, 20, 80, 160];
    let benchmark = ParameterizedBenchmark::new(
        "ndarray",
        |bencher, &axis_len| {
            let arr = Array3::<f32>::random(
                [axis_len * 2, axis_len, axis_len].f(),
                Uniform::new(-10., 10.),
            );
            let view = arr.slice(s![..;2, .., ..]);
            bencher.iter(|| black_box(view.sum()))
        },
        axis_lens,
    )
    .with_function("nditer", |bencher, &axis_len| {
        let arr = Array3::<f32>::random(
            [axis_len * 2, axis_len, axis_len].f(),
            Uniform::new(-10., 10.),
        );
        let view = arr.slice(s![..;2, .., ..]);
        bencher.iter(|| black_box(view.producer().cloned().pairwise_sum()))
    })
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("pairwise_sum_equal_lengths_discontiguous0_ix3_f", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = pairwise_sum_equal_lengths_ix2, pairwise_sum_equal_lengths_ix2_f, pairwise_sum_equal_lengths_discontiguous1_ix2, pairwise_sum_equal_lengths_discontiguous0_ix3, pairwise_sum_equal_lengths_discontiguous1_ix3, pairwise_sum_equal_lengths_discontiguous0_ix3_f
}
criterion_main!(benches);
