//! Quick dot-product benchmark: reviero vs pulp vs wide vs scalar.
//!
//! ```text
//! taskset -c 2 env RUSTFLAGS="-C target-cpu=native" cargo bench --bench dot
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use half::{bf16, f16};
use std::hint::black_box;
use std::ops::{Add, Mul};
use std::time::Duration;

use wide::{f32x8 as WideF32x8, f64x4 as WideF64x4};

const SIZES: &[usize] = &[513, 1027, 4099, 10_007, 1 << 20];

type DotFn<T> = fn(&[T], &[T]) -> T;

#[inline(never)]
fn scalar_dot<T>(x: &[T], y: &[T]) -> T
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    let n = x.len();
    let mut acc = [T::default(); 8];
    let mut i = 0;

    while i + 8 <= n {
        acc[0] = x[i] * y[i] + acc[0];
        acc[1] = x[i + 1] * y[i + 1] + acc[1];
        acc[2] = x[i + 2] * y[i + 2] + acc[2];
        acc[3] = x[i + 3] * y[i + 3] + acc[3];
        acc[4] = x[i + 4] * y[i + 4] + acc[4];
        acc[5] = x[i + 5] * y[i + 5] + acc[5];
        acc[6] = x[i + 6] * y[i + 6] + acc[6];
        acc[7] = x[i + 7] * y[i + 7] + acc[7];
        i += 8;
    }

    let mut result = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    while i < n {
        result = x[i] * y[i] + result;
        i += 1;
    }
    result
}

macro_rules! define_impls {
    ($ty:ident, $wvec:ident, $lanes:expr) => {
        paste::paste! {
            struct [<RevieroDot $ty>]<'a>(&'a [$ty], &'a [$ty]);

            impl reviero::SimdFn for [<RevieroDot $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: reviero::traits::Simd>(self, simd: S) -> $ty {
                    let (xs, xt) = reviero::utils::[<as_simd_ $ty>](&simd, self.0);
                    let (ys, yt) = reviero::utils::[<as_simd_ $ty>](&simd, self.1);

                    let mut acc = [simd.[<splat_ $ty>](0.0); 8];
                    let mut it = xs.zip(ys);

                    while it.len() >= 8 {
                        let (x0, y0) = it.next().unwrap();
                        let (x1, y1) = it.next().unwrap();
                        let (x2, y2) = it.next().unwrap();
                        let (x3, y3) = it.next().unwrap();
                        let (x4, y4) = it.next().unwrap();
                        let (x5, y5) = it.next().unwrap();
                        let (x6, y6) = it.next().unwrap();
                        let (x7, y7) = it.next().unwrap();

                        acc[0] = simd.[<mul_add_ $ty>](x0, y0, acc[0]);
                        acc[1] = simd.[<mul_add_ $ty>](x1, y1, acc[1]);
                        acc[2] = simd.[<mul_add_ $ty>](x2, y2, acc[2]);
                        acc[3] = simd.[<mul_add_ $ty>](x3, y3, acc[3]);
                        acc[4] = simd.[<mul_add_ $ty>](x4, y4, acc[4]);
                        acc[5] = simd.[<mul_add_ $ty>](x5, y5, acc[5]);
                        acc[6] = simd.[<mul_add_ $ty>](x6, y6, acc[6]);
                        acc[7] = simd.[<mul_add_ $ty>](x7, y7, acc[7]);
                    }

                    for (x, y) in it {
                        acc[0] = simd.[<mul_add_ $ty>](x, y, acc[0]);
                    }

                    let sum = simd.[<add_ $ty>](
                        simd.[<add_ $ty>](
                            simd.[<add_ $ty>](acc[0], acc[1]),
                            simd.[<add_ $ty>](acc[2], acc[3]),
                        ),
                        simd.[<add_ $ty>](
                            simd.[<add_ $ty>](acc[4], acc[5]),
                            simd.[<add_ $ty>](acc[6], acc[7]),
                        ),
                    );
                    let mut result = simd.[<reduce_sum_ $ty>](sum);
                    for (&x, &y) in xt.iter().zip(yt) {
                        result += x * y;
                    }
                    result
                }
            }

            #[inline(never)]
            fn [<reviero_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                reviero::dispatch([<RevieroDot $ty>](x, y))
            }

            struct [<PulpDot $ty>]<'a>(&'a [$ty], &'a [$ty]);

            impl pulp::WithSimd for [<PulpDot $ty>]<'_> {
                type Output = $ty;

                #[inline(always)]
                fn with_simd<S: pulp::Simd>(self, simd: S) -> $ty {
                    let (xs, xt) = S::[<as_simd_ $ty s>](self.0);
                    let (ys, yt) = S::[<as_simd_ $ty s>](self.1);

                    let mut acc = [simd.[<splat_ $ty s>](0.0); 8];
                    let mut i = 0;

                    while i + 8 <= xs.len() {
                        acc[0] = simd.[<mul_add_ $ty s>](xs[i], ys[i], acc[0]);
                        acc[1] = simd.[<mul_add_ $ty s>](xs[i + 1], ys[i + 1], acc[1]);
                        acc[2] = simd.[<mul_add_ $ty s>](xs[i + 2], ys[i + 2], acc[2]);
                        acc[3] = simd.[<mul_add_ $ty s>](xs[i + 3], ys[i + 3], acc[3]);
                        acc[4] = simd.[<mul_add_ $ty s>](xs[i + 4], ys[i + 4], acc[4]);
                        acc[5] = simd.[<mul_add_ $ty s>](xs[i + 5], ys[i + 5], acc[5]);
                        acc[6] = simd.[<mul_add_ $ty s>](xs[i + 6], ys[i + 6], acc[6]);
                        acc[7] = simd.[<mul_add_ $ty s>](xs[i + 7], ys[i + 7], acc[7]);
                        i += 8;
                    }
                    while i < xs.len() {
                        acc[0] = simd.[<mul_add_ $ty s>](xs[i], ys[i], acc[0]);
                        i += 1;
                    }

                    let sum = simd.[<add_ $ty s>](
                        simd.[<add_ $ty s>](
                            simd.[<add_ $ty s>](acc[0], acc[1]),
                            simd.[<add_ $ty s>](acc[2], acc[3]),
                        ),
                        simd.[<add_ $ty s>](
                            simd.[<add_ $ty s>](acc[4], acc[5]),
                            simd.[<add_ $ty s>](acc[6], acc[7]),
                        ),
                    );
                    let mut result = simd.[<reduce_sum_ $ty s>](sum);
                    for (&x, &y) in xt.iter().zip(yt) {
                        result += x * y;
                    }
                    result
                }
            }

            #[inline(never)]
            fn [<pulp_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                pulp::Arch::new().dispatch([<PulpDot $ty>](x, y))
            }

            #[inline(never)]
            fn [<wide_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                let n = x.len();
                let lanes = $lanes;
                let mut acc = [$wvec::splat(0.0); 8];
                let mut i = 0;

                while i + 8 * lanes <= n {
                    acc[0] = $wvec::new(x[i..i + lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i..i + lanes].try_into().unwrap()), acc[0]);
                    acc[1] = $wvec::new(x[i + lanes..i + 2 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + lanes..i + 2 * lanes].try_into().unwrap()), acc[1]);
                    acc[2] = $wvec::new(x[i + 2 * lanes..i + 3 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 2 * lanes..i + 3 * lanes].try_into().unwrap()), acc[2]);
                    acc[3] = $wvec::new(x[i + 3 * lanes..i + 4 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 3 * lanes..i + 4 * lanes].try_into().unwrap()), acc[3]);
                    acc[4] = $wvec::new(x[i + 4 * lanes..i + 5 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 4 * lanes..i + 5 * lanes].try_into().unwrap()), acc[4]);
                    acc[5] = $wvec::new(x[i + 5 * lanes..i + 6 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 5 * lanes..i + 6 * lanes].try_into().unwrap()), acc[5]);
                    acc[6] = $wvec::new(x[i + 6 * lanes..i + 7 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 6 * lanes..i + 7 * lanes].try_into().unwrap()), acc[6]);
                    acc[7] = $wvec::new(x[i + 7 * lanes..i + 8 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 7 * lanes..i + 8 * lanes].try_into().unwrap()), acc[7]);
                    i += 8 * lanes;
                }
                while i + lanes <= n {
                    acc[0] = $wvec::new(x[i..i + lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i..i + lanes].try_into().unwrap()), acc[0]);
                    i += lanes;
                }

                let sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                let mut result = sum.reduce_add();
                while i < n {
                    result += x[i] * y[i];
                    i += 1;
                }
                result
            }

            fn [<dot_impls_ $ty>]() -> [(&'static str, DotFn<$ty>); 4] {
                [
                    ("reviero", [<reviero_ $ty>]),
                    ("pulp", [<pulp_ $ty>]),
                    ("wide", [<wide_ $ty>]),
                    ("scalar", scalar_dot::<$ty>),
                ]
            }
        }
    };
}

define_impls!(f32, WideF32x8, 8);
define_impls!(f64, WideF64x4, 4);

macro_rules! define_half_impls {
    ($ty:ident) => {
        paste::paste! {
            struct [<RevieroDot $ty>]<'a>(&'a [$ty], &'a [$ty]);

            impl reviero::SimdFn for [<RevieroDot $ty>]<'_> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: reviero::traits::Simd>(self, simd: S) -> $ty {
                    let (xs, xt) = reviero::utils::[<as_simd_ $ty>](&simd, self.0);
                    let (ys, yt) = reviero::utils::[<as_simd_ $ty>](&simd, self.1);

                    let mut acc = [simd.[<splat_ $ty>](<$ty>::from_f32(0.0)); 8];
                    let mut it = xs.zip(ys);

                    while it.len() >= 8 {
                        let (x0, y0) = it.next().unwrap();
                        let (x1, y1) = it.next().unwrap();
                        let (x2, y2) = it.next().unwrap();
                        let (x3, y3) = it.next().unwrap();
                        let (x4, y4) = it.next().unwrap();
                        let (x5, y5) = it.next().unwrap();
                        let (x6, y6) = it.next().unwrap();
                        let (x7, y7) = it.next().unwrap();

                        acc[0] = simd.[<mul_add_ $ty>](x0, y0, acc[0]);
                        acc[1] = simd.[<mul_add_ $ty>](x1, y1, acc[1]);
                        acc[2] = simd.[<mul_add_ $ty>](x2, y2, acc[2]);
                        acc[3] = simd.[<mul_add_ $ty>](x3, y3, acc[3]);
                        acc[4] = simd.[<mul_add_ $ty>](x4, y4, acc[4]);
                        acc[5] = simd.[<mul_add_ $ty>](x5, y5, acc[5]);
                        acc[6] = simd.[<mul_add_ $ty>](x6, y6, acc[6]);
                        acc[7] = simd.[<mul_add_ $ty>](x7, y7, acc[7]);
                    }

                    for (x, y) in it {
                        acc[0] = simd.[<mul_add_ $ty>](x, y, acc[0]);
                    }

                    let sum = simd.[<add_ $ty>](
                        simd.[<add_ $ty>](
                            simd.[<add_ $ty>](acc[0], acc[1]),
                            simd.[<add_ $ty>](acc[2], acc[3]),
                        ),
                        simd.[<add_ $ty>](
                            simd.[<add_ $ty>](acc[4], acc[5]),
                            simd.[<add_ $ty>](acc[6], acc[7]),
                        ),
                    );
                    let mut result = simd.[<reduce_sum_ $ty>](sum);
                    for (&x, &y) in xt.iter().zip(yt) {
                        result += x * y;
                    }
                    result
                }
            }

            #[inline(never)]
            fn [<reviero_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                reviero::dispatch([<RevieroDot $ty>](x, y))
            }

            fn [<dot_impls_ $ty>]() -> [(&'static str, DotFn<$ty>); 2] {
                [
                    ("reviero", [<reviero_ $ty>]),
                    ("scalar", scalar_dot::<$ty>),
                ]
            }
        }
    };
}

define_half_impls!(f16);
define_half_impls!(bf16);

fn bench_dot<T: Copy>(c: &mut Criterion, label: &str, one: T, two: T, impls: &[(&str, DotFn<T>)]) {
    let mut group = c.benchmark_group(label);
    for &n in SIZES {
        let x = vec![one; n];
        let y = vec![two; n];
        group.throughput(Throughput::Elements(n as u64));
        for &(name, f) in impls {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| black_box(f(black_box(x.as_slice()), black_box(y.as_slice()))))
            });
        }
    }
    group.finish();
}

fn bench_f32(c: &mut Criterion) {
    bench_dot(c, "dot_f32", 1.0_f32, 2.0_f32, &dot_impls_f32());
}

fn bench_f64(c: &mut Criterion) {
    bench_dot(c, "dot_f64", 1.0_f64, 2.0_f64, &dot_impls_f64());
}

fn bench_f16(c: &mut Criterion) {
    bench_dot(c, "dot_f16", f16::from_f32(1.0), f16::from_f32(2.0), &dot_impls_f16());
}

fn bench_bf16(c: &mut Criterion) {
    bench_dot(c, "dot_bf16", bf16::from_f32(1.0), bf16::from_f32(2.0), &dot_impls_bf16());
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(700));
    targets = bench_f32, bench_f64, bench_f16, bench_bf16
}
criterion_main!(benches);
