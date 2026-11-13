//! Quick dot-product benchmark: reviero vs pulp vs wide vs scalar.
//!
//! ```text
//! taskset -c 2 env RUSTFLAGS="-C target-cpu=native" cargo bench --bench dot
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use wide::{f32x8 as WideF32x8, f64x4 as WideF64x4};

const SIZES: &[usize] = &[513, 1027, 4099, 10_007, 1 << 20];

type DotFn<T> = fn(&[T], &[T]) -> T;

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

                    let mut acc = [simd.[<splat_ $ty>](0.0); 4];
                    let mut it = xs.zip(ys);

                    while it.len() >= 4 {
                        let (x0, y0) = it.next().unwrap();
                        let (x1, y1) = it.next().unwrap();
                        let (x2, y2) = it.next().unwrap();
                        let (x3, y3) = it.next().unwrap();

                        acc[0] = simd.[<mul_add_ $ty>](x0, y0, acc[0]);
                        acc[1] = simd.[<mul_add_ $ty>](x1, y1, acc[1]);
                        acc[2] = simd.[<mul_add_ $ty>](x2, y2, acc[2]);
                        acc[3] = simd.[<mul_add_ $ty>](x3, y3, acc[3]);
                    }

                    for (x, y) in it {
                        acc[0] = simd.[<mul_add_ $ty>](x, y, acc[0]);
                    }

                    let sum = simd.[<add_ $ty>](
                        simd.[<add_ $ty>](acc[0], acc[1]),
                        simd.[<add_ $ty>](acc[2], acc[3]),
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

                    let mut acc = [simd.[<splat_ $ty s>](0.0); 4];
                    let mut i = 0;

                    while i + 4 <= xs.len() {
                        acc[0] = simd.[<mul_add_ $ty s>](xs[i], ys[i], acc[0]);
                        acc[1] = simd.[<mul_add_ $ty s>](xs[i + 1], ys[i + 1], acc[1]);
                        acc[2] = simd.[<mul_add_ $ty s>](xs[i + 2], ys[i + 2], acc[2]);
                        acc[3] = simd.[<mul_add_ $ty s>](xs[i + 3], ys[i + 3], acc[3]);
                        i += 4;
                    }
                    while i < xs.len() {
                        acc[0] = simd.[<mul_add_ $ty s>](xs[i], ys[i], acc[0]);
                        i += 1;
                    }

                    let sum = simd.[<add_ $ty s>](
                        simd.[<add_ $ty s>](acc[0], acc[1]),
                        simd.[<add_ $ty s>](acc[2], acc[3]),
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
                let mut acc = [$wvec::splat(0.0); 4];
                let mut i = 0;

                while i + 4 * lanes <= n {
                    acc[0] = $wvec::new(x[i..i + lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i..i + lanes].try_into().unwrap()), acc[0]);
                    acc[1] = $wvec::new(x[i + lanes..i + 2 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + lanes..i + 2 * lanes].try_into().unwrap()), acc[1]);
                    acc[2] = $wvec::new(x[i + 2 * lanes..i + 3 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 2 * lanes..i + 3 * lanes].try_into().unwrap()), acc[2]);
                    acc[3] = $wvec::new(x[i + 3 * lanes..i + 4 * lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i + 3 * lanes..i + 4 * lanes].try_into().unwrap()), acc[3]);
                    i += 4 * lanes;
                }
                while i + lanes <= n {
                    acc[0] = $wvec::new(x[i..i + lanes].try_into().unwrap())
                        .mul_add($wvec::new(y[i..i + lanes].try_into().unwrap()), acc[0]);
                    i += lanes;
                }

                let mut result = ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add();
                while i < n {
                    result += x[i] * y[i];
                    i += 1;
                }
                result
            }

            #[inline(never)]
            fn [<scalar_ $ty>](x: &[$ty], y: &[$ty]) -> $ty {
                x.iter().zip(y).map(|(a, b)| a * b).sum()
            }

            fn [<dot_impls_ $ty>]() -> [(&'static str, DotFn<$ty>); 4] {
                [
                    ("reviero", [<reviero_ $ty>]),
                    ("pulp", [<pulp_ $ty>]),
                    ("wide", [<wide_ $ty>]),
                    ("scalar", [<scalar_ $ty>]),
                ]
            }
        }
    };
}

define_impls!(f32, WideF32x8, 8);
define_impls!(f64, WideF64x4, 4);

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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(700));
    targets = bench_f32, bench_f64
}
criterion_main!(benches);
