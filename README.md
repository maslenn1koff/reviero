# reviero

[![crates.io](https://img.shields.io/crates/v/reviero?style=flat&labelColor=white&color=white&logoColor=black)](https://crates.io/crates/reviero)
[![CI](https://img.shields.io/github/actions/workflow/status/maslenn1koff/reviero/rust.yml?style=flat&label=CI&labelColor=white&color=white)](https://github.com/maslenn1koff/reviero/actions/workflows/rust.yml)
[![MSRV](https://img.shields.io/crates/msrv/reviero?style=flat&labelColor=white&color=white)](https://crates.io/crates/reviero)
[![License](https://img.shields.io/badge/license-MIT-white?style=flat&labelColor=white&color=white)](LICENSE)

**Safe and cross-platform SIMD for Rust.**

`reviero` provides portable SIMD operations for `f32`, `f64`, `f16` and `bf16` with runtime CPU dispatch, allowing the same code to run across different architectures and instruction sets.

## Installation

Add `reviero` to your `Cargo.toml`:

```toml
[dependencies]
reviero = "0.2"
```

## Usage

### Manual Vectorization

`reviero` provides a low-level SIMD interface that allows you to implement vectorized operations manually while keeping the implementation portable across supported architectures.

The following example implements a vectorized sum:

```rust
pub struct Sum<'a>(&'a [f32]);

impl<'a> reviero::SimdFn for Sum<'a> {
    type Result = f32;

    #[inline(always)]
    fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
        let (head, tail) = reviero::utils::as_simd_f32(&simd, self.0);

        let mut sum = simd.splat_f32(0.0);

        for vector in head {
            sum = simd.add_f32(sum, vector);
        }

        let mut sum = simd.reduce_sum_f32(sum);

        for &x in tail {
            sum += x;
        }

        sum
    }
}

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let sum = reviero::dispatch(Sum(&x));
    println!("sum: {sum}");
}
```

`SimdFn::call` must be marked `#[inline(always)]`. This lets the selected backend's target features propagate into the operation and allows its vector instructions to be inlined; without it the operation stays correct, but may lose most of the performance benefit.

### Runtime Dispatch

`reviero::dispatch()` automatically selects an appropriate SIMD implementation based on the capabilities of the current CPU.

## Benchmarks

Criterion dot-product benchmarks on an Intel Core i5-12600H. For `f32` and `f64` inputs each chart compares `reviero`, `pulp`, `wide`, and `scalar`; for `f16` and `bf16` inputs it compares `reviero` and `scalar`. Sizes range from 513 to 1,048,576 elements. Higher throughput is better; values are in billions of elements per second (Gelem/s).

### Native target

Built with `RUSTFLAGS="-C target-cpu=native"`.

![Dot-product throughput, native target](https://github.com/maslenn1koff/reviero/raw/main/assets/dot-product-benchmarks.svg)

### Default target

Built without the `target-cpu` flag.

![Dot-product throughput, default target](https://github.com/maslenn1koff/reviero/raw/main/assets/dot-product-benchmarks-no-native.svg)

See the raw Criterion output for the [native run](https://github.com/maslenn1koff/reviero/blob/main/assets/raw_bench_native.txt) and the [default-target run](https://github.com/maslenn1koff/reviero/blob/main/assets/raw_bench.txt).
