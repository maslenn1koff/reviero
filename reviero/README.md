# reviero

[![Status](https://img.shields.io/badge/status-early_development-white)](#)
[![License](https://img.shields.io/badge/license-MIT-white)](LICENSE)

**Safe and cross-platform SIMD for Rust.**

`reviero` provides portable SIMD operations with runtime CPU dispatch, allowing the same code to run across different architectures and instruction sets.

## Installation

Add `reviero` to your `Cargo.toml`:

```toml
[dependencies]
reviero = "0.1"
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
        let (head, tail) = reviero::utils::as_simd_f32s(&simd, self.0);

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

## Benchmarks?

**go to github!**