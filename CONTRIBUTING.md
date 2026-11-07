# Contributing to reviero

Thanks for your interest in contributing to `reviero`! Contributions of bug
reports, fixes, tests, documentation, and performance improvements are welcome.

Before participating, please read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

The Rust crate is in the `reviero/` directory. Install Rust 1.89 or newer with
the `rustfmt` and `clippy` components, then run development commands from that
directory:

```sh
cd reviero
cargo test
```

## Before opening a pull request

Run the same checks used by CI:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features
cargo test --all --all-features
```

CI treats warnings as errors. To check locally with the same setting, run the
commands with `RUSTFLAGS="-D warnings"`.

The project also checks the AArch64 target. If you have that target installed,
you can run:

```sh
cargo check --target aarch64-unknown-linux-gnu --all-targets --all-features
```

## Code and tests

- Keep changes focused and follow the existing Rust style. `rustfmt.toml`
  configures formatting.
- Public API changes should include clear rustdoc and, where useful, an
  example. Keep the crate-level documentation in `src/lib.rs` accurate.
- Implement operations through the generic `traits::Simd` interface so the
  same operation can run on every supported backend. Use
  `utils::as_simd_f32s` or `utils::as_simd_f64s` to split slices into complete
  vectors and a scalar tail.
- Mark `SimdFn::call` implementations with `#[inline(always)]` so backend
  target features can propagate into the operation.
- Add or update tests for behavioral changes. Account for backend-dependent
  floating-point rounding in reductions and fused multiply-add operations;
  compare floating-point results with an appropriate tolerance rather than
  assuming bit-for-bit equality.
- Keep architecture-specific operations behind the existing backend structure
  and ensure runtime feature detection remains consistent with the target
  features used by each implementation.

## Pull requests

Please open a pull request against `main` with a concise description of the
change, its motivation, and the checks you ran. Include relevant test results
and call out any public API or floating-point behavior changes.
