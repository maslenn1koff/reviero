//! Cross-backend accuracy test.
//!
//! Every available backend executes the same generic suite and its output is
//! compared against the scalar fallback backend. Floating-point results are
//! compared with a small tolerance (FMA may round once instead of twice),
//! while masks and bitwise results must match exactly.

use super::fallback::Fallback;
use super::test_utils::{SUITE_N, Suite, SuiteHalfType, SuiteOp};
use crate::SimdFn;
use crate::traits::Simd;

fn assert_close_f32(name: &str, index: usize, want: f32, got: f32) {
    let tolerance = 1e-6 * want.abs().max(1.0);
    assert!((want - got).abs() <= tolerance, "{name}: f32 float result #{index}: expected {want:?}, got {got:?}");
}

fn assert_close_f64(name: &str, index: usize, want: f64, got: f64) {
    let tolerance = 1e-12 * want.abs().max(1.0);
    assert!((want - got).abs() <= tolerance, "{name}: f64 float result #{index}: expected {want:?}, got {got:?}");
}

fn assert_close_scalar_f32(name: &str, what: &str, want: f32, got: f32) {
    let tolerance = 1e-5 * want.abs().max(1.0);
    assert!((want - got).abs() <= tolerance, "{name}: f32 {what}: expected {want:?}, got {got:?}");
}

fn assert_close_scalar_f64(name: &str, what: &str, want: f64, got: f64) {
    let tolerance = 1e-12 * want.abs().max(1.0);
    assert!((want - got).abs() <= tolerance, "{name}: f64 {what}: expected {want:?}, got {got:?}");
}

fn assert_close_half(name: &str, kind: &str, what: &str, want: f32, got: f32, relative: f32) {
    let tolerance = relative * want.abs().max(1.0);
    assert!((want - got).abs() <= tolerance, "{name}: {kind} {what}: expected {want:?}, got {got:?}");
}

fn assert_half_suite(name: &str, kind: &str, expected: &SuiteHalfType, actual: &SuiteHalfType, relative: f32) {
    assert_eq!(actual.floats.len(), expected.floats.len(), "{name}: {kind} float group count");
    for (group, (want_group, got_group)) in expected.floats.iter().zip(actual.floats.iter()).enumerate() {
        assert_eq!(got_group.len(), want_group.len(), "{name}: {kind} float group #{group} length");
        for (index, (&want, &got)) in want_group.iter().zip(got_group.iter()).enumerate() {
            assert_close_half(name, kind, &format!("float result #{}", group * SUITE_N + index), want, got, relative);
        }
    }
    assert_eq!(actual.masks, expected.masks, "{name}: {kind} comparison masks");
    assert_close_half(name, kind, "reduce_sum", expected.sum, actual.sum, relative);
    assert_close_half(name, kind, "reduce_max", expected.max, actual.max, relative);
    assert_close_half(name, kind, "reduce_min", expected.min, actual.min, relative);
}

fn assert_suite(name: &str, expected: &Suite, actual: &Suite) {
    assert_eq!(actual.single.floats.len(), expected.single.floats.len(), "{name}: f32 float group count");
    for (group, (want_group, got_group)) in expected.single.floats.iter().zip(actual.single.floats.iter()).enumerate() {
        assert_eq!(got_group.len(), want_group.len(), "{name}: f32 float group #{group} length");
        for (index, (&want, &got)) in want_group.iter().zip(got_group.iter()).enumerate() {
            assert_close_f32(name, group * SUITE_N + index, want, got);
        }
    }
    assert_eq!(actual.single.masks, expected.single.masks, "{name}: f32 comparison masks");
    assert_eq!(actual.single.bits, expected.single.bits, "{name}: f32 bitwise results");
    assert_close_scalar_f32(name, "reduce_sum", expected.single.sum, actual.single.sum);
    assert_eq!(actual.single.max, expected.single.max, "{name}: f32 reduce_max");
    assert_eq!(actual.single.min, expected.single.min, "{name}: f32 reduce_min");

    assert_eq!(actual.double.floats.len(), expected.double.floats.len(), "{name}: f64 float group count");
    for (group, (want_group, got_group)) in expected.double.floats.iter().zip(actual.double.floats.iter()).enumerate() {
        assert_eq!(got_group.len(), want_group.len(), "{name}: f64 float group #{group} length");
        for (index, (&want, &got)) in want_group.iter().zip(got_group.iter()).enumerate() {
            assert_close_f64(name, group * SUITE_N + index, want, got);
        }
    }
    assert_eq!(actual.double.masks, expected.double.masks, "{name}: f64 comparison masks");
    assert_eq!(actual.double.bits, expected.double.bits, "{name}: f64 bitwise results");
    assert_close_scalar_f64(name, "reduce_sum", expected.double.sum, actual.double.sum);
    assert_eq!(actual.double.max, expected.double.max, "{name}: f64 reduce_max");
    assert_eq!(actual.double.min, expected.double.min, "{name}: f64 reduce_min");

    // Half-precision results are widened to `f32`; allow half/bf16-sized
    // rounding differences between native and software implementations.
    assert_half_suite(name, "f16", &expected.half.f16, &actual.half.f16, 2e-3);
    assert_half_suite(name, "bf16", &expected.half.bf16, &actual.half.bf16, 1e-2);
}

#[test]
fn all_backends_match_fallback() {
    let expected = SuiteOp.call(Fallback);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            let actual = unsafe { super::x86::sse2::vectorize(SuiteOp) };
            assert_suite("sse2", &expected, &actual);

            if std::arch::is_x86_feature_detected!("f16c") {
                let actual = unsafe { super::x86::sse2::vectorize_f16c(SuiteOp) };
                assert_suite("sse2+f16c", &expected, &actual);
            }
        } else {
            eprintln!("sse2 backend skipped: CPU does not support sse2");
        }

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            let actual = unsafe { super::x86::avx2::vectorize(SuiteOp) };
            assert_suite("avx2+fma", &expected, &actual);

            if std::arch::is_x86_feature_detected!("f16c") {
                let actual = unsafe { super::x86::avx2::vectorize_f16c(SuiteOp) };
                assert_suite("avx2+fma+f16c", &expected, &actual);
            }
        } else {
            eprintln!("avx2 backend skipped: CPU does not support avx2+fma");
        }

        if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma") {
            let actual = unsafe { super::x86::avx512::vectorize(SuiteOp) };
            assert_suite("avx512f+fma", &expected, &actual);

            if std::arch::is_x86_feature_detected!("avx512fp16") && std::arch::is_x86_feature_detected!("avx512vl") && std::arch::is_x86_feature_detected!("avx512bf16") {
                let actual = unsafe { super::x86::avx512::vectorize_native(SuiteOp) };
                assert_suite("avx512f+fma+native", &expected, &actual);
            }
        } else {
            eprintln!("avx512 backend skipped: CPU does not support avx512f+fma");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let actual = unsafe { super::aarch64::neon::vectorize(SuiteOp) };
        assert_suite("neon", &expected, &actual);

        if std::arch::is_aarch64_feature_detected!("fp16") {
            let actual = unsafe { super::aarch64::neon::vectorize_f16(SuiteOp) };
            assert_suite("neon+fp16", &expected, &actual);
        }
    }
}

/// Inputs whose lanes cover every NaN placement:
/// 0: `a` is NaN, 1: `b` is NaN, 2: both finite, 3: both NaN.
fn nan_inputs_f32(lanes: usize) -> (Vec<f32>, Vec<f32>) {
    let lhs = (0..lanes).map(|lane| if lane % 4 == 0 || lane % 4 == 3 { f32::NAN } else { lane as f32 }).collect();
    let rhs = (0..lanes)
        .map(|lane| match lane % 4 {
            1 | 3 => f32::NAN,
            _ => lane as f32 + 1.0,
        })
        .collect();
    (lhs, rhs)
}

fn nan_inputs_f64(lanes: usize) -> (Vec<f64>, Vec<f64>) {
    let lhs = (0..lanes).map(|lane| if lane % 4 == 0 || lane % 4 == 3 { f64::NAN } else { lane as f64 }).collect();
    let rhs = (0..lanes)
        .map(|lane| match lane % 4 {
            1 | 3 => f64::NAN,
            _ => lane as f64 + 1.0,
        })
        .collect();
    (lhs, rhs)
}

struct NanResults {
    lhs_f32: Vec<f32>,
    rhs_f32: Vec<f32>,
    min_f32: Vec<f32>,
    max_f32: Vec<f32>,
    lhs_f64: Vec<f64>,
    rhs_f64: Vec<f64>,
    min_f64: Vec<f64>,
    max_f64: Vec<f64>,
}

struct NanOp;

impl SimdFn for NanOp {
    type Result = NanResults;

    #[inline(always)]
    fn call<S: Simd>(self, simd: S) -> NanResults {
        let lanes = S::f32_lanes;
        let (lhs_f32, rhs_f32) = nan_inputs_f32(lanes);
        let a = simd.load_f32(&lhs_f32);
        let b = simd.load_f32(&rhs_f32);
        let mut min_f32 = vec![0.0; lanes];
        let mut max_f32 = vec![0.0; lanes];
        simd.store_f32(&mut min_f32, simd.min_f32(a, b));
        simd.store_f32(&mut max_f32, simd.max_f32(a, b));

        let lanes = S::f64_lanes;
        let (lhs_f64, rhs_f64) = nan_inputs_f64(lanes);
        let a = simd.load_f64(&lhs_f64);
        let b = simd.load_f64(&rhs_f64);
        let mut min_f64 = vec![0.0; lanes];
        let mut max_f64 = vec![0.0; lanes];
        simd.store_f64(&mut min_f64, simd.min_f64(a, b));
        simd.store_f64(&mut max_f64, simd.max_f64(a, b));

        NanResults {
            lhs_f32,
            rhs_f32,
            min_f32,
            max_f32,
            lhs_f64,
            rhs_f64,
            min_f64,
            max_f64,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum NanRule {
    /// x86 `minps`/`maxps` style: the second operand wins when either lane is NaN.
    SecondOperand,
    /// AArch64 `FMIN`/`FMAX` style: any NaN propagates.
    Propagate,
    /// Rust `f32::min`/`f32::max` style: ignore a single NaN.
    Number,
}

fn expected_min_f32(rule: NanRule, a: f32, b: f32) -> f32 {
    match rule {
        NanRule::SecondOperand if a.is_nan() || b.is_nan() => b,
        NanRule::Propagate if a.is_nan() || b.is_nan() => f32::NAN,
        _ => a.min(b),
    }
}

fn expected_max_f32(rule: NanRule, a: f32, b: f32) -> f32 {
    match rule {
        NanRule::SecondOperand if a.is_nan() || b.is_nan() => b,
        NanRule::Propagate if a.is_nan() || b.is_nan() => f32::NAN,
        _ => a.max(b),
    }
}

fn expected_min_f64(rule: NanRule, a: f64, b: f64) -> f64 {
    match rule {
        NanRule::SecondOperand if a.is_nan() || b.is_nan() => b,
        NanRule::Propagate if a.is_nan() || b.is_nan() => f64::NAN,
        _ => a.min(b),
    }
}

fn expected_max_f64(rule: NanRule, a: f64, b: f64) -> f64 {
    match rule {
        NanRule::SecondOperand if a.is_nan() || b.is_nan() => b,
        NanRule::Propagate if a.is_nan() || b.is_nan() => f64::NAN,
        _ => a.max(b),
    }
}

fn assert_nan_eq_f32(name: &str, what: &str, lane: usize, got: f32, want: f32) {
    if want.is_nan() {
        assert!(got.is_nan(), "{name}: {what} f32 lane {lane}: expected NaN, got {got:?}");
    } else {
        assert_eq!(got.to_bits(), want.to_bits(), "{name}: {what} f32 lane {lane}: expected {want:?}, got {got:?}");
    }
}

fn assert_nan_eq_f64(name: &str, what: &str, lane: usize, got: f64, want: f64) {
    if want.is_nan() {
        assert!(got.is_nan(), "{name}: {what} f64 lane {lane}: expected NaN, got {got:?}");
    } else {
        assert_eq!(got.to_bits(), want.to_bits(), "{name}: {what} f64 lane {lane}: expected {want:?}, got {got:?}");
    }
}

fn assert_nan_min_max(name: &str, results: &NanResults, rule: NanRule) {
    for lane in 0..results.min_f32.len() {
        let a = results.lhs_f32[lane];
        let b = results.rhs_f32[lane];
        assert_nan_eq_f32(name, "min", lane, results.min_f32[lane], expected_min_f32(rule, a, b));
        assert_nan_eq_f32(name, "max", lane, results.max_f32[lane], expected_max_f32(rule, a, b));
    }

    for lane in 0..results.min_f64.len() {
        let a = results.lhs_f64[lane];
        let b = results.rhs_f64[lane];
        assert_nan_eq_f64(name, "min", lane, results.min_f64[lane], expected_min_f64(rule, a, b));
        assert_nan_eq_f64(name, "max", lane, results.max_f64[lane], expected_max_f64(rule, a, b));
    }
}

#[test]
fn dispatch_matches_fallback() {
    let expected = SuiteOp.call(Fallback);
    let actual = crate::dispatch(SuiteOp);
    assert_suite("dispatch", &expected, &actual);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn backend_is_cached_after_dispatch() {
    let _ = crate::dispatch(SuiteOp);
    assert_ne!(crate::BACKEND.load(std::sync::atomic::Ordering::Relaxed), 0, "backend was not cached");
}

#[test]
fn nan_min_max_semantics() {
    assert_nan_min_max("fallback", &NanOp.call(Fallback), NanRule::Number);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        macro_rules! check {
            ($name:literal, $call:expr) => {{
                let actual = unsafe { $call(NanOp) };
                assert_nan_min_max($name, &actual, NanRule::SecondOperand);
            }};
        }

        if std::arch::is_x86_feature_detected!("sse2") {
            check!("sse2", super::x86::sse2::vectorize);
            if std::arch::is_x86_feature_detected!("f16c") {
                check!("sse2+f16c", super::x86::sse2::vectorize_f16c);
            }
        }

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            check!("avx2+fma", super::x86::avx2::vectorize);
            if std::arch::is_x86_feature_detected!("f16c") {
                check!("avx2+fma+f16c", super::x86::avx2::vectorize_f16c);
            }
        }

        if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma") {
            check!("avx512f+fma", super::x86::avx512::vectorize);
            if std::arch::is_x86_feature_detected!("avx512fp16") && std::arch::is_x86_feature_detected!("avx512vl") && std::arch::is_x86_feature_detected!("avx512bf16") {
                check!("avx512f+fma+native", super::x86::avx512::vectorize_native);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let actual = unsafe { super::aarch64::neon::vectorize(NanOp) };
        assert_nan_min_max("neon", &actual, NanRule::Propagate);

        if std::arch::is_aarch64_feature_detected!("fp16") {
            let actual = unsafe { super::aarch64::neon::vectorize_f16(NanOp) };
            assert_nan_min_max("neon+fp16", &actual, NanRule::Propagate);
        }
    }
}
