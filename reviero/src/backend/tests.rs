//! Cross-backend accuracy test.
//!
//! Every available backend executes the same generic suite and its output is
//! compared against the scalar fallback backend. Floating-point results are
//! compared with a small tolerance (FMA may round once instead of twice),
//! while masks and bitwise results must match exactly.

use super::fallback::Fallback;
use super::test_utils::{SUITE_N, Suite, SuiteHalfType, SuiteOp};
use crate::SimdFn;

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
