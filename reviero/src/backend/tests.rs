//! Cross-backend accuracy test.
//!
//! Every available backend executes the same generic suite and its output is
//! compared against the scalar fallback backend. Floating-point results are
//! compared with a small tolerance (FMA may round once instead of twice),
//! while masks and bitwise results must match exactly.

use super::fallback::Fallback;
use super::test_utils::{SUITE_N, Suite, SuiteOp};
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
}

#[test]
fn all_backends_match_fallback() {
    let expected = SuiteOp.call(Fallback);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            let actual = unsafe { super::x86_64::sse2::vectorize(SuiteOp) };
            assert_suite("sse2", &expected, &actual);
        } else {
            eprintln!("sse2 backend skipped: CPU does not support sse2");
        }

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            let actual = unsafe { super::x86_64::avx2::vectorize(SuiteOp) };
            assert_suite("avx2+fma", &expected, &actual);
        } else {
            eprintln!("avx2 backend skipped: CPU does not support avx2+fma");
        }

        if std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma") {
            let actual = unsafe { super::x86_64::avx512::vectorize(SuiteOp) };
            assert_suite("avx512f+fma", &expected, &actual);
        } else {
            eprintln!("avx512 backend skipped: CPU does not support avx512f+fma");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let actual = unsafe { super::aarch64::neon::vectorize(SuiteOp) };
        assert_suite("neon", &expected, &actual);
    }
}
