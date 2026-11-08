//! Shared helpers for backend tests.
//!
//! The `check_*` functions are generic over `Simd` and are meant to be called
//! from a `SimdFn` implementation that is itself executed inside a backend's
//! `vectorize` wrapper, so the target features are enabled for the whole call.

use crate::SimdFn;
use crate::traits::{Simd, SimdMask};

/// Generates the standard per-backend test module.
///
/// The enclosing module must expose a `run(check: Check)` helper that executes
/// a [`Check`] on the backend under test.
#[cfg(test)]
#[macro_export]
macro_rules! backend_tests {
    ($backend:ty, $f32_lanes:expr, $f64_lanes:expr) => {
        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn lanes() {
                assert_eq!(<$backend as $crate::traits::SimdTypes>::f32s_lanes, $f32_lanes);
                assert_eq!(<$backend as $crate::traits::SimdTypes>::f64s_lanes, $f64_lanes);
            }

            #[test]
            fn io() {
                super::run($crate::backend::test_utils::Check::Io);
            }

            #[test]
            fn float_ops() {
                super::run($crate::backend::test_utils::Check::Float);
            }

            #[test]
            fn mask_ops() {
                super::run($crate::backend::test_utils::Check::Mask);
            }

            #[test]
            fn bit_ops() {
                super::run($crate::backend::test_utils::Check::Bits);
            }

            #[test]
            fn reduce_ops() {
                super::run($crate::backend::test_utils::Check::Reduce);
            }
        }
    };
}

/// The operation group executed by [`Run`].
#[derive(Clone, Copy, Debug)]
pub enum Check {
    Io,
    Float,
    Mask,
    Bits,
    Reduce,
}

/// Adapter that lets a [`Check`] be executed through any backend.
pub struct Run(pub Check);

impl SimdFn for Run {
    type Result = ();

    #[inline(always)]
    fn call<S: Simd>(self, simd: S) {
        match self.0 {
            Check::Io => {
                check_io_f32(simd);
                check_io_f64(simd);
            }
            Check::Float => {
                check_float_f32(simd);
                check_float_f64(simd);
            }
            Check::Mask => {
                check_mask_f32(simd);
                check_mask_f64(simd);
            }
            Check::Bits => {
                check_bits_f32(simd);
                check_bits_f64(simd);
            }
            Check::Reduce => {
                check_reduce_f32(simd);
                check_reduce_f64(simd);
            }
        }
    }
}

#[inline(always)]
fn check_io_f32<S: Simd>(simd: S) {
    let lanes = S::f32s_lanes;
    let src: Vec<f32> = (0..lanes).map(|lane| lane as f32 * 0.5 - 2.0).collect();

    let value = simd.load_f32s(&src);
    let mut dst = vec![f32::NAN; lanes];
    simd.store_f32s(&mut dst, value);
    assert_eq!(dst, src, "f32 load/store");

    let value = simd.splat_f32s(1.25);
    simd.store_f32s(&mut dst, value);
    assert!(dst.iter().all(|&value| value == 1.25), "f32 splat: {dst:?}");
}

#[inline(always)]
fn check_io_f64<S: Simd>(simd: S) {
    let lanes = S::f64s_lanes;
    let src: Vec<f64> = (0..lanes).map(|lane| lane as f64 * 0.5 - 2.0).collect();

    let value = simd.load_f64s(&src);
    let mut dst = vec![f64::NAN; lanes];
    simd.store_f64s(&mut dst, value);
    assert_eq!(dst, src, "f64 load/store");

    let value = simd.splat_f64s(1.25);
    simd.store_f64s(&mut dst, value);
    assert!(dst.iter().all(|&value| value == 1.25), "f64 splat: {dst:?}");
}

#[inline(always)]
fn check_float_f32<S: Simd>(simd: S) {
    let lanes = S::f32s_lanes;
    let a: Vec<f32> = (0..lanes).map(|lane| (lane + 1) as f32).collect();
    let b: Vec<f32> = vec![2.0; lanes];
    let c: Vec<f32> = (0..lanes).map(|lane| lane as f32 * 0.5).collect();
    let signed: Vec<f32> = (0..lanes).map(|lane| if lane % 2 == 0 { a[lane] } else { -a[lane] }).collect();
    let squares: Vec<f32> = (0..lanes)
        .map(|lane| {
            let root = (lane + 1) as f32;
            root * root
        })
        .collect();

    let av = simd.load_f32s(&a);
    let bv = simd.load_f32s(&b);
    let cv = simd.load_f32s(&c);
    let sv = simd.load_f32s(&signed);
    let qv = simd.load_f32s(&squares);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f32s(&mut out, $value);
            let expected: Vec<f32> = $expected;
            for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("add", simd.add_f32s(av, bv), a.iter().map(|value| value + 2.0).collect::<Vec<_>>());
    check!("sub", simd.sub_f32s(av, bv), a.iter().map(|value| value - 2.0).collect::<Vec<_>>());
    check!("mul", simd.mul_f32s(av, bv), a.iter().map(|value| value * 2.0).collect::<Vec<_>>());
    check!("div", simd.div_f32s(av, bv), a.iter().map(|value| value / 2.0).collect::<Vec<_>>());
    check!("min", simd.min_f32s(av, bv), a.iter().map(|value| value.min(2.0)).collect::<Vec<_>>());
    check!("max", simd.max_f32s(av, bv), a.iter().map(|value| value.max(2.0)).collect::<Vec<_>>());
    check!("abs", simd.abs_f32s(sv), signed.iter().map(|value| value.abs()).collect::<Vec<_>>());
    check!("neg", simd.neg_f32s(sv), signed.iter().map(|value| -value).collect::<Vec<_>>());
    check!("sqrt", simd.sqrt_f32s(qv), (1..=lanes).map(|value| value as f32).collect::<Vec<_>>());
    check!("mul_add", simd.mul_add_f32s(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 + c[lane]).collect::<Vec<_>>());
    check!("mul_sub", simd.mul_sub_f32s(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 - c[lane]).collect::<Vec<_>>());
    check!(
        "nmul_add",
        simd.nmul_add_f32s(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) + c[lane]).collect::<Vec<_>>()
    );
    check!(
        "nmul_sub",
        simd.nmul_sub_f32s(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) - c[lane]).collect::<Vec<_>>()
    );

    // Cancelling to zero must keep the sign of zero consistent with `a*b - c`
    // and `-(a*b) - c` (i.e. +0.0), not just match under float equality.
    let ones = vec![1.0f32; lanes];
    let neg_ones = vec![-1.0f32; lanes];
    let ones_v = simd.load_f32s(&ones);
    let neg_ones_v = simd.load_f32s(&neg_ones);

    simd.store_f32s(&mut out, simd.mul_sub_f32s(ones_v, ones_v, ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f32.to_bits(), "mul_sub f32 zero sign lane {lane}");
    }

    simd.store_f32s(&mut out, simd.nmul_sub_f32s(ones_v, ones_v, neg_ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f32.to_bits(), "nmul_sub f32 zero sign lane {lane}");
    }
}

#[inline(always)]
fn check_float_f64<S: Simd>(simd: S) {
    let lanes = S::f64s_lanes;
    let a: Vec<f64> = (0..lanes).map(|lane| (lane + 1) as f64).collect();
    let b: Vec<f64> = vec![2.0; lanes];
    let c: Vec<f64> = (0..lanes).map(|lane| lane as f64 * 0.5).collect();
    let signed: Vec<f64> = (0..lanes).map(|lane| if lane % 2 == 0 { a[lane] } else { -a[lane] }).collect();
    let squares: Vec<f64> = (0..lanes)
        .map(|lane| {
            let root = (lane + 1) as f64;
            root * root
        })
        .collect();

    let av = simd.load_f64s(&a);
    let bv = simd.load_f64s(&b);
    let cv = simd.load_f64s(&c);
    let sv = simd.load_f64s(&signed);
    let qv = simd.load_f64s(&squares);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f64s(&mut out, $value);
            let expected: Vec<f64> = $expected;
            for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("add", simd.add_f64s(av, bv), a.iter().map(|value| value + 2.0).collect::<Vec<_>>());
    check!("sub", simd.sub_f64s(av, bv), a.iter().map(|value| value - 2.0).collect::<Vec<_>>());
    check!("mul", simd.mul_f64s(av, bv), a.iter().map(|value| value * 2.0).collect::<Vec<_>>());
    check!("div", simd.div_f64s(av, bv), a.iter().map(|value| value / 2.0).collect::<Vec<_>>());
    check!("min", simd.min_f64s(av, bv), a.iter().map(|value| value.min(2.0)).collect::<Vec<_>>());
    check!("max", simd.max_f64s(av, bv), a.iter().map(|value| value.max(2.0)).collect::<Vec<_>>());
    check!("abs", simd.abs_f64s(sv), signed.iter().map(|value| value.abs()).collect::<Vec<_>>());
    check!("neg", simd.neg_f64s(sv), signed.iter().map(|value| -value).collect::<Vec<_>>());
    check!("sqrt", simd.sqrt_f64s(qv), (1..=lanes).map(|value| value as f64).collect::<Vec<_>>());
    check!("mul_add", simd.mul_add_f64s(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 + c[lane]).collect::<Vec<_>>());
    check!("mul_sub", simd.mul_sub_f64s(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 - c[lane]).collect::<Vec<_>>());
    check!(
        "nmul_add",
        simd.nmul_add_f64s(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) + c[lane]).collect::<Vec<_>>()
    );
    check!(
        "nmul_sub",
        simd.nmul_sub_f64s(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) - c[lane]).collect::<Vec<_>>()
    );

    // Same zero-sign checks as for f32.
    let ones = vec![1.0f64; lanes];
    let neg_ones = vec![-1.0f64; lanes];
    let ones_v = simd.load_f64s(&ones);
    let neg_ones_v = simd.load_f64s(&neg_ones);

    simd.store_f64s(&mut out, simd.mul_sub_f64s(ones_v, ones_v, ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f64.to_bits(), "mul_sub f64 zero sign lane {lane}");
    }

    simd.store_f64s(&mut out, simd.nmul_sub_f64s(ones_v, ones_v, neg_ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f64.to_bits(), "nmul_sub f64 zero sign lane {lane}");
    }
}

#[inline(always)]
fn check_mask_f32<S: Simd>(simd: S) {
    let lanes = S::f32s_lanes;
    let a: Vec<f32> = (0..lanes).map(|lane| lane as f32).collect();
    let b: Vec<f32> = (0..lanes).map(|lane| (lanes - 1 - lane) as f32).collect();
    let av = simd.load_f32s(&a);
    let bv = simd.load_f32s(&b);
    let one = simd.splat_f32s(1.0);
    let zero = simd.splat_f32s(0.0);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $mask:expr, $cmp:expr) => {{
            let selected = simd.select_f32s($mask, one, zero);
            simd.store_f32s(&mut out, selected);
            for lane in 0..lanes {
                let want = if $cmp(a[lane], b[lane]) { 1.0 } else { 0.0 };
                assert_eq!(out[lane], want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("eq", simd.eq_f32s(av, bv), |x, y| x == y);
    check!("ne", simd.ne_f32s(av, bv), |x, y| x != y);
    check!("lt", simd.lt_f32s(av, bv), |x, y| x < y);
    check!("le", simd.le_f32s(av, bv), |x, y| x <= y);
    check!("gt", simd.gt_f32s(av, bv), |x, y| x > y);
    check!("ge", simd.ge_f32s(av, bv), |x, y| x >= y);

    let all = simd.eq_f32s(av, av);
    let none = simd.lt_f32s(av, av);
    assert!(all.all(), "f32 mask all");
    assert!(all.any(), "f32 mask any");
    assert!(!all.none(), "f32 mask none");
    assert!(none.none(), "f32 empty mask none");
    assert!(!none.any(), "f32 empty mask any");
    assert!(!none.all(), "f32 empty mask all");
    assert!(all.and(all).all(), "f32 mask and");
    assert!(all.and(none).none(), "f32 mask and empty");
    assert!(none.or(all).all(), "f32 mask or");
    assert!(none.not().all(), "f32 mask not");
    assert!(all.not().none(), "f32 mask not all");

    // Mixed (partially set) masks must survive and/or/not/select correctly.
    if lanes >= 2 {
        let left: Vec<f32> = (0..lanes).map(|lane| lane as f32).collect();
        let right: Vec<f32> = (0..lanes).map(|lane| (lane + lane % 2) as f32).collect();
        let lv = simd.load_f32s(&left);
        let rv = simd.load_f32s(&right);
        let mixed = simd.eq_f32s(lv, rv);

        assert!(mixed.any(), "f32 mixed mask any");
        assert!(!mixed.all(), "f32 mixed mask all");
        assert!(!mixed.none(), "f32 mixed mask none");

        simd.store_f32s(&mut out, simd.select_f32s(mixed, one, zero));
        for (lane, &value) in out.iter().enumerate() {
            let want = if lane % 2 == 0 { 1.0 } else { 0.0 };
            assert_eq!(value, want, "f32 mixed select lane {lane}");
        }

        let full = simd.eq_f32s(lv, lv);
        assert!(!mixed.and(full).all(), "f32 mixed and all");
        assert!(mixed.or(full).all(), "f32 mixed or all");
        assert!(mixed.not().any(), "f32 mixed not any");
        assert!(mixed.and(full.not()).none(), "f32 mixed and empty");
    }
}

#[inline(always)]
fn check_mask_f64<S: Simd>(simd: S) {
    let lanes = S::f64s_lanes;
    let a: Vec<f64> = (0..lanes).map(|lane| lane as f64).collect();
    let b: Vec<f64> = (0..lanes).map(|lane| (lanes - 1 - lane) as f64).collect();
    let av = simd.load_f64s(&a);
    let bv = simd.load_f64s(&b);
    let one = simd.splat_f64s(1.0);
    let zero = simd.splat_f64s(0.0);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $mask:expr, $cmp:expr) => {{
            let selected = simd.select_f64s($mask, one, zero);
            simd.store_f64s(&mut out, selected);
            for lane in 0..lanes {
                let want = if $cmp(a[lane], b[lane]) { 1.0 } else { 0.0 };
                assert_eq!(out[lane], want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("eq", simd.eq_f64s(av, bv), |x, y| x == y);
    check!("ne", simd.ne_f64s(av, bv), |x, y| x != y);
    check!("lt", simd.lt_f64s(av, bv), |x, y| x < y);
    check!("le", simd.le_f64s(av, bv), |x, y| x <= y);
    check!("gt", simd.gt_f64s(av, bv), |x, y| x > y);
    check!("ge", simd.ge_f64s(av, bv), |x, y| x >= y);

    let all = simd.eq_f64s(av, av);
    let none = simd.lt_f64s(av, av);
    assert!(all.all(), "f64 mask all");
    assert!(all.any(), "f64 mask any");
    assert!(!all.none(), "f64 mask none");
    assert!(none.none(), "f64 empty mask none");
    assert!(!none.any(), "f64 empty mask any");
    assert!(!none.all(), "f64 empty mask all");
    assert!(all.and(all).all(), "f64 mask and");
    assert!(all.and(none).none(), "f64 mask and empty");
    assert!(none.or(all).all(), "f64 mask or");
    assert!(none.not().all(), "f64 mask not");
    assert!(all.not().none(), "f64 mask not all");

    // Mixed (partially set) masks must survive and/or/not/select correctly.
    if lanes >= 2 {
        let left: Vec<f64> = (0..lanes).map(|lane| lane as f64).collect();
        let right: Vec<f64> = (0..lanes).map(|lane| (lane + lane % 2) as f64).collect();
        let lv = simd.load_f64s(&left);
        let rv = simd.load_f64s(&right);
        let mixed = simd.eq_f64s(lv, rv);

        assert!(mixed.any(), "f64 mixed mask any");
        assert!(!mixed.all(), "f64 mixed mask all");
        assert!(!mixed.none(), "f64 mixed mask none");

        simd.store_f64s(&mut out, simd.select_f64s(mixed, one, zero));
        for (lane, &value) in out.iter().enumerate() {
            let want = if lane % 2 == 0 { 1.0 } else { 0.0 };
            assert_eq!(value, want, "f64 mixed select lane {lane}");
        }

        let full = simd.eq_f64s(lv, lv);
        assert!(!mixed.and(full).all(), "f64 mixed and all");
        assert!(mixed.or(full).all(), "f64 mixed or all");
        assert!(mixed.not().any(), "f64 mixed not any");
        assert!(mixed.and(full.not()).none(), "f64 mixed and empty");
    }
}

#[inline(always)]
fn check_bits_f32<S: Simd>(simd: S) {
    let lanes = S::f32s_lanes;
    // Keep the exponent bits small enough that all bitwise results stay finite.
    let a: Vec<f32> = (0..lanes).map(|lane| 1.0 + (lane % 8) as f32 * 0.125).collect();
    let b: Vec<f32> = (0..lanes).map(|lane| 0.5 + (lane % 8) as f32 * 0.0625).collect();
    let av = simd.load_f32s(&a);
    let bv = simd.load_f32s(&b);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f32s(&mut out, $value);
            for lane in 0..lanes {
                let want: u32 = $expected(a[lane], b[lane]);
                assert_eq!(out[lane].to_bits(), want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("bitand", simd.bitand_f32s(av, bv), |x: f32, y: f32| x.to_bits() & y.to_bits());
    check!("bitor", simd.bitor_f32s(av, bv), |x: f32, y: f32| x.to_bits() | y.to_bits());
    check!("bitxor", simd.bitxor_f32s(av, bv), |x: f32, y: f32| x.to_bits() ^ y.to_bits());
    check!("bitandnot", simd.bitandnot_f32s(av, bv), |x: f32, y: f32| !x.to_bits() & y.to_bits());

    simd.store_f32s(&mut out, simd.bitnot_f32s(av));
    for lane in 0..lanes {
        assert_eq!(out[lane].to_bits(), !a[lane].to_bits(), "bitnot f32 lane {lane}");
    }
}

#[inline(always)]
fn check_bits_f64<S: Simd>(simd: S) {
    let lanes = S::f64s_lanes;
    let a: Vec<f64> = (0..lanes).map(|lane| 1.0 + (lane % 8) as f64 * 0.125).collect();
    let b: Vec<f64> = (0..lanes).map(|lane| 0.5 + (lane % 8) as f64 * 0.0625).collect();
    let av = simd.load_f64s(&a);
    let bv = simd.load_f64s(&b);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f64s(&mut out, $value);
            for lane in 0..lanes {
                let want: u64 = $expected(a[lane], b[lane]);
                assert_eq!(out[lane].to_bits(), want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("bitand", simd.bitand_f64s(av, bv), |x: f64, y: f64| x.to_bits() & y.to_bits());
    check!("bitor", simd.bitor_f64s(av, bv), |x: f64, y: f64| x.to_bits() | y.to_bits());
    check!("bitxor", simd.bitxor_f64s(av, bv), |x: f64, y: f64| x.to_bits() ^ y.to_bits());
    check!("bitandnot", simd.bitandnot_f64s(av, bv), |x: f64, y: f64| !x.to_bits() & y.to_bits());

    simd.store_f64s(&mut out, simd.bitnot_f64s(av));
    for lane in 0..lanes {
        assert_eq!(out[lane].to_bits(), !a[lane].to_bits(), "bitnot f64 lane {lane}");
    }
}

#[inline(always)]
fn check_reduce_f32<S: Simd>(simd: S) {
    let lanes = S::f32s_lanes;
    let values: Vec<f32> = (0..lanes).map(|lane| lane as f32 - 2.0).collect();
    let vector = simd.load_f32s(&values);

    assert_eq!(simd.reduce_sum_f32s(vector), values.iter().sum::<f32>(), "reduce_sum f32");
    assert_eq!(simd.reduce_max_f32s(vector), values.iter().copied().fold(f32::NEG_INFINITY, f32::max), "reduce_max f32");
    assert_eq!(simd.reduce_min_f32s(vector), values.iter().copied().fold(f32::INFINITY, f32::min), "reduce_min f32");
}

#[inline(always)]
fn check_reduce_f64<S: Simd>(simd: S) {
    let lanes = S::f64s_lanes;
    let values: Vec<f64> = (0..lanes).map(|lane| lane as f64 - 2.0).collect();
    let vector = simd.load_f64s(&values);

    assert_eq!(simd.reduce_sum_f64s(vector), values.iter().sum::<f64>(), "reduce_sum f64");
    assert_eq!(simd.reduce_max_f64s(vector), values.iter().copied().fold(f64::NEG_INFINITY, f64::max), "reduce_max f64");
    assert_eq!(simd.reduce_min_f64s(vector), values.iter().copied().fold(f64::INFINITY, f64::min), "reduce_min f64");
}

pub const SUITE_N: usize = 64;

fn suite_a_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| (index % 17) as f32 - 8.0).collect()
}

fn suite_b_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| ((index % 7) + 1) as f32 * 0.25).collect()
}

fn suite_c_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| ((index % 5) as f32 - 2.0) * 0.5).collect()
}

fn suite_sqrt_f32() -> Vec<f32> {
    (0..SUITE_N)
        .map(|index| {
            let root = (index % 16 + 1) as f32;
            root * root
        })
        .collect()
}

fn suite_reduce_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| (index % 7) as f32).collect()
}

fn suite_bits_a_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| 1.0 + (index % 8) as f32 * 0.125).collect()
}

fn suite_bits_b_f32() -> Vec<f32> {
    (0..SUITE_N).map(|index| 0.5 + (index % 8) as f32 * 0.0625).collect()
}

fn suite_a_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| (index % 31) as f64 - 15.0).collect()
}

fn suite_b_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| ((index % 7) + 1) as f64 * 0.25).collect()
}

fn suite_c_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| ((index % 5) as f64 - 2.0) * 0.5).collect()
}

fn suite_sqrt_f64() -> Vec<f64> {
    (0..SUITE_N)
        .map(|index| {
            let root = (index % 16 + 1) as f64;
            root * root
        })
        .collect()
}

fn suite_reduce_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| (index % 7) as f64).collect()
}

fn suite_bits_a_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| 1.0 + (index % 8) as f64 * 0.125).collect()
}

fn suite_bits_b_f64() -> Vec<f64> {
    (0..SUITE_N).map(|index| 0.5 + (index % 8) as f64 * 0.0625).collect()
}

/// Results of the generic f32 suite.
///
/// Each outer vector holds one operation group; every inner vector has
/// `SUITE_N` entries, so the layout is independent of the backend's lane count.
#[derive(Clone, Debug)]
pub struct SuiteF32 {
    pub floats: Vec<Vec<f32>>,
    pub masks: Vec<Vec<u8>>,
    pub bits: Vec<Vec<u32>>,
    pub sum: f32,
    pub max: f32,
    pub min: f32,
}

/// Results of the generic f64 suite.
///
/// Each outer vector holds one operation group; every inner vector has
/// `SUITE_N` entries, so the layout is independent of the backend's lane count.
#[derive(Clone, Debug)]
pub struct SuiteF64 {
    pub floats: Vec<Vec<f64>>,
    pub masks: Vec<Vec<u8>>,
    pub bits: Vec<Vec<u64>>,
    pub sum: f64,
    pub max: f64,
    pub min: f64,
}

/// Combined results for both scalar widths.
#[derive(Clone, Debug)]
pub struct Suite {
    pub single: SuiteF32,
    pub double: SuiteF64,
}

/// Executes the full generic suite on a single backend.
pub struct SuiteOp;

impl SimdFn for SuiteOp {
    type Result = Suite;

    #[inline(always)]
    fn call<S: Simd>(self, simd: S) -> Suite {
        Suite {
            single: suite_f32(simd),
            double: suite_f64(simd),
        }
    }
}

#[inline(always)]
pub fn suite_f32<S: Simd>(simd: S) -> SuiteF32 {
    let lanes = S::f32s_lanes;
    assert_eq!(SUITE_N % lanes, 0, "suite length must be divisible by f32 lane count");

    let a = suite_a_f32();
    let b = suite_b_f32();
    let c = suite_c_f32();
    let sqrt_input = suite_sqrt_f32();
    let reduce_input = suite_reduce_f32();
    let bits_a = suite_bits_a_f32();
    let bits_b = suite_bits_b_f32();

    let one = simd.splat_f32s(1.0);
    let zero = simd.splat_f32s(0.0);

    let mut floats: Vec<Vec<f32>> = (0..13).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut masks: Vec<Vec<u8>> = (0..6).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut bits: Vec<Vec<u32>> = (0..5).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut buffer = vec![0.0f32; lanes];

    let mut sum = 0.0f32;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;

    macro_rules! emit_float {
        ($group:expr, $value:expr) => {{
            simd.store_f32s(&mut buffer, $value);
            floats[$group].extend_from_slice(&buffer);
        }};
    }

    macro_rules! emit_mask {
        ($group:expr, $value:expr) => {{
            simd.store_f32s(&mut buffer, $value);
            masks[$group].extend(buffer.iter().map(|&value| u8::from(value != 0.0)));
        }};
    }

    macro_rules! emit_bits {
        ($group:expr, $value:expr) => {{
            simd.store_f32s(&mut buffer, $value);
            bits[$group].extend(buffer.iter().map(|value| value.to_bits()));
        }};
    }

    for start in (0..SUITE_N).step_by(lanes) {
        let end = start + lanes;
        let av = simd.load_f32s(&a[start..end]);
        let bv = simd.load_f32s(&b[start..end]);
        let cv = simd.load_f32s(&c[start..end]);
        let qv = simd.load_f32s(&sqrt_input[start..end]);
        let rv = simd.load_f32s(&reduce_input[start..end]);
        let ba = simd.load_f32s(&bits_a[start..end]);
        let bb = simd.load_f32s(&bits_b[start..end]);

        emit_float!(0, simd.add_f32s(av, bv));
        emit_float!(1, simd.sub_f32s(av, bv));
        emit_float!(2, simd.mul_f32s(av, bv));
        emit_float!(3, simd.div_f32s(av, bv));
        emit_float!(4, simd.min_f32s(av, bv));
        emit_float!(5, simd.max_f32s(av, bv));
        emit_float!(6, simd.abs_f32s(av));
        emit_float!(7, simd.neg_f32s(av));
        emit_float!(8, simd.sqrt_f32s(qv));
        emit_float!(9, simd.mul_add_f32s(av, bv, cv));
        emit_float!(10, simd.mul_sub_f32s(av, bv, cv));
        emit_float!(11, simd.nmul_add_f32s(av, bv, cv));
        emit_float!(12, simd.nmul_sub_f32s(av, bv, cv));

        emit_mask!(0, simd.select_f32s(simd.eq_f32s(av, bv), one, zero));
        emit_mask!(1, simd.select_f32s(simd.ne_f32s(av, bv), one, zero));
        emit_mask!(2, simd.select_f32s(simd.lt_f32s(av, bv), one, zero));
        emit_mask!(3, simd.select_f32s(simd.le_f32s(av, bv), one, zero));
        emit_mask!(4, simd.select_f32s(simd.gt_f32s(av, bv), one, zero));
        emit_mask!(5, simd.select_f32s(simd.ge_f32s(av, bv), one, zero));

        emit_bits!(0, simd.bitand_f32s(ba, bb));
        emit_bits!(1, simd.bitor_f32s(ba, bb));
        emit_bits!(2, simd.bitxor_f32s(ba, bb));
        emit_bits!(3, simd.bitandnot_f32s(ba, bb));
        emit_bits!(4, simd.bitnot_f32s(ba));

        sum += simd.reduce_sum_f32s(rv);
        max = max.max(simd.reduce_max_f32s(rv));
        min = min.min(simd.reduce_min_f32s(rv));
    }

    SuiteF32 {
        floats,
        masks,
        bits,
        sum,
        max,
        min,
    }
}

#[inline(always)]
pub fn suite_f64<S: Simd>(simd: S) -> SuiteF64 {
    let lanes = S::f64s_lanes;
    assert_eq!(SUITE_N % lanes, 0, "suite length must be divisible by f64 lane count");

    let a = suite_a_f64();
    let b = suite_b_f64();
    let c = suite_c_f64();
    let sqrt_input = suite_sqrt_f64();
    let reduce_input = suite_reduce_f64();
    let bits_a = suite_bits_a_f64();
    let bits_b = suite_bits_b_f64();

    let one = simd.splat_f64s(1.0);
    let zero = simd.splat_f64s(0.0);

    let mut floats: Vec<Vec<f64>> = (0..13).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut masks: Vec<Vec<u8>> = (0..6).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut bits: Vec<Vec<u64>> = (0..5).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut buffer = vec![0.0f64; lanes];

    let mut sum = 0.0f64;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;

    macro_rules! emit_float {
        ($group:expr, $value:expr) => {{
            simd.store_f64s(&mut buffer, $value);
            floats[$group].extend_from_slice(&buffer);
        }};
    }

    macro_rules! emit_mask {
        ($group:expr, $value:expr) => {{
            simd.store_f64s(&mut buffer, $value);
            masks[$group].extend(buffer.iter().map(|&value| u8::from(value != 0.0)));
        }};
    }

    macro_rules! emit_bits {
        ($group:expr, $value:expr) => {{
            simd.store_f64s(&mut buffer, $value);
            bits[$group].extend(buffer.iter().map(|value| value.to_bits()));
        }};
    }

    for start in (0..SUITE_N).step_by(lanes) {
        let end = start + lanes;
        let av = simd.load_f64s(&a[start..end]);
        let bv = simd.load_f64s(&b[start..end]);
        let cv = simd.load_f64s(&c[start..end]);
        let qv = simd.load_f64s(&sqrt_input[start..end]);
        let rv = simd.load_f64s(&reduce_input[start..end]);
        let ba = simd.load_f64s(&bits_a[start..end]);
        let bb = simd.load_f64s(&bits_b[start..end]);

        emit_float!(0, simd.add_f64s(av, bv));
        emit_float!(1, simd.sub_f64s(av, bv));
        emit_float!(2, simd.mul_f64s(av, bv));
        emit_float!(3, simd.div_f64s(av, bv));
        emit_float!(4, simd.min_f64s(av, bv));
        emit_float!(5, simd.max_f64s(av, bv));
        emit_float!(6, simd.abs_f64s(av));
        emit_float!(7, simd.neg_f64s(av));
        emit_float!(8, simd.sqrt_f64s(qv));
        emit_float!(9, simd.mul_add_f64s(av, bv, cv));
        emit_float!(10, simd.mul_sub_f64s(av, bv, cv));
        emit_float!(11, simd.nmul_add_f64s(av, bv, cv));
        emit_float!(12, simd.nmul_sub_f64s(av, bv, cv));

        emit_mask!(0, simd.select_f64s(simd.eq_f64s(av, bv), one, zero));
        emit_mask!(1, simd.select_f64s(simd.ne_f64s(av, bv), one, zero));
        emit_mask!(2, simd.select_f64s(simd.lt_f64s(av, bv), one, zero));
        emit_mask!(3, simd.select_f64s(simd.le_f64s(av, bv), one, zero));
        emit_mask!(4, simd.select_f64s(simd.gt_f64s(av, bv), one, zero));
        emit_mask!(5, simd.select_f64s(simd.ge_f64s(av, bv), one, zero));

        emit_bits!(0, simd.bitand_f64s(ba, bb));
        emit_bits!(1, simd.bitor_f64s(ba, bb));
        emit_bits!(2, simd.bitxor_f64s(ba, bb));
        emit_bits!(3, simd.bitandnot_f64s(ba, bb));
        emit_bits!(4, simd.bitnot_f64s(ba));

        sum += simd.reduce_sum_f64s(rv);
        max = max.max(simd.reduce_max_f64s(rv));
        min = min.min(simd.reduce_min_f64s(rv));
    }

    SuiteF64 {
        floats,
        masks,
        bits,
        sum,
        max,
        min,
    }
}
