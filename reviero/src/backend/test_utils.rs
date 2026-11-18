//! Shared helpers for backend tests.
//!
//! The `check_*` functions are generic over `Simd` and are meant to be called
//! from a `SimdFn` implementation that is itself executed inside a backend's
//! `vectorize` wrapper, so the target features are enabled for the whole call.

use crate::SimdFn;
use crate::traits::{FloatSimd, HalfSimd, Simd, SimdMask};
use half::{bf16, f16};

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
                use $crate::traits::{FloatSimd, HalfSimd};

                assert_eq!(<$backend as FloatSimd>::f32_lanes, $f32_lanes);
                assert_eq!(<$backend as FloatSimd>::f64_lanes, $f64_lanes);
                assert_eq!(<$backend as HalfSimd>::f16_lanes, $f32_lanes);
                assert_eq!(<$backend as HalfSimd>::bf16_lanes, $f32_lanes);
                assert_eq!(<$backend as HalfSimd>::m16_lanes, $f32_lanes);
            }

            #[test]
            fn as_simd() {
                super::run($crate::backend::test_utils::Check::Utils);
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

            #[test]
            fn half_io() {
                super::run($crate::backend::test_utils::Check::HalfIo);
            }

            #[test]
            fn half_convert() {
                super::run($crate::backend::test_utils::Check::HalfConvert);
            }

            #[test]
            fn half_float_ops() {
                super::run($crate::backend::test_utils::Check::HalfFloat);
            }

            #[test]
            fn half_mask_ops() {
                super::run($crate::backend::test_utils::Check::HalfMask);
            }

            #[test]
            fn half_bit_ops() {
                super::run($crate::backend::test_utils::Check::HalfBits);
            }

            #[test]
            fn half_reduce_ops() {
                super::run($crate::backend::test_utils::Check::HalfReduce);
            }
        }
    };
}

/// The operation group executed by [`Run`].
#[derive(Clone, Copy, Debug)]
pub enum Check {
    Utils,
    Io,
    Float,
    Mask,
    Bits,
    Reduce,
    HalfIo,
    HalfConvert,
    HalfFloat,
    HalfMask,
    HalfBits,
    HalfReduce,
}

/// Adapter that lets a [`Check`] be executed through any backend.
pub struct Run(pub Check);

impl SimdFn for Run {
    type Result = ();

    #[inline(always)]
    fn call<S: Simd>(self, simd: S) {
        match self.0 {
            Check::Utils => {
                check_utils_f32(simd);
                check_utils_f64(simd);
                check_utils_f16(simd);
                check_utils_bf16(simd);
            }
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
            Check::HalfIo => {
                check_io_f16(simd);
                check_io_bf16(simd);
            }
            Check::HalfConvert => {
                check_convert_f16(simd);
                check_convert_bf16(simd);
            }
            Check::HalfFloat => {
                check_float_f16(simd);
                check_float_bf16(simd);
            }
            Check::HalfMask => {
                check_mask_f16(simd);
                check_mask_bf16(simd);
            }
            Check::HalfBits => {
                check_bits_f16(simd);
                check_bits_bf16(simd);
            }
            Check::HalfReduce => {
                check_reduce_f16(simd);
                check_reduce_bf16(simd);
            }
        }
    }
}

macro_rules! define_utils_check {
    ($name:ident, $ty:ident, $lanes:ident, $convert:expr) => {
        #[inline(always)]
        fn $name<S: Simd>(simd: S) {
            let lanes = S::$lanes;
            let len = lanes * 3 + lanes / 2 + 1;
            let data: Vec<$ty> = (0..=len).map(|index| ($convert)(index)).collect();

            // Start one element into the buffer so the loads are not naturally
            // aligned to the backend's vector width.
            let src = &data[1..];

            paste::paste! {
                let (vectors, tail) = crate::utils::[<as_simd_ $ty>](&simd, src);
                let full = src.len() / lanes * lanes;

                assert_eq!(vectors.len(), full / lanes, concat!(stringify!($ty), " vector count"));
                assert_eq!(tail.len(), src.len() - full, concat!(stringify!($ty), " tail length"));

                let mut out = vec![<$ty>::default(); lanes];
                for (index, vector) in vectors.enumerate() {
                    simd.[<store_ $ty>](&mut out, vector);
                    assert_eq!(
                        out.as_slice(),
                        &src[index * lanes..(index + 1) * lanes],
                        concat!(stringify!($ty), " vector #{} contents"),
                        index
                    );
                }

                assert_eq!(tail, &src[full..], concat!(stringify!($ty), " tail contents"));

                let (vectors, tail) = crate::utils::[<as_simd_ $ty>](&simd, &[]);
                assert_eq!(vectors.len(), 0, concat!(stringify!($ty), " empty vector count"));
                assert!(tail.is_empty(), concat!(stringify!($ty), " empty tail"));
            }
        }
    };
}

define_utils_check!(check_utils_f32, f32, f32_lanes, |index: usize| index as f32 + 0.25);
define_utils_check!(check_utils_f64, f64, f64_lanes, |index: usize| index as f64 + 0.25);
define_utils_check!(check_utils_f16, f16, f16_lanes, |index: usize| f16::from_f32(index as f32 + 0.25));
define_utils_check!(check_utils_bf16, bf16, bf16_lanes, |index: usize| bf16::from_f32(index as f32 + 0.25));

#[inline(always)]
fn check_io_f32<S: Simd>(simd: S) {
    let lanes = S::f32_lanes;
    let src: Vec<f32> = (0..lanes).map(|lane| lane as f32 * 0.5 - 2.0).collect();

    let value = simd.load_f32(&src);
    let mut dst = vec![f32::NAN; lanes];
    simd.store_f32(&mut dst, value);
    assert_eq!(dst, src, "f32 load/store");

    let value = simd.splat_f32(1.25);
    simd.store_f32(&mut dst, value);
    assert!(dst.iter().all(|&value| value == 1.25), "f32 splat: {dst:?}");
}

#[inline(always)]
fn check_io_f64<S: Simd>(simd: S) {
    let lanes = S::f64_lanes;
    let src: Vec<f64> = (0..lanes).map(|lane| lane as f64 * 0.5 - 2.0).collect();

    let value = simd.load_f64(&src);
    let mut dst = vec![f64::NAN; lanes];
    simd.store_f64(&mut dst, value);
    assert_eq!(dst, src, "f64 load/store");

    let value = simd.splat_f64(1.25);
    simd.store_f64(&mut dst, value);
    assert!(dst.iter().all(|&value| value == 1.25), "f64 splat: {dst:?}");
}

#[inline(always)]
fn check_float_f32<S: Simd>(simd: S) {
    let lanes = S::f32_lanes;
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

    let av = simd.load_f32(&a);
    let bv = simd.load_f32(&b);
    let cv = simd.load_f32(&c);
    let sv = simd.load_f32(&signed);
    let qv = simd.load_f32(&squares);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f32(&mut out, $value);
            let expected: Vec<f32> = $expected;
            for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("add", simd.add_f32(av, bv), a.iter().map(|value| value + 2.0).collect::<Vec<_>>());
    check!("sub", simd.sub_f32(av, bv), a.iter().map(|value| value - 2.0).collect::<Vec<_>>());
    check!("mul", simd.mul_f32(av, bv), a.iter().map(|value| value * 2.0).collect::<Vec<_>>());
    check!("div", simd.div_f32(av, bv), a.iter().map(|value| value / 2.0).collect::<Vec<_>>());
    check!("min", simd.min_f32(av, bv), a.iter().map(|value| value.min(2.0)).collect::<Vec<_>>());
    check!("max", simd.max_f32(av, bv), a.iter().map(|value| value.max(2.0)).collect::<Vec<_>>());
    check!("abs", simd.abs_f32(sv), signed.iter().map(|value| value.abs()).collect::<Vec<_>>());
    check!("neg", simd.neg_f32(sv), signed.iter().map(|value| -value).collect::<Vec<_>>());
    check!("sqrt", simd.sqrt_f32(qv), (1..=lanes).map(|value| value as f32).collect::<Vec<_>>());
    check!("mul_add", simd.mul_add_f32(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 + c[lane]).collect::<Vec<_>>());
    check!("mul_sub", simd.mul_sub_f32(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 - c[lane]).collect::<Vec<_>>());
    check!(
        "nmul_add",
        simd.nmul_add_f32(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) + c[lane]).collect::<Vec<_>>()
    );
    check!(
        "nmul_sub",
        simd.nmul_sub_f32(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) - c[lane]).collect::<Vec<_>>()
    );

    // Cancelling to zero must keep the sign of zero consistent with `a*b - c`
    // and `-(a*b) - c` (i.e. +0.0), not just match under float equality.
    let ones = vec![1.0f32; lanes];
    let neg_ones = vec![-1.0f32; lanes];
    let ones_v = simd.load_f32(&ones);
    let neg_ones_v = simd.load_f32(&neg_ones);

    simd.store_f32(&mut out, simd.mul_sub_f32(ones_v, ones_v, ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f32.to_bits(), "mul_sub f32 zero sign lane {lane}");
    }

    simd.store_f32(&mut out, simd.nmul_sub_f32(ones_v, ones_v, neg_ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f32.to_bits(), "nmul_sub f32 zero sign lane {lane}");
    }
}

#[inline(always)]
fn check_float_f64<S: Simd>(simd: S) {
    let lanes = S::f64_lanes;
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

    let av = simd.load_f64(&a);
    let bv = simd.load_f64(&b);
    let cv = simd.load_f64(&c);
    let sv = simd.load_f64(&signed);
    let qv = simd.load_f64(&squares);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f64(&mut out, $value);
            let expected: Vec<f64> = $expected;
            for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("add", simd.add_f64(av, bv), a.iter().map(|value| value + 2.0).collect::<Vec<_>>());
    check!("sub", simd.sub_f64(av, bv), a.iter().map(|value| value - 2.0).collect::<Vec<_>>());
    check!("mul", simd.mul_f64(av, bv), a.iter().map(|value| value * 2.0).collect::<Vec<_>>());
    check!("div", simd.div_f64(av, bv), a.iter().map(|value| value / 2.0).collect::<Vec<_>>());
    check!("min", simd.min_f64(av, bv), a.iter().map(|value| value.min(2.0)).collect::<Vec<_>>());
    check!("max", simd.max_f64(av, bv), a.iter().map(|value| value.max(2.0)).collect::<Vec<_>>());
    check!("abs", simd.abs_f64(sv), signed.iter().map(|value| value.abs()).collect::<Vec<_>>());
    check!("neg", simd.neg_f64(sv), signed.iter().map(|value| -value).collect::<Vec<_>>());
    check!("sqrt", simd.sqrt_f64(qv), (1..=lanes).map(|value| value as f64).collect::<Vec<_>>());
    check!("mul_add", simd.mul_add_f64(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 + c[lane]).collect::<Vec<_>>());
    check!("mul_sub", simd.mul_sub_f64(av, bv, cv), (0..lanes).map(|lane| a[lane] * 2.0 - c[lane]).collect::<Vec<_>>());
    check!(
        "nmul_add",
        simd.nmul_add_f64(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) + c[lane]).collect::<Vec<_>>()
    );
    check!(
        "nmul_sub",
        simd.nmul_sub_f64(av, bv, cv),
        (0..lanes).map(|lane| -(a[lane] * 2.0) - c[lane]).collect::<Vec<_>>()
    );

    // Same zero-sign checks as for f32.
    let ones = vec![1.0f64; lanes];
    let neg_ones = vec![-1.0f64; lanes];
    let ones_v = simd.load_f64(&ones);
    let neg_ones_v = simd.load_f64(&neg_ones);

    simd.store_f64(&mut out, simd.mul_sub_f64(ones_v, ones_v, ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f64.to_bits(), "mul_sub f64 zero sign lane {lane}");
    }

    simd.store_f64(&mut out, simd.nmul_sub_f64(ones_v, ones_v, neg_ones_v));
    for (lane, &value) in out.iter().enumerate() {
        assert_eq!(value.to_bits(), 0.0f64.to_bits(), "nmul_sub f64 zero sign lane {lane}");
    }
}

#[inline(always)]
fn check_mask_f32<S: FloatSimd>(simd: S) {
    let lanes = S::f32_lanes;
    let a: Vec<f32> = (0..lanes).map(|lane| lane as f32).collect();
    let b: Vec<f32> = (0..lanes).map(|lane| (lanes - 1 - lane) as f32).collect();
    let av = simd.load_f32(&a);
    let bv = simd.load_f32(&b);
    let one = simd.splat_f32(1.0);
    let zero = simd.splat_f32(0.0);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $mask:expr, $cmp:expr) => {{
            let selected = simd.select_f32($mask, one, zero);
            simd.store_f32(&mut out, selected);
            for lane in 0..lanes {
                let want = if $cmp(a[lane], b[lane]) { 1.0 } else { 0.0 };
                assert_eq!(out[lane], want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("eq", simd.eq_f32(av, bv), |x, y| x == y);
    check!("ne", simd.ne_f32(av, bv), |x, y| x != y);
    check!("lt", simd.lt_f32(av, bv), |x, y| x < y);
    check!("le", simd.le_f32(av, bv), |x, y| x <= y);
    check!("gt", simd.gt_f32(av, bv), |x, y| x > y);
    check!("ge", simd.ge_f32(av, bv), |x, y| x >= y);

    let all = simd.eq_f32(av, av);
    let none = simd.lt_f32(av, av);
    assert!(all.all(), "f32 mask all");
    assert!(all.any(), "f32 mask any");
    assert!(all.any(), "f32 mask any");
    assert!(!none.any(), "f32 empty mask any");
    assert!(!none.any(), "f32 empty mask any");
    assert!(!none.all(), "f32 empty mask all");
    assert!(all.and(all).all(), "f32 mask and");
    assert!(!all.and(none).any(), "f32 mask and empty");
    assert!(none.or(all).all(), "f32 mask or");
    assert!(none.not().all(), "f32 mask not");
    assert!(!all.not().any(), "f32 mask not all");

    let mask_bits = (1u32 << lanes) - 1;
    assert_eq!(S::m32::splat(true).to_bits(), mask_bits, "f32 splat mask bits");
    assert_eq!(S::m32::splat(false).to_bits(), 0, "f32 empty splat mask bits");
    let bits = 0b1010_0101 & mask_bits;
    assert_eq!(S::m32::from_bits(bits).to_bits(), bits, "f32 mask bit roundtrip");

    // Mixed (partially set) masks must survive and/or/not/select correctly.
    if lanes >= 2 {
        let left: Vec<f32> = (0..lanes).map(|lane| lane as f32).collect();
        let right: Vec<f32> = (0..lanes).map(|lane| (lane + lane % 2) as f32).collect();
        let lv = simd.load_f32(&left);
        let rv = simd.load_f32(&right);
        let mixed = simd.eq_f32(lv, rv);

        assert!(mixed.any(), "f32 mixed mask any");
        assert!(!mixed.all(), "f32 mixed mask all");
        assert!(mixed.any(), "f32 mixed mask any");

        simd.store_f32(&mut out, simd.select_f32(mixed, one, zero));
        for (lane, &value) in out.iter().enumerate() {
            let want = if lane % 2 == 0 { 1.0 } else { 0.0 };
            assert_eq!(value, want, "f32 mixed select lane {lane}");
        }

        let full = simd.eq_f32(lv, lv);
        assert!(!mixed.and(full).all(), "f32 mixed and all");
        assert!(mixed.or(full).all(), "f32 mixed or all");
        assert!(mixed.not().any(), "f32 mixed not any");
        assert!(!mixed.and(full.not()).any(), "f32 mixed and empty");
    }

    // Ordered comparisons involving NaN must follow IEEE semantics on every
    // backend: `eq` is false, `ne` is true and all ordering comparisons are
    // false.
    let nan = f32::NAN;
    let nan_lhs: Vec<f32> = (0..lanes).map(|lane| if lane % 2 == 0 { nan } else { lane as f32 }).collect();
    let nan_rhs: Vec<f32> = (0..lanes).map(|lane| if lane % 2 == 0 { lane as f32 } else { nan }).collect();
    let nan_lhs = simd.load_f32(&nan_lhs);
    let nan_rhs = simd.load_f32(&nan_rhs);

    assert!(!simd.eq_f32(nan_lhs, nan_rhs).any(), "f32 NaN eq");
    assert!(simd.ne_f32(nan_lhs, nan_rhs).all(), "f32 NaN ne");
    assert!(!simd.lt_f32(nan_lhs, nan_rhs).any(), "f32 NaN lt");
    assert!(!simd.le_f32(nan_lhs, nan_rhs).any(), "f32 NaN le");
    assert!(!simd.gt_f32(nan_lhs, nan_rhs).any(), "f32 NaN gt");
    assert!(!simd.ge_f32(nan_lhs, nan_rhs).any(), "f32 NaN ge");
}

#[inline(always)]
fn check_mask_f64<S: FloatSimd>(simd: S) {
    let lanes = S::f64_lanes;
    let a: Vec<f64> = (0..lanes).map(|lane| lane as f64).collect();
    let b: Vec<f64> = (0..lanes).map(|lane| (lanes - 1 - lane) as f64).collect();
    let av = simd.load_f64(&a);
    let bv = simd.load_f64(&b);
    let one = simd.splat_f64(1.0);
    let zero = simd.splat_f64(0.0);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $mask:expr, $cmp:expr) => {{
            let selected = simd.select_f64($mask, one, zero);
            simd.store_f64(&mut out, selected);
            for lane in 0..lanes {
                let want = if $cmp(a[lane], b[lane]) { 1.0 } else { 0.0 };
                assert_eq!(out[lane], want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("eq", simd.eq_f64(av, bv), |x, y| x == y);
    check!("ne", simd.ne_f64(av, bv), |x, y| x != y);
    check!("lt", simd.lt_f64(av, bv), |x, y| x < y);
    check!("le", simd.le_f64(av, bv), |x, y| x <= y);
    check!("gt", simd.gt_f64(av, bv), |x, y| x > y);
    check!("ge", simd.ge_f64(av, bv), |x, y| x >= y);

    let all = simd.eq_f64(av, av);
    let none = simd.lt_f64(av, av);
    assert!(all.all(), "f64 mask all");
    assert!(all.any(), "f64 mask any");
    assert!(all.any(), "f64 mask any");
    assert!(!none.any(), "f64 empty mask any");
    assert!(!none.any(), "f64 empty mask any");
    assert!(!none.all(), "f64 empty mask all");
    assert!(all.and(all).all(), "f64 mask and");
    assert!(!all.and(none).any(), "f64 mask and empty");
    assert!(none.or(all).all(), "f64 mask or");
    assert!(none.not().all(), "f64 mask not");
    assert!(!all.not().any(), "f64 mask not all");

    let mask_bits = (1u32 << lanes) - 1;
    assert_eq!(S::m64::splat(true).to_bits(), mask_bits, "f64 splat mask bits");
    assert_eq!(S::m64::splat(false).to_bits(), 0, "f64 empty splat mask bits");
    let bits = 0b1010_0101 & mask_bits;
    assert_eq!(S::m64::from_bits(bits).to_bits(), bits, "f64 mask bit roundtrip");

    // Mixed (partially set) masks must survive and/or/not/select correctly.
    if lanes >= 2 {
        let left: Vec<f64> = (0..lanes).map(|lane| lane as f64).collect();
        let right: Vec<f64> = (0..lanes).map(|lane| (lane + lane % 2) as f64).collect();
        let lv = simd.load_f64(&left);
        let rv = simd.load_f64(&right);
        let mixed = simd.eq_f64(lv, rv);

        assert!(mixed.any(), "f64 mixed mask any");
        assert!(!mixed.all(), "f64 mixed mask all");
        assert!(mixed.any(), "f64 mixed mask any");

        simd.store_f64(&mut out, simd.select_f64(mixed, one, zero));
        for (lane, &value) in out.iter().enumerate() {
            let want = if lane % 2 == 0 { 1.0 } else { 0.0 };
            assert_eq!(value, want, "f64 mixed select lane {lane}");
        }

        let full = simd.eq_f64(lv, lv);
        assert!(!mixed.and(full).all(), "f64 mixed and all");
        assert!(mixed.or(full).all(), "f64 mixed or all");
        assert!(mixed.not().any(), "f64 mixed not any");
        assert!(!mixed.and(full.not()).any(), "f64 mixed and empty");
    }

    // Same IEEE semantics for NaN as the f32 checks above.
    let nan = f64::NAN;
    let nan_lhs: Vec<f64> = (0..lanes).map(|lane| if lane % 2 == 0 { nan } else { lane as f64 }).collect();
    let nan_rhs: Vec<f64> = (0..lanes).map(|lane| if lane % 2 == 0 { lane as f64 } else { nan }).collect();
    let nan_lhs = simd.load_f64(&nan_lhs);
    let nan_rhs = simd.load_f64(&nan_rhs);

    assert!(!simd.eq_f64(nan_lhs, nan_rhs).any(), "f64 NaN eq");
    assert!(simd.ne_f64(nan_lhs, nan_rhs).all(), "f64 NaN ne");
    assert!(!simd.lt_f64(nan_lhs, nan_rhs).any(), "f64 NaN lt");
    assert!(!simd.le_f64(nan_lhs, nan_rhs).any(), "f64 NaN le");
    assert!(!simd.gt_f64(nan_lhs, nan_rhs).any(), "f64 NaN gt");
    assert!(!simd.ge_f64(nan_lhs, nan_rhs).any(), "f64 NaN ge");
}

#[inline(always)]
fn check_bits_f32<S: Simd>(simd: S) {
    let lanes = S::f32_lanes;
    // Keep the exponent bits small enough that all bitwise results stay finite.
    let a: Vec<f32> = (0..lanes).map(|lane| 1.0 + (lane % 8) as f32 * 0.125).collect();
    let b: Vec<f32> = (0..lanes).map(|lane| 0.5 + (lane % 8) as f32 * 0.0625).collect();
    let av = simd.load_f32(&a);
    let bv = simd.load_f32(&b);

    let mut out = vec![0.0f32; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f32(&mut out, $value);
            for lane in 0..lanes {
                let want: u32 = $expected(a[lane], b[lane]);
                assert_eq!(out[lane].to_bits(), want, concat!($name, " f32 lane {}"), lane);
            }
        }};
    }

    check!("bitand", simd.bitand_f32(av, bv), |x: f32, y: f32| x.to_bits() & y.to_bits());
    check!("bitor", simd.bitor_f32(av, bv), |x: f32, y: f32| x.to_bits() | y.to_bits());
    check!("bitxor", simd.bitxor_f32(av, bv), |x: f32, y: f32| x.to_bits() ^ y.to_bits());
    check!("bitandnot", simd.bitandnot_f32(av, bv), |x: f32, y: f32| !x.to_bits() & y.to_bits());

    simd.store_f32(&mut out, simd.bitnot_f32(av));
    for lane in 0..lanes {
        assert_eq!(out[lane].to_bits(), !a[lane].to_bits(), "bitnot f32 lane {lane}");
    }
}

#[inline(always)]
fn check_bits_f64<S: Simd>(simd: S) {
    let lanes = S::f64_lanes;
    let a: Vec<f64> = (0..lanes).map(|lane| 1.0 + (lane % 8) as f64 * 0.125).collect();
    let b: Vec<f64> = (0..lanes).map(|lane| 0.5 + (lane % 8) as f64 * 0.0625).collect();
    let av = simd.load_f64(&a);
    let bv = simd.load_f64(&b);

    let mut out = vec![0.0f64; lanes];
    macro_rules! check {
        ($name:literal, $value:expr, $expected:expr) => {{
            simd.store_f64(&mut out, $value);
            for lane in 0..lanes {
                let want: u64 = $expected(a[lane], b[lane]);
                assert_eq!(out[lane].to_bits(), want, concat!($name, " f64 lane {}"), lane);
            }
        }};
    }

    check!("bitand", simd.bitand_f64(av, bv), |x: f64, y: f64| x.to_bits() & y.to_bits());
    check!("bitor", simd.bitor_f64(av, bv), |x: f64, y: f64| x.to_bits() | y.to_bits());
    check!("bitxor", simd.bitxor_f64(av, bv), |x: f64, y: f64| x.to_bits() ^ y.to_bits());
    check!("bitandnot", simd.bitandnot_f64(av, bv), |x: f64, y: f64| !x.to_bits() & y.to_bits());

    simd.store_f64(&mut out, simd.bitnot_f64(av));
    for lane in 0..lanes {
        assert_eq!(out[lane].to_bits(), !a[lane].to_bits(), "bitnot f64 lane {lane}");
    }
}

#[inline(always)]
fn check_reduce_f32<S: Simd>(simd: S) {
    let lanes = S::f32_lanes;
    let values: Vec<f32> = (0..lanes).map(|lane| lane as f32 - 2.0).collect();
    let vector = simd.load_f32(&values);

    assert_eq!(simd.reduce_sum_f32(vector), values.iter().sum::<f32>(), "reduce_sum f32");
    assert_eq!(simd.reduce_max_f32(vector), values.iter().copied().fold(f32::NEG_INFINITY, f32::max), "reduce_max f32");
    assert_eq!(simd.reduce_min_f32(vector), values.iter().copied().fold(f32::INFINITY, f32::min), "reduce_min f32");
}

#[inline(always)]
fn check_reduce_f64<S: Simd>(simd: S) {
    let lanes = S::f64_lanes;
    let values: Vec<f64> = (0..lanes).map(|lane| lane as f64 - 2.0).collect();
    let vector = simd.load_f64(&values);

    assert_eq!(simd.reduce_sum_f64(vector), values.iter().sum::<f64>(), "reduce_sum f64");
    assert_eq!(simd.reduce_max_f64(vector), values.iter().copied().fold(f64::NEG_INFINITY, f64::max), "reduce_max f64");
    assert_eq!(simd.reduce_min_f64(vector), values.iter().copied().fold(f64::INFINITY, f64::min), "reduce_min f64");
}

macro_rules! define_half_io_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            let src: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(lane as f32 * 0.5 - 2.0)).collect();

            paste::paste! {
                let value = simd.[<load_ $ty>](&src);
                let mut dst = vec![<$ty>::from_f32(f32::NAN); lanes];
                simd.[<store_ $ty>](&mut dst, value);
                assert_eq!(dst, src, concat!(stringify!($ty), " load/store"));

                let value = simd.[<splat_ $ty>](<$ty>::from_f32(1.25));
                simd.[<store_ $ty>](&mut dst, value);
                assert!(
                    dst.iter().all(|&value| value == <$ty>::from_f32(1.25)),
                    concat!(stringify!($ty), " splat: {:?}"),
                    dst
                );
            }
        }
    };
}

define_half_io_check!(check_io_f16, f16, f16_lanes);
define_half_io_check!(check_io_bf16, bf16, bf16_lanes);

macro_rules! define_half_convert_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            if S::f32_lanes != lanes {
                return;
            }

            let src: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32((lane as f32 - 3.5) * 0.25)).collect();
            let raw: Vec<f32> = (0..lanes).map(|lane| 1.0 + (lane as f32 + 1.0) / 1024.0).collect();
            let expected: Vec<$ty> = raw.iter().map(|&value| <$ty>::from_f32(value)).collect();

            paste::paste! {
                let vector = simd.[<load_ $ty>](&src);
                let wide = simd.[<$ty _to_f32>](vector);

                let mut out32 = vec![0.0f32; lanes];
                simd.store_f32(&mut out32, wide);
                for (lane, (&got, &want)) in out32.iter().zip(src.iter()).enumerate() {
                    assert_eq!(got, want.to_f32(), concat!(stringify!($ty), "_to_f32 lane {}"), lane);
                }

                let mut out = vec![<$ty>::from_f32(0.0); lanes];
                simd.[<store_ $ty>](&mut out, simd.[<f32_to_ $ty>](wide));
                assert_eq!(out, src, concat!("f32_to_", stringify!($ty), " roundtrip"));

                let raw = simd.load_f32(&raw);
                simd.[<store_ $ty>](&mut out, simd.[<f32_to_ $ty>](raw));
                for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(got, want, concat!("f32_to_", stringify!($ty), " rounding lane {}"), lane);
                }
            }

            let lane_mask = (1u32 << lanes) - 1;
            let bits = 0b1010_0101 & lane_mask;
            let mask = S::m16::from_bits(bits);
            assert_eq!(mask.to_bits() & lane_mask, bits, concat!(stringify!($ty), " mask from_bits"));
            let mask32 = simd.m16_to_m32(mask);
            assert_eq!(mask32.to_bits() & ((1u32 << S::f32_lanes) - 1), bits, concat!(stringify!($ty), " m16_to_m32"));
            assert_eq!(simd.m32_to_m16(mask32).to_bits() & lane_mask, bits, concat!(stringify!($ty), " m32_to_m16"));
        }
    };
}

define_half_convert_check!(check_convert_f16, f16, f16_lanes);
define_half_convert_check!(check_convert_bf16, bf16, bf16_lanes);

macro_rules! define_half_float_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            let a: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32((lane + 1) as f32)).collect();
            let b: Vec<$ty> = vec![<$ty>::from_f32(2.0); lanes];
            let c: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(lane as f32 * 0.5)).collect();
            let signed: Vec<$ty> = (0..lanes)
                .map(|lane| if lane % 2 == 0 { a[lane] } else { <$ty>::from_f32(-a[lane].to_f32()) })
                .collect();
            let squares: Vec<$ty> = (0..lanes)
                .map(|lane| {
                    let root = (lane + 1) as f32;
                    <$ty>::from_f32(root * root)
                })
                .collect();

            paste::paste! {
                let av = simd.[<load_ $ty>](&a);
                let bv = simd.[<load_ $ty>](&b);
                let cv = simd.[<load_ $ty>](&c);
                let sv = simd.[<load_ $ty>](&signed);
                let qv = simd.[<load_ $ty>](&squares);

                let mut out = vec![<$ty>::from_f32(0.0); lanes];
                macro_rules! check {
                    ($label:literal, $value:expr, $expected:expr) => {{
                        simd.[<store_ $ty>](&mut out, $value);
                        let expected: Vec<$ty> = $expected;
                        for (lane, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
                            assert_eq!(got, want, concat!($label, " ", stringify!($ty), " lane {}"), lane);
                        }
                    }};
                }

                check!(
                    "add",
                    simd.[<add_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32() + 2.0)).collect::<Vec<_>>()
                );
                check!(
                    "sub",
                    simd.[<sub_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32() - 2.0)).collect::<Vec<_>>()
                );
                check!(
                    "mul",
                    simd.[<mul_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32() * 2.0)).collect::<Vec<_>>()
                );
                check!(
                    "div",
                    simd.[<div_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32() / 2.0)).collect::<Vec<_>>()
                );
                check!(
                    "min",
                    simd.[<min_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32().min(2.0))).collect::<Vec<_>>()
                );
                check!(
                    "max",
                    simd.[<max_ $ty>](av, bv),
                    a.iter().map(|value| <$ty>::from_f32(value.to_f32().max(2.0))).collect::<Vec<_>>()
                );
                check!(
                    "abs",
                    simd.[<abs_ $ty>](sv),
                    signed.iter().map(|value| <$ty>::from_f32(value.to_f32().abs())).collect::<Vec<_>>()
                );
                check!(
                    "neg",
                    simd.[<neg_ $ty>](sv),
                    signed.iter().map(|value| <$ty>::from_f32(-value.to_f32())).collect::<Vec<_>>()
                );
                check!(
                    "sqrt",
                    simd.[<sqrt_ $ty>](qv),
                    (1..=lanes).map(|value| <$ty>::from_f32(value as f32)).collect::<Vec<_>>()
                );
                check!(
                    "mul_add",
                    simd.[<mul_add_ $ty>](av, bv, cv),
                    (0..lanes).map(|lane| <$ty>::from_f32(a[lane].to_f32() * 2.0 + c[lane].to_f32())).collect::<Vec<_>>()
                );
                check!(
                    "mul_sub",
                    simd.[<mul_sub_ $ty>](av, bv, cv),
                    (0..lanes).map(|lane| <$ty>::from_f32(a[lane].to_f32() * 2.0 - c[lane].to_f32())).collect::<Vec<_>>()
                );
                check!(
                    "nmul_add",
                    simd.[<nmul_add_ $ty>](av, bv, cv),
                    (0..lanes).map(|lane| <$ty>::from_f32(-(a[lane].to_f32() * 2.0) + c[lane].to_f32())).collect::<Vec<_>>()
                );
                check!(
                    "nmul_sub",
                    simd.[<nmul_sub_ $ty>](av, bv, cv),
                    (0..lanes).map(|lane| <$ty>::from_f32(-(a[lane].to_f32() * 2.0) - c[lane].to_f32())).collect::<Vec<_>>()
                );
            }
        }
    };
}

define_half_float_check!(check_float_f16, f16, f16_lanes);
define_half_float_check!(check_float_bf16, bf16, bf16_lanes);

macro_rules! define_half_mask_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            let a: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(lane as f32)).collect();
            let b: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32((lanes - 1 - lane) as f32)).collect();

            paste::paste! {
                let av = simd.[<load_ $ty>](&a);
                let bv = simd.[<load_ $ty>](&b);
                let one = simd.[<splat_ $ty>](<$ty>::from_f32(1.0));
                let zero = simd.[<splat_ $ty>](<$ty>::from_f32(0.0));

                let mut out = vec![<$ty>::from_f32(0.0); lanes];
                macro_rules! check {
                    ($label:literal, $mask:expr, $cmp:expr) => {{
                        let selected = simd.[<select_ $ty>]($mask, one, zero);
                        simd.[<store_ $ty>](&mut out, selected);
                        for lane in 0..lanes {
                            let want = if $cmp(a[lane].to_f32(), b[lane].to_f32()) {
                                <$ty>::from_f32(1.0)
                            } else {
                                <$ty>::from_f32(0.0)
                            };
                            assert_eq!(out[lane], want, concat!($label, " ", stringify!($ty), " lane {}"), lane);
                        }
                    }};
                }

                check!("eq", simd.[<eq_ $ty>](av, bv), |x, y| x == y);
                check!("ne", simd.[<ne_ $ty>](av, bv), |x, y| x != y);
                check!("lt", simd.[<lt_ $ty>](av, bv), |x, y| x < y);
                check!("le", simd.[<le_ $ty>](av, bv), |x, y| x <= y);
                check!("gt", simd.[<gt_ $ty>](av, bv), |x, y| x > y);
                check!("ge", simd.[<ge_ $ty>](av, bv), |x, y| x >= y);

                let all = simd.[<eq_ $ty>](av, av);
                let none = simd.[<lt_ $ty>](av, av);
                assert!(all.all(), concat!(stringify!($ty), " mask all"));
                assert!(all.any(), concat!(stringify!($ty), " mask any"));
                assert!(!none.any(), concat!(stringify!($ty), " empty mask any"));
                assert!(!none.all(), concat!(stringify!($ty), " empty mask all"));
                assert!(all.and(all).all(), concat!(stringify!($ty), " mask and"));
                assert!(!all.and(none).any(), concat!(stringify!($ty), " mask and empty"));
                assert!(none.or(all).all(), concat!(stringify!($ty), " mask or"));
                assert!(none.not().all(), concat!(stringify!($ty), " mask not"));
                assert!(!all.not().any(), concat!(stringify!($ty), " mask not all"));

                let mask_bits = (1u32 << lanes) - 1;
                assert_eq!(S::m16::splat(true).to_bits() & mask_bits, mask_bits, concat!(stringify!($ty), " splat mask bits"));
                assert_eq!(S::m16::splat(false).to_bits() & mask_bits, 0, concat!(stringify!($ty), " empty splat mask bits"));
                let bits = 0b1010_0101 & mask_bits;
                assert_eq!(S::m16::from_bits(bits).to_bits() & mask_bits, bits, concat!(stringify!($ty), " mask bit roundtrip"));

                if lanes >= 2 {
                    let left: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(lane as f32)).collect();
                    let right: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32((lane + lane % 2) as f32)).collect();
                    let lv = simd.[<load_ $ty>](&left);
                    let rv = simd.[<load_ $ty>](&right);
                    let mixed = simd.[<eq_ $ty>](lv, rv);

                    assert!(mixed.any(), concat!(stringify!($ty), " mixed mask any"));
                    assert!(!mixed.all(), concat!(stringify!($ty), " mixed mask all"));

                    simd.[<store_ $ty>](&mut out, simd.[<select_ $ty>](mixed, one, zero));
                    for (lane, &value) in out.iter().enumerate() {
                        let want = if lane % 2 == 0 { <$ty>::from_f32(1.0) } else { <$ty>::from_f32(0.0) };
                        assert_eq!(value, want, concat!(stringify!($ty), " mixed select lane {}"), lane);
                    }

                    let full = simd.[<eq_ $ty>](lv, lv);
                    assert!(!mixed.and(full).all(), concat!(stringify!($ty), " mixed and all"));
                    assert!(mixed.or(full).all(), concat!(stringify!($ty), " mixed or all"));
                    assert!(mixed.not().any(), concat!(stringify!($ty), " mixed not any"));
                    assert!(!mixed.and(full.not()).any(), concat!(stringify!($ty), " mixed and empty"));
                }
            }
        }
    };
}

define_half_mask_check!(check_mask_f16, f16, f16_lanes);
define_half_mask_check!(check_mask_bf16, bf16, bf16_lanes);

// Half bitwise operations act on the backend's internal representation (the
// packed half format for native backends, or a widened `f32` for software
// ones), so the raw bit patterns are backend-specific. Test algebraic
// identities that must hold for every representation instead.
macro_rules! define_half_bits_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            let a: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(1.0 + (lane % 8) as f32 * 0.125)).collect();
            let b: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(0.5 + (lane % 8) as f32 * 0.0625)).collect();
            let zero = vec![<$ty>::from_f32(0.0); lanes];

            paste::paste! {
                let av = simd.[<load_ $ty>](&a);
                let bv = simd.[<load_ $ty>](&b);
                let zv = simd.[<load_ $ty>](&zero);

                let mut out = vec![<$ty>::from_f32(0.0); lanes];
                macro_rules! check {
                    ($label:literal, $value:expr, $expected:expr) => {{
                        simd.[<store_ $ty>](&mut out, $value);
                        let expected: &[$ty] = $expected;
                        assert_eq!(out.as_slice(), expected, concat!($label, " ", stringify!($ty)));
                    }};
                }

                check!("bitand self", simd.[<bitand_ $ty>](av, av), &a);
                check!("bitor self", simd.[<bitor_ $ty>](av, av), &a);
                check!("bitxor self", simd.[<bitxor_ $ty>](av, av), &zero);
                check!("bitandnot self", simd.[<bitandnot_ $ty>](av, av), &zero);
                check!("bitnot twice", simd.[<bitnot_ $ty>](simd.[<bitnot_ $ty>](av)), &a);
                check!("bitand zero", simd.[<bitand_ $ty>](av, zv), &zero);
                check!("bitor zero", simd.[<bitor_ $ty>](av, zv), &a);
                check!("bitxor zero", simd.[<bitxor_ $ty>](av, zv), &a);
                check!("bitandnot zero lhs", simd.[<bitandnot_ $ty>](zv, av), &a);
                check!("bitandnot zero rhs", simd.[<bitandnot_ $ty>](av, zv), &zero);

                let mut lhs = vec![<$ty>::from_f32(0.0); lanes];
                let mut rhs = vec![<$ty>::from_f32(0.0); lanes];

                simd.[<store_ $ty>](&mut lhs, simd.[<bitand_ $ty>](av, bv));
                simd.[<store_ $ty>](&mut rhs, simd.[<bitand_ $ty>](bv, av));
                assert_eq!(lhs, rhs, concat!("bitand ", stringify!($ty), " commutative"));

                simd.[<store_ $ty>](&mut lhs, simd.[<bitor_ $ty>](av, bv));
                simd.[<store_ $ty>](&mut rhs, simd.[<bitor_ $ty>](bv, av));
                assert_eq!(lhs, rhs, concat!("bitor ", stringify!($ty), " commutative"));

                simd.[<store_ $ty>](&mut lhs, simd.[<bitxor_ $ty>](av, bv));
                simd.[<store_ $ty>](&mut rhs, simd.[<bitxor_ $ty>](bv, av));
                assert_eq!(lhs, rhs, concat!("bitxor ", stringify!($ty), " commutative"));
            }
        }
    };
}

define_half_bits_check!(check_bits_f16, f16, f16_lanes);
define_half_bits_check!(check_bits_bf16, bf16, bf16_lanes);

macro_rules! define_half_reduce_check {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        fn $name<S: HalfSimd>(simd: S) {
            let lanes = S::$lanes;
            let values: Vec<$ty> = (0..lanes).map(|lane| <$ty>::from_f32(lane as f32 - 2.0)).collect();

            paste::paste! {
                let vector = simd.[<load_ $ty>](&values);
                let sum: f32 = values.iter().map(|value| value.to_f32()).sum();
                let max = values.iter().map(|value| value.to_f32()).fold(f32::NEG_INFINITY, f32::max);
                let min = values.iter().map(|value| value.to_f32()).fold(f32::INFINITY, f32::min);

                assert_eq!(simd.[<reduce_sum_ $ty>](vector), <$ty>::from_f32(sum), concat!("reduce_sum ", stringify!($ty)));
                assert_eq!(simd.[<reduce_max_ $ty>](vector), <$ty>::from_f32(max), concat!("reduce_max ", stringify!($ty)));
                assert_eq!(simd.[<reduce_min_ $ty>](vector), <$ty>::from_f32(min), concat!("reduce_min ", stringify!($ty)));
            }
        }
    };
}

define_half_reduce_check!(check_reduce_f16, f16, f16_lanes);
define_half_reduce_check!(check_reduce_bf16, bf16, bf16_lanes);

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

/// Results of the generic f16 or bf16 suite.
///
/// Values are widened to `f32` after being stored, so backends with different
/// internal representations can be compared with a single tolerance.
#[derive(Clone, Debug)]
pub struct SuiteHalfType {
    pub floats: Vec<Vec<f32>>,
    pub masks: Vec<Vec<u8>>,
    pub sum: f32,
    pub max: f32,
    pub min: f32,
}

/// Results of the half-precision suites.
#[derive(Clone, Debug)]
pub struct SuiteHalf {
    pub f16: SuiteHalfType,
    pub bf16: SuiteHalfType,
}

/// Combined results for all scalar widths.
#[derive(Clone, Debug)]
pub struct Suite {
    pub single: SuiteF32,
    pub double: SuiteF64,
    pub half: SuiteHalf,
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
            half: SuiteHalf {
                f16: suite_f16(simd),
                bf16: suite_bf16(simd),
            },
        }
    }
}

macro_rules! define_half_suite {
    ($name:ident, $ty:ident, $lanes:ident) => {
        #[inline(always)]
        pub fn $name<S: HalfSimd>(simd: S) -> SuiteHalfType {
            paste::paste! {
            let lanes = S::$lanes;
            assert_eq!(SUITE_N % lanes, 0, concat!("suite length must be divisible by ", stringify!($ty), " lane count"));

            let a: Vec<$ty> = suite_a_f32().into_iter().map(<$ty>::from_f32).collect();
            let b: Vec<$ty> = suite_b_f32().into_iter().map(<$ty>::from_f32).collect();
            let c: Vec<$ty> = suite_c_f32().into_iter().map(<$ty>::from_f32).collect();
            let sqrt_input: Vec<$ty> = suite_sqrt_f32().into_iter().map(<$ty>::from_f32).collect();
            let reduce_input: Vec<$ty> = suite_reduce_f32().into_iter().map(<$ty>::from_f32).collect();

            let one = simd.[<splat_ $ty>](<$ty>::from_f32(1.0));
            let zero = simd.[<splat_ $ty>](<$ty>::from_f32(0.0));

            let mut floats: Vec<Vec<f32>> = (0..13).map(|_| Vec::with_capacity(SUITE_N)).collect();
            let mut masks: Vec<Vec<u8>> = (0..6).map(|_| Vec::with_capacity(SUITE_N)).collect();
            let mut buffer = vec![<$ty>::from_f32(0.0); lanes];

            let mut sum = 0.0f32;
            let mut max = f32::NEG_INFINITY;
            let mut min = f32::INFINITY;

            macro_rules! emit_float {
                ($group:expr, $value:expr) => {{
                    simd.[<store_ $ty>](&mut buffer, $value);
                    floats[$group].extend(buffer.iter().map(|value| value.to_f32()));
                }};
            }

            macro_rules! emit_mask {
                ($group:expr, $value:expr) => {{
                    simd.[<store_ $ty>](&mut buffer, $value);
                    masks[$group].extend(buffer.iter().map(|value| u8::from(*value != <$ty>::from_f32(0.0))));
                }};
            }

            for start in (0..SUITE_N).step_by(lanes) {
                let end = start + lanes;
                let av = simd.[<load_ $ty>](&a[start..end]);
                let bv = simd.[<load_ $ty>](&b[start..end]);
                let cv = simd.[<load_ $ty>](&c[start..end]);
                let qv = simd.[<load_ $ty>](&sqrt_input[start..end]);
                let rv = simd.[<load_ $ty>](&reduce_input[start..end]);

                emit_float!(0, simd.[<add_ $ty>](av, bv));
                emit_float!(1, simd.[<sub_ $ty>](av, bv));
                emit_float!(2, simd.[<mul_ $ty>](av, bv));
                emit_float!(3, simd.[<div_ $ty>](av, bv));
                emit_float!(4, simd.[<min_ $ty>](av, bv));
                emit_float!(5, simd.[<max_ $ty>](av, bv));
                emit_float!(6, simd.[<abs_ $ty>](av));
                emit_float!(7, simd.[<neg_ $ty>](av));
                emit_float!(8, simd.[<sqrt_ $ty>](qv));
                emit_float!(9, simd.[<mul_add_ $ty>](av, bv, cv));
                emit_float!(10, simd.[<mul_sub_ $ty>](av, bv, cv));
                emit_float!(11, simd.[<nmul_add_ $ty>](av, bv, cv));
                emit_float!(12, simd.[<nmul_sub_ $ty>](av, bv, cv));

                emit_mask!(0, simd.[<select_ $ty>](simd.[<eq_ $ty>](av, bv), one, zero));
                emit_mask!(1, simd.[<select_ $ty>](simd.[<ne_ $ty>](av, bv), one, zero));
                emit_mask!(2, simd.[<select_ $ty>](simd.[<lt_ $ty>](av, bv), one, zero));
                emit_mask!(3, simd.[<select_ $ty>](simd.[<le_ $ty>](av, bv), one, zero));
                emit_mask!(4, simd.[<select_ $ty>](simd.[<gt_ $ty>](av, bv), one, zero));
                emit_mask!(5, simd.[<select_ $ty>](simd.[<ge_ $ty>](av, bv), one, zero));

                sum += simd.[<reduce_sum_ $ty>](rv).to_f32();
                max = max.max(simd.[<reduce_max_ $ty>](rv).to_f32());
                min = min.min(simd.[<reduce_min_ $ty>](rv).to_f32());
            }

            SuiteHalfType {
                floats,
                masks,
                sum,
                max,
                min,
            }
            }
        }
    };
}

define_half_suite!(suite_f16, f16, f16_lanes);
define_half_suite!(suite_bf16, bf16, bf16_lanes);

#[inline(always)]
pub fn suite_f32<S: Simd>(simd: S) -> SuiteF32 {
    let lanes = S::f32_lanes;
    assert_eq!(SUITE_N % lanes, 0, "suite length must be divisible by f32 lane count");

    let a = suite_a_f32();
    let b = suite_b_f32();
    let c = suite_c_f32();
    let sqrt_input = suite_sqrt_f32();
    let reduce_input = suite_reduce_f32();
    let bits_a = suite_bits_a_f32();
    let bits_b = suite_bits_b_f32();

    let one = simd.splat_f32(1.0);
    let zero = simd.splat_f32(0.0);

    let mut floats: Vec<Vec<f32>> = (0..13).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut masks: Vec<Vec<u8>> = (0..6).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut bits: Vec<Vec<u32>> = (0..5).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut buffer = vec![0.0f32; lanes];

    let mut sum = 0.0f32;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;

    macro_rules! emit_float {
        ($group:expr, $value:expr) => {{
            simd.store_f32(&mut buffer, $value);
            floats[$group].extend_from_slice(&buffer);
        }};
    }

    macro_rules! emit_mask {
        ($group:expr, $value:expr) => {{
            simd.store_f32(&mut buffer, $value);
            masks[$group].extend(buffer.iter().map(|&value| u8::from(value != 0.0)));
        }};
    }

    macro_rules! emit_bits {
        ($group:expr, $value:expr) => {{
            simd.store_f32(&mut buffer, $value);
            bits[$group].extend(buffer.iter().map(|value| value.to_bits()));
        }};
    }

    for start in (0..SUITE_N).step_by(lanes) {
        let end = start + lanes;
        let av = simd.load_f32(&a[start..end]);
        let bv = simd.load_f32(&b[start..end]);
        let cv = simd.load_f32(&c[start..end]);
        let qv = simd.load_f32(&sqrt_input[start..end]);
        let rv = simd.load_f32(&reduce_input[start..end]);
        let ba = simd.load_f32(&bits_a[start..end]);
        let bb = simd.load_f32(&bits_b[start..end]);

        emit_float!(0, simd.add_f32(av, bv));
        emit_float!(1, simd.sub_f32(av, bv));
        emit_float!(2, simd.mul_f32(av, bv));
        emit_float!(3, simd.div_f32(av, bv));
        emit_float!(4, simd.min_f32(av, bv));
        emit_float!(5, simd.max_f32(av, bv));
        emit_float!(6, simd.abs_f32(av));
        emit_float!(7, simd.neg_f32(av));
        emit_float!(8, simd.sqrt_f32(qv));
        emit_float!(9, simd.mul_add_f32(av, bv, cv));
        emit_float!(10, simd.mul_sub_f32(av, bv, cv));
        emit_float!(11, simd.nmul_add_f32(av, bv, cv));
        emit_float!(12, simd.nmul_sub_f32(av, bv, cv));

        emit_mask!(0, simd.select_f32(simd.eq_f32(av, bv), one, zero));
        emit_mask!(1, simd.select_f32(simd.ne_f32(av, bv), one, zero));
        emit_mask!(2, simd.select_f32(simd.lt_f32(av, bv), one, zero));
        emit_mask!(3, simd.select_f32(simd.le_f32(av, bv), one, zero));
        emit_mask!(4, simd.select_f32(simd.gt_f32(av, bv), one, zero));
        emit_mask!(5, simd.select_f32(simd.ge_f32(av, bv), one, zero));

        emit_bits!(0, simd.bitand_f32(ba, bb));
        emit_bits!(1, simd.bitor_f32(ba, bb));
        emit_bits!(2, simd.bitxor_f32(ba, bb));
        emit_bits!(3, simd.bitandnot_f32(ba, bb));
        emit_bits!(4, simd.bitnot_f32(ba));

        sum += simd.reduce_sum_f32(rv);
        max = max.max(simd.reduce_max_f32(rv));
        min = min.min(simd.reduce_min_f32(rv));
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
    let lanes = S::f64_lanes;
    assert_eq!(SUITE_N % lanes, 0, "suite length must be divisible by f64 lane count");

    let a = suite_a_f64();
    let b = suite_b_f64();
    let c = suite_c_f64();
    let sqrt_input = suite_sqrt_f64();
    let reduce_input = suite_reduce_f64();
    let bits_a = suite_bits_a_f64();
    let bits_b = suite_bits_b_f64();

    let one = simd.splat_f64(1.0);
    let zero = simd.splat_f64(0.0);

    let mut floats: Vec<Vec<f64>> = (0..13).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut masks: Vec<Vec<u8>> = (0..6).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut bits: Vec<Vec<u64>> = (0..5).map(|_| Vec::with_capacity(SUITE_N)).collect();
    let mut buffer = vec![0.0f64; lanes];

    let mut sum = 0.0f64;
    let mut max = f64::NEG_INFINITY;
    let mut min = f64::INFINITY;

    macro_rules! emit_float {
        ($group:expr, $value:expr) => {{
            simd.store_f64(&mut buffer, $value);
            floats[$group].extend_from_slice(&buffer);
        }};
    }

    macro_rules! emit_mask {
        ($group:expr, $value:expr) => {{
            simd.store_f64(&mut buffer, $value);
            masks[$group].extend(buffer.iter().map(|&value| u8::from(value != 0.0)));
        }};
    }

    macro_rules! emit_bits {
        ($group:expr, $value:expr) => {{
            simd.store_f64(&mut buffer, $value);
            bits[$group].extend(buffer.iter().map(|value| value.to_bits()));
        }};
    }

    for start in (0..SUITE_N).step_by(lanes) {
        let end = start + lanes;
        let av = simd.load_f64(&a[start..end]);
        let bv = simd.load_f64(&b[start..end]);
        let cv = simd.load_f64(&c[start..end]);
        let qv = simd.load_f64(&sqrt_input[start..end]);
        let rv = simd.load_f64(&reduce_input[start..end]);
        let ba = simd.load_f64(&bits_a[start..end]);
        let bb = simd.load_f64(&bits_b[start..end]);

        emit_float!(0, simd.add_f64(av, bv));
        emit_float!(1, simd.sub_f64(av, bv));
        emit_float!(2, simd.mul_f64(av, bv));
        emit_float!(3, simd.div_f64(av, bv));
        emit_float!(4, simd.min_f64(av, bv));
        emit_float!(5, simd.max_f64(av, bv));
        emit_float!(6, simd.abs_f64(av));
        emit_float!(7, simd.neg_f64(av));
        emit_float!(8, simd.sqrt_f64(qv));
        emit_float!(9, simd.mul_add_f64(av, bv, cv));
        emit_float!(10, simd.mul_sub_f64(av, bv, cv));
        emit_float!(11, simd.nmul_add_f64(av, bv, cv));
        emit_float!(12, simd.nmul_sub_f64(av, bv, cv));

        emit_mask!(0, simd.select_f64(simd.eq_f64(av, bv), one, zero));
        emit_mask!(1, simd.select_f64(simd.ne_f64(av, bv), one, zero));
        emit_mask!(2, simd.select_f64(simd.lt_f64(av, bv), one, zero));
        emit_mask!(3, simd.select_f64(simd.le_f64(av, bv), one, zero));
        emit_mask!(4, simd.select_f64(simd.gt_f64(av, bv), one, zero));
        emit_mask!(5, simd.select_f64(simd.ge_f64(av, bv), one, zero));

        emit_bits!(0, simd.bitand_f64(ba, bb));
        emit_bits!(1, simd.bitor_f64(ba, bb));
        emit_bits!(2, simd.bitxor_f64(ba, bb));
        emit_bits!(3, simd.bitandnot_f64(ba, bb));
        emit_bits!(4, simd.bitnot_f64(ba));

        sum += simd.reduce_sum_f64(rv);
        max = max.max(simd.reduce_max_f64(rv));
        min = min.min(simd.reduce_min_f64(rv));
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
