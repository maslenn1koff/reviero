#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    SimdFn,
    traits::{SimdBitOps, SimdFloatOps, SimdIo, SimdMask, SimdMaskOps, SimdReduceOps, SimdTypes},
};

/// AVX2 comparison mask for eight `f32` lanes.
#[derive(Copy, Clone, Debug)]
pub struct AvxMaskF32(__m256);

impl SimdMask for AvxMaskF32 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm256_or_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm256_xor_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) == 0b1111_1111 }
    }
}

/// AVX2 comparison mask for four `f64` lanes.
#[derive(Copy, Clone, Debug)]
pub struct AvxMaskF64(__m256d);

impl SimdMask for AvxMaskF64 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm256_or_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm256_xor_pd(self.0, _mm256_castsi256_pd(_mm256_set1_epi64x(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_pd(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm256_movemask_pd(self.0) == 0b1111 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Avx2Fma;

impl Default for Avx2Fma {
    fn default() -> Self {
        Avx2Fma
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx2Fma)
}

impl SimdTypes for Avx2Fma {
    type f32s = __m256;
    type f64s = __m256d;

    type m32s = AvxMaskF32;
    type m64s = AvxMaskF64;

    const f32s_lanes: usize = 8;
    const f64s_lanes: usize = 4;
}

impl SimdIo for Avx2Fma {
    #[inline(always)]
    fn splat_f32s(self, value: f32) -> Self::f32s {
        unsafe { _mm256_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64s(self, value: f64) -> Self::f64s {
        unsafe { _mm256_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32s(self, src: &[f32]) -> Self::f32s {
        assert!(src.len() >= Self::f32s_lanes);
        unsafe { _mm256_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64s(self, src: &[f64]) -> Self::f64s {
        assert!(src.len() >= Self::f64s_lanes);
        unsafe { _mm256_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32s(self, dst: &mut [f32], value: Self::f32s) {
        assert!(dst.len() >= Self::f32s_lanes);
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64s(self, dst: &mut [f64], value: Self::f64s) {
        assert!(dst.len() >= Self::f64s_lanes);
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32s(self, src: &[f32]) -> Self::f32s {
        unsafe { _mm256_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64s(self, src: &[f64]) -> Self::f64s {
        unsafe { _mm256_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32s(self, dst: &mut [f32], value: Self::f32s) {
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64s(self, dst: &mut [f64], value: Self::f64s) {
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr(), value) }
    }
}

impl SimdFloatOps for Avx2Fma {
    #[inline(always)]
    fn abs_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm256_andnot_pd(_mm256_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm256_xor_ps(a, _mm256_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm256_xor_pd(a, _mm256_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm256_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm256_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm256_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm256_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm256_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm256_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm256_fnmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm256_fnmsub_pd(a, b, c) }
    }
}

impl SimdMaskOps for Avx2Fma {
    #[inline(always)]
    fn eq_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn eq_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_blendv_ps(b, a, mask.0) }
    }

    #[inline(always)]
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_blendv_pd(b, a, mask.0) }
    }
}

impl SimdBitOps for Avx2Fma {
    #[inline(always)]
    fn bitand_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm256_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm256_andnot_pd(a, b) }
    }
}

impl SimdReduceOps for Avx2Fma {
    #[inline(always)]
    fn reduce_max_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_max_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_max_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_max_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_min_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_min_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_min_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_min_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_min_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{Check, Run};
    use crate::traits::SimdTypes;

    fn run(check: Check) {
        if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
            eprintln!("avx2 backend test skipped: CPU does not support avx2+fma");
            return;
        }

        unsafe { vectorize(Run(check)) };
    }

    #[test]
    fn lanes() {
        assert_eq!(Avx2Fma::f32s_lanes, 8);
        assert_eq!(Avx2Fma::f64s_lanes, 4);
    }

    #[test]
    fn io() {
        run(Check::Io);
    }

    #[test]
    fn float_ops() {
        run(Check::Float);
    }

    #[test]
    fn mask_ops() {
        run(Check::Mask);
    }

    #[test]
    fn bit_ops() {
        run(Check::Bits);
    }

    #[test]
    fn reduce_ops() {
        run(Check::Reduce);
    }
}
