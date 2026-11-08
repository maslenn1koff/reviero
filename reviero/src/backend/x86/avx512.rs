#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    SimdFn,
    traits::{SimdBitOps, SimdFloatOps, SimdIo, SimdMask, SimdMaskOps, SimdReduceOps, SimdTypes},
};

#[derive(Copy, Clone, Debug)]
pub struct Avx512MaskF32(__mmask16); // 16 bits

impl SimdMask for Avx512MaskF32 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u16::MAX
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Avx512MaskF64(__mmask8); // 8 bits, all used

impl SimdMask for Avx512MaskF64 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0 == u8::MAX
    }
}

macro_rules! bit_helper {
    ($name:ident, $ty:ty, $to_int:ident, $from_int:ident, $op:ident) => {
        #[inline(always)]
        unsafe fn $name(a: $ty, b: $ty) -> $ty {
            unsafe { $from_int($op($to_int(a), $to_int(b))) }
        }
    };
}

bit_helper!(and_ps, __m512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_and_si512);
bit_helper!(or_ps, __m512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_or_si512);
bit_helper!(xor_ps, __m512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_xor_si512);
bit_helper!(andnot_ps, __m512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_andnot_si512);

bit_helper!(and_pd, __m512d, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_and_si512);
bit_helper!(or_pd, __m512d, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_or_si512);
bit_helper!(xor_pd, __m512d, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_xor_si512);
bit_helper!(andnot_pd, __m512d, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_andnot_si512);

#[inline(always)]
unsafe fn split_f32s(a: __m512) -> (__m256, __m256) {
    unsafe {
        let lo = _mm512_castps512_ps256(a);
        let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(a)));
        (lo, hi)
    }
}

#[inline(always)]
unsafe fn split_f64s(a: __m512d) -> (__m256d, __m256d) {
    unsafe { (_mm512_castpd512_pd256(a), _mm512_extractf64x4_pd::<1>(a)) }
}

#[derive(Copy, Clone, Debug)]
pub struct Avx512;

impl Default for Avx512 {
    fn default() -> Self {
        Avx512
    }
}

#[target_feature(enable = "avx512f", enable = "fma")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx512)
}

impl SimdTypes for Avx512 {
    type f32s = __m512;
    type f64s = __m512d;

    type m32s = Avx512MaskF32;
    type m64s = Avx512MaskF64;

    const f32s_lanes: usize = 16;
    const f64s_lanes: usize = 8;
}

impl SimdIo for Avx512 {
    #[inline(always)]
    fn splat_f32s(self, value: f32) -> Self::f32s {
        unsafe { _mm512_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64s(self, value: f64) -> Self::f64s {
        unsafe { _mm512_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32s(self, src: &[f32]) -> Self::f32s {
        assert!(src.len() >= Self::f32s_lanes);
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64s(self, src: &[f64]) -> Self::f64s {
        assert!(src.len() >= Self::f64s_lanes);
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32s(self, dst: &mut [f32], value: Self::f32s) {
        assert!(dst.len() >= Self::f32s_lanes);
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64s(self, dst: &mut [f64], value: Self::f64s) {
        assert!(dst.len() >= Self::f64s_lanes);
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32s(self, src: &[f32]) -> Self::f32s {
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64s(self, src: &[f64]) -> Self::f64s {
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32s(self, dst: &mut [f32], value: Self::f32s) {
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64s(self, dst: &mut [f64], value: Self::f64s) {
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) }
    }
}

impl SimdFloatOps for Avx512 {
    #[inline(always)]
    fn abs_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm512_abs_ps(a) }
    }

    #[inline(always)]
    fn abs_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm512_abs_pd(a) }
    }

    #[inline(always)]
    fn add_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { xor_ps(a, _mm512_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { xor_pd(a, _mm512_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm512_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm512_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm512_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm512_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm512_fnmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm512_fnmsub_pd(a, b, c) }
    }
}

impl SimdMaskOps for Avx512 {
    #[inline(always)]
    fn eq_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn eq_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm512_mask_blend_ps(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm512_mask_blend_pd(mask.0, b, a) }
    }
}

impl SimdBitOps for Avx512 {
    #[inline(always)]
    fn bitand_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe {
            let x = _mm512_castps_si512(a);
            _mm512_castsi512_ps(_mm512_ternarylogic_epi32::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitnot_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe {
            let x = _mm512_castpd_si512(a);
            _mm512_castsi512_pd(_mm512_ternarylogic_epi64::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitandnot_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { andnot_pd(a, b) }
    }
}

impl SimdReduceOps for Avx512 {
    #[inline(always)]
    fn reduce_max_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let (lo, hi) = split_f32s(a);
            let s = _mm256_max_ps(lo, hi);
            let s = _mm_max_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_max_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_max_pd(lo, hi);
            let s = _mm_max_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
            let s = _mm_max_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let (lo, hi) = split_f32s(a);
            let s = _mm256_min_ps(lo, hi);
            let s = _mm_min_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            let s = _mm_min_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_min_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_min_pd(lo, hi);
            let s = _mm_min_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
            let s = _mm_min_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let (lo, hi) = split_f32s(a);
            let s = _mm256_add_ps(lo, hi);
            let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_add_pd(lo, hi);
            let s = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
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
        if !std::arch::is_x86_feature_detected!("avx512f") || !std::arch::is_x86_feature_detected!("fma") {
            eprintln!("avx512 backend test skipped: CPU does not support avx512f+fma");
            return;
        }

        unsafe { vectorize(Run(check)) };
    }

    #[test]
    fn lanes() {
        assert_eq!(Avx512::f32s_lanes, 16);
        assert_eq!(Avx512::f64s_lanes, 8);
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
