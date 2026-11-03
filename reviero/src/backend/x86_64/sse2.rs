#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    SimdFn,
    traits::{SimdBitOps, SimdFloatOps, SimdIo, SimdMask, SimdMaskOps, SimdReduceOps, SimdTypes},
};

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskF32(__m128); // 4 lanes

impl SimdMask for Sse2MaskF32 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_ps(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_ps(self.0, _mm_castsi128_ps(_mm_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_ps(self.0) == 0b1111 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskF64(__m128d); // 2 lanes

impl SimdMask for Sse2MaskF64 {
    #[inline(always)]
    fn and(self, other: Self) -> Self {
        unsafe { Self(_mm_and_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        unsafe { Self(_mm_or_pd(self.0, other.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self(_mm_xor_pd(self.0, _mm_castsi128_pd(_mm_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm_movemask_pd(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm_movemask_pd(self.0) == 0b11 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2;

impl Default for Sse2 {
    fn default() -> Self {
        Sse2
    }
}

#[target_feature(enable = "sse2")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Sse2)
}

impl SimdTypes for Sse2 {
    type f32s = __m128;
    type f64s = __m128d;

    type m32s = Sse2MaskF32;
    type m64s = Sse2MaskF64;

    const f32s_lanes: usize = 4;
    const f64s_lanes: usize = 2;
}

impl SimdIo for Sse2 {
    #[inline(always)]
    fn splat_f32s(self, value: f32) -> Self::f32s {
        unsafe { _mm_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64s(self, value: f64) -> Self::f64s {
        unsafe { _mm_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32s(self, src: &[f32]) -> Self::f32s {
        assert!(src.len() >= Self::f32s_lanes);
        unsafe { _mm_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64s(self, src: &[f64]) -> Self::f64s {
        assert!(src.len() >= Self::f64s_lanes);
        unsafe { _mm_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32s(self, dst: &mut [f32], value: Self::f32s) {
        assert!(dst.len() >= Self::f32s_lanes);
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64s(self, dst: &mut [f64], value: Self::f64s) {
        assert!(dst.len() >= Self::f64s_lanes);
        unsafe { _mm_storeu_pd(dst.as_mut_ptr(), value) };
    }
}

impl SimdFloatOps for Sse2 {
    #[inline(always)]
    fn abs_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm_andnot_pd(_mm_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm_xor_ps(a, _mm_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm_xor_pd(a, _mm_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm_add_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm_add_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm_sub_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm_sub_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn nmul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm_sub_ps(c, _mm_mul_ps(a, b)) }
    }

    #[inline(always)]
    fn nmul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm_sub_pd(c, _mm_mul_pd(a, b)) }
    }

    #[inline(always)]
    fn nmul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { _mm_sub_ps(_mm_xor_ps(_mm_mul_ps(a, b), _mm_set1_ps(-0.0)), c) }
    }

    #[inline(always)]
    fn nmul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { _mm_sub_pd(_mm_xor_pd(_mm_mul_pd(a, b), _mm_set1_pd(-0.0)), c) }
    }
}

impl SimdMaskOps for Sse2 {
    #[inline(always)]
    fn eq_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpeq_ps(a, b)) }
    }

    #[inline(always)]
    fn eq_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpeq_pd(a, b)) }
    }

    #[inline(always)]
    fn ge_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpge_ps(a, b)) }
    }

    #[inline(always)]
    fn ge_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpge_pd(a, b)) }
    }

    #[inline(always)]
    fn gt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpgt_ps(a, b)) }
    }

    #[inline(always)]
    fn gt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpgt_pd(a, b)) }
    }

    #[inline(always)]
    fn le_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmple_ps(a, b)) }
    }

    #[inline(always)]
    fn le_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmple_pd(a, b)) }
    }

    #[inline(always)]
    fn lt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmplt_ps(a, b)) }
    }

    #[inline(always)]
    fn lt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmplt_pd(a, b)) }
    }

    #[inline(always)]
    fn ne_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpneq_ps(a, b)) }
    }

    #[inline(always)]
    fn ne_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpneq_pd(a, b)) }
    }

    #[inline(always)]
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_or_ps(_mm_and_ps(mask.0, a), _mm_andnot_ps(mask.0, b)) }
    }

    #[inline(always)]
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_or_pd(_mm_and_pd(mask.0, a), _mm_andnot_pd(mask.0, b)) }
    }
}

impl SimdBitOps for Sse2 {
    #[inline(always)]
    fn bitand_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { _mm_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { _mm_andnot_pd(a, b) }
    }
}

impl SimdReduceOps for Sse2 {
    #[inline(always)]
    fn reduce_max_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_max_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_max_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_min_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_min_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32s(self, a: Self::f32s) -> f32 {
        unsafe {
            let s = _mm_add_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64s(self, a: Self::f64s) -> f64 {
        unsafe {
            let s = _mm_add_sd(a, _mm_unpackhi_pd(a, a));
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
        if !std::arch::is_x86_feature_detected!("sse2") {
            eprintln!("sse2 backend test skipped: CPU does not support sse2");
            return;
        }

        unsafe { vectorize(Run(check)) };
    }

    #[test]
    fn lanes() {
        assert_eq!(Sse2::f32s_lanes, 4);
        assert_eq!(Sse2::f64s_lanes, 2);
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
