#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    SimdFn,
    traits::{Float, Simd, SimdMask},
};

#[derive(Copy, Clone, Debug)]
pub struct Sse2MaskF32(__m128); // 4 lanes

impl SimdMask for Sse2MaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_castsi128_ps(_mm_set1_epi32(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        unsafe {
            Self(_mm_castsi128_ps(_mm_set_epi32(
                if v & 8 != 0 { -1 } else { 0 },
                if v & 4 != 0 { -1 } else { 0 },
                if v & 2 != 0 { -1 } else { 0 },
                if v & 1 != 0 { -1 } else { 0 },
            )))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        unsafe { _mm_movemask_ps(self.0) as u32 }
    }

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
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm_castsi128_pd(_mm_set1_epi32(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        unsafe { Self(_mm_castsi128_pd(_mm_set_epi64x(if v & 2 != 0 { -1 } else { 0 }, if v & 1 != 0 { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        unsafe { _mm_movemask_pd(self.0) as u32 }
    }

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

impl Float for Sse2 {
    type f32 = __m128;
    type f64 = __m128d;

    type m32s = Sse2MaskF32;
    type m64s = Sse2MaskF64;

    const f32_lanes: usize = 4;
    const f64_lanes: usize = 2;
    const m64_lanes: usize = Self::f64_lanes;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { _mm_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { _mm_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { _mm_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { _mm_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { _mm_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { _mm_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { _mm_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { _mm_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { _mm_storeu_pd(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_andnot_pd(_mm_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, _mm_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, _mm_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_add_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_add_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(_mm_mul_ps(a, b), c) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(_mm_mul_pd(a, b), c) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(c, _mm_mul_ps(a, b)) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(c, _mm_mul_pd(a, b)) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm_sub_ps(_mm_xor_ps(_mm_mul_ps(a, b), _mm_set1_ps(-0.0)), c) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm_sub_pd(_mm_xor_pd(_mm_mul_pd(a, b), _mm_set1_pd(-0.0)), c) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpeq_ps(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpeq_pd(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpge_ps(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpge_pd(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpgt_ps(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpgt_pd(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmple_ps(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmple_pd(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmplt_ps(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmplt_pd(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        unsafe { Sse2MaskF32(_mm_cmpneq_ps(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        unsafe { Sse2MaskF64(_mm_cmpneq_pd(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32s, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_or_ps(_mm_and_ps(mask.0, a), _mm_andnot_ps(mask.0, b)) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64s, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_or_pd(_mm_and_pd(mask.0, a), _mm_andnot_pd(mask.0, b)) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm_andnot_pd(a, b) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_max_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_max_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_min_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_min_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_add_ps(a, _mm_movehl_ps(a, a));
            let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_add_sd(a, _mm_unpackhi_pd(a, a));
            _mm_cvtsd_f64(s)
        }
    }
}

impl Simd for Sse2 {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    if !std::arch::is_x86_feature_detected!("sse2") {
        eprintln!("sse2 backend test skipped: CPU does not support sse2");
        return;
    }

    unsafe { vectorize(crate::backend::test_utils::Run(check)) };
}

#[cfg(test)]
crate::backend_tests!(Sse2, 4, 2);
