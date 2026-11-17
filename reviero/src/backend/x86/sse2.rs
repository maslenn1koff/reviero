use half::{bf16, f16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    SimdFn,
    traits::{FloatSimd, HalfSimd, Simd, SimdMask},
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

pub trait HalfConv {
    unsafe fn splat_f16(value: f16) -> __m128;
    unsafe fn splat_bf16(value: bf16) -> __m128;
    unsafe fn load_f16(src: &[f16]) -> __m128;
    unsafe fn load_bf16(src: &[bf16]) -> __m128;
    unsafe fn store_f16(dst: &mut [f16], value: __m128);
    unsafe fn store_bf16(dst: &mut [bf16], value: __m128);
}

#[derive(Clone, Copy)]
pub struct HalfConvSoftware;

#[derive(Clone, Copy)]
pub struct HalfConvF16c;

impl HalfConv for HalfConvSoftware {
    #[inline(always)]
    unsafe fn splat_f16(value: f16) -> __m128 {
        unsafe { _mm_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn splat_bf16(value: bf16) -> __m128 {
        unsafe { _mm_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn load_f16(src: &[f16]) -> __m128 {
        let mut fsrc = [0.0f32; 4];

        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { _mm_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_bf16(src: &[bf16]) -> __m128 {
        let mut fsrc = [0.0f32; 4];

        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { _mm_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_f16(dst: &mut [f16], value: __m128) {
        let mut fdst = [0.0f32; 4];
        unsafe { _mm_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, value) in dst.iter_mut().zip(fdst) {
            *dst = f16::from_f32(value);
        }
    }

    #[inline(always)]
    unsafe fn store_bf16(dst: &mut [bf16], value: __m128) {
        let mut fdst = [0.0f32; 4];
        unsafe { _mm_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, value) in dst.iter_mut().zip(fdst) {
            *dst = bf16::from_f32(value);
        }
    }
}

impl HalfConv for HalfConvF16c {
    #[inline(always)]
    unsafe fn splat_f16(value: f16) -> __m128 {
        unsafe { _mm_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn splat_bf16(value: bf16) -> __m128 {
        unsafe { _mm_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn load_f16(src: &[f16]) -> __m128 {
        unsafe {
            let packed = _mm_loadl_epi64(src.as_ptr().cast());
            _mm_cvtph_ps(packed)
        }
    }

    #[inline(always)]
    unsafe fn load_bf16(src: &[bf16]) -> __m128 {
        unsafe {
            let packed = _mm_loadl_epi64(src.as_ptr().cast());

            let expanded = _mm_unpacklo_epi16(packed, _mm_setzero_si128());

            let expanded = _mm_slli_epi32::<16>(expanded);

            _mm_castsi128_ps(expanded)
        }
    }

    #[inline(always)]
    unsafe fn store_f16(dst: &mut [f16], value: __m128) {
        unsafe {
            let packed = _mm_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);

            _mm_storel_epi64(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    unsafe fn store_bf16(dst: &mut [bf16], value: __m128) {
        unsafe {
            let bits = _mm_castps_si128(value);

            let lsb = _mm_and_si128(_mm_srli_epi32::<16>(bits), _mm_set1_epi32(1));

            let rounded = _mm_add_epi32(bits, _mm_add_epi32(_mm_set1_epi32(0x7FFF), lsb));

            let mut upper = _mm_srli_epi32::<16>(rounded);

            let exponent = _mm_and_si128(bits, _mm_set1_epi32(0x7F800000));

            let mantissa = _mm_and_si128(bits, _mm_set1_epi32(0x007FFFFF));

            let is_inf_or_nan = _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7F800000));

            let has_mantissa = _mm_cmpgt_epi32(mantissa, _mm_setzero_si128());

            let is_nan = _mm_and_si128(is_inf_or_nan, has_mantissa);

            upper = _mm_or_si128(upper, _mm_and_si128(is_nan, _mm_set1_epi32(0x40)));

            let shifted = _mm_sub_epi32(upper, _mm_set1_epi32(0x8000));

            let packed = _mm_packs_epi32(shifted, _mm_setzero_si128());

            let packed = _mm_xor_si128(packed, _mm_set1_epi16(i16::MIN));

            _mm_storel_epi64(dst.as_mut_ptr().cast(), packed);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sse2<C: HalfConv>(pub C);

#[target_feature(enable = "sse2")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Sse2(HalfConvSoftware))
}

#[target_feature(enable = "sse2", enable = "f16c")]
pub unsafe fn vectorize_f16c<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Sse2(HalfConvF16c))
}

impl<C: HalfConv + Sync + Send + Copy + 'static> FloatSimd for Sse2<C> {
    type f32 = __m128;
    type f64 = __m128d;
    type m32 = Sse2MaskF32;
    type m64 = Sse2MaskF64;

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
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpeq_ps(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpeq_pd(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpge_ps(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpge_pd(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpgt_ps(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpgt_pd(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmple_ps(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmple_pd(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmplt_ps(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmplt_pd(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Sse2MaskF32(_mm_cmpneq_ps(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Sse2MaskF64(_mm_cmpneq_pd(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm_or_ps(_mm_and_ps(mask.0, a), _mm_andnot_ps(mask.0, b)) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
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

impl<C: HalfConv + Sync + Send + Copy + 'static> HalfSimd for Sse2<C> {
    type f16 = __m128;
    type bf16 = __m128;
    type m16 = Sse2MaskF32;
    const f16_lanes: usize = 4;

    #[rustfmt::skip] #[inline(always)]    fn f16_to_f32(self, a: Self::f16) -> Self::f32 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_f16(self, a: Self::f32) -> Self::f16 { a }
    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> Self::f32 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: Self::f32) -> Self::bf16 { a }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> Self::m32 { a }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: Self::m32) -> Self::m16 { a }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        unsafe { C::splat_f16(value) }
    }

    #[inline(always)]
    fn splat_bf16(self, value: bf16) -> Self::bf16 {
        unsafe { C::splat_bf16(value) }
    }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        assert!(src.len() >= Self::f16_lanes);
        unsafe { C::load_f16(src) }
    }

    #[inline(always)]
    fn load_bf16(self, src: &[bf16]) -> Self::bf16 {
        assert!(src.len() >= Self::f16_lanes);
        unsafe { C::load_bf16(src) }
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        assert!(dst.len() >= Self::f16_lanes);
        unsafe { C::store_f16(dst, value) }
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        assert!(dst.len() >= Self::f16_lanes);
        unsafe { C::store_bf16(dst, value) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f16(self, src: &[f16]) -> Self::f16 {
        unsafe { C::load_f16(src) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_bf16(self, src: &[bf16]) -> Self::bf16 {
        unsafe { C::load_bf16(src) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f16(self, dst: &mut [f16], value: Self::f16) {
        unsafe { C::store_f16(dst, value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        unsafe { C::store_bf16(dst, value) }
    }

    #[inline(always)]
    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(a), _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }

    #[inline(always)]
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 {
        unsafe { _mm_castsi128_ps(_mm_xor_si128(_mm_castps_si128(a), _mm_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm_castsi128_ps(_mm_andnot_si128(_mm_castps_si128(a), _mm_castps_si128(b))) }
    }
}

impl<C: HalfConv + Sync + Send + Copy + 'static> Simd for Sse2<C> {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    if !std::arch::is_x86_feature_detected!("sse2") {
        eprintln!("sse2 backend test skipped: CPU does not support sse2");
        return;
    }

    unsafe { vectorize(crate::backend::test_utils::Run(check)) };

    if std::arch::is_x86_feature_detected!("f16c") {
        unsafe { vectorize_f16c(crate::backend::test_utils::Run(check)) };
    }
}

#[cfg(test)]
crate::backend_tests!(Sse2<HalfConvSoftware>, 4, 2);
