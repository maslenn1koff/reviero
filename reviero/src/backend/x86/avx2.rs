use crate::{
    SimdFn,
    traits::{FloatSimd, HalfSimd, Simd, SimdMask},
};
use half::{bf16, f16};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 comparison mask for eight `f32` lanes.
#[derive(Copy, Clone, Debug)]
pub struct AvxMaskF32(__m256);

impl SimdMask for AvxMaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm256_castsi256_ps(_mm256_set1_epi32(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        unsafe {
            Self(_mm256_castsi256_ps(_mm256_set_epi32(
                if v & 128 != 0 { -1 } else { 0 },
                if v & 64 != 0 { -1 } else { 0 },
                if v & 32 != 0 { -1 } else { 0 },
                if v & 16 != 0 { -1 } else { 0 },
                if v & 8 != 0 { -1 } else { 0 },
                if v & 4 != 0 { -1 } else { 0 },
                if v & 2 != 0 { -1 } else { 0 },
                if v & 1 != 0 { -1 } else { 0 },
            )))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        unsafe { _mm256_movemask_ps(self.0) as u32 }
    }

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
    fn splat(v: bool) -> Self {
        unsafe { Self(_mm256_castsi256_pd(_mm256_set1_epi64x(if v { -1 } else { 0 }))) }
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        unsafe {
            Self(_mm256_castsi256_pd(_mm256_set_epi64x(
                if v & 8 != 0 { -1 } else { 0 },
                if v & 4 != 0 { -1 } else { 0 },
                if v & 2 != 0 { -1 } else { 0 },
                if v & 1 != 0 { -1 } else { 0 },
            )))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        unsafe { _mm256_movemask_pd(self.0) as u32 }
    }

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

pub trait HalfConv {
    unsafe fn splat_f16(value: f16) -> __m256;
    unsafe fn splat_bf16(value: bf16) -> __m256;
    unsafe fn load_f16(src: &[f16]) -> __m256;
    unsafe fn load_bf16(src: &[bf16]) -> __m256;
    unsafe fn store_f16(dst: &mut [f16], value: __m256);
    unsafe fn store_bf16(dst: &mut [bf16], value: __m256);
}

#[derive(Clone, Copy)]
struct HalfConvSoftware;

#[derive(Clone, Copy)]
struct HalfConvF16c;

impl HalfConv for HalfConvSoftware {
    #[inline(always)]
    unsafe fn splat_f16(value: f16) -> __m256 {
        unsafe { _mm256_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn splat_bf16(value: bf16) -> __m256 {
        unsafe { _mm256_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn load_f16(src: &[f16]) -> __m256 {
        let mut fsrc = [0f32; 8];
        for (fsr, src) in fsrc.iter_mut().zip(src) {
            *fsr = src.to_f32();
        }

        unsafe { _mm256_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_bf16(src: &[bf16]) -> __m256 {
        let mut fsrc = [0f32; 8];
        for (fsr, src) in fsrc.iter_mut().zip(src) {
            *fsr = src.to_f32();
        }

        unsafe { _mm256_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_f16(dst: &mut [f16], value: __m256) {
        let mut fdst = [0f32; 8];
        unsafe { _mm256_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, fdst) in dst.iter_mut().zip(fdst) {
            *dst = f16::from_f32(fdst);
        }
    }

    #[inline(always)]
    unsafe fn store_bf16(dst: &mut [bf16], value: __m256) {
        let mut fdst = [0f32; 8];
        unsafe { _mm256_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, fdst) in dst.iter_mut().zip(fdst) {
            *dst = bf16::from_f32(fdst);
        }
    }
}

impl HalfConv for HalfConvF16c {
    #[inline(always)]
    unsafe fn splat_f16(value: f16) -> __m256 {
        unsafe { _mm256_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn splat_bf16(value: bf16) -> __m256 {
        unsafe { _mm256_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    unsafe fn load_f16(src: &[f16]) -> __m256 {
        unsafe {
            let packed = _mm_loadu_si128(src.as_ptr().cast());
            _mm256_cvtph_ps(packed)
        }
    }

    #[inline(always)]
    unsafe fn load_bf16(src: &[bf16]) -> __m256 {
        unsafe {
            let packed = _mm_loadu_si128(src.as_ptr().cast());
            let x = _mm256_cvtepu16_epi32(packed);
            let x = _mm256_slli_epi32(x, 16);
            _mm256_castsi256_ps(x)
        }
    }

    #[inline(always)]
    unsafe fn store_f16(dst: &mut [f16], value: __m256) {
        unsafe {
            let packed = _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(dst.as_mut_ptr().cast(), packed);
        }
    }

    #[inline(always)]
    unsafe fn store_bf16(dst: &mut [bf16], value: __m256) {
        unsafe {
            let bits = _mm256_castps_si256(value);

            let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));

            let rounded = _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb));

            let mut upper = _mm256_srli_epi32::<16>(rounded);

            let exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7F800000));
            let mantissa = _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF));

            let is_inf_or_nan = _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7F800000));
            let has_mantissa = _mm256_cmpgt_epi32(mantissa, _mm256_setzero_si256());
            let is_nan = _mm256_and_si256(is_inf_or_nan, has_mantissa);

            upper = _mm256_or_si256(upper, _mm256_and_si256(is_nan, _mm256_set1_epi32(0x40)));

            // Convert unsigned 16-bit values into the signed i16 range.
            let shifted = _mm256_sub_epi32(upper, _mm256_set1_epi32(0x8000));

            let packed = _mm_packs_epi32(_mm256_castsi256_si128(shifted), _mm256_extracti128_si256::<1>(shifted));

            let packed = _mm_xor_si128(packed, _mm_set1_epi16(i16::MIN));

            _mm_storeu_si128(dst.as_mut_ptr().cast(), packed);
        }
    }
}

#[derive(Copy, Clone)]
pub struct Avx2Fma<C: HalfConv>(pub C);

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx2Fma(HalfConvSoftware))
}

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn vectorize_f16c<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx2Fma(HalfConvF16c))
}

impl<C: HalfConv + Sync + Send + Copy + 'static> FloatSimd for Avx2Fma<C> {
    type f32 = __m256;
    type f64 = __m256d;

    type m32 = AvxMaskF32;
    type m64 = AvxMaskF64;

    const f32_lanes: usize = 8;
    const f64_lanes: usize = 4;
    const m64_lanes: usize = Self::f64_lanes;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { _mm256_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { _mm256_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { _mm256_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { _mm256_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { _mm256_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { _mm256_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm256_andnot_pd(_mm256_set1_pd(-0.0), a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm256_xor_ps(a, _mm256_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm256_xor_pd(a, _mm256_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm256_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm256_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm256_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm256_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm256_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm256_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm256_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm256_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm256_fnmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm256_fnmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { AvxMaskF32(_mm256_cmp_ps::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { AvxMaskF64(_mm256_cmp_pd::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_blendv_ps(b, a, mask.0) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_blendv_pd(b, a, mask.0) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm256_andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm256_andnot_pd(a, b) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_max_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_max_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_max_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_min_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_min_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_min_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_min_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_min_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe {
            let s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_movehdup_ps(s));
            _mm_cvtss_f32(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let s = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
            let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }
}

impl<C: HalfConv + Sync + Send + Copy + 'static> HalfSimd for Avx2Fma<C> {
    type f16 = __m256;
    type bf16 = __m256;
    type m16 = AvxMaskF32;
    const f16_lanes: usize = 8;

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
        unsafe { _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(a), _mm256_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_castsi256_ps(_mm256_andnot_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm256_castsi256_ps(_mm256_and_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }

    #[inline(always)]
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 {
        unsafe { _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(a), _mm256_set1_epi32(-1))) }
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { _mm256_castsi256_ps(_mm256_andnot_si256(_mm256_castps_si256(a), _mm256_castps_si256(b))) }
    }
}

impl<C: HalfConv + Sync + Send + Copy + 'static> Simd for Avx2Fma<C> {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        eprintln!("avx2 backend test skipped: CPU does not support avx2+fma");
        return;
    }

    unsafe { vectorize(crate::backend::test_utils::Run(check)) };
}

#[cfg(test)]
crate::backend_tests!(Avx2Fma<HalfConvSoftware>, 8, 4);
