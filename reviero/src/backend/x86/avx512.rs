#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::{bf16, f16};

use crate::{
    SimdFn,
    traits::{FloatSimd, HalfSimd, Simd, SimdMask},
};

#[derive(Copy, Clone, Debug)]
pub struct Avx512MaskF32(__mmask16); // 16 bits

impl SimdMask for Avx512MaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(if v { u16::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        Self(v as u16)
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        self.0 as u32
    }

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
    fn splat(v: bool) -> Self {
        Self(if v { u8::MAX } else { 0 })
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        Self(v as u8)
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        self.0 as u32
    }

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

pub trait HalfImpl: Copy + Send + Sync + 'static {
    type f16: Copy + Send + Sync + 'static;
    type bf16: Copy + Send + Sync + 'static;
    type m16: SimdMask + Copy + Send + Sync + 'static;

    const f16_lanes: usize = 16;

    fn f16_to_f32(self, a: Self::f16) -> __m512;
    fn f32_to_f16(self, a: __m512) -> Self::f16;
    fn bf16_to_f32(self, a: Self::bf16) -> __m512;
    fn f32_to_bf16(self, a: __m512) -> Self::bf16;
    fn m16_to_m32(self, a: Self::m16) -> Avx512MaskF32;
    fn m32_to_m16(self, a: Avx512MaskF32) -> Self::m16;

    fn splat_f16(self, value: f16) -> Self::f16;
    fn splat_bf16(self, value: bf16) -> Self::bf16;
    fn load_f16(self, src: &[f16]) -> Self::f16;
    fn load_bf16(self, src: &[bf16]) -> Self::bf16;
    fn store_f16(self, dst: &mut [f16], value: Self::f16);
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16);

    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16;
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16;
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16;
    fn bitnot_f16(self, a: Self::f16) -> Self::f16;
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16;

    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16;
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16;
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16;
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16;
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16;

    #[inline(always)]
    fn abs_f16(self, a: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_abs_ps(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn neg_f16(self, a: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { xor_ps(self.f16_to_f32(a), _mm512_set1_ps(-0.0)) })
    }

    #[inline(always)]
    fn sqrt_f16(self, a: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_sqrt_ps(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn add_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_add_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn sub_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_sub_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_mul_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn div_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_div_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn min_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_min_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn max_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_max_ps(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_fmadd_ps(self.f16_to_f32(a), self.f16_to_f32(b), self.f16_to_f32(c)) })
    }

    #[inline(always)]
    fn mul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_fmsub_ps(self.f16_to_f32(a), self.f16_to_f32(b), self.f16_to_f32(c)) })
    }

    #[inline(always)]
    fn nmul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_fnmadd_ps(self.f16_to_f32(a), self.f16_to_f32(b), self.f16_to_f32(c)) })
    }

    #[inline(always)]
    fn nmul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_fnmsub_ps(self.f16_to_f32(a), self.f16_to_f32(b), self.f16_to_f32(c)) })
    }

    #[inline(always)]
    fn reduce_sum_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { _mm512_reduce_add_ps(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_max_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { _mm512_reduce_max_ps(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_min_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { _mm512_reduce_min_ps(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn eq_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ne_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn lt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn le_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn gt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ge_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_GE_OQ>(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { _mm512_mask_blend_ps(self.m16_to_m32(mask).0, self.f16_to_f32(b), self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn abs_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_abs_ps(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn neg_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { xor_ps(self.bf16_to_f32(a), _mm512_set1_ps(-0.0)) })
    }

    #[inline(always)]
    fn sqrt_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_sqrt_ps(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn add_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_add_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn sub_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_sub_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_mul_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn div_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_div_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn min_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_min_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn max_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_max_ps(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_fmadd_ps(self.bf16_to_f32(a), self.bf16_to_f32(b), self.bf16_to_f32(c)) })
    }

    #[inline(always)]
    fn mul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_fmsub_ps(self.bf16_to_f32(a), self.bf16_to_f32(b), self.bf16_to_f32(c)) })
    }

    #[inline(always)]
    fn nmul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_fnmadd_ps(self.bf16_to_f32(a), self.bf16_to_f32(b), self.bf16_to_f32(c)) })
    }

    #[inline(always)]
    fn nmul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_fnmsub_ps(self.bf16_to_f32(a), self.bf16_to_f32(b), self.bf16_to_f32(c)) })
    }

    #[inline(always)]
    fn reduce_sum_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { _mm512_reduce_add_ps(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_max_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { _mm512_reduce_max_ps(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_min_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { _mm512_reduce_min_ps(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn eq_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ne_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn lt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn le_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn gt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ge_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(Avx512MaskF32(unsafe { _mm512_cmp_ps_mask::<_CMP_GE_OQ>(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn select_bf16(self, mask: Self::m16, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { _mm512_mask_blend_ps(self.m16_to_m32(mask).0, self.bf16_to_f32(b), self.bf16_to_f32(a)) })
    }
}

#[derive(Clone, Copy)]
pub struct HalfImplSoftware;

#[derive(Clone, Copy)]
pub struct HalfImplNative;

impl HalfImpl for HalfImplSoftware {
    type f16 = __m512;
    type bf16 = __m512;
    type m16 = Avx512MaskF32;

    #[rustfmt::skip] #[inline(always)]    fn f16_to_f32(self, a: Self::f16) -> __m512 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_f16(self, a: __m512) -> Self::f16 { a }
    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> __m512 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: __m512) -> Self::bf16 { a }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> Avx512MaskF32 { a }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: Avx512MaskF32) -> Self::m16 { a }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        unsafe { _mm512_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    fn splat_bf16(self, value: bf16) -> Self::bf16 {
        unsafe { _mm512_set1_ps(value.to_f32()) }
    }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        let mut fsrc = [0.0f32; 16];
        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { _mm512_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    fn load_bf16(self, src: &[bf16]) -> Self::bf16 {
        let mut fsrc = [0.0f32; 16];
        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { _mm512_loadu_ps(fsrc.as_ptr()) }
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        let mut fdst = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, value) in dst.iter_mut().zip(fdst) {
            *dst = f16::from_f32(value);
        }
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        let mut fdst = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(fdst.as_mut_ptr(), value) };

        for (dst, value) in dst.iter_mut().zip(fdst) {
            *dst = bf16::from_f32(value);
        }
    }

    #[inline(always)]
    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        unsafe {
            let x = _mm512_castps_si512(a);
            _mm512_castsi512_ps(_mm512_ternarylogic_epi32::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 {
        unsafe {
            let x = _mm512_castps_si512(a);
            _mm512_castsi512_ps(_mm512_ternarylogic_epi32::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { andnot_ps(a, b) }
    }
}

#[inline(always)]
unsafe fn f16_to_bits(a: __m256h) -> __m256i {
    unsafe { std::mem::transmute::<__m256h, __m256i>(a) }
}

#[inline(always)]
unsafe fn f16_from_bits(a: __m256i) -> __m256h {
    unsafe { std::mem::transmute::<__m256i, __m256h>(a) }
}

#[inline(always)]
unsafe fn bf16_to_bits(a: __m256bh) -> __m256i {
    unsafe { std::mem::transmute::<__m256bh, __m256i>(a) }
}

#[inline(always)]
unsafe fn bf16_from_bits(a: __m256i) -> __m256bh {
    unsafe { std::mem::transmute::<__m256i, __m256bh>(a) }
}

impl HalfImpl for HalfImplNative {
    type f16 = __m256h;
    type bf16 = __m256bh;
    type m16 = Avx512MaskF32;

    #[inline(always)]
    fn f16_to_f32(self, a: Self::f16) -> __m512 {
        let bits = unsafe { f16_to_bits(a) };
        unsafe { _mm512_cvtph_ps(bits) }
    }

    #[inline(always)]
    fn f32_to_f16(self, a: __m512) -> Self::f16 {
        let bits = unsafe { _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT) };
        unsafe { f16_from_bits(bits) }
    }

    #[inline(always)]
    fn bf16_to_f32(self, a: Self::bf16) -> __m512 {
        unsafe { _mm512_cvtpbh_ps(a) }
    }

    #[inline(always)]
    fn f32_to_bf16(self, a: __m512) -> Self::bf16 {
        unsafe { _mm512_cvtneps_pbh(a) }
    }

    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> Avx512MaskF32 { a }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: Avx512MaskF32) -> Self::m16 { a }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_set1_epi16(value.to_bits() as i16)) }
    }

    #[inline(always)]
    fn splat_bf16(self, value: bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_set1_epi16(value.to_bits() as i16)) }
    }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_loadu_si256(src.as_ptr().cast())) }
    }

    #[inline(always)]
    fn load_bf16(self, src: &[bf16]) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_loadu_si256(src.as_ptr().cast())) }
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), f16_to_bits(value)) };
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), bf16_to_bits(value)) };
    }

    #[inline(always)]
    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_and_si256(f16_to_bits(a), f16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_or_si256(f16_to_bits(a), f16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_xor_si256(f16_to_bits(a), f16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_xor_si256(f16_to_bits(a), _mm256_set1_epi16(-1))) }
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_andnot_si256(f16_to_bits(a), f16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_and_si256(bf16_to_bits(a), bf16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_or_si256(bf16_to_bits(a), bf16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_xor_si256(bf16_to_bits(a), bf16_to_bits(b))) }
    }

    #[inline(always)]
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_xor_si256(bf16_to_bits(a), _mm256_set1_epi16(-1))) }
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { bf16_from_bits(_mm256_andnot_si256(bf16_to_bits(a), bf16_to_bits(b))) }
    }

    #[inline(always)]
    fn abs_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { _mm256_abs_ph(a) }
    }

    #[inline(always)]
    fn neg_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { f16_from_bits(_mm256_xor_si256(f16_to_bits(a), _mm256_set1_epi16(i16::MIN))) }
    }

    #[inline(always)]
    fn sqrt_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { _mm256_sqrt_ph(a) }
    }

    #[inline(always)]
    fn add_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_add_ph(a, b) }
    }

    #[inline(always)]
    fn sub_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_sub_ph(a, b) }
    }

    #[inline(always)]
    fn mul_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_mul_ph(a, b) }
    }

    #[inline(always)]
    fn div_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_div_ph(a, b) }
    }

    #[inline(always)]
    fn min_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_min_ph(a, b) }
    }

    #[inline(always)]
    fn max_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_max_ph(a, b) }
    }

    #[inline(always)]
    fn mul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { _mm256_fmadd_ph(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { _mm256_fmsub_ph(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { _mm256_fnmadd_ph(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { _mm256_fnmsub_ph(a, b, c) }
    }

    #[inline(always)]
    fn eq_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { Avx512MaskF32(_mm256_cmp_ph_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { _mm256_mask_blend_ph(mask.0, b, a) }
    }
}

#[derive(Copy, Clone)]
pub struct Avx512<H: HalfImpl>(pub H);

impl Default for Avx512<HalfImplSoftware> {
    fn default() -> Self {
        Avx512(HalfImplSoftware)
    }
}

#[target_feature(enable = "avx512f", enable = "fma")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    if std::arch::is_x86_feature_detected!("avx512fp16") && std::arch::is_x86_feature_detected!("avx512vl") && std::arch::is_x86_feature_detected!("avx512bf16") {
        unsafe { vectorize_native(op) }
    } else {
        op.call(Avx512(HalfImplSoftware))
    }
}

#[target_feature(enable = "avx512f", enable = "fma", enable = "avx512fp16", enable = "avx512vl", enable = "avx512bf16")]
pub unsafe fn vectorize_native<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Avx512(HalfImplNative))
}

impl<H: HalfImpl> FloatSimd for Avx512<H> {
    type f32 = __m512;
    type f64 = __m512d;

    type m32 = Avx512MaskF32;
    type m64 = Avx512MaskF64;

    const f32_lanes: usize = 16;
    const f64_lanes: usize = 8;
    const m64_lanes: usize = Self::f64_lanes;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { _mm512_set1_ps(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { _mm512_set1_pd(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { _mm512_loadu_ps(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { _mm512_loadu_pd(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { _mm512_storeu_ps(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { _mm512_storeu_pd(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_abs_ps(a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_abs_pd(a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_add_ps(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_add_pd(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_sub_ps(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_sub_pd(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_mul_ps(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_mul_pd(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_div_ps(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_div_pd(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_max_ps(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_max_pd(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_min_ps(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_min_pd(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { xor_ps(a, _mm512_set1_ps(-0.0)) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { xor_pd(a, _mm512_set1_pd(-0.0)) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { _mm512_sqrt_ps(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { _mm512_sqrt_pd(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fnmadd_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fnmadd_pd(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { _mm512_fnmsub_ps(a, b, c) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { _mm512_fnmsub_pd(a, b, c) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LE_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        unsafe { Avx512MaskF32(_mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        unsafe { Avx512MaskF64(_mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(a, b)) }
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { _mm512_mask_blend_ps(mask.0, b, a) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { _mm512_mask_blend_pd(mask.0, b, a) }
    }

    #[inline(always)]
    fn bitand_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { and_ps(a, b) }
    }

    #[inline(always)]
    fn bitor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { or_ps(a, b) }
    }

    #[inline(always)]
    fn bitxor_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { xor_ps(a, b) }
    }

    #[inline(always)]
    fn bitand_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { and_pd(a, b) }
    }

    #[inline(always)]
    fn bitor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { or_pd(a, b) }
    }

    #[inline(always)]
    fn bitxor_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { xor_pd(a, b) }
    }

    #[inline(always)]
    fn bitnot_f32(self, a: Self::f32) -> Self::f32 {
        unsafe {
            let x = _mm512_castps_si512(a);
            _mm512_castsi512_ps(_mm512_ternarylogic_epi32::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe {
            let x = _mm512_castpd_si512(a);
            _mm512_castsi512_pd(_mm512_ternarylogic_epi64::<0x55>(x, x, x))
        }
    }

    #[inline(always)]
    fn bitandnot_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { andnot_ps(a, b) }
    }

    #[inline(always)]
    fn bitandnot_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { andnot_pd(a, b) }
    }

    #[inline(always)]
    fn reduce_max_f32(self, a: Self::f32) -> f32 {
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
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_max_pd(lo, hi);
            let s = _mm_max_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
            let s = _mm_max_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
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
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_min_pd(lo, hi);
            let s = _mm_min_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
            let s = _mm_min_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
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
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe {
            let (lo, hi) = split_f64s(a);
            let s = _mm256_add_pd(lo, hi);
            let s = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd::<1>(s));
            let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
            _mm_cvtsd_f64(s)
        }
    }
}

impl<H: HalfImpl> HalfSimd for Avx512<H> {
    type f16 = H::f16;
    type bf16 = H::bf16;
    type m16 = H::m16;

    const f16_lanes: usize = H::f16_lanes;

    #[rustfmt::skip] #[inline(always)]    fn f16_to_f32(self, a: Self::f16) -> Self::f32 { self.0.f16_to_f32(a) }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_f16(self, a: Self::f32) -> Self::f16 { self.0.f32_to_f16(a) }
    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> Self::f32 { self.0.bf16_to_f32(a) }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: Self::f32) -> Self::bf16 { self.0.f32_to_bf16(a) }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> Self::m32 { self.0.m16_to_m32(a) }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: Self::m32) -> Self::m16 { self.0.m32_to_m16(a) }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        self.0.splat_f16(value)
    }

    #[inline(always)]
    fn splat_bf16(self, value: bf16) -> Self::bf16 {
        self.0.splat_bf16(value)
    }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        assert!(src.len() >= Self::f16_lanes);
        self.0.load_f16(src)
    }

    #[inline(always)]
    fn load_bf16(self, src: &[bf16]) -> Self::bf16 {
        assert!(src.len() >= Self::f16_lanes);
        self.0.load_bf16(src)
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        assert!(dst.len() >= Self::f16_lanes);
        self.0.store_f16(dst, value)
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        assert!(dst.len() >= Self::f16_lanes);
        self.0.store_bf16(dst, value)
    }

    #[inline(always)]
    unsafe fn load_unchecked_f16(self, src: &[f16]) -> Self::f16 {
        self.0.load_f16(src)
    }

    #[inline(always)]
    unsafe fn load_unchecked_bf16(self, src: &[bf16]) -> Self::bf16 {
        self.0.load_bf16(src)
    }

    #[inline(always)]
    unsafe fn store_unchecked_f16(self, dst: &mut [f16], value: Self::f16) {
        self.0.store_f16(dst, value)
    }

    #[inline(always)]
    unsafe fn store_unchecked_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        self.0.store_bf16(dst, value)
    }

    #[inline(always)]
    fn abs_f16(self, a: Self::f16) -> Self::f16 {
        self.0.abs_f16(a)
    }

    #[inline(always)]
    fn neg_f16(self, a: Self::f16) -> Self::f16 {
        self.0.neg_f16(a)
    }

    #[inline(always)]
    fn sqrt_f16(self, a: Self::f16) -> Self::f16 {
        self.0.sqrt_f16(a)
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        self.0.bitnot_f16(a)
    }

    #[inline(always)]
    fn add_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.add_f16(a, b)
    }

    #[inline(always)]
    fn sub_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.sub_f16(a, b)
    }

    #[inline(always)]
    fn mul_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.mul_f16(a, b)
    }

    #[inline(always)]
    fn div_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.div_f16(a, b)
    }

    #[inline(always)]
    fn min_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.min_f16(a, b)
    }

    #[inline(always)]
    fn max_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.max_f16(a, b)
    }

    #[inline(always)]
    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.bitand_f16(a, b)
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.bitor_f16(a, b)
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.bitxor_f16(a, b)
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.bitandnot_f16(a, b)
    }

    #[inline(always)]
    fn mul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.0.mul_add_f16(a, b, c)
    }

    #[inline(always)]
    fn mul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.0.mul_sub_f16(a, b, c)
    }

    #[inline(always)]
    fn nmul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.0.nmul_add_f16(a, b, c)
    }

    #[inline(always)]
    fn nmul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.0.nmul_sub_f16(a, b, c)
    }

    #[inline(always)]
    fn reduce_sum_f16(self, a: Self::f16) -> f16 {
        self.0.reduce_sum_f16(a)
    }

    #[inline(always)]
    fn reduce_max_f16(self, a: Self::f16) -> f16 {
        self.0.reduce_max_f16(a)
    }

    #[inline(always)]
    fn reduce_min_f16(self, a: Self::f16) -> f16 {
        self.0.reduce_min_f16(a)
    }

    #[inline(always)]
    fn eq_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.eq_f16(a, b)
    }

    #[inline(always)]
    fn ne_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.ne_f16(a, b)
    }

    #[inline(always)]
    fn lt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.lt_f16(a, b)
    }

    #[inline(always)]
    fn le_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.le_f16(a, b)
    }

    #[inline(always)]
    fn gt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.gt_f16(a, b)
    }

    #[inline(always)]
    fn ge_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.0.ge_f16(a, b)
    }

    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.0.select_f16(mask, a, b)
    }

    #[inline(always)]
    fn abs_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.0.abs_bf16(a)
    }

    #[inline(always)]
    fn neg_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.0.neg_bf16(a)
    }

    #[inline(always)]
    fn sqrt_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.0.sqrt_bf16(a)
    }

    #[inline(always)]
    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.0.bitnot_bf16(a)
    }

    #[inline(always)]
    fn add_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.add_bf16(a, b)
    }

    #[inline(always)]
    fn sub_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.sub_bf16(a, b)
    }

    #[inline(always)]
    fn mul_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.mul_bf16(a, b)
    }

    #[inline(always)]
    fn div_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.div_bf16(a, b)
    }

    #[inline(always)]
    fn min_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.min_bf16(a, b)
    }

    #[inline(always)]
    fn max_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.max_bf16(a, b)
    }

    #[inline(always)]
    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.bitand_bf16(a, b)
    }

    #[inline(always)]
    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.bitor_bf16(a, b)
    }

    #[inline(always)]
    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.bitxor_bf16(a, b)
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.bitandnot_bf16(a, b)
    }

    #[inline(always)]
    fn mul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.0.mul_add_bf16(a, b, c)
    }

    #[inline(always)]
    fn mul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.0.mul_sub_bf16(a, b, c)
    }

    #[inline(always)]
    fn nmul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.0.nmul_add_bf16(a, b, c)
    }

    #[inline(always)]
    fn nmul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.0.nmul_sub_bf16(a, b, c)
    }

    #[inline(always)]
    fn reduce_sum_bf16(self, a: Self::bf16) -> bf16 {
        self.0.reduce_sum_bf16(a)
    }

    #[inline(always)]
    fn reduce_max_bf16(self, a: Self::bf16) -> bf16 {
        self.0.reduce_max_bf16(a)
    }

    #[inline(always)]
    fn reduce_min_bf16(self, a: Self::bf16) -> bf16 {
        self.0.reduce_min_bf16(a)
    }

    #[inline(always)]
    fn eq_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.eq_bf16(a, b)
    }

    #[inline(always)]
    fn ne_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.ne_bf16(a, b)
    }

    #[inline(always)]
    fn lt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.lt_bf16(a, b)
    }

    #[inline(always)]
    fn le_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.le_bf16(a, b)
    }

    #[inline(always)]
    fn gt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.gt_bf16(a, b)
    }

    #[inline(always)]
    fn ge_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.0.ge_bf16(a, b)
    }

    #[inline(always)]
    fn select_bf16(self, mask: Self::m16, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.0.select_bf16(mask, a, b)
    }
}

impl<H: HalfImpl> Simd for Avx512<H> {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    if !std::arch::is_x86_feature_detected!("avx512f") || !std::arch::is_x86_feature_detected!("fma") {
        eprintln!("avx512 backend test skipped: CPU does not support avx512f+fma");
        return;
    }

    unsafe { vectorize(crate::backend::test_utils::Run(check)) };
}

#[cfg(test)]
crate::backend_tests!(Avx512<HalfImplSoftware>, 16, 8);
