#![allow(clippy::incompatible_msrv)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use half::{bf16, f16};

use crate::{
    SimdFn,
    traits::{FloatSimd, HalfSimd, Simd, SimdMask},
};

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF32(uint32x4_t); // 4 lanes

impl SimdMask for NeonMaskF32 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(unsafe { vdupq_n_u32(if v { u32::MAX } else { 0 }) })
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        let lanes = [
            if v & 1 != 0 { u32::MAX } else { 0 },
            if v & 2 != 0 { u32::MAX } else { 0 },
            if v & 4 != 0 { u32::MAX } else { 0 },
            if v & 8 != 0 { u32::MAX } else { 0 },
        ];
        Self(unsafe { vld1q_u32(lanes.as_ptr()) })
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        let mut lanes = [0; 4];
        unsafe { vst1q_u32(lanes.as_mut_ptr(), self.0) };
        lanes.iter().enumerate().fold(0, |bits, (i, &lane)| bits | (u32::from(lane != 0) << i))
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(unsafe { vandq_u32(self.0, other.0) })
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(unsafe { vorrq_u32(self.0, other.0) })
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { vmvnq_u32(self.0) })
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u32(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { vminvq_u32(self.0) != 0 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF64(uint64x2_t); // 2 lanes

impl SimdMask for NeonMaskF64 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(unsafe { vdupq_n_u64(if v { u64::MAX } else { 0 }) })
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        let lanes = [if v & 1 != 0 { u64::MAX } else { 0 }, if v & 2 != 0 { u64::MAX } else { 0 }];
        Self(unsafe { vld1q_u64(lanes.as_ptr()) })
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        let mut lanes = [0; 2];
        unsafe { vst1q_u64(lanes.as_mut_ptr(), self.0) };
        lanes.iter().enumerate().fold(0, |bits, (i, &lane)| bits | (u32::from(lane != 0) << i))
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(unsafe { vandq_u64(self.0, other.0) })
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(unsafe { vorrq_u64(self.0, other.0) })
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { veorq_u64(self.0, vdupq_n_u64(u64::MAX)) })
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_u64(self.0)) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { vminvq_u32(vreinterpretq_u32_u64(self.0)) != 0 }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF16(uint16x8_t); // 8 lanes

impl SimdMask for NeonMaskF16 {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        Self(unsafe { vdupq_n_u16(if v { u16::MAX } else { 0 }) })
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        let lanes = [
            if v & 1 != 0 { u16::MAX } else { 0 },
            if v & 2 != 0 { u16::MAX } else { 0 },
            if v & 4 != 0 { u16::MAX } else { 0 },
            if v & 8 != 0 { u16::MAX } else { 0 },
            if v & 16 != 0 { u16::MAX } else { 0 },
            if v & 32 != 0 { u16::MAX } else { 0 },
            if v & 64 != 0 { u16::MAX } else { 0 },
            if v & 128 != 0 { u16::MAX } else { 0 },
        ];
        Self(unsafe { vld1q_u16(lanes.as_ptr()) })
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        let mut lanes = [0; 8];
        unsafe { vst1q_u16(lanes.as_mut_ptr(), self.0) };
        lanes.iter().enumerate().fold(0, |bits, (i, &lane)| bits | (u32::from(lane != 0) << i))
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(unsafe { vandq_u16(self.0, other.0) })
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        Self(unsafe { vorrq_u16(self.0, other.0) })
    }

    #[inline(always)]
    fn not(self) -> Self {
        Self(unsafe { vmvnq_u16(self.0) })
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u16(self.0) != 0 }
    }

    #[inline(always)]
    fn all(self) -> bool {
        unsafe { vminvq_u16(self.0) != 0 }
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

bit_helper!(and_ps, float32x4_t, vreinterpretq_u32_f32, vreinterpretq_f32_u32, vandq_u32);
bit_helper!(or_ps, float32x4_t, vreinterpretq_u32_f32, vreinterpretq_f32_u32, vorrq_u32);
bit_helper!(xor_ps, float32x4_t, vreinterpretq_u32_f32, vreinterpretq_f32_u32, veorq_u32);

bit_helper!(and_pd, float64x2_t, vreinterpretq_u64_f64, vreinterpretq_f64_u64, vandq_u64);
bit_helper!(or_pd, float64x2_t, vreinterpretq_u64_f64, vreinterpretq_f64_u64, vorrq_u64);
bit_helper!(xor_pd, float64x2_t, vreinterpretq_u64_f64, vreinterpretq_f64_u64, veorq_u64);

#[inline(always)]
unsafe fn andnot_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b), vreinterpretq_u32_f32(a))) }
}

#[inline(always)]
unsafe fn andnot_pd(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    unsafe { vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(b), vreinterpretq_u64_f64(a))) }
}

#[inline(always)]
unsafe fn f16x8_to_bits(a: float16x8_t) -> uint16x8_t {
    unsafe { std::mem::transmute::<float16x8_t, uint16x8_t>(a) }
}

#[inline(always)]
unsafe fn f16x8_from_bits(a: uint16x8_t) -> float16x8_t {
    unsafe { std::mem::transmute::<uint16x8_t, float16x8_t>(a) }
}

#[inline(always)]
unsafe fn mask_f16x8_to_f32(a: NeonMaskF16) -> NeonMaskF32 {
    unsafe {
        let tested = vtstq_u16(a.0, a.0);
        NeonMaskF32(vreinterpretq_u32_s32(vmovl_s16(vreinterpret_s16_u16(vget_low_u16(tested)))))
    }
}

#[inline(always)]
unsafe fn mask_f32_to_f16x8(a: NeonMaskF32) -> NeonMaskF16 {
    unsafe {
        let narrowed = vmovn_u32(a.0);
        NeonMaskF16(vcombine_u16(narrowed, narrowed))
    }
}

pub trait HalfImpl: Copy + Send + Sync + 'static {
    type f16: Copy + Send + Sync + 'static;
    type bf16: Copy + Send + Sync + 'static;
    type m16: SimdMask + Copy + Send + Sync + 'static;

    const f16_lanes: usize = 4;
    const bf16_lanes: usize = Self::f16_lanes;

    fn f16_to_f32(self, a: Self::f16) -> float32x4_t;
    fn f32_to_f16(self, a: float32x4_t) -> Self::f16;
    fn bf16_to_f32(self, a: Self::bf16) -> float32x4_t;
    fn f32_to_bf16(self, a: float32x4_t) -> Self::bf16;
    fn m16_to_m32(self, a: Self::m16) -> NeonMaskF32;
    fn m32_to_m16(self, a: NeonMaskF32) -> Self::m16;

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
        self.f32_to_f16(unsafe { vabsq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn neg_f16(self, a: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vnegq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn sqrt_f16(self, a: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vsqrtq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn add_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vaddq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn sub_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vsubq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vmulq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn div_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vdivq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn min_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vminq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn max_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vmaxq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vfmaq_f32(self.f16_to_f32(c), self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vfmaq_f32(vnegq_f32(self.f16_to_f32(c)), self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn nmul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vfmsq_f32(self.f16_to_f32(c), self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn nmul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vfmsq_f32(vnegq_f32(self.f16_to_f32(c)), self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn reduce_sum_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { vaddvq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_max_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { vmaxvq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_min_f16(self, a: Self::f16) -> f16 {
        f16::from_f32(unsafe { vminvq_f32(self.f16_to_f32(a)) })
    }

    #[inline(always)]
    fn eq_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vceqq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ne_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vmvnq_u32(vceqq_f32(self.f16_to_f32(a), self.f16_to_f32(b))) }))
    }

    #[inline(always)]
    fn lt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcltq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn le_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcleq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn gt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcgtq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ge_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcgeq_f32(self.f16_to_f32(a), self.f16_to_f32(b)) }))
    }

    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(unsafe { vbslq_f32(self.m16_to_m32(mask).0, self.f16_to_f32(a), self.f16_to_f32(b)) })
    }

    #[inline(always)]
    fn abs_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vabsq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn neg_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vnegq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn sqrt_bf16(self, a: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vsqrtq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn add_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vaddq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn sub_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vsubq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vmulq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn div_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vdivq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn min_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vminq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn max_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vmaxq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vfmaq_f32(self.bf16_to_f32(c), self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn mul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vfmaq_f32(vnegq_f32(self.bf16_to_f32(c)), self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn nmul_add_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vfmsq_f32(self.bf16_to_f32(c), self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn nmul_sub_bf16(self, a: Self::bf16, b: Self::bf16, c: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vfmsq_f32(vnegq_f32(self.bf16_to_f32(c)), self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }

    #[inline(always)]
    fn reduce_sum_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { vaddvq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_max_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { vmaxvq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn reduce_min_bf16(self, a: Self::bf16) -> bf16 {
        bf16::from_f32(unsafe { vminvq_f32(self.bf16_to_f32(a)) })
    }

    #[inline(always)]
    fn eq_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vceqq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ne_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vmvnq_u32(vceqq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b))) }))
    }

    #[inline(always)]
    fn lt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcltq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn le_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcleq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn gt_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcgtq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn ge_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::m16 {
        self.m32_to_m16(NeonMaskF32(unsafe { vcgeq_f32(self.bf16_to_f32(a), self.bf16_to_f32(b)) }))
    }

    #[inline(always)]
    fn select_bf16(self, mask: Self::m16, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(unsafe { vbslq_f32(self.m16_to_m32(mask).0, self.bf16_to_f32(a), self.bf16_to_f32(b)) })
    }
}

#[derive(Clone, Copy)]
pub struct HalfImplSoftware;

#[derive(Clone, Copy)]
pub struct HalfImplF16;

impl HalfImpl for HalfImplSoftware {
    type f16 = float32x4_t;
    type bf16 = float32x4_t;
    type m16 = NeonMaskF32;

    #[rustfmt::skip] #[inline(always)]    fn f16_to_f32(self, a: Self::f16) -> float32x4_t { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_f16(self, a: float32x4_t) -> Self::f16 { a }
    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> float32x4_t { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: float32x4_t) -> Self::bf16 { a }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> NeonMaskF32 { a }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: NeonMaskF32) -> Self::m16 { a }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        unsafe { vdupq_n_f32(value.to_f32()) }
    }

    #[inline(always)]
    fn splat_bf16(self, value: bf16) -> Self::bf16 {
        unsafe { vdupq_n_f32(value.to_f32()) }
    }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        let mut fsrc = [0.0f32; 4];
        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { vld1q_f32(fsrc.as_ptr()) }
    }

    #[inline(always)]
    fn load_bf16(self, src: &[bf16]) -> Self::bf16 {
        let mut fsrc = [0.0f32; 4];
        for (dst, src) in fsrc.iter_mut().zip(src) {
            *dst = src.to_f32();
        }

        unsafe { vld1q_f32(fsrc.as_ptr()) }
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        let mut fdst = [0.0f32; 4];
        unsafe { vst1q_f32(fdst.as_mut_ptr(), value) };

        for (dst, value) in dst.iter_mut().zip(fdst) {
            *dst = f16::from_f32(value);
        }
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        let mut fdst = [0.0f32; 4];
        unsafe { vst1q_f32(fdst.as_mut_ptr(), value) };

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
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
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
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
    }

    #[inline(always)]
    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        unsafe { andnot_ps(a, b) }
    }
}

impl HalfImpl for HalfImplF16 {
    type f16 = float16x8_t;
    type bf16 = float32x4_t;
    type m16 = NeonMaskF16;

    const f16_lanes: usize = 8;
    const bf16_lanes: usize = 4;

    #[inline(always)]
    fn f16_to_f32(self, a: Self::f16) -> float32x4_t {
        unsafe { vcvt_f32_f16(vget_low_f16(a)) }
    }

    #[inline(always)]
    fn f32_to_f16(self, a: float32x4_t) -> Self::f16 {
        unsafe { vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(vdupq_n_f32(0.0))) }
    }

    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> float32x4_t { HalfImplSoftware.bf16_to_f32(a) }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: float32x4_t) -> Self::bf16 { HalfImplSoftware.f32_to_bf16(a) }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> NeonMaskF32 { unsafe { mask_f16x8_to_f32(a) } }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: NeonMaskF32) -> Self::m16 { unsafe { mask_f32_to_f16x8(a) } }

    #[inline(always)]
    fn splat_f16(self, value: f16) -> Self::f16 {
        unsafe { f16x8_from_bits(vdupq_n_u16(value.to_bits())) }
    }

    #[rustfmt::skip] #[inline(always)]    fn splat_bf16(self, value: bf16) -> Self::bf16 { HalfImplSoftware.splat_bf16(value) }

    #[inline(always)]
    fn load_f16(self, src: &[f16]) -> Self::f16 {
        unsafe { f16x8_from_bits(vld1q_u16(src.as_ptr().cast())) }
    }

    #[rustfmt::skip] #[inline(always)]    fn load_bf16(self, src: &[bf16]) -> Self::bf16 { HalfImplSoftware.load_bf16(src) }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        unsafe { vst1q_u16(dst.as_mut_ptr().cast(), f16x8_to_bits(value)) };
    }

    #[rustfmt::skip] #[inline(always)]    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) { HalfImplSoftware.store_bf16(dst, value) }

    #[inline(always)]
    fn bitand_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16x8_from_bits(vandq_u16(f16x8_to_bits(a), f16x8_to_bits(b))) }
    }

    #[inline(always)]
    fn bitor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16x8_from_bits(vorrq_u16(f16x8_to_bits(a), f16x8_to_bits(b))) }
    }

    #[inline(always)]
    fn bitxor_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16x8_from_bits(veorq_u16(f16x8_to_bits(a), f16x8_to_bits(b))) }
    }

    #[inline(always)]
    fn bitnot_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { f16x8_from_bits(vmvnq_u16(f16x8_to_bits(a))) }
    }

    #[inline(always)]
    fn bitandnot_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { f16x8_from_bits(vbicq_u16(f16x8_to_bits(b), f16x8_to_bits(a))) }
    }

    #[rustfmt::skip] #[inline(always)]    fn bitand_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 { HalfImplSoftware.bitand_bf16(a, b) }
    #[rustfmt::skip] #[inline(always)]    fn bitor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 { HalfImplSoftware.bitor_bf16(a, b) }
    #[rustfmt::skip] #[inline(always)]    fn bitxor_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 { HalfImplSoftware.bitxor_bf16(a, b) }
    #[rustfmt::skip] #[inline(always)]    fn bitnot_bf16(self, a: Self::bf16) -> Self::bf16 { HalfImplSoftware.bitnot_bf16(a) }
    #[rustfmt::skip] #[inline(always)]    fn bitandnot_bf16(self, a: Self::bf16, b: Self::bf16) -> Self::bf16 { HalfImplSoftware.bitandnot_bf16(a, b) }

    #[inline(always)]
    fn abs_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { vabsq_f16(a) }
    }

    #[inline(always)]
    fn neg_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { vnegq_f16(a) }
    }

    #[inline(always)]
    fn sqrt_f16(self, a: Self::f16) -> Self::f16 {
        unsafe { vsqrtq_f16(a) }
    }

    #[inline(always)]
    fn add_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vaddq_f16(a, b) }
    }

    #[inline(always)]
    fn sub_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vsubq_f16(a, b) }
    }

    #[inline(always)]
    fn mul_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vmulq_f16(a, b) }
    }

    #[inline(always)]
    fn div_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vdivq_f16(a, b) }
    }

    #[inline(always)]
    fn min_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vminq_f16(a, b) }
    }

    #[inline(always)]
    fn max_f16(self, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vmaxq_f16(a, b) }
    }

    #[inline(always)]
    fn mul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { vfmaq_f16(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { vfmaq_f16(vnegq_f16(c), a, b) }
    }

    #[inline(always)]
    fn nmul_add_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { vfmsq_f16(c, a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f16(self, a: Self::f16, b: Self::f16, c: Self::f16) -> Self::f16 {
        unsafe { vfmsq_f16(vnegq_f16(c), a, b) }
    }

    #[inline(always)]
    fn reduce_sum_f16(self, a: Self::f16) -> f16 {
        unsafe {
            let pair = vpaddq_f16(a, a);
            let quad = vpaddq_f16(pair, pair);
            let total = vpadd_f16(vget_low_f16(quad), vget_low_f16(quad));
            f16::from_bits(vget_lane_u16::<0>(vreinterpret_u16_f16(total)))
        }
    }

    #[inline(always)]
    fn reduce_max_f16(self, a: Self::f16) -> f16 {
        unsafe {
            let mut value = a;
            value = vmaxq_f16(value, vextq_f16::<4>(value, value));
            value = vmaxq_f16(value, vextq_f16::<2>(value, value));
            value = vmaxq_f16(value, vextq_f16::<1>(value, value));
            f16::from_bits(vget_lane_u16::<0>(vget_low_u16(vreinterpretq_u16_f16(value))))
        }
    }

    #[inline(always)]
    fn reduce_min_f16(self, a: Self::f16) -> f16 {
        unsafe {
            let mut value = a;
            value = vminq_f16(value, vextq_f16::<4>(value, value));
            value = vminq_f16(value, vextq_f16::<2>(value, value));
            value = vminq_f16(value, vextq_f16::<1>(value, value));
            f16::from_bits(vget_lane_u16::<0>(vget_low_u16(vreinterpretq_u16_f16(value))))
        }
    }

    #[inline(always)]
    fn eq_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vceqq_f16(a, b)) }
    }

    #[inline(always)]
    fn ne_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vmvnq_u16(vceqq_f16(a, b))) }
    }

    #[inline(always)]
    fn lt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vcltq_f16(a, b)) }
    }

    #[inline(always)]
    fn le_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vcleq_f16(a, b)) }
    }

    #[inline(always)]
    fn gt_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vcgtq_f16(a, b)) }
    }

    #[inline(always)]
    fn ge_f16(self, a: Self::f16, b: Self::f16) -> Self::m16 {
        unsafe { NeonMaskF16(vcgeq_f16(a, b)) }
    }

    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        unsafe { vbslq_f16(mask.0, a, b) }
    }
}

#[derive(Copy, Clone)]
pub struct Neon<H: HalfImpl>(pub H);

impl Default for Neon<HalfImplSoftware> {
    fn default() -> Self {
        Neon(HalfImplSoftware)
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    if std::arch::is_aarch64_feature_detected!("fp16") {
        unsafe { vectorize_f16(op) }
    } else {
        op.call(Neon(HalfImplSoftware))
    }
}

#[target_feature(enable = "neon", enable = "fp16")]
pub unsafe fn vectorize_f16<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Neon(HalfImplF16))
}

impl<H: HalfImpl> FloatSimd for Neon<H> {
    type f32 = float32x4_t;
    type f64 = float64x2_t;

    type m32 = NeonMaskF32;
    type m64 = NeonMaskF64;

    const f32_lanes: usize = 4;
    const f64_lanes: usize = 2;
    const m64_lanes: usize = Self::f64_lanes;

    #[inline(always)]
    fn splat_f32(self, value: f32) -> Self::f32 {
        unsafe { vdupq_n_f32(value) }
    }

    #[inline(always)]
    fn splat_f64(self, value: f64) -> Self::f64 {
        unsafe { vdupq_n_f64(value) }
    }

    #[inline(always)]
    fn load_f32(self, src: &[f32]) -> Self::f32 {
        assert!(src.len() >= Self::f32_lanes);
        unsafe { vld1q_f32(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64(self, src: &[f64]) -> Self::f64 {
        assert!(src.len() >= Self::f64_lanes);
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32(self, dst: &mut [f32], value: Self::f32) {
        assert!(dst.len() >= Self::f32_lanes);
        unsafe { vst1q_f32(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64(self, dst: &mut [f64], value: Self::f64) {
        assert!(dst.len() >= Self::f64_lanes);
        unsafe { vst1q_f64(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    unsafe fn load_unchecked_f32(self, src: &[f32]) -> Self::f32 {
        unsafe { vld1q_f32(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn load_unchecked_f64(self, src: &[f64]) -> Self::f64 {
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f32(self, dst: &mut [f32], value: Self::f32) {
        unsafe { vst1q_f32(dst.as_mut_ptr(), value) }
    }

    #[inline(always)]
    unsafe fn store_unchecked_f64(self, dst: &mut [f64], value: Self::f64) {
        unsafe { vst1q_f64(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn abs_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vabsq_f32(a) }
    }

    #[inline(always)]
    fn abs_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vabsq_f64(a) }
    }

    #[inline(always)]
    fn add_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vaddq_f32(a, b) }
    }

    #[inline(always)]
    fn add_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vaddq_f64(a, b) }
    }

    #[inline(always)]
    fn sub_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vsubq_f32(a, b) }
    }

    #[inline(always)]
    fn sub_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vsubq_f64(a, b) }
    }

    #[inline(always)]
    fn mul_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vmulq_f32(a, b) }
    }

    #[inline(always)]
    fn mul_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vmulq_f64(a, b) }
    }

    #[inline(always)]
    fn div_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vdivq_f32(a, b) }
    }

    #[inline(always)]
    fn div_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vdivq_f64(a, b) }
    }

    #[inline(always)]
    fn max_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vmaxq_f32(a, b) }
    }

    #[inline(always)]
    fn max_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vmaxq_f64(a, b) }
    }

    #[inline(always)]
    fn min_f32(self, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vminq_f32(a, b) }
    }

    #[inline(always)]
    fn min_f64(self, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vminq_f64(a, b) }
    }

    #[inline(always)]
    fn neg_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vnegq_f32(a) }
    }

    #[inline(always)]
    fn neg_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vnegq_f64(a) }
    }

    #[inline(always)]
    fn sqrt_f32(self, a: Self::f32) -> Self::f32 {
        unsafe { vsqrtq_f32(a) }
    }

    #[inline(always)]
    fn sqrt_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vsqrtq_f64(a) }
    }

    #[inline(always)]
    fn mul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline(always)]
    fn mul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmaq_f64(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        // `a * b - c`; negating `c` keeps the sign of zero correct.
        unsafe { vfmaq_f32(vnegq_f32(c), a, b) }
    }

    #[inline(always)]
    fn mul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmaq_f64(vnegq_f64(c), a, b) }
    }

    #[inline(always)]
    fn nmul_add_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        // `-(a * b) + c`.
        unsafe { vfmsq_f32(c, a, b) }
    }

    #[inline(always)]
    fn nmul_add_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmsq_f64(c, a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f32(self, a: Self::f32, b: Self::f32, c: Self::f32) -> Self::f32 {
        // `-(a * b) - c`.
        unsafe { vfmsq_f32(vnegq_f32(c), a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f64(self, a: Self::f64, b: Self::f64, c: Self::f64) -> Self::f64 {
        unsafe { vfmsq_f64(vnegq_f64(c), a, b) }
    }

    #[inline(always)]
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vceqq_f32(a, b) })
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { vceqq_f64(a, b) })
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vcgeq_f32(a, b) })
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { vcgeq_f64(a, b) })
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vcgtq_f32(a, b) })
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { vcgtq_f64(a, b) })
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vcleq_f32(a, b) })
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { vcleq_f64(a, b) })
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vcltq_f32(a, b) })
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { vcltq_f64(a, b) })
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32 {
        NeonMaskF32(unsafe { vmvnq_u32(vceqq_f32(a, b)) })
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64 {
        NeonMaskF64(unsafe { veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX)) })
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vbslq_f32(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        unsafe { vbslq_f64(mask.0, a, b) }
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
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
    }

    #[inline(always)]
    fn bitnot_f64(self, a: Self::f64) -> Self::f64 {
        unsafe { vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vdupq_n_u64(u64::MAX))) }
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
        unsafe { vmaxvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_max_f64(self, a: Self::f64) -> f64 {
        unsafe { vmaxvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_min_f32(self, a: Self::f32) -> f32 {
        unsafe { vminvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_min_f64(self, a: Self::f64) -> f64 {
        unsafe { vminvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_sum_f32(self, a: Self::f32) -> f32 {
        unsafe { vaddvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_sum_f64(self, a: Self::f64) -> f64 {
        unsafe { vaddvq_f64(a) }
    }
}

impl<H: HalfImpl> HalfSimd for Neon<H> {
    type f16 = H::f16;
    type bf16 = H::bf16;
    type m16 = H::m16;

    const f16_lanes: usize = H::f16_lanes;
    const bf16_lanes: usize = H::bf16_lanes;

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
        assert!(src.len() >= Self::bf16_lanes);
        self.0.load_bf16(src)
    }

    #[inline(always)]
    fn store_f16(self, dst: &mut [f16], value: Self::f16) {
        assert!(dst.len() >= Self::f16_lanes);
        self.0.store_f16(dst, value)
    }

    #[inline(always)]
    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) {
        assert!(dst.len() >= Self::bf16_lanes);
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

impl<H: HalfImpl> Simd for Neon<H> {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    unsafe { vectorize(crate::backend::test_utils::Run(check)) };
}

#[cfg(test)]
crate::backend_tests!(Neon<HalfImplSoftware>, 4, 2);

#[cfg(test)]
mod fp16_lanes {
    use super::*;
    use crate::traits::{FloatSimd, HalfSimd};

    #[test]
    fn native_half_is_eight_lanes() {
        assert_eq!(<Neon<HalfImplF16> as FloatSimd>::f32_lanes, 4);
        assert_eq!(<Neon<HalfImplF16> as HalfSimd>::f16_lanes, 8);
        assert_eq!(<Neon<HalfImplF16> as HalfSimd>::bf16_lanes, 4);
    }
}
