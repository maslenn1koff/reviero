#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::{
    SimdFn,
    traits::{Float, Simd, SimdMask},
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

#[derive(Copy, Clone, Debug)]
pub struct Neon;

impl Default for Neon {
    fn default() -> Self {
        Neon
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn vectorize<Op: SimdFn>(op: Op) -> Op::Result {
    op.call(Neon)
}

impl Float for Neon {
    type f32 = float32x4_t;
    type f64 = float64x2_t;

    type m32s = NeonMaskF32;
    type m64s = NeonMaskF64;

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
    fn eq_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vceqq_f32(a, b) })
    }

    #[inline(always)]
    fn eq_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { vceqq_f64(a, b) })
    }

    #[inline(always)]
    fn ge_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vcgeq_f32(a, b) })
    }

    #[inline(always)]
    fn ge_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { vcgeq_f64(a, b) })
    }

    #[inline(always)]
    fn gt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vcgtq_f32(a, b) })
    }

    #[inline(always)]
    fn gt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { vcgtq_f64(a, b) })
    }

    #[inline(always)]
    fn le_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vcleq_f32(a, b) })
    }

    #[inline(always)]
    fn le_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { vcleq_f64(a, b) })
    }

    #[inline(always)]
    fn lt_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vcltq_f32(a, b) })
    }

    #[inline(always)]
    fn lt_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { vcltq_f64(a, b) })
    }

    #[inline(always)]
    fn ne_f32(self, a: Self::f32, b: Self::f32) -> Self::m32s {
        NeonMaskF32(unsafe { vmvnq_u32(vceqq_f32(a, b)) })
    }

    #[inline(always)]
    fn ne_f64(self, a: Self::f64, b: Self::f64) -> Self::m64s {
        NeonMaskF64(unsafe { veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX)) })
    }

    #[inline(always)]
    fn select_f32(self, mask: Self::m32s, a: Self::f32, b: Self::f32) -> Self::f32 {
        unsafe { vbslq_f32(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64s, a: Self::f64, b: Self::f64) -> Self::f64 {
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

impl Simd for Neon {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    unsafe { vectorize(crate::backend::test_utils::Run(check)) };
}

#[cfg(test)]
crate::backend_tests!(Neon, 4, 2);
