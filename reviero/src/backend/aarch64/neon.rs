#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::{
    SimdFn,
    traits::{SimdBitOps, SimdFloatOps, SimdIo, SimdMask, SimdMaskOps, SimdReduceOps, SimdTypes},
};

#[derive(Copy, Clone, Debug)]
pub struct NeonMaskF32(uint32x4_t); // 4 lanes

impl SimdMask for NeonMaskF32 {
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

impl SimdTypes for Neon {
    type f32s = float32x4_t;
    type f64s = float64x2_t;

    type m32s = NeonMaskF32;
    type m64s = NeonMaskF64;

    const f32s_lanes: usize = 4;
    const f64s_lanes: usize = 2;
}

impl SimdIo for Neon {
    #[inline(always)]
    fn splat_f32s(self, value: f32) -> Self::f32s {
        unsafe { vdupq_n_f32(value) }
    }

    #[inline(always)]
    fn splat_f64s(self, value: f64) -> Self::f64s {
        unsafe { vdupq_n_f64(value) }
    }

    #[inline(always)]
    fn load_f32s(self, src: &[f32]) -> Self::f32s {
        assert!(src.len() >= Self::f32s_lanes);
        unsafe { vld1q_f32(src.as_ptr()) }
    }

    #[inline(always)]
    fn load_f64s(self, src: &[f64]) -> Self::f64s {
        assert!(src.len() >= Self::f64s_lanes);
        unsafe { vld1q_f64(src.as_ptr()) }
    }

    #[inline(always)]
    fn store_f32s(self, dst: &mut [f32], value: Self::f32s) {
        assert!(dst.len() >= Self::f32s_lanes);
        unsafe { vst1q_f32(dst.as_mut_ptr(), value) };
    }

    #[inline(always)]
    fn store_f64s(self, dst: &mut [f64], value: Self::f64s) {
        assert!(dst.len() >= Self::f64s_lanes);
        unsafe { vst1q_f64(dst.as_mut_ptr(), value) };
    }
}

impl SimdFloatOps for Neon {
    #[inline(always)]
    fn abs_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { vabsq_f32(a) }
    }

    #[inline(always)]
    fn abs_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { vabsq_f64(a) }
    }

    #[inline(always)]
    fn add_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vaddq_f32(a, b) }
    }

    #[inline(always)]
    fn add_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vaddq_f64(a, b) }
    }

    #[inline(always)]
    fn sub_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vsubq_f32(a, b) }
    }

    #[inline(always)]
    fn sub_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vsubq_f64(a, b) }
    }

    #[inline(always)]
    fn mul_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vmulq_f32(a, b) }
    }

    #[inline(always)]
    fn mul_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vmulq_f64(a, b) }
    }

    #[inline(always)]
    fn div_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vdivq_f32(a, b) }
    }

    #[inline(always)]
    fn div_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vdivq_f64(a, b) }
    }

    #[inline(always)]
    fn max_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vmaxq_f32(a, b) }
    }

    #[inline(always)]
    fn max_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vmaxq_f64(a, b) }
    }

    #[inline(always)]
    fn min_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vminq_f32(a, b) }
    }

    #[inline(always)]
    fn min_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vminq_f64(a, b) }
    }

    #[inline(always)]
    fn neg_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { vnegq_f32(a) }
    }

    #[inline(always)]
    fn neg_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { vnegq_f64(a) }
    }

    #[inline(always)]
    fn sqrt_f32s(self, a: Self::f32s) -> Self::f32s {
        unsafe { vsqrtq_f32(a) }
    }

    #[inline(always)]
    fn sqrt_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { vsqrtq_f64(a) }
    }

    #[inline(always)]
    fn mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { vfmaq_f32(c, a, b) }
    }

    #[inline(always)]
    fn mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { vfmaq_f64(c, a, b) }
    }

    #[inline(always)]
    fn mul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { vnegq_f32(vfmsq_f32(c, a, b)) }
    }

    #[inline(always)]
    fn mul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { vnegq_f64(vfmsq_f64(c, a, b)) }
    }

    #[inline(always)]
    fn nmul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { vfmsq_f32(c, a, b) }
    }

    #[inline(always)]
    fn nmul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { vfmsq_f64(c, a, b) }
    }

    #[inline(always)]
    fn nmul_sub_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        unsafe { vnegq_f32(vfmaq_f32(c, a, b)) }
    }

    #[inline(always)]
    fn nmul_sub_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        unsafe { vnegq_f64(vfmaq_f64(c, a, b)) }
    }
}

impl SimdMaskOps for Neon {
    #[inline(always)]
    fn eq_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vceqq_f32(a, b) })
    }

    #[inline(always)]
    fn eq_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { vceqq_f64(a, b) })
    }

    #[inline(always)]
    fn ge_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vcgeq_f32(a, b) })
    }

    #[inline(always)]
    fn ge_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { vcgeq_f64(a, b) })
    }

    #[inline(always)]
    fn gt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vcgtq_f32(a, b) })
    }

    #[inline(always)]
    fn gt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { vcgtq_f64(a, b) })
    }

    #[inline(always)]
    fn le_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vcleq_f32(a, b) })
    }

    #[inline(always)]
    fn le_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { vcleq_f64(a, b) })
    }

    #[inline(always)]
    fn lt_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vcltq_f32(a, b) })
    }

    #[inline(always)]
    fn lt_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { vcltq_f64(a, b) })
    }

    #[inline(always)]
    fn ne_f32s(self, a: Self::f32s, b: Self::f32s) -> Self::m32s {
        NeonMaskF32(unsafe { vmvnq_u32(vceqq_f32(a, b)) })
    }

    #[inline(always)]
    fn ne_f64s(self, a: Self::f64s, b: Self::f64s) -> Self::m64s {
        NeonMaskF64(unsafe { veorq_u64(vceqq_f64(a, b), vdupq_n_u64(u64::MAX)) })
    }

    #[inline(always)]
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        unsafe { vbslq_f32(mask.0, a, b) }
    }

    #[inline(always)]
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        unsafe { vbslq_f64(mask.0, a, b) }
    }
}

impl SimdBitOps for Neon {
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
        unsafe { vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a))) }
    }

    #[inline(always)]
    fn bitnot_f64s(self, a: Self::f64s) -> Self::f64s {
        unsafe { vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vdupq_n_u64(u64::MAX))) }
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

impl SimdReduceOps for Neon {
    #[inline(always)]
    fn reduce_max_f32s(self, a: Self::f32s) -> f32 {
        unsafe { vmaxvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_max_f64s(self, a: Self::f64s) -> f64 {
        unsafe { vmaxvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_min_f32s(self, a: Self::f32s) -> f32 {
        unsafe { vminvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_min_f64s(self, a: Self::f64s) -> f64 {
        unsafe { vminvq_f64(a) }
    }

    #[inline(always)]
    fn reduce_sum_f32s(self, a: Self::f32s) -> f32 {
        unsafe { vaddvq_f32(a) }
    }

    #[inline(always)]
    fn reduce_sum_f64s(self, a: Self::f64s) -> f64 {
        unsafe { vaddvq_f64(a) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{Check, Run};
    use crate::traits::SimdTypes;

    fn run(check: Check) {
        unsafe { vectorize(Run(check)) };
    }

    #[test]
    fn lanes() {
        assert_eq!(Neon::f32s_lanes, 4);
        assert_eq!(Neon::f64s_lanes, 2);
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
