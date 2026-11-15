use crate::traits::{FloatSimd, HalfSimd, Simd, SimdMask};
use half::{bf16, f16};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! define_io_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<splat_ $ty>](self, value: $ty) -> Self::[<$ty>] {
                value
            }

            #[inline(always)]
            fn [<load_ $ty>](self, src: &[$ty]) -> Self::[<$ty>] {
                src[0]
            }

            #[inline(always)]
            fn [<store_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]) {
                dst[0] = value;
            }

            #[inline(always)]
            unsafe fn [<load_unchecked_ $ty>](self, src: &[$ty]) -> Self::[<$ty>] {
                src[0]
            }

            #[inline(always)]
            unsafe fn [<store_unchecked_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]) {
                dst[0] = value;
            }
        }
    };
}

macro_rules! define_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<$name>]()
                }
            )*
        }
    };
}

macro_rules! define_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$ty>] {
                    a.[<$name>](b)
                }
            )*
        }
    };
}

macro_rules! define_ternary_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<mul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                a * b + c
            }

            #[inline(always)]
            fn [<mul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                a * b - c
            }

            #[inline(always)]
            fn [<nmul_add _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                -(a * b) + c
            }

            #[inline(always)]
            fn [<nmul_sub _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>] {
                -(a * b) - c
            }
        }
    };
}

macro_rules! define_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$mask>] {
                    FallbackMask(a.[<$name>](&b))
                }
            )*
        }
    };
}

macro_rules! define_bit_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<bitand_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitandnot_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(!a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() | b.to_bits())
            }

            #[inline(always)]
            fn [<bitxor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(a.to_bits() ^ b.to_bits())
            }

            #[inline(always)]
            fn [<bitnot_ $ty>](self, a: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(!a.to_bits())
            }
        }
    };
}

macro_rules! define_half_bit_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<bitand_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(
                    $ty::from_f32(a).to_bits() & $ty::from_f32(b).to_bits()
                ).to_f32()
            }

            #[inline(always)]
            fn [<bitandnot_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(
                    !$ty::from_f32(a).to_bits() & $ty::from_f32(b).to_bits()
                ).to_f32()
            }

            #[inline(always)]
            fn [<bitor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(
                    $ty::from_f32(a).to_bits() | $ty::from_f32(b).to_bits()
                ).to_f32()
            }

            #[inline(always)]
            fn [<bitxor_ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(
                    $ty::from_f32(a).to_bits() ^ $ty::from_f32(b).to_bits()
                ).to_f32()
            }

            #[inline(always)]
            fn [<bitnot_ $ty>](self, a: Self::$ty) -> Self::$ty {
                <$ty>::from_bits(
                    !$ty::from_f32(a).to_bits()
                ).to_f32()
            }
        }
    };
}

macro_rules! define_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<reduce_ $name _ $ty>](self, a: $ty) -> $ty {
                    a
                }
            )*
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Fallback;

impl Default for Fallback {
    fn default() -> Self {
        Fallback
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FallbackMask(bool);

impl SimdMask for FallbackMask {
    #[inline(always)]
    fn splat(v: bool) -> Self {
        FallbackMask(v)
    }

    #[inline(always)]
    fn from_bits(v: u32) -> Self {
        FallbackMask(v & 1 != 0)
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        self.0 as u32
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        FallbackMask(self.0 && other.0)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        FallbackMask(self.0 || other.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        FallbackMask(!self.0)
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }
}

impl FloatSimd for Fallback {
    type f32 = f32;
    type f64 = f64;

    type m32 = FallbackMask;
    type m64 = FallbackMask;

    const f32_lanes: usize = 1;
    const f64_lanes: usize = 1;
    const m64_lanes: usize = Self::f64_lanes;

    define_io_ops!(f32);
    define_io_ops!(f64);

    define_unary_ops!(f32, abs, neg, sqrt);
    define_unary_ops!(f64, abs, neg, sqrt);

    define_binary_ops!(f32, add, sub, mul, div, min, max);
    define_binary_ops!(f64, add, sub, mul, div, min, max);

    define_ternary_ops!(f32);
    define_ternary_ops!(f64);

    define_compare_ops!(f32, m32, eq, ne, lt, le, gt, ge);
    define_compare_ops!(f64, m64, eq, ne, lt, le, gt, ge);

    #[inline(always)]
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32 {
        if mask.0 { a } else { b }
    }

    #[inline(always)]
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64 {
        if mask.0 { a } else { b }
    }

    define_bit_ops!(f32);
    define_bit_ops!(f64);

    define_reduce_ops!(f32, sum, max, min);
    define_reduce_ops!(f64, sum, max, min);
}

impl HalfSimd for Fallback {
    type f16 = f32;
    type bf16 = f32;
    type m16 = FallbackMask;
    const f16_lanes: usize = 1;

    #[rustfmt::skip] #[inline(always)]    fn f16_to_f32(self, a: Self::f16) -> Self::f32 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_f16(self, a: Self::f32) -> Self::f16 { a }
    #[rustfmt::skip] #[inline(always)]    fn bf16_to_f32(self, a: Self::bf16) -> Self::f32 { a }
    #[rustfmt::skip] #[inline(always)]    fn f32_to_bf16(self, a: Self::f32) -> Self::bf16 { a }
    #[rustfmt::skip] #[inline(always)]    fn m16_to_m32(self, a: Self::m16) -> Self::m32 { a }
    #[rustfmt::skip] #[inline(always)]    fn m32_to_m16(self, a: Self::m32) -> Self::m16 { a }

    #[rustfmt::skip] #[inline(always)]    fn splat_f16(self, value: f16) -> Self::f16 { value.to_f32() }
    #[rustfmt::skip] #[inline(always)]    fn splat_bf16(self, value: bf16) -> Self::bf16 { value.to_f32() }
    #[rustfmt::skip] #[inline(always)]    fn load_f16(self, src: &[f16]) -> Self::f16 { src[0].to_f32() }
    #[rustfmt::skip] #[inline(always)]    fn load_bf16(self, src: &[bf16]) -> Self::bf16 { src[0].to_f32() }
    #[rustfmt::skip] #[inline(always)]    unsafe fn load_unchecked_f16(self, src: &[f16]) -> Self::f16 { src[0].to_f32() }
    #[rustfmt::skip] #[inline(always)]    unsafe fn load_unchecked_bf16(self, src: &[bf16]) -> Self::bf16 { src[0].to_f32() }
    #[rustfmt::skip] #[inline(always)]    fn store_f16(self, dst: &mut [f16], value: Self::f16) { dst[0] = f16::from_f32(value) }
    #[rustfmt::skip] #[inline(always)]    fn store_bf16(self, dst: &mut [bf16], value: Self::bf16) { dst[0] = bf16::from_f32(value) }
    #[rustfmt::skip] #[inline(always)]    unsafe fn store_unchecked_f16(self, dst: &mut [f16], value: Self::f16) { dst[0] = f16::from_f32(value) }
    #[rustfmt::skip] #[inline(always)]    unsafe fn store_unchecked_bf16(self, dst: &mut [bf16], value: Self::bf16) { dst[0] = bf16::from_f32(value) }

    define_half_bit_ops!(f16);
    define_half_bit_ops!(bf16);
}

impl Simd for Fallback {}

#[cfg(test)]
fn run(check: crate::backend::test_utils::Check) {
    use crate::SimdFn;

    crate::backend::test_utils::Run(check).call(Fallback);
}

#[cfg(test)]
crate::backend_tests!(Fallback, 1, 1);
