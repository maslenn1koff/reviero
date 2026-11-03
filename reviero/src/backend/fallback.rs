use crate::traits::{SimdBitOps, SimdFloatOps, SimdIo, SimdMask, SimdMaskOps, SimdReduceOps, SimdTypes};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! define_io_ops {
    ($ty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<splat_ $ty s>](self, value: $ty) -> Self::[<$ty s>] {
                value
            }

            #[inline(always)]
            fn [<load_ $ty s>](self, src: &[$ty]) -> Self::[<$ty s>] {
                src[0]
            }

            #[inline(always)]
            fn [<store_ $ty s>](self, dst: &mut [$ty], value: Self::[<$ty s>]) {
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
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>]) -> Self::[<$ty s>] {
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
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>]) -> Self::[<$ty s>] {
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
            fn [<mul_add _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>], c: Self::[<$ty s>]) -> Self::[<$ty s>] {
                a * b + c
            }

            #[inline(always)]
            fn [<mul_sub _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>], c: Self::[<$ty s>]) -> Self::[<$ty s>] {
                a * b - c
            }

            #[inline(always)]
            fn [<nmul_add _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>], c: Self::[<$ty s>]) -> Self::[<$ty s>] {
                -(a * b) + c
            }

            #[inline(always)]
            fn [<nmul_sub _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>], c: Self::[<$ty s>]) -> Self::[<$ty s>] {
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
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>]) -> Self::[<$mask>] {
                    FallbackMask(a.[<$name>](&b))
                }
            )*
        }
    };
}

macro_rules! define_bit_ops {
    ($ty:ident, $uty:ident) => {
        paste::paste! {
            #[inline(always)]
            fn [<bitand_ $ty s>](self, a: $ty, b: $ty) -> $ty {
                <$ty>::from_bits(a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitandnot_ $ty s>](self, a: $ty, b: $ty) -> $ty {
                <$ty>::from_bits(!a.to_bits() & b.to_bits())
            }

            #[inline(always)]
            fn [<bitor_ $ty s>](self, a: $ty, b: $ty) -> $ty {
                <$ty>::from_bits(a.to_bits() | b.to_bits())
            }

            #[inline(always)]
            fn [<bitxor_ $ty s>](self, a: $ty, b: $ty) -> $ty {
                <$ty>::from_bits(a.to_bits() ^ b.to_bits())
            }

            #[inline(always)]
            fn [<bitnot_ $ty s>](self, a: $ty) -> $ty {
                <$ty>::from_bits(!a.to_bits())
            }
        }
    };
}

macro_rules! define_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        paste::paste! {
            $(
                #[inline(always)]
                fn [<reduce_ $name _ $ty s>](self, a: $ty) -> $ty {
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

impl SimdTypes for Fallback {
    type f32s = f32;
    type f64s = f64;

    type m32s = FallbackMask;
    type m64s = FallbackMask;

    const f32s_lanes: usize = 1;
    const f64s_lanes: usize = 1;
}

impl SimdIo for Fallback {
    define_io_ops!(f32);
    define_io_ops!(f64);
}

impl SimdFloatOps for Fallback {
    define_unary_ops!(f32, abs, neg, sqrt);
    define_unary_ops!(f64, abs, neg, sqrt);

    define_binary_ops!(f32, add, sub, mul, div, min, max);
    define_binary_ops!(f64, add, sub, mul, div, min, max);

    define_ternary_ops!(f32);
    define_ternary_ops!(f64);
}

impl SimdMaskOps for Fallback {
    define_compare_ops!(f32, m32s, eq, ne, lt, le, gt, ge);
    define_compare_ops!(f64, m64s, eq, ne, lt, le, gt, ge);

    #[inline(always)]
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s {
        if mask.0 { a } else { b }
    }
    
    #[inline(always)]
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s {
        if mask.0 { a } else { b }
    }
}

impl SimdBitOps for Fallback {
    define_bit_ops!(f32, u32);
    define_bit_ops!(f64, u64);
}

impl SimdReduceOps for Fallback {
    define_reduce_ops!(f32, sum, max, min);
    define_reduce_ops!(f64, sum, max, min);
}
