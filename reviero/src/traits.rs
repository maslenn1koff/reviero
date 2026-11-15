use half::{bf16, f16};
use std::fmt::Debug;

macro_rules! define_simd_io_ops {
    ($ty:ident) => {
        paste::paste! {
            /// Returns a vector with every lane set to `value`.
            fn [<splat_ $ty>](self, value: $ty) -> Self::[<$ty>];

            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Panics
            ///
            /// Panics if `src.len()` is shorter than the lane count.
            fn [<load_ $ty>](self, src: &[$ty]) -> Self::[<$ty>];

            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Panics
            ///
            /// Panics if `dst.len()` is shorter than the lane count.
            fn [<store_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]);

            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Safety
            ///
            /// UB if `src.len()` is shorter than the lane count.
            unsafe fn [<load_unchecked_ $ty>](self, src: &[$ty]) -> Self::[<$ty>];

            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Safety
            ///
            /// UB if `dst.len()` is shorter than the lane count.
            unsafe fn [<store_unchecked_ $ty>](self, dst: &mut [$ty], value: Self::[<$ty>]);
        }
    };
}

macro_rules! define_simd_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise unary operation.
                fn [<$name _ $ty>](self, a: Self::[<$ty>]) -> Self::[<$ty>];
            }
        )*
    };
}

macro_rules! define_simd_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise binary operation.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::[<$ty>];
            }
        )*
    };
}

macro_rules! define_simd_ternary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise ternary operation.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>], c: Self::[<$ty>]) -> Self::[<$ty>];
            }
        )*
    };
}

macro_rules! define_simd_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Horizontal reduction over all lanes.
                ///
                /// The reduction order is backend-dependent.
                /// Floating-point results may differ from sequential scalar accumulation
                /// due to floating-point non-associativity.
                fn [<reduce_ $name _ $ty>](self, a: Self::[<$ty>]) -> $ty;
            }
        )*
    };
}

macro_rules! define_simd_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise comparison producing a lane mask.
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::$mask;
            }
        )*
    };
}

macro_rules! define_halfsimd_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise unary operation.
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::$ty) -> Self::$ty {
                    self.[<f32_to_ $ty>](
                        self.[<$name _f32>](self.[<$ty _to_f32>](a))
                    )
                }
            }
        )*
    };
}

macro_rules! define_halfsimd_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise binary operation.
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::$ty, b: Self::$ty) -> Self::$ty {
                    self.[<f32_to_ $ty>](
                        self.[<$name _f32>](self.[<$ty _to_f32>](a), self.[<$ty _to_f32>](b))
                    )
                }
            }
        )*
    };
}

macro_rules! define_halfsimd_ternary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise ternary operation.
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::$ty, b: Self::$ty, c: Self::$ty) -> Self::$ty {
                    self.[<f32_to_ $ty>](
                        self.[<$name _f32>](self.[<$ty _to_f32>](a), self.[<$ty _to_f32>](b), self.[<$ty _to_f32>](c))
                    )
                }
            }
        )*
    };
}

macro_rules! define_halfsimd_reduce_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Horizontal reduction over all lanes.
                ///
                /// The reduction order is backend-dependent.
                /// Floating-point results may differ from sequential scalar accumulation
                /// due to floating-point non-associativity.
                #[inline(always)]
                fn [<reduce_ $name _ $ty>](self, a: Self::$ty) -> $ty {
                    $ty::from_f32(self.[<reduce_ $name _f32>](self.[<$ty _to_f32>](a)))
                }
            }
        )*
    };
}

macro_rules! define_halfsimd_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise comparison producing a lane mask.
                #[inline(always)]
                fn [<$name _ $ty>](self, a: Self::[<$ty>], b: Self::[<$ty>]) -> Self::$mask {
                    self.[<m32_to_ $mask>](
                        self.[<$name _f32>](self.[<$ty _to_f32>](a), self.[<$ty _to_f32>](b))
                    )
                }
            }
        )*
    };
}

pub trait Simd: FloatSimd {}

pub trait SimdMask: Copy + Debug + Send + Sync + 'static {
    fn splat(v: bool) -> Self;
    fn from_bits(v: u32) -> Self;
    fn to_bits(self) -> u32;

    fn and(self, other: Self) -> Self;
    fn or(self, other: Self) -> Self;
    fn not(self) -> Self;
    fn any(self) -> bool;
    fn all(self) -> bool;
}

pub trait FloatSimd: Copy + Send + Sync + 'static {
    type f32: Copy + Send + Sync + 'static;
    type f64: Copy + Send + Sync + 'static;
    type m32: SimdMask + Copy + Send + Sync + 'static;
    type m64: SimdMask + Copy + Send + Sync + 'static;

    const f32_lanes: usize;
    const f64_lanes: usize;
    const m32_lanes: usize = Self::f32_lanes;
    const m64_lanes: usize = Self::f64_lanes;

    define_simd_io_ops!(f32);
    define_simd_io_ops!(f64);

    // Single instruction on every mainstream ISA.
    define_simd_unary_ops!(f32, abs, neg, sqrt, bitnot);
    define_simd_unary_ops!(f64, abs, neg, sqrt, bitnot);

    // NaN handling of `min`/`max` is backend-defined (follows the native
    // instruction, e.g. `minps`), not the semantics of `f32::min`.
    define_simd_binary_ops!(f32, add, sub, mul, div, min, max, bitand, bitor, bitxor, bitandnot);
    define_simd_binary_ops!(f64, add, sub, mul, div, min, max, bitand, bitor, bitxor, bitandnot);

    // `a * b + c`. If the target architecture supports FMA, the operation is performed as a single unit; otherwise, it is executed as two separate operations.
    // Consequently, the rounding result may vary depending on the backend.
    define_simd_ternary_ops!(f32, mul_add, mul_sub, nmul_add, nmul_sub);
    define_simd_ternary_ops!(f64, mul_add, mul_sub, nmul_add, nmul_sub);

    define_simd_reduce_ops!(f32, sum, max, min);
    define_simd_reduce_ops!(f64, sum, max, min);

    define_simd_compare_ops!(f32, m32, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(f64, m64, eq, ne, lt, le, gt, ge);

    /// Per lane: `mask ? a : b`.
    fn select_f32(self, mask: Self::m32, a: Self::f32, b: Self::f32) -> Self::f32;
    /// Per lane: `mask ? a : b`.
    fn select_f64(self, mask: Self::m64, a: Self::f64, b: Self::f64) -> Self::f64;
}

pub trait HalfSimd: FloatSimd {
    type f16: Copy + Send + Sync + 'static;
    type bf16: Copy + Send + Sync + 'static;
    type m16: SimdMask + Copy + Send + Sync + 'static;

    const f16_lanes: usize;
    const bf16_lanes: usize = Self::f16_lanes;
    const m16_lanes: usize = Self::f16_lanes;

    define_simd_io_ops!(f16);
    define_simd_io_ops!(bf16);

    fn f16_to_f32(self, a: Self::f16) -> Self::f32;
    fn f32_to_f16(self, a: Self::f32) -> Self::f16;
    fn bf16_to_f32(self, a: Self::bf16) -> Self::f32;
    fn f32_to_bf16(self, a: Self::f32) -> Self::bf16;
    fn m16_to_m32(self, a: Self::m16) -> Self::m32;
    fn m32_to_m16(self, a: Self::m32) -> Self::m16;

    define_halfsimd_unary_ops!(f16, abs, neg, sqrt);
    define_halfsimd_unary_ops!(bf16, abs, neg, sqrt);

    define_simd_unary_ops!(f16, bitnot);
    define_simd_unary_ops!(bf16, bitnot);

    define_halfsimd_binary_ops!(f16, add, sub, mul, div, min, max);
    define_halfsimd_binary_ops!(bf16, add, sub, mul, div, min, max);

    define_simd_binary_ops!(f16, bitand, bitor, bitxor, bitandnot);
    define_simd_binary_ops!(bf16, bitand, bitor, bitxor, bitandnot);

    define_halfsimd_ternary_ops!(f16, mul_add, mul_sub, nmul_add, nmul_sub);
    define_halfsimd_ternary_ops!(bf16, mul_add, mul_sub, nmul_add, nmul_sub);

    define_halfsimd_reduce_ops!(f16, max, min, sum);
    define_halfsimd_reduce_ops!(bf16, max, min, sum);

    define_halfsimd_compare_ops!(f16, m16, eq, ne, lt, le, gt, ge);
    define_halfsimd_compare_ops!(bf16, m16, eq, ne, lt, le, gt, ge);

    /// Per lane: `mask ? a : b`.
    #[inline(always)]
    fn select_f16(self, mask: Self::m16, a: Self::f16, b: Self::f16) -> Self::f16 {
        self.f32_to_f16(self.select_f32(self.m16_to_m32(mask), self.f16_to_f32(a), self.f16_to_f32(b)))
    }

    /// Per lane: `mask ? a : b`.
    #[inline(always)]
    fn select_bf16(self, mask: Self::m16, a: Self::bf16, b: Self::bf16) -> Self::bf16 {
        self.f32_to_bf16(self.select_f32(self.m16_to_m32(mask), self.bf16_to_f32(a), self.bf16_to_f32(b)))
    }
}
