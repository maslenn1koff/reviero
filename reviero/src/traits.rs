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

pub trait Simd: Float {}

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

pub trait Float: Copy + Send + Sync + 'static {
    type f32: Copy + Send + Sync + 'static;
    type f64: Copy + Send + Sync + 'static;
    type m32s: SimdMask + Copy + Send + Sync + 'static;
    type m64s: SimdMask + Copy + Send + Sync + 'static;

    const f32_lanes: usize;
    const f64_lanes: usize;
    const m32_lanes: usize = Self::f32_lanes;
    const m64_lanes: usize = Self::f32_lanes;

    define_simd_io_ops!(f32);
    define_simd_io_ops!(f64);

    // Single instruction on every mainstream ISA.
    define_simd_unary_ops!(f32, abs, neg, sqrt);
    define_simd_unary_ops!(f64, abs, neg, sqrt);

    define_simd_binary_ops!(f32, add, sub, mul, div);
    define_simd_binary_ops!(f64, add, sub, mul, div);

    // NaN handling of `min`/`max` is backend-defined (follows the native
    // instruction, e.g. `minps`), not the semantics of `f32::min`.
    define_simd_binary_ops!(f32, min, max);
    define_simd_binary_ops!(f64, min, max);

    // `a * b + c`. If the target architecture supports FMA, the operation is performed as a single unit; otherwise, it is executed as two separate operations.
    // Consequently, the rounding result may vary depending on the backend.
    define_simd_ternary_ops!(f32, mul_add, mul_sub, nmul_add, nmul_sub);
    define_simd_ternary_ops!(f64, mul_add, mul_sub, nmul_add, nmul_sub);

    define_simd_reduce_ops!(f32, sum, max, min);
    define_simd_reduce_ops!(f64, sum, max, min);

    define_simd_compare_ops!(f32, m32s, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(f64, m64s, eq, ne, lt, le, gt, ge);

    /// Per lane: `mask ? a : b`.
    fn select_f32(self, mask: Self::m32s, a: Self::f32, b: Self::f32) -> Self::f32;
    /// Per lane: `mask ? a : b`.
    fn select_f64(self, mask: Self::m64s, a: Self::f64, b: Self::f64) -> Self::f64;

    // Bitwise ops on the float bit patterns (and/or/xor are one instruction).
    // Enough for sign tricks: abs = and, neg = xor, copysign = and + or.
    define_simd_unary_ops!(f32, bitnot);
    define_simd_unary_ops!(f64, bitnot);
    define_simd_binary_ops!(f32, bitand, bitor, bitxor, bitandnot);
    define_simd_binary_ops!(f64, bitand, bitor, bitxor, bitandnot);
}
