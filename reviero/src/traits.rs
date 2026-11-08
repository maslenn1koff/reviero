use std::fmt::Debug;

macro_rules! define_simd_io_ops {
    ($ty:ident) => {
        paste::paste! {
            /// Returns a vector with every lane set to `value`.
            fn [<splat_ $ty s>](self, value: $ty) -> Self::[<$ty s>];

            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Panics
            ///
            /// Panics if `src.len()` is shorter than the lane count.
            fn [<load_ $ty s>](self, src: &[$ty]) -> Self::[<$ty s>];

            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Panics
            ///
            /// Panics if `dst.len()` is shorter than the lane count.
            fn [<store_ $ty s>](self, dst: &mut [$ty], value: Self::[<$ty s>]);

            /// Loads the first `LANES` elements of `src`.
            ///
            /// # Safety
            /// 
            /// UB if `src.len()` is shorter than the lane count.
            unsafe fn [<load_unchecked_ $ty s>](self, src: &[$ty]) -> Self::[<$ty s>];

            /// Stores the vector into the first `LANES` elements of `dst`.
            ///
            /// # Safety
            /// 
            /// UB if `dst.len()` is shorter than the lane count.
            unsafe fn [<store_unchecked_ $ty s>](self, dst: &mut [$ty], value: Self::[<$ty s>]);
        }
    };
}

macro_rules! define_simd_unary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise unary operation.
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>]) -> Self::[<$ty s>];
            }
        )*
    };
}

macro_rules! define_simd_binary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise binary operation.
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>]) -> Self::[<$ty s>];
            }
        )*
    };
}

macro_rules! define_simd_ternary_ops {
    ($ty:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise ternary operation.
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>], c: Self::[<$ty s>]) -> Self::[<$ty s>];
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
                fn [<reduce_ $name _ $ty s>](self, a: Self::[<$ty s>]) -> $ty;
            }
        )*
    };
}

macro_rules! define_simd_compare_ops {
    ($ty:ident, $mask:ident, $($name:ident),*) => {
        $(
            paste::paste! {
                /// Element-wise comparison producing a lane mask.
                fn [<$name _ $ty s>](self, a: Self::[<$ty s>], b: Self::[<$ty s>]) -> Self::$mask;
            }
        )*
    };
}

pub trait Simd: SimdIo + SimdFloatOps + SimdMaskOps + SimdBitOps + SimdReduceOps {}

impl<T> Simd for T where T: SimdIo + SimdFloatOps + SimdMaskOps + SimdBitOps + SimdReduceOps {}

pub trait SimdMask: Copy + Debug + Send + Sync + 'static {
    fn and(self, other: Self) -> Self;
    fn or(self, other: Self) -> Self;
    fn not(self) -> Self;

    /// True if at least one lane is set (`movemask` + test).
    fn any(self) -> bool;
    /// True if every lane is set.
    fn all(self) -> bool;

    /// True if no lane is set.
    #[inline(always)]
    fn none(self) -> bool {
        !self.any()
    }
}

pub trait SimdTypes: Copy + Debug + Default + Send + Sync + 'static {
    type f32s: Copy + Debug + Send + Sync + 'static;
    type f64s: Copy + Debug + Send + Sync + 'static;

    type m32s: SimdMask;
    type m64s: SimdMask;

    const f32s_lanes: usize;
    const f64s_lanes: usize;
}

pub trait SimdIo: SimdTypes {
    define_simd_io_ops!(f32);
    define_simd_io_ops!(f64);
}

pub trait SimdFloatOps: SimdTypes {
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
}

pub trait SimdMaskOps: SimdTypes {
    define_simd_compare_ops!(f32, m32s, eq, ne, lt, le, gt, ge);
    define_simd_compare_ops!(f64, m64s, eq, ne, lt, le, gt, ge);

    /// Per lane: `mask ? a : b`.
    fn select_f32s(self, mask: Self::m32s, a: Self::f32s, b: Self::f32s) -> Self::f32s;
    /// Per lane: `mask ? a : b`.
    fn select_f64s(self, mask: Self::m64s, a: Self::f64s, b: Self::f64s) -> Self::f64s;
}

pub trait SimdBitOps: SimdTypes {
    // Bitwise ops on the float bit patterns (and/or/xor are one instruction).
    // Enough for sign tricks: abs = and, neg = xor, copysign = and + or.
    define_simd_unary_ops!(f32, bitnot);
    define_simd_unary_ops!(f64, bitnot);
    define_simd_binary_ops!(f32, bitand, bitor, bitxor, bitandnot);
    define_simd_binary_ops!(f64, bitand, bitor, bitxor, bitandnot);
}

pub trait SimdReduceOps: SimdTypes {
    define_simd_reduce_ops!(f32, sum, max, min);
    define_simd_reduce_ops!(f64, sum, max, min);
}
