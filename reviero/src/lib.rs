//! Portable SIMD operations with runtime CPU dispatch.
//!
//! `reviero` provides a shared set of vector operations for `f32` and `f64`.
//! Implement an operation as [`SimdFn`] and pass it to [`dispatch`]; the
//! operation is then run using the best supported backend selected for the
//! current CPU. This lets one implementation work across supported
//! architectures without requiring callers to use architecture-specific
//! intrinsics.
//!
//! On x86 and x86_64, dispatch prefers AVX-512F with FMA, then AVX2 with FMA,
//! then SSE2, and otherwise uses the scalar fallback. On AArch64, it uses
//! NEON. The backend is detected once and reused. Floating-point results can
//! vary between backends because reductions and fused multiply-add operations
//! can change rounding.
//!
//! # Implementing an operation
//!
//! The [`SimdFn::call`] method is generic over the selected SIMD implementation.
//! Use [`utils::as_simd_f32s`] or [`utils::as_simd_f64s`] to process complete
//! vectors; each utility also returns the remaining elements for scalar
//! handling.
//!
//! ```
//! use reviero::{SimdFn, dispatch};
//!
//! struct Sum<'a>(&'a [f32]);
//!
//! impl SimdFn for Sum<'_> {
//!     type Result = f32;
//!
//!     #[inline(always)]
//!     fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
//!         let (vectors, tail) = reviero::utils::as_simd_f32s::<S>(self.0);
//!         let mut sum = simd.splat_f32s(0.0);
//!
//!         for vector in vectors {
//!             sum = simd.add_f32s(sum, vector);
//!         }
//!
//!         let mut result = simd.reduce_sum_f32s(sum);
//!         for &value in tail {
//!             result += value;
//!         }
//!         result
//!     }
//! }
//!
//! let values = [1.0, 2.0, 3.0, 4.0, 5.0];
//! assert_eq!(dispatch(Sum(&values)), 15.0);
//! ```
//!
//! The available vector operations, masks, and lane counts are defined by the
//! traits in [`traits`]. Their concrete lane counts depend on the selected
//! backend.

#![allow(non_upper_case_globals, non_camel_case_types)]

mod backend;
pub mod traits;
pub mod utils;

#[cfg(not(target_arch = "aarch64"))]
use crate::backend::fallback::Fallback;
use crate::traits::Simd;
use std::sync::LazyLock;

enum Backend {
    #[cfg(not(target_arch = "aarch64"))]
    Fallback,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512f,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Backend {
    fn new() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Backend::Neon
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
                    return Backend::Avx512f;
                } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return Backend::Avx2Fma;
                } else if is_x86_feature_detected!("sse2") {
                    return Backend::Sse2;
                }
            }

            Backend::Fallback
        }
    }
}

static BACKEND: LazyLock<Backend> = LazyLock::new(Backend::new);

/// A SIMD operation that [`dispatch`] can run on the best available backend.
///
/// `dispatch` monomorphizes the operation for every supported backend. The
/// implementation of [`SimdFn::call`] must be annotated with
/// `#[inline(always)]`; this is what lets the selected backend's target
/// features propagate into the operation and lets its vector instructions be
/// inlined into the caller. Without it the operation remains correct, but it
/// can lose most of the performance benefit of the SIMD backend.
pub trait SimdFn {
    type Result;

    /// Runs this operation with the SIMD backend represented by `simd`.
    ///
    /// Implementations should be marked `#[inline(always)]`.
    fn call<S: Simd>(self, simd: S) -> Self::Result;
}

/// Runs `op` using the best SIMD backend supported by the current CPU.
///
/// The backend is detected once and then reused. On x86/x86_64 the preference
/// is AVX-512F + FMA, AVX2 + FMA, SSE2, then the scalar fallback. On AArch64
/// the NEON backend is used.
#[inline(always)]
pub fn dispatch<Op: SimdFn>(op: Op) -> Op::Result {
    match &*BACKEND {
        #[cfg(not(target_arch = "aarch64"))]
        Backend::Fallback => op.call(Fallback),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => unsafe { backend::x86_64::sse2::vectorize(op) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2Fma => unsafe { backend::x86_64::avx2::vectorize(op) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx512f => unsafe { backend::x86_64::avx512::vectorize(op) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { backend::aarch64::neon::vectorize(op) },
    }
}
