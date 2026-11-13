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
//!         let (vectors, tail) = reviero::utils::as_simd_f32(&simd, self.0);
//!         let mut sum = simd.splat_f32(0.0);
//!
//!         for vector in vectors {
//!             sum = simd.add_f32(sum, vector);
//!         }
//!
//!         let mut result = simd.reduce_sum_f32(sum);
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

use crate::traits::Simd;
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
static BACKEND: AtomicU8 = AtomicU8::new(0);

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[repr(u8)]
#[derive(Clone, Copy)]
enum Backend {
    #[cfg(not(target_arch = "aarch64"))]
    Fallback = 1,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 = 2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma = 3,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512f = 4,
    #[cfg(target_arch = "aarch64")]
    Neon = 5,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[cold]
#[inline(never)]
fn detect_backend() -> u8 {
    let backend = Backend::new() as u8;
    BACKEND.store(backend, Ordering::Relaxed);
    backend
}

/// Runs `op` using the best SIMD backend supported by the current CPU.
///
/// The backend is detected once and then reused. On x86/x86_64 the preference
/// is AVX-512F + FMA, AVX2 + FMA, SSE2, then the scalar fallback. On AArch64
/// the NEON backend is used.
#[inline(always)]
pub fn dispatch<Op: SimdFn>(op: Op) -> Op::Result {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    let backend = match BACKEND.load(Ordering::Relaxed) {
        0 => detect_backend(),
        backend => backend,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if backend == Backend::Avx2Fma as u8 {
            return unsafe { crate::backend::x86::avx2::vectorize(op) };
        } else if backend == Backend::Avx512f as u8 {
            return unsafe { crate::backend::x86::avx512::vectorize(op) };
        } else if backend == Backend::Sse2 as u8 {
            return unsafe { crate::backend::x86::sse2::vectorize(op) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    if backend == Backend::Neon as u8 {
        return unsafe { crate::backend::aarch64::neon::vectorize(op) };
    }

    op.call(crate::backend::fallback::Fallback)
}
