#![allow(non_upper_case_globals, non_camel_case_types)]

mod backend;
pub mod traits;
pub mod utils;

use crate::{backend::fallback::Fallback, traits::Simd};
use std::sync::LazyLock;

enum Backend {
    Fallback,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512f,
}

impl Backend {
    fn new() -> Self {
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

static BACKEND: LazyLock<Backend> = LazyLock::new(Backend::new);

pub trait SimdFn {
    type Result;
    fn call<S: Simd>(self, simd: S) -> Self::Result;
}

#[inline(always)]
pub fn dispatch<Op: SimdFn>(op: Op) -> Op::Result {
    match &*BACKEND {
        Backend::Fallback => op.call(Fallback),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => unsafe { backend::x86_64::sse2::vectorize(op) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2Fma => unsafe { backend::x86_64::avx2::vectorize(op) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx512f => unsafe { backend::x86_64::avx512::vectorize(op) },
    }
}
