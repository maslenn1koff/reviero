#![allow(non_upper_case_globals, non_camel_case_types)]

mod backend;
pub mod traits;
pub mod utils;

use crate::{backend::fallback::Fallback, traits::Simd};
use std::sync::LazyLock;

enum Backend {
    Fallback,
}

impl Backend {
    fn new() -> Self {
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
    }
}
