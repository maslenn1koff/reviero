use crate::traits::Simd;
use std::iter::ExactSizeIterator;

/// Splits a slice of `f32` values into an iterator of SIMD vectors and a remaining slice.
pub fn as_simd_f32s<'a, S: Simd>(src: &'a [f32]) -> (impl ExactSizeIterator<Item = S::f32s> + 'a, &'a [f32]) {
    assert!(S::f32s_lanes > 0);

    let count = src.len() / S::f32s_lanes;
    let (head, tail) = src.split_at(count * S::f32s_lanes);
    let simd = S::default();
    let vectors = head.chunks_exact(S::f32s_lanes).map(move |chunk| simd.load_f32s(chunk));

    (vectors, tail)
}

/// Splits a slice of `f64` values into an iterator of SIMD vectors and a remaining slice.
pub fn as_simd_f64s<'a, S: Simd>(src: &'a [f64]) -> (impl ExactSizeIterator<Item = S::f64s> + 'a, &'a [f64]) {
    assert!(S::f64s_lanes > 0);

    let count = src.len() / S::f64s_lanes;
    let (head, tail) = src.split_at(count * S::f64s_lanes);
    let simd = S::default();
    let vectors = head.chunks_exact(S::f64s_lanes).map(move |chunk| simd.load_f64s(chunk));

    (vectors, tail)
}
