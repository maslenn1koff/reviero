use crate::traits::Simd;
use std::iter::ExactSizeIterator;

macro_rules! define_as_simd {
    ($ty:ident) => {
        paste::paste! {
            /// Splits a slice of `$ty` values into an iterator of SIMD vectors and a remaining slice.
            pub fn [<as_simd_ $ty s>]<'a, S: Simd>(simd: &'a S, src: &'a [$ty]) -> (impl ExactSizeIterator<Item = S::[<$ty s>]> + 'a , &'a [$ty]) {
                assert!(S::[<$ty s _lanes>] > 0);

                let count = src.len() / S::[<$ty s _lanes>];
                let (head, tail) = src.split_at(count * S::[<$ty s _lanes>]);
                let vectors = head.chunks_exact(S::[<$ty s _lanes>]).map(move |chunk| simd.[<load_ $ty s>](chunk));

                (vectors, tail)
            }
        }
    };
}

define_as_simd!(f32);
define_as_simd!(f64);
