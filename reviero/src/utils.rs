use crate::traits::Simd;
use std::iter::ExactSizeIterator;

macro_rules! define_as_simd {
    ($ty:ident) => {
        paste::paste! {
            /// Splits a slice of `$ty` values into an iterator of SIMD vectors and a remaining slice.
            pub fn [<as_simd_ $ty>]<'a, S: Simd>(simd: &'a S, src: &'a [$ty]) -> (impl ExactSizeIterator<Item = S::$ty> + 'a , &'a [$ty]) {
                assert!(S::[<$ty _lanes>] > 0);

                let count = src.len() / S::[<$ty _lanes>];
                let (head, tail) = src.split_at(count * S::[<$ty _lanes>]);
                let vectors = head.chunks_exact(S::[<$ty _lanes>]).map(move |chunk| unsafe { simd.[<load_unchecked_ $ty>](chunk) });

                (vectors, tail)
            }
        }
    };
}

define_as_simd!(f32);
define_as_simd!(f64);
