use reviero::{SimdFn, dispatch};
use std::vec;

macro_rules! define_dot {
    ($ty:ty) => {
        paste::paste! {
            struct [<Dot $ty>]<'a>(&'a [[<$ty>]], &'a [[<$ty>]]);

            impl <'a> SimdFn for [<Dot $ty>]<'a> {
                type Result = $ty;

                #[inline(always)]
                fn call<S: reviero::traits::Simd>(self, simd: S) -> Self::Result {
                    assert_eq!(self.0.len(), self.1.len());

                    let (x_head, x_tail) = reviero::utils::[<as_simd_ $ty s>]::<S>(self.0);
                    let (y_head, y_tail) = reviero::utils::[<as_simd_ $ty s>]::<S>(self.1);

                    let mut acc = [simd.[<splat_ $ty s>](0.0); 4];

                    let mut vectors = x_head.zip(y_head);
                    while vectors.len() >= 4 {
                        let (x0, y0) = vectors.next().unwrap();
                        let (x1, y1) = vectors.next().unwrap();
                        let (x2, y2) = vectors.next().unwrap();
                        let (x3, y3) = vectors.next().unwrap();

                        acc[0] = simd.[<mul_add_ $ty s>](x0, y0, acc[0]);
                        acc[1] = simd.[<mul_add_ $ty s>](x1, y1, acc[1]);
                        acc[2] = simd.[<mul_add_ $ty s>](x2, y2, acc[2]);
                        acc[3] = simd.[<mul_add_ $ty s>](x3, y3, acc[3]);
                    }

                    for (x, y) in vectors {
                        acc[0] = simd.[<mul_add_ $ty s>](x, y, acc[0]);
                    }

                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    acc[1] = simd.[<add_ $ty s>](acc[2], acc[3]);
                    acc[0] = simd.[<add_ $ty s>](acc[0], acc[1]);
                    let mut product = simd.[<reduce_sum_ $ty s>](acc[0]);

                    for (x, y) in x_tail.iter().zip(y_tail.iter()) {
                        product += x * y;
                    }

                    product
                }
            }
        }
    };
}

define_dot!(f32);
define_dot!(f64);

fn main() {
    let x = vec![1.0; 4096 * 4096];
    let y = vec![2.0; 4096 * 4096];

    let product = dispatch(Dotf32(&x, &y));
    println!("dot product(f32): {product}");

    let x = vec![1.0; 4096 * 4096];
    let y = vec![2.0; 4096 * 4096];

    let product = dispatch(Dotf64(&x, &y));
    println!("dot product(f64): {product}");
}
