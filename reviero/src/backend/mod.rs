pub mod fallback;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_64;
