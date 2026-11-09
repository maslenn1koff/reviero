pub mod fallback;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(test)]
mod test_utils;

#[cfg(test)]
mod tests;
