//! Seeded RNG used everywhere randomness is needed (RANSAC sampling, BRIEF
//! pattern generation, synthetic data). The core must stay deterministic:
//! never use system RNG or time.

pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [0, n). Requires n > 0.
    #[inline]
    pub fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Two independent standard-normal samples (Box-Muller).
    pub fn next_gaussian_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        (r * th.cos(), r * th.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = XorShift64::new(1);
        let mut b = XorShift64::new(1);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f64_in_range() {
        let mut r = XorShift64::new(3);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
