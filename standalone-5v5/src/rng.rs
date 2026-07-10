//! Tiny deterministic PRNG (xorshift128+ style). Seeded, reproducible, zero deps.

#[derive(Clone)]
pub struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 to spread the seed into the two state words.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut sm = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        let s0 = sm();
        let s1 = sm() | 1;
        Rng { s0, s1 }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s0;
        let y = self.s1;
        self.s0 = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s1 = x;
        x.wrapping_add(y)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn f01(&mut self) -> f32 {
        // top 24 bits -> [0,1)
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Uniform f32 in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f01()
    }

    /// Approx standard normal via sum-of-uniforms (Irwin–Hall, k=6, centered).
    #[inline]
    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.f01();
        }
        // sum of 6 uniforms has mean 3, var 6/12=0.5 -> std ~0.707
        mean + std * (s - 3.0) / 0.7071
    }

    /// Sample an index from a probability distribution (must sum ~1).
    #[inline]
    pub fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        let r = self.f01();
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.len() - 1
    }
}
