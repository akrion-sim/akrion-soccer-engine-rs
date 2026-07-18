//! Minimal fully-connected MLP with manual backprop + Adam. Zero deps.
//!
//! Hidden layers use tanh; the output layer is linear. Used for both the actor
//! (output = action logits) and the critic (output = 1 value). Everything is
//! hand-rolled so every gradient is inspectable — the opposite of a black box.

use crate::rng::Rng;
use std::io::{Error, ErrorKind};
use std::path::Path;

#[derive(Clone)]
pub struct Mlp {
    sizes: Vec<usize>, // e.g. [in, 64, 64, out]
    w: Vec<Vec<f32>>,  // per layer, row-major [out*in]
    b: Vec<Vec<f32>>,  // per layer [out]
    gw: Vec<Vec<f32>>, // grad accumulators
    gb: Vec<Vec<f32>>,
    mw: Vec<Vec<f32>>, // Adam moments
    vw: Vec<Vec<f32>>,
    mb: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
    t: f32,          // Adam timestep
    grad_count: f32, // samples accumulated since last step
}

impl Mlp {
    pub fn new(sizes: &[usize], rng: &mut Rng) -> Self {
        let n = sizes.len() - 1;
        let mut w = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let (mut gw, mut gb) = (Vec::new(), Vec::new());
        let (mut mw, mut vw, mut mb, mut vb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for l in 0..n {
            let (fin, fout) = (sizes[l], sizes[l + 1]);
            // He-ish init for tanh: scale ~ sqrt(1/fin).
            let scale = (1.0 / fin as f32).sqrt();
            let mut wl = vec![0.0f32; fout * fin];
            for x in wl.iter_mut() {
                *x = rng.normal(0.0, scale);
            }
            w.push(wl);
            b.push(vec![0.0f32; fout]);
            gw.push(vec![0.0f32; fout * fin]);
            gb.push(vec![0.0f32; fout]);
            mw.push(vec![0.0f32; fout * fin]);
            vw.push(vec![0.0f32; fout * fin]);
            mb.push(vec![0.0f32; fout]);
            vb.push(vec![0.0f32; fout]);
        }
        Mlp {
            sizes: sizes.to_vec(),
            w,
            b,
            gw,
            gb,
            mw,
            vw,
            mb,
            vb,
            t: 0.0,
            grad_count: 0.0,
        }
    }

    #[allow(dead_code)]
    pub fn out_dim(&self) -> usize {
        *self.sizes.last().unwrap()
    }

    pub fn in_dim(&self) -> usize {
        self.sizes[0]
    }

    /// Forward pass, returning per-layer post-activations (acts[0] == input).
    pub fn forward(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let n = self.w.len();
        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(n + 1);
        acts.push(x.to_vec());
        for l in 0..n {
            let (fin, fout) = (self.sizes[l], self.sizes[l + 1]);
            let a_prev = &acts[l];
            let mut z = vec![0.0f32; fout];
            for o in 0..fout {
                let mut s = self.b[l][o];
                let base = o * fin;
                for i in 0..fin {
                    s += self.w[l][base + i] * a_prev[i];
                }
                // tanh on hidden layers, linear on the last.
                z[o] = if l + 1 < n { s.tanh() } else { s };
            }
            acts.push(z);
        }
        acts
    }

    #[inline]
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let acts = self.forward(x);
        acts.into_iter().last().unwrap()
    }

    /// Accumulate gradients for one sample given dL/d(output) (pre-activation of
    /// the linear output layer, which equals the output itself).
    pub fn backward(&mut self, acts: &[Vec<f32>], d_out: &[f32]) {
        let n = self.w.len();
        let mut delta = d_out.to_vec(); // grad wrt this layer's pre-activation
        for l in (0..n).rev() {
            let (fin, fout) = (self.sizes[l], self.sizes[l + 1]);
            let a_prev = &acts[l];
            for o in 0..fout {
                let d = delta[o];
                self.gb[l][o] += d;
                let base = o * fin;
                for i in 0..fin {
                    self.gw[l][base + i] += d * a_prev[i];
                }
            }
            if l > 0 {
                // Propagate to previous layer, then apply tanh'(a) = 1 - a^2.
                let mut dprev = vec![0.0f32; fin];
                for o in 0..fout {
                    let d = delta[o];
                    let base = o * fin;
                    for i in 0..fin {
                        dprev[i] += d * self.w[l][base + i];
                    }
                }
                let a_this = &acts[l];
                for i in 0..fin {
                    dprev[i] *= 1.0 - a_this[i] * a_this[i];
                }
                delta = dprev;
            }
        }
        self.grad_count += 1.0;
    }

    /// Adam step over the mean-accumulated gradient, then zero the accumulators.
    pub fn step(&mut self, lr: f32) {
        if self.grad_count == 0.0 {
            return;
        }
        let inv = 1.0 / self.grad_count;
        self.t += 1.0;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powf(self.t);
        let bc2 = 1.0 - b2.powf(self.t);
        for l in 0..self.w.len() {
            for k in 0..self.w[l].len() {
                let g = self.gw[l][k] * inv;
                self.mw[l][k] = b1 * self.mw[l][k] + (1.0 - b1) * g;
                self.vw[l][k] = b2 * self.vw[l][k] + (1.0 - b2) * g * g;
                let mhat = self.mw[l][k] / bc1;
                let vhat = self.vw[l][k] / bc2;
                self.w[l][k] -= lr * mhat / (vhat.sqrt() + eps);
                self.gw[l][k] = 0.0;
            }
            for k in 0..self.b[l].len() {
                let g = self.gb[l][k] * inv;
                self.mb[l][k] = b1 * self.mb[l][k] + (1.0 - b1) * g;
                self.vb[l][k] = b2 * self.vb[l][k] + (1.0 - b2) * g * g;
                let mhat = self.mb[l][k] / bc1;
                let vhat = self.vb[l][k] / bc2;
                self.b[l][k] -= lr * mhat / (vhat.sqrt() + eps);
                self.gb[l][k] = 0.0;
            }
        }
        self.grad_count = 0.0;
    }
}

impl Mlp {
    /// Persist weights to a simple text file: line 1 = sizes; then per layer a
    /// weights line and a biases line (space-separated f32). Zero deps.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut s = String::new();
        s.push_str(
            &self
                .sizes
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        );
        s.push('\n');
        for l in 0..self.w.len() {
            s.push_str(
                &self.w[l]
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            s.push('\n');
            s.push_str(
                &self.b[l]
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            s.push('\n');
        }
        std::fs::write(path, s)
    }

    /// Load weights saved by `save` (fresh Mlp; Adam state reset).
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let txt = std::fs::read_to_string(path)?;
        let mut lines = txt.lines();
        let sizes_line = lines
            .next()
            .ok_or_else(|| invalid_data("missing layer-size line"))?;
        let sizes: Vec<usize> = parse_values(sizes_line, "layer size")?;
        if sizes.len() < 2 || sizes.contains(&0) {
            return Err(invalid_data(
                "model must have at least input and output layers",
            ));
        }
        let mut seed = Rng::new(0);
        let mut m = Mlp::new(&sizes, &mut seed);
        for l in 0..m.w.len() {
            let wl_line = lines
                .next()
                .ok_or_else(|| invalid_data(format!("missing weights for layer {l}")))?;
            let bl_line = lines
                .next()
                .ok_or_else(|| invalid_data(format!("missing biases for layer {l}")))?;
            let wl: Vec<f32> = parse_values(wl_line, "weight")?;
            let bl: Vec<f32> = parse_values(bl_line, "bias")?;
            if wl.len() != m.w[l].len() {
                return Err(invalid_data(format!(
                    "layer {l} has {} weights, expected {}",
                    wl.len(),
                    m.w[l].len()
                )));
            }
            if bl.len() != m.b[l].len() {
                return Err(invalid_data(format!(
                    "layer {l} has {} biases, expected {}",
                    bl.len(),
                    m.b[l].len()
                )));
            }
            m.w[l] = wl;
            m.b[l] = bl;
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(invalid_data("extra non-empty lines after final layer"));
        }
        Ok(m)
    }
}

fn parse_values<T: std::str::FromStr>(line: &str, label: &str) -> std::io::Result<Vec<T>> {
    line.split_whitespace()
        .map(|x| {
            x.parse::<T>()
                .map_err(|_| invalid_data(format!("invalid {label}: {x}")))
        })
        .collect()
}

fn invalid_data(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

/// Numerically stable masked softmax. Masked entries get probability 0.
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if mask[i] && l > m {
            m = l;
        }
    }
    let mut out = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for i in 0..logits.len() {
        if mask[i] {
            let e = (logits[i] - m).exp();
            out[i] = e;
            sum += e;
        }
    }
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_softmax_zeroes_illegal_actions_and_normalizes() {
        let probs = masked_softmax(&[1.0, 5.0, -1.0], &[true, false, true]);
        assert_eq!(probs[1], 0.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[2]);
    }

    #[test]
    fn model_save_load_round_trips_predictions() {
        let mut rng = Rng::new(42);
        let model = Mlp::new(&[3, 4, 2], &mut rng);
        let path = std::env::temp_dir().join(format!(
            "fiveaside-model-roundtrip-{}-{}.txt",
            std::process::id(),
            rng.next_u64()
        ));
        model.save(&path).unwrap();
        let loaded = Mlp::load(&path).unwrap();
        let x = [0.25, -0.5, 0.75];
        let a = model.predict(&x);
        let b = loaded.predict(&x);
        assert_eq!(model.out_dim(), loaded.out_dim());
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn malformed_model_load_returns_invalid_data() {
        let path = std::env::temp_dir().join(format!(
            "fiveaside-bad-model-{}-{}.txt",
            std::process::id(),
            99
        ));
        std::fs::write(&path, "3 2\n1 2\n").unwrap();
        let err = match Mlp::load(&path) {
            Ok(_) => panic!("malformed model loaded successfully"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let _ = std::fs::remove_file(path);
    }
}
