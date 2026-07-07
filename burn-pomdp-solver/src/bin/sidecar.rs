//! The SIDECAR inference server — matches Codex's engine-side hook contract exactly:
//!   HTTP POST /infer  {entities:[f32;184], legal_action_mask:[bool;A], action_labels:[str;A], agent}
//!               -> {logits:[f64;A], value:f64}
//! Loads the trained Burn POMDP-solver, keeps per-agent GRU belief, serves a decision per request.
//! The engine (gated SOCCER_BURN_POMDP_SIDECAR + _URL) queries this each tick; on timeout it falls
//! back to its own policy — so this is a safe, reversible A/B lever.

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_pomdp_solver::adapter::{ENTITY_DIM, FIELD_MOTION_DIM, N_ENTITIES};
use burn_pomdp_solver::{PomdpActorCritic, PomdpConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

type B = NdArray;

#[derive(Deserialize)]
struct Req {
    entities: Vec<f32>,
    #[serde(default)]
    legal_action_mask: Vec<bool>,
    #[serde(default)]
    agent: usize,
    #[serde(default)]
    reset: bool,
}

#[derive(Serialize)]
struct Resp {
    logits: Vec<f64>,
    value: f64,
    contract_version: u32,
}

struct Server {
    net: PomdpActorCritic<B>,
    dev: <B as burn::tensor::backend::Backend>::Device,
    belief: HashMap<usize, Tensor<B, 3>>, // per-agent GRU hidden
    n_actions: usize,
    sample: bool, // SIDECAR_SAMPLE: stochastic (explore) for on-policy collection vs argmax for eval
    rng: u64,
}

impl Server {
    fn infer(&mut self, req: &Req) -> Resp {
        // entities -> (1, 23, 8); pad/truncate defensively to FIELD_MOTION_DIM
        let mut fm = req.entities.clone();
        fm.resize(FIELD_MOTION_DIM, 0.0);
        let ent = Tensor::<B, 1>::from_data(TensorData::new(fm, [FIELD_MOTION_DIM]), &self.dev)
            .reshape([1, N_ENTITIES, ENTITY_DIM]);
        if req.reset {
            self.belief.remove(&req.agent);
        }
        let prev = self.belief.get(&req.agent).cloned();
        let (logits, value, new_hidden) = self.net.step_infer(ent, prev);
        self.belief.insert(req.agent, new_hidden);
        let mut lg: Vec<f64> = logits.into_data().to_vec::<f32>().unwrap().iter().map(|&x| x as f64).collect();
        lg.resize(self.n_actions, f64::NEG_INFINITY);
        // mask illegal actions (engine also enforces, but be correct)
        if req.legal_action_mask.len() == lg.len() {
            for (i, &ok) in req.legal_action_mask.iter().enumerate() {
                if !ok {
                    lg[i] = f64::NEG_INFINITY;
                }
            }
        }
        let v = value.into_data().to_vec::<f32>().unwrap().first().copied().unwrap_or(0.0) as f64;
        // Stochastic collection: sample a legal action from softmax(logits) and rewrite logits so the
        // engine's argmax picks it. Gives on-policy EXPLORATION so fine-tuning can improve, not just
        // entrench the deterministic policy. Eval leaves sample=false (argmax = best play).
        if self.sample {
            let legal: Vec<usize> = (0..lg.len()).filter(|&i| lg[i].is_finite()).collect();
            if !legal.is_empty() {
                let maxl = legal.iter().map(|&i| lg[i]).fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = legal.iter().map(|&i| (lg[i] - maxl).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>().max(1e-9);
                self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((self.rng >> 33) as f64 / (1u64 << 31) as f64) * sum;
                let mut cum = 0.0;
                let mut chosen = *legal.last().unwrap();
                for (k, &i) in legal.iter().enumerate() {
                    cum += exps[k];
                    if u <= cum {
                        chosen = i;
                        break;
                    }
                }
                for (i, v) in lg.iter_mut().enumerate() {
                    *v = if i == chosen { 0.0 } else { f64::NEG_INFINITY };
                }
            }
        }
        Resp { logits: lg, value: v, contract_version: 1 }
    }
}

fn read_body(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let mut content_len = 0usize;
    let mut header_end = None;
    loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if header_end.is_none() {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let headers = String::from_utf8_lossy(&buf[..pos]).to_lowercase();
                for line in headers.lines() {
                    if let Some(v) = line.strip_prefix("content-length:") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                }
            }
        }
        if let Some(he) = header_end {
            if buf.len() >= he + content_len {
                return Some(String::from_utf8_lossy(&buf[he..he + content_len]).to_string());
            }
        }
    }
    None
}

fn main() {
    let dev = Default::default();
    let n_actions: usize = std::env::var("N_ACTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(73);
    let model = std::env::var("MODEL").unwrap_or_else(|_| "/tmp/burn-pomdp-big".into());
    let cfg = PomdpConfig::new(ENTITY_DIM, n_actions).with_model_dim(96).with_hidden_dim(128).with_n_heads(6);
    let net = cfg.init::<B>(&dev).load(&model, &dev).expect("load trained model");
    let addr = std::env::var("SOCCER_BURN_POMDP_SIDECAR_URL")
        .ok()
        .map(|u| u.trim_start_matches("http://").split('/').next().unwrap_or("127.0.0.1:8091").to_string())
        .unwrap_or_else(|| "127.0.0.1:8091".into());
    let listener = TcpListener::bind(&addr).expect("bind sidecar");
    let mut srv = Server { net, dev, belief: HashMap::new(), n_actions };
    println!("burn-pomdp sidecar listening on http://{addr}/infer  model={model}.bin  n_actions={n_actions}");
    let mut nreq = 0usize;
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let parsed = read_body(&mut stream).and_then(|b| serde_json::from_str::<Req>(&b).ok());
        nreq += 1;
        if nreq <= 3 || nreq % 500 == 0 {
            eprintln!("SIDECAR_SERVER served {nreq} requests (parsed_ok={})", parsed.is_some());
        }
        let resp = match parsed {
            Some(req) => srv.infer(&req),
            None => Resp { logits: vec![], value: 0.0, contract_version: 1 },
        };
        let body = serde_json::to_string(&resp).unwrap_or_else(|_| "{}".into());
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
    }
}
