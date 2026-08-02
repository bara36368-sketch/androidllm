//! androidllm_rs - PyO3 accelerator core for androidllm.
//!
//! Fuses the per-layer llama forward (matmuls, rope, attention, swiglu,
//! rms_norm) into a single native call, plus the head matmul and multinomial
//! sampling. Semantics mirror androidllm/models/llama.py + engine._sample
//! so outputs match the numpy path (modulo f32 accumulation).
//!
//! Weights arrive as f16 numpy arrays (dequantized by Python), row-major
//! (in, out). A matmul computes out[n] = sum_k a[k] * w[k * out + n] for a
//! single-row activation a -- i.e. numpy's `a @ w.T` without building w.T.

use half::f16;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2,
            PyReadwriteArray3, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict};
use rayon::prelude::*;
use std::sync::OnceLock;

fn thread_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("ANDROIDLLM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .clamp(1, 64);
        rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()
    })
}

// -- matmul: out[n] = sum_k a[k] * w[k * w_out + n], f32 accumulate ----------

/// Single-row f16 weight matmul. `w` is (in, out) row-major f16.
/// Parallelised over output columns when large enough.
fn mm(a: &[f32], w: &[f16], w_out: usize, w_in: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w_out];
    if w_out >= 256 {
        let a_owned: Vec<f32> = a.to_vec();
        thread_pool().install(|| {
            out.par_chunks_mut(64).enumerate().for_each(|(ci, block)| {
                let n0 = ci * 64;
                for (i, o) in block.iter_mut().enumerate() {
                    let n = n0 + i;
                    if n >= w_out {
                        break;
                    }
                    let base = n * w_in;
                    let mut acc = 0.0f32;
                    for k in 0..w_in {
                        acc += w[base + k].to_f32() * a_owned[k];
                    }
                    *o = acc;
                }
            });
        });
    } else {
        for n in 0..w_out {
            let base = n * w_in;
            let mut acc = 0.0f32;
            for k in 0..w_in {
                acc += w[base + k].to_f32() * a[k];
            }
            out[n] = acc;
        }
    }
    out
}

// -- rms_norm (matches numpy: x / sqrt(mean(x^2) + eps) * w, f32 math) --------

fn rms_norm(x: &[f32], w: &[f16], eps: f64) -> Vec<f32> {
    let mut mean = 0.0f64;
    for &v in x {
        mean += (v as f64) * (v as f64);
    }
    mean /= x.len() as f64;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| (v as f32 * scale as f32) * g.to_f32())
        .collect()
}

/// Matches apply_rope: split x in halves, rotate with cos/sin from the table.
/// Applies per-head: x is a flat (heads * head_dim) vector.
fn apply_rope(x: &[f32], rope_row: &[f16], heads: usize, head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut r = vec![0.0f32; x.len()];
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let c = rope_row[i].to_f32();
            let s = rope_row[half + i].to_f32();
            let a = x[base + i];
            let b = x[base + half + i];
            r[base + i] = a * c - b * s;
            r[base + half + i] = a * s + b * c;
        }
    }
    r
}

// -- fused layer forward -----------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn layer_forward_inner(
    x: &[f32],
    q: &[f16], k: &[f16], v: &[f16], o: &[f16],
    gate: &[f16], up: &[f16], down: &[f16],
    n_in: &[f16], n_post: &[f16],
    q_out: usize, q_in: usize, k_out: usize, k_in: usize,
    v_out: usize, v_in: usize, o_out: usize, o_in: usize,
    g_out: usize, g_in: usize, u_out: usize, u_in: usize,
    d_out: usize, d_in: usize,
    kv_k: &mut [f16], kv_v: &mut [f16],
    kv_stride: usize,
    pos: usize,
    rope: &[f16], // (max_len, head_dim) with cos in first half of each row
    heads: usize, kv_heads: usize, head_dim: usize,
    eps: f64,
) -> Vec<f32> {
    let hidden = x.len();
    let half = head_dim / 2;
    let rope_row = &rope[pos * head_dim..(pos + 1) * head_dim];

    let q_lin = mm(x, q, q_out, q_in); // (heads*head_dim)
    let k_lin = mm(x, k, k_out, k_in); // (kv_heads*head_dim)
    let v_lin = mm(x, v, v_out, v_in);
    let q_r = apply_rope(&q_lin, rope_row, heads, head_dim);
    let k_r = apply_rope(&k_lin, rope_row, kv_heads, head_dim);

    for h in 0..kv_heads {
        let hd = h * head_dim;
        for d in 0..head_dim {
            let idx = pos * kv_stride + hd + d;
            kv_k[idx] = f16::from_f32(k_r[hd + d]);
            kv_v[idx] = f16::from_f32(v_lin[hd + d]);
        }
    }

    // group-query attention, expanding kv heads by factor g
    let g = heads / kv_heads;
    let tlen = pos + 1;
    let inv = 1.0 / (head_dim as f64).sqrt() as f32;
    let mut scores = vec![0.0f32; heads * tlen];
    for h in 0..heads {
        let kh = h / g;
        let qh = &q_r[h * head_dim..(h + 1) * head_dim];
        let kvb = kh * head_dim;
        let row = &mut scores[h * tlen..(h + 1) * tlen];
        for t in 0..tlen {
            let kb = t * kv_stride + kvb;
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += qh[d] * kv_k[kb + d].to_f32();
            }
            row[t] = acc * inv;
        }
        // softmax
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        let invs = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= invs;
        }
    }

    let mut ctx = vec![0.0f32; heads * head_dim];
    for h in 0..heads {
        let kh = h / g;
        let hd = h * head_dim;
        let kvb = kh * head_dim;
        for t in 0..tlen {
            let p = scores[h * tlen + t];
            let vb = t * kv_stride + kvb;
            for d in 0..head_dim {
                ctx[hd + d] += p * kv_v[vb + d].to_f32();
            }
        }
    }
    let o_lin = mm(&ctx, o, o_out, o_in);

    let mut xo = vec![0.0f32; hidden];
    for i in 0..hidden {
        xo[i] = x[i] + o_lin[i];
    }

    let xn = rms_norm(&xo, n_in, eps);
    let gate_lin = mm(&xn, gate, g_out, g_in);
    let up_lin = mm(&xn, up, u_out, u_in);
    let mut mlp = vec![0.0f32; g_out];
    for i in 0..g_out {
        let gt = gate_lin[i];
        mlp[i] = gt * (1.0 / (1.0 + (-gt).exp())) * up_lin[i];
    }
    let down_lin = mm(&mlp, down, d_out, d_in);
    for i in 0..hidden {
        xo[i] += down_lin[i];
    }
    rms_norm(&xo, n_post, eps)
}

// -- PyO3 surface ------------------------------------------------------------

fn weight_of(
    layer: &Bound<'_, PyDict>,
    name: &str,
) -> PyResult<(Vec<f16>, usize, usize)> {
    let arr: PyReadonlyArray2<f16> = layer
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("layer missing key: {name}")))?
        .extract()?;
    let a = arr.as_array();
    let (out, inp) = (a.shape()[0], a.shape()[1]);
    Ok((a.iter().copied().collect(), inp, out))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn layer_forward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f16>,
    layer: &Bound<'_, PyDict>,
    kv_k: PyReadwriteArray3<'_, f16>,
    kv_v: PyReadwriteArray3<'_, f16>,
    pos: usize,
    rope: PyReadonlyArray2<'_, f16>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f16>>> {
    let mut kv_k = kv_k;
    let mut kv_v = kv_v;
    let xv: Vec<f32> = x.as_array().iter().map(|h| h.to_f32()).collect();
    let hidden = xv.len();
    let (q, q_in, q_out) = weight_of(layer, "q")?;
    let (k, k_in, k_out) = weight_of(layer, "k")?;
    let (v, v_in, v_out) = weight_of(layer, "v")?;
    let (o, o_in, o_out) = weight_of(layer, "o")?;
    let (gate, g_in, g_out) = weight_of(layer, "gate")?;
    let (up, u_in, u_out) = weight_of(layer, "up")?;
    let (down, d_in, d_out) = weight_of(layer, "down")?;
    let n_in: PyReadonlyArray1<f16> = layer
        .get_item("n_in")?
        .ok_or_else(|| PyValueError::new_err("layer missing key: n_in"))?
        .extract()?;
    let n_post: PyReadonlyArray1<f16> = layer
        .get_item("n_post")?
        .ok_or_else(|| PyValueError::new_err("layer missing key: n_post"))?
        .extract()?;

    let kv_shape = kv_k.as_array().shape().to_vec();
    let kv_stride = kv_shape[1] * kv_shape[2];
    if kv_stride != kv_heads * head_dim {
        return Err(PyValueError::new_err("kv cache shape mismatch"));
    }
    let max_len = kv_shape[0];
    if pos >= max_len {
        return Err(PyValueError::new_err("pos out of kv cache range"));
    }

    let mut kk: Vec<f16> = kv_k.as_array().iter().copied().collect();
    let mut vv: Vec<f16> = kv_v.as_array().iter().copied().collect();
    let rope_flat: Vec<f16> = rope.as_array().iter().copied().collect();

    let out = layer_forward_inner(
        &xv,
        &q, &k, &v, &o, &gate, &up, &down,
        &n_in.as_array().iter().copied().collect::<Vec<_>>(),
        &n_post.as_array().iter().copied().collect::<Vec<_>>(),
        q_out, q_in, k_out, k_in, v_out, v_in, o_out, o_in,
        g_out, g_in, u_out, u_in, d_out, d_in,
        &mut kk, &mut vv,
        kv_stride, pos, &rope_flat, heads, kv_heads, head_dim, eps,
    );

    // write KV cache back into numpy buffers
    for (i, item) in kv_k.as_array_mut().iter_mut().enumerate() {
        *item = kk[i];
    }
    for (i, item) in kv_v.as_array_mut().iter_mut().enumerate() {
        *item = vv[i];
    }

    let out_f16: Vec<f16> = out.iter().map(|&v| f16::from_f32(v)).collect();
    let _ = py;
    let arr = numpy::ndarray::Array2::from_shape_vec((1, hidden), out_f16)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py))
}

#[pyfunction]
fn head_logits<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f16>,
    final_norm: PyReadonlyArray1<'_, f16>,
    w: PyReadonlyArray2<'_, f16>,
    eps: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let xv: Vec<f32> = x.as_array().iter().map(|h| h.to_f32()).collect();
    let nv: Vec<f16> = final_norm.as_array().iter().copied().collect();
    let h = rms_norm(&xv, &nv, eps);
    let wv: Vec<f16> = w.as_array().iter().copied().collect();
    let shape = w.as_array().shape().to_vec();
    let (vocab, hidden) = (shape[0], shape[1]);
    let mut out = vec![0.0f32; vocab];
    thread_pool().install(|| {
        out.par_chunks_mut(256).enumerate().for_each(|(ci, block)| {
            let v0 = ci * 256;
            for (i, o) in block.iter_mut().enumerate() {
                let v = v0 + i;
                if v >= vocab {
                    break;
                }
                let base = v * hidden;
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    acc += wv[base + k].to_f32() * h[k];
                }
                *o = acc;
            }
        });
    });
    let arr = numpy::ndarray::Array2::from_shape_vec((1, vocab), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.to_pyarray(py))
}

// -- sampling (matches engine._sample, deterministic per seed) ---------------

#[pyfunction]
fn sample(
    logits: PyReadonlyArray1<'_, f32>,
    temperature: f64,
    top_p: f64,
    min_p: f64,
    seed: u64,
) -> usize {
    let lv: Vec<f32> = logits.as_array().iter().copied().collect();
    if temperature <= 0.0 {
        return argmax(&lv);
    }
    let temp = temperature.max(1e-9);
    let mx = lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = lv
        .iter()
        .map(|&l| ((l - mx) / temp as f32).exp())
        .collect();
    let total: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= total;
    }
    if min_p > 0.0 && min_p < 1.0 {
        let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
        let keep: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p >= min_p as f32 * pmax)
            .map(|(i, _)| i)
            .collect();
        if !keep.is_empty() {
            let sub: Vec<f32> = keep.iter().map(|&i| probs[i]).collect();
            let s: f32 = sub.iter().sum();
            let mut rng = XorShift64::new(seed);
            return keep[draw(&mut rng, &sub, s)];
        }
    }
    if top_p < 1.0 {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cum = 0.0f32;
        let mut keep = Vec::new();
        for &i in &order {
            cum += probs[i];
            keep.push(i);
            if cum > top_p as f32 {
                break;
            }
        }
        if keep.is_empty() {
            keep.push(order[0]);
        }
        let sub: Vec<f32> = keep.iter().map(|&i| probs[i]).collect();
        let s: f32 = sub.iter().sum();
        let mut rng = XorShift64::new(seed);
        return keep[draw(&mut rng, &sub, s)];
    }
    let s: f32 = probs.iter().sum();
    let mut rng = XorShift64::new(seed);
    draw(&mut rng, &probs, s)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn draw(rng: &mut XorShift64, probs: &[f32], total: f32) -> usize {
    let r: f32 = (rng.next() % 1_000_000_000) as f32 / 1_000_000_000.0;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p / total;
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[pymodule]
fn androidllm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layer_forward, m)?)?;
    m.add_function(wrap_pyfunction!(head_logits, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}
