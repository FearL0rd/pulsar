//! Kimi-K3 forward path.
//!
//! Reference: llama.cpp PR #26185 src/models/kimi-k3.cpp (saved under
//! docs/ref/) + the K3 tech report S2. Each block is 3 KDA layers then 1
//! gated MLA layer; on top of that sit Attention Residuals over depth and
//! a latent MoE whose routed experts run at n_expert_latent.
//!
//! Decode-shaped: one token per pass, prefill loops tokens. The KDA
//! recurrence (conv window + delta state) is sequential, so a batched
//! pass would have to loop inside the kernels anyway; qwen35 earned its
//! batched variants only after the single-token path was proven, and the
//! same order applies here.
//!
//! NoPE: K3 keeps the 64 rope-tail dims in the K/Q rows but never rotates
//! them. Rather than branch the hot MLA kernel, we hand it a rope config
//! whose rotation IS the identity: with ext_factor 0 the kernel takes
//! theta = freq_scale * theta_extrap, so freq_scale 0 gives theta 0, and
//! with attn_factor 1 that is exactly cos 1 / sin 0. Not an approximation
//! - see mla_rope_yarn in mla_kernels.inc. The q-side tail is simply
//! never passed through mla_rope_tail.

use super::{Attn, Ffn, K3W, LayerW, MatW, Model, Result, State};
use kernels::DeviceBuf;

/// Matmul over either weight encoding (mirrors qwen35::matw).
fn matw(out: &mut DeviceBuf, w: &MatW, x: &DeviceBuf, xq: &DeviceBuf, in_dim: u32, out_dim: u32, t: u32) -> Result {
    match w {
        MatW::Q8(b) => kernels::matmul_q8_0(out, b, x, in_dim, out_dim, t)?,
        MatW::Kq(k) => kernels::matmul_kq(out, &k.w, xq, in_dim, out_dim, t, k.row_bytes, k.quant)?,
    }
    Ok(())
}

/// The identity-rotation config described in the module header.
fn nope_cfg() -> kernels::RopeCfg {
    kernels::RopeCfg {
        n_ctx_orig: 1,
        freq_base: 10_000.0,
        freq_scale: 0.0,  // -> theta 0
        ext_factor: 0.0,  // -> no extrapolation mix, so theta stays 0
        attn_factor: 1.0, // -> cos 1, sin 0
        beta_fast: 32.0,
        beta_slow: 1.0,
        kq_mult: 1.0,
    }
}

/// Per-KDA-layer recurrent state, on that layer's owner card.
struct KdaState {
    /// delta-rule state [n_head][head_dim][head_dim]
    s: DeviceBuf,
    /// one conv window per stream, [conv_k - 1][d_inner] each. K3 gives
    /// q, k and v their own conv weights, so they need their own windows
    /// (qwen35 packs one window because its projection is fused).
    conv_q: DeviceBuf,
    conv_k: DeviceBuf,
    conv_v: DeviceBuf,
    dev: i32,
}

/// Scratch for whichever card a layer lands on. K3's non-expert stack is
/// ~30GB, so the layer-split planner spreads it and every buffer a layer
/// touches has to exist on that layer's card.
struct K3Scratch {
    dev: i32,
    /// hop buffers for the residual in / attention out
    normed_a: DeviceBuf,
    attn_out_a: DeviceBuf,
    xq: DeviceBuf,
    // KDA
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    cq: DeviceBuf,
    ck: DeviceBuf,
    cv: DeviceBuf,
    g: DeviceBuf,
    beta: DeviceBuf,
    fa: DeviceBuf,
    o: DeviceBuf,
    gate: DeviceBuf,
    // MLA
    q_rank: DeviceBuf,
    q_rank_norm: DeviceBuf,
    mq: DeviceBuf,
    kv_raw: DeviceBuf,
    kv_norm: DeviceBuf,
    qk_low: DeviceBuf,
    heads: DeviceBuf,
    selected: DeviceBuf,
}

impl K3Scratch {
    fn new(m: &Model, dev: i32, ctx: u32) -> Result<K3Scratch> {
        let s = m.shape;
        kernels::set_device(dev)?;
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let d_inner = (s.n_head * s.kda_head_dim) as usize;
        let conv_state = (s.ssm_conv_k as usize - 1) * d_inner;
        let _ = conv_state;
        Ok(K3Scratch {
            dev,
            normed_a: f32s(s.n_embd as usize)?,
            attn_out_a: f32s(s.n_embd as usize)?,
            // q8_K blocks: 256 values -> 292 bytes, over the widest input
            // any matw on this card sees (n_embd or the latent width)
            xq: DeviceBuf::alloc(
                (s.n_embd.max(s.n_expert_latent).max(s.n_ff_exp) as usize).div_ceil(256) * 292,
            )?,
            q: f32s(d_inner)?,
            k: f32s(d_inner)?,
            v: f32s(d_inner)?,
            cq: f32s(d_inner)?,
            ck: f32s(d_inner)?,
            cv: f32s(d_inner)?,
            g: f32s(d_inner)?,
            beta: f32s(s.n_head as usize)?,
            fa: f32s(s.kda_head_dim as usize)?,
            o: f32s(d_inner)?,
            gate: f32s(d_inner)?,
            q_rank: f32s(s.n_lora_q as usize)?,
            q_rank_norm: f32s(s.n_lora_q as usize)?,
            mq: f32s((s.n_head * s.qk_dim()) as usize)?,
            kv_raw: f32s((s.n_kv_lora + s.qk_rope) as usize)?,
            kv_norm: f32s(s.n_kv_lora as usize)?,
            qk_low: f32s((s.n_head * s.n_kv_lora) as usize)?,
            heads: f32s((s.n_head * s.value_mla) as usize)?,
            // one u32 per visible row: mla_fill_selected_range writes
            // `visible` entries, which grows to the full context
            selected: DeviceBuf::alloc(ctx as usize * 4)?,
        })
    }
}

pub(super) struct K3Rt {
    states: Vec<Option<KdaState>>,
    scratch: Vec<K3Scratch>,
    /// AttnRes checkpoint bank, [n_ckpt][n_embd] on the primary. One
    /// checkpoint is banked every attn_res_block layers; at 93 layers and
    /// block 12 that is 8 of them, so this is ~230KB, not a KV cache.
    ckpt: DeviceBuf,
    /// how many checkpoints are live for the token in flight
    n_ckpt: u32,
    /// latent-space MoE input/output, [n_expert_latent]
    lat_in: DeviceBuf,
    lat_out: DeviceBuf,
    /// AttnRes output (must not alias the stream it mixes)
    mixed: DeviceBuf,
    /// routed-expert result after the up-projection, kept separate so the
    /// shared-expert add has a distinct source and destination
    routed_out: DeviceBuf,
}

impl K3Rt {
    pub fn new(m: &Model, ctx: u32) -> Result<K3Rt> {
        let s = m.shape;
        let primary = kernels::get_device();
        let d_inner = (s.n_head * s.kda_head_dim) as usize;
        let hd = s.kda_head_dim as usize;

        let mut states = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer as usize {
            let is_kda = matches!(&m.layers[il].attn, Attn::K3(w) if w.kda.is_some());
            if !is_kda {
                states.push(None);
                continue;
            }
            let dev = m.attn_layer_dev.get(il).copied().unwrap_or(primary);
            kernels::set_device(dev)?;
            let sbytes = s.n_head as usize * hd * hd * 4;
            let cbytes = (s.ssm_conv_k as usize - 1) * d_inner * 4;
            let mut st = KdaState {
                s: DeviceBuf::alloc(sbytes)?,
                conv_q: DeviceBuf::alloc(cbytes)?,
                conv_k: DeviceBuf::alloc(cbytes)?,
                conv_v: DeviceBuf::alloc(cbytes)?,
                dev,
            };
            kernels::zero(&mut st.s, sbytes)?;
            kernels::zero(&mut st.conv_q, cbytes)?;
            kernels::zero(&mut st.conv_k, cbytes)?;
            kernels::zero(&mut st.conv_v, cbytes)?;
            states.push(Some(st));
        }

        // one scratch set per card any layer actually lands on
        let mut devs: Vec<i32> = m.attn_layer_dev.iter().copied().collect();
        devs.push(primary);
        devs.sort_unstable();
        devs.dedup();
        let mut scratch = Vec::with_capacity(devs.len());
        for d in devs {
            scratch.push(K3Scratch::new(m, d, ctx)?);
        }

        kernels::set_device(primary)?;
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let max_ckpt = if s.attn_res_block == 0 {
            0
        } else {
            s.n_exec_layer.div_ceil(s.attn_res_block)
        };
        let lat = if s.n_expert_latent > 0 { s.n_expert_latent } else { s.n_embd };
        Ok(K3Rt {
            states,
            scratch,
            ckpt: f32s((max_ckpt.max(1) * s.n_embd) as usize)?,
            n_ckpt: 0,
            lat_in: f32s(lat as usize)?,
            lat_out: f32s(lat as usize)?,
            mixed: f32s(s.n_embd as usize)?,
            routed_out: f32s(s.n_embd as usize)?,
        })
    }

    fn reset(&mut self) -> Result {
        let primary = kernels::get_device();
        for st in self.states.iter_mut().flatten() {
            kernels::set_device(st.dev)?;
            for b in [&mut st.s, &mut st.conv_q, &mut st.conv_k, &mut st.conv_v] {
                let n = b.bytes();
                kernels::zero(b, n)?;
            }
        }
        kernels::set_device(primary)?;
        self.n_ckpt = 0;
        Ok(())
    }

    fn sc(&mut self, dev: i32) -> Result<&mut K3Scratch> {
        self.scratch
            .iter_mut()
            .find(|x| x.dev == dev)
            .ok_or_else(|| "k3: no scratch for owner device".into())
    }
}

impl Model {
    pub(super) fn forward_k3(&self, st: &mut State, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.k3.take().ok_or("k3 state missing")?;
        let r = self.forward_k3_inner(st, &mut rt, tokens, pos0, rows);
        st.k3 = Some(rt);
        r
    }

    fn forward_k3_inner(&self, st: &mut State, rt: &mut K3Rt, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        if pos0 == 0 {
            rt.reset()?;
        }
        let primary = kernels::get_device();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + i as u32;
            let ids = [tok as i32];
            st.tok.write(0, kernels::as_bytes(&ids))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, 1)?;
            rt.n_ckpt = 0;
            for il in 0..s.n_exec_layer as usize {
                self.eval_k3_layer(st, rt, il, &self.layers[il], pos)?;
            }
            kernels::set_device(primary)?;
        }
        if rows == 0 {
            return Ok(None);
        }
        if rows != 1 {
            return Err("k3: decode path emits one row per pass".into());
        }
        // final AttnRes mix over the full bank, then the head
        if s.attn_res_block > 0 {
            let w = self
                .output_res_score
                .as_ref()
                .ok_or("k3: output_res_score missing")?;
            kernels::k3_attn_res(
                &mut rt.mixed,
                &st.cur,
                Some(&rt.ckpt).filter(|_| rt.n_ckpt > 0),
                w,
                1,
                s.n_embd,
                rt.n_ckpt,
                s.rms_eps,
            )?;
            kernels::copy_d2d(&mut st.cur, 0, &rt.mixed, 0, s.n_embd as usize * 4)?;
        }
        kernels::rms_norm(&mut st.normed, &st.cur, &self.output_norm, s.n_embd, 1, s.rms_eps)?;
        self.head_logits(st, 1)?;
        kernels::sync()?;
        Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
    }

    fn eval_k3_layer(&self, st: &mut State, rt: &mut K3Rt, il: usize, l: &LayerW, pos: u32) -> Result {
        let s = self.shape;
        let eps = s.rms_eps;
        let primary = kernels::get_device();
        let Attn::K3(w) = &l.attn else {
            return Err("k3 layer without K3 attn weights".into());
        };
        let d_inner = s.n_head * s.kda_head_dim;
        let n_embd = s.n_embd;

        // ---- AttnRes mix #1, then bank this layer's RAW input.
        // Order matters: the checkpoint is the raw residual entering the
        // layer, not the mixed value the layer goes on to read.
        let banked = s.attn_res_block > 0 && il % s.attn_res_block as usize == 0;
        if s.attn_res_block > 0 {
            kernels::k3_attn_res(
                &mut rt.mixed,
                &st.cur,
                Some(&rt.ckpt).filter(|_| rt.n_ckpt > 0),
                &w.attn_res_score,
                1,
                n_embd,
                rt.n_ckpt,
                eps,
            )?;
            if banked {
                kernels::copy_d2d(
                    &mut rt.ckpt,
                    rt.n_ckpt as usize * n_embd as usize * 4,
                    &st.cur,
                    0,
                    n_embd as usize * 4,
                )?;
                rt.n_ckpt += 1;
            }
            kernels::rms_norm(&mut st.normed, &rt.mixed, &l.attn_norm, n_embd, 1, eps)?;
        } else {
            kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, n_embd, 1, eps)?;
        }

        // ---- attention half, on this layer's owner card
        let a_dev = self.attn_layer_dev.get(il).copied().unwrap_or(primary);
        if a_dev != primary {
            let bytes = n_embd as usize * 4;
            let sc = rt.sc(a_dev)?;
            kernels::copy_across(&mut sc.normed_a, &st.normed, bytes)?;
            kernels::set_device(a_dev)?;
        }

        if let Some(kda) = &w.kda {
            let sc = rt.scratch.iter_mut().find(|x| x.dev == a_dev).ok_or("k3 scratch")?;
            let xin: *const DeviceBuf = if a_dev != primary { &sc.normed_a } else { &st.normed };
            let xin = unsafe { &*xin };
            if matches!(kda.wq, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, xin, n_embd, 1)?;
            }
            matw(&mut sc.q, &kda.wq, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.k, &kda.wk, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.v, &kda.wv, xin, &sc.xq, n_embd, d_inner, 1)?;
            matw(&mut sc.gate, &kda.wg, xin, &sc.xq, n_embd, d_inner, 1)?;
            // decay logits through the rank-head_dim bottleneck, then
            // beta per head; both finished on-device by k3_kda_coeffs
            matw(&mut sc.fa, &kda.f_a, xin, &sc.xq, n_embd, s.kda_head_dim, 1)?;
            matw(&mut sc.g, &kda.f_b, &sc.fa, &sc.xq, s.kda_head_dim, d_inner, 1)?;
            matw(&mut sc.beta, &kda.beta_w, xin, &sc.xq, n_embd, s.n_head, 1)?;
            kernels::k3_kda_coeffs(
                &mut sc.g, &mut sc.beta, &kda.a, &kda.dt_bias,
                1, s.n_head, s.kda_head_dim, s.kda_gate_lb,
            )?;

            let ks = rt.states[il].as_mut().ok_or("k3 kda state missing")?;
            kernels::qwen35_conv_step(&mut sc.cq, &sc.q, &kda.conv_q, &mut ks.conv_q, d_inner, s.ssm_conv_k)?;
            kernels::qwen35_conv_step(&mut sc.ck, &sc.k, &kda.conv_k, &mut ks.conv_k, d_inner, s.ssm_conv_k)?;
            kernels::qwen35_conv_step(&mut sc.cv, &sc.v, &kda.conv_v, &mut ks.conv_v, d_inner, s.ssm_conv_k)?;
            kernels::qwen35_l2_norm(&mut sc.cq, s.n_head, s.kda_head_dim, eps)?;
            kernels::qwen35_l2_norm(&mut sc.ck, s.n_head, s.kda_head_dim, eps)?;
            kernels::k3_kda_step(
                &mut sc.o, &mut ks.s, &sc.cq, &sc.ck, &sc.cv, &sc.g, &sc.beta,
                s.n_head, s.kda_head_dim,
            )?;
            // per-head rms norm, then the full-rank sigmoid output gate
            kernels::gqa_head_rms_norm(&mut sc.o, Some(&kda.ssm_norm), s.n_head, s.kda_head_dim, eps)?;
            kernels::qwen35_sigmoid_gate(&mut sc.o, &sc.gate, d_inner)?;
            if matches!(kda.out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, &sc.o, d_inner, 1)?;
            }
            if a_dev != primary {
                matw(&mut sc.attn_out_a, &kda.out, &sc.o, &sc.xq, d_inner, n_embd, 1)?;
                kernels::copy_across(&mut st.attn_out, &sc.attn_out_a, n_embd as usize * 4)?;
                kernels::set_device(primary)?;
            } else {
                matw(&mut st.attn_out, &kda.out, &sc.o, &sc.xq, d_inner, n_embd, 1)?;
            }
        } else if let Some(mla) = &w.mla {
            let rope = nope_cfg();
            let kv_raw_dim = s.n_kv_lora + s.qk_rope;
            let sc = rt.scratch.iter_mut().find(|x| x.dev == a_dev).ok_or("k3 scratch")?;
            let xin: *const DeviceBuf = if a_dev != primary { &sc.normed_a } else { &st.normed };
            let xin = unsafe { &*xin };
            kernels::matmul_q8_0(&mut sc.q_rank, &mla.q_a, xin, n_embd, s.n_lora_q, 1)?;
            kernels::rms_norm(&mut sc.q_rank_norm, &sc.q_rank, &mla.q_a_norm, s.n_lora_q, 1, eps)?;
            kernels::matmul_q8_0(&mut sc.mq, &mla.q_b, &sc.q_rank_norm, s.n_lora_q, s.n_head * s.qk_dim(), 1)?;
            // NoPE: no mla_rope_tail on q
            kernels::matmul_q8_0(&mut sc.kv_raw, &mla.kv_a_mqa, xin, n_embd, kv_raw_dim, 1)?;
            kernels::mla_kv_lora_rms_norm(&mut sc.kv_norm, &sc.kv_raw, &mla.kv_a_norm, 1, kv_raw_dim, s.n_kv_lora, eps)?;
            kernels::mla_store_compact_kv(
                &mut st.kcache[il], &mut st.vcache[il], &sc.kv_norm, &sc.kv_raw,
                pos, 1, st.ctx, kv_raw_dim, s.n_kv_lora, s.qk_rope, st.kvq_lat,
            )?;
            let visible = pos + 1;
            kernels::mla_fill_selected_range(&mut sc.selected, 1, pos, visible, st.ctx)?;
            kernels::mla_qk_lowrank(&mut sc.qk_low, &sc.mq, &mla.k_b, 1, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_dim())?;
            kernels::mla_attention(
                &mut sc.heads, &sc.mq, &sc.qk_low, &st.kcache[il], &st.vcache[il],
                &mla.v_b, &sc.selected, 1, visible, st.ctx, s.n_head, s.n_kv_lora,
                s.qk_nope, s.qk_rope, s.value_mla, &rope, st.kvq_lat,
            )?;
            // K3's output gate reads the NORMED LAYER INPUT, not the
            // attention result - reusing heads here would be wrong.
            if matches!(mla.gate, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, xin, n_embd, 1)?;
            }
            matw(&mut sc.gate, &mla.gate, xin, &sc.xq, n_embd, s.n_head * s.value_mla, 1)?;
            kernels::qwen35_sigmoid_gate(&mut sc.heads, &sc.gate, s.n_head * s.value_mla)?;
            if matches!(mla.out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut sc.xq, &sc.heads, s.n_head * s.value_mla, 1)?;
            }
            if a_dev != primary {
                matw(&mut sc.attn_out_a, &mla.out, &sc.heads, &sc.xq, s.n_head * s.value_mla, n_embd, 1)?;
                kernels::copy_across(&mut st.attn_out, &sc.attn_out_a, n_embd as usize * 4)?;
                kernels::set_device(primary)?;
            } else {
                matw(&mut st.attn_out, &mla.out, &sc.heads, &sc.xq, s.n_head * s.value_mla, n_embd, 1)?;
            }
        } else {
            return Err("k3 layer with neither kda nor mla".into());
        }

        // On a banked layer the residual RESTARTS from the attention
        // output; elsewhere it accumulates. This is the AttnRes contract:
        // the banked value is what carries the prefix forward, so adding
        // it back in here would double-count it.
        if banked {
            kernels::copy_d2d(&mut st.after_attn, 0, &st.attn_out, 0, n_embd as usize * 4)?;
        } else {
            kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, n_embd)?;
        }

        // ---- FFN half: AttnRes mix #2, then dense or latent MoE
        if s.attn_res_block > 0 {
            kernels::k3_attn_res(
                &mut rt.mixed,
                &st.after_attn,
                Some(&rt.ckpt).filter(|_| rt.n_ckpt > 0),
                &w.ffn_res_score,
                1,
                n_embd,
                rt.n_ckpt,
                eps,
            )?;
            kernels::rms_norm(&mut st.normed, &rt.mixed, &l.ffn_norm, n_embd, 1, eps)?;
        } else {
            kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, n_embd, 1, eps)?;
        }
        self.k3_ffn(st, rt, il, l, w)?;
        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_embd)?;
        Ok(())
    }

    /// Dense FFN on the leading layer, latent MoE everywhere else.
    fn k3_ffn(&self, st: &mut State, rt: &mut K3Rt, il: usize, l: &LayerW, w: &K3W) -> Result {
        let s = self.shape;
        let act = s.moe_act_op;
        if let Ffn::Dense { gate, up, down } = &l.ffn {
            kernels::matmul_q8_0(&mut st.gate_act, gate, &st.normed, s.n_embd, s.n_ff_dense, 1)?;
            kernels::matmul_q8_0(&mut st.up_act, up, &st.normed, s.n_embd, s.n_ff_dense, 1)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, s.n_ff_dense, 0.0, 1.0, act)?;
            kernels::matmul_q8_0(&mut st.ffn_out, down, &st.ffn_mid, s.n_ff_dense, s.n_embd, 1)?;
            return Ok(());
        }
        let Ffn::Moe { gate_inp, probs_b, shexp, gate_exps, up_exps, down_exps, .. } = &l.ffn else {
            return Err("k3 layer without a dense or MoE ffn".into());
        };
        let routed = w.routed.as_ref().ok_or("k3 MoE layer without latent projections")?;

        // The router scores the FULL-WIDTH input while the experts consume
        // the latent one, so the logits come off `normed`, not lat_in.
        kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert, 1)?;
        kernels::router_select(
            &mut st.router_selected,
            &mut st.router_weights,
            &st.router_logits,
            probs_b,
            s.n_expert,
            s.n_expert_used,
            s.expert_weight_scale,
            1,
            0, // sigmoid router with the probs_b correction bias
            0,
        )?;

        // shared experts stay at n_embd and read the un-projected input
        let _ = shexp; // K3 carries its own native-quant triple in K3W
        if let Some(sh) = &w.shexp {
            let ffw = s.n_ff_shexp;
            kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, 1)?;
            matw(&mut st.gate_act, &sh.gate, &st.normed, &st.xq, s.n_embd, ffw, 1)?;
            matw(&mut st.up_act, &sh.up, &st.normed, &st.xq, s.n_embd, ffw, 1)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, ffw, 0.0, 1.0, act)?;
            kernels::quantize_q8_k(&mut st.midq, &st.ffn_mid, ffw, 1)?;
            matw(&mut st.shared_out, &sh.down, &st.ffn_mid, &st.midq, ffw, s.n_embd, 1)?;
        } else {
            kernels::zero(&mut st.shared_out, s.n_embd as usize * 4)?;
        }

        // routed half runs in the latent space
        let sc_dev = kernels::get_device();
        let sc = rt.scratch.iter_mut().find(|x| x.dev == sc_dev).ok_or("k3 scratch")?;
        if matches!(routed.down, MatW::Kq(_)) {
            kernels::quantize_q8_k(&mut sc.xq, &st.normed, s.n_embd, 1)?;
        }
        matw(&mut rt.lat_in, &routed.down, &st.normed, &sc.xq, s.n_embd, s.n_expert_latent, 1)?;
        kernels::quantize_q8_k(&mut st.xq, &rt.lat_in, s.n_expert_latent, 1)?;
        kernels::sync()?;
        let selected = st.router_selected.read_i32(s.n_expert_used as usize)?;
        self.dsv4_moe(st, &selected, gate_exps, up_exps, down_exps, 0, 1)?;
        // moe_out is latent-wide here, not n_embd
        if let Some(n) = &routed.norm {
            kernels::rms_norm(&mut rt.lat_out, &st.moe_out, n, s.n_expert_latent, 1, s.rms_eps)?;
        } else {
            kernels::copy_d2d(&mut rt.lat_out, 0, &st.moe_out, 0, s.n_expert_latent as usize * 4)?;
        }
        if matches!(routed.up, MatW::Kq(_)) {
            kernels::quantize_q8_k(&mut sc.xq, &rt.lat_out, s.n_expert_latent, 1)?;
        }
        matw(&mut rt.routed_out, &routed.up, &rt.lat_out, &sc.xq, s.n_expert_latent, s.n_embd, 1)?;
        kernels::add(&mut st.ffn_out, &rt.routed_out, &st.shared_out, s.n_embd)?;
        Ok(())
    }
}
