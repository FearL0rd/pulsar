//! Check that every tensor the K3 loader asks for is present, with the
//! dims the config implies, without uploading anything:
//!   cargo run --release -p engine --example k3-tensors -- <model-00001-of-N.gguf>
//! Catches a wrong tensor name or a mis-parsed dim before a load that
//! would otherwise spend minutes moving weights first.
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("k3-tensors needs the linux engine build");
}

#[cfg(target_os = "linux")]
fn main() {
    let path = std::env::args().nth(1).expect("usage: k3-tensors <model.gguf>");
    let (shards, g) = engine::parse_header(std::path::Path::new(&path)).expect("parse gguf");
    let s = engine::Shape::from_gguf(&g).expect("shape");
    println!("shards: {}, tensors: {}", shards.len(), g.tensors.len());

    let mut missing: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut n_kda = 0usize;
    let mut n_mla = 0usize;

    // want(name, dims): dims are checked only where a mismatch would mean
    // the config parse is wrong, not merely that the file differs.
    let mut want = |name: String, dims: &[u64]| {
        match g.tensor(&name) {
            None => missing.push(name),
            Some(t) => {
                if !dims.is_empty() && t.dims.len() >= dims.len() && t.dims[..dims.len()] != *dims {
                    wrong.push(format!("{name}: want {dims:?} got {:?}", t.dims));
                }
            }
        }
    };

    let d_inner = (s.n_head * s.kda_head_dim) as u64;
    let n_embd = s.n_embd as u64;
    let latent = s.n_expert_latent as u64;

    for il in 0..s.n_exec_layer {
        let t = |n: &str| format!("blk.{il}.{n}");
        want(t("attn_norm.weight"), &[n_embd]);
        want(t("ffn_norm.weight"), &[n_embd]);
        if s.attn_res_block > 0 {
            want(t("attn_res_score.weight"), &[n_embd]);
            want(t("ffn_res_score.weight"), &[n_embd]);
        }
        let is_kda = g.tensor(&t("ssm_a")).is_some();
        if is_kda {
            n_kda += 1;
            want(t("attn_q.weight"), &[n_embd, d_inner]);
            want(t("attn_k.weight"), &[n_embd, d_inner]);
            want(t("attn_v.weight"), &[n_embd, d_inner]);
            for c in ["q", "k", "v"] {
                want(t(&format!("ssm_conv1d_{c}.weight")), &[s.ssm_conv_k as u64]);
            }
            want(t("ssm_f_a.weight"), &[n_embd, s.kda_head_dim as u64]);
            want(t("ssm_f_b.weight"), &[s.kda_head_dim as u64, d_inner]);
            want(t("ssm_beta.weight"), &[n_embd, s.n_head as u64]);
            want(t("ssm_a"), &[s.n_head as u64]);
            want(t("ssm_dt.bias"), &[d_inner]);
            want(t("ssm_g.weight"), &[n_embd, d_inner]);
            want(t("ssm_norm.weight"), &[s.kda_head_dim as u64]);
        } else {
            n_mla += 1;
            want(t("attn_q_a.weight"), &[n_embd, s.n_lora_q as u64]);
            want(t("attn_q_a_norm.weight"), &[s.n_lora_q as u64]);
            want(
                t("attn_q_b.weight"),
                &[s.n_lora_q as u64, (s.n_head * s.qk_dim()) as u64],
            );
            want(
                t("attn_kv_a_mqa.weight"),
                &[n_embd, (s.n_kv_lora + s.qk_rope) as u64],
            );
            want(t("attn_kv_a_norm.weight"), &[s.n_kv_lora as u64]);
            want(
                t("attn_k_b.weight"),
                &[s.qk_nope as u64, s.n_kv_lora as u64, s.n_head as u64],
            );
            want(
                t("attn_v_b.weight"),
                &[s.n_kv_lora as u64, s.value_mla as u64, s.n_head as u64],
            );
            want(t("attn_gate.weight"), &[n_embd, (s.n_head * s.value_mla) as u64]);
        }
        want(t("attn_output.weight"), &[(s.n_head * s.value_mla) as u64, n_embd]);

        if il < s.n_leading_dense {
            want(t("ffn_gate.weight"), &[n_embd, s.n_ff_dense as u64]);
            want(t("ffn_up.weight"), &[n_embd, s.n_ff_dense as u64]);
            want(t("ffn_down.weight"), &[s.n_ff_dense as u64, n_embd]);
        } else {
            want(t("ffn_gate_inp.weight"), &[n_embd, s.n_expert as u64]);
            want(t("exp_probs_b.bias"), &[s.n_expert as u64]);
            want(t("ffn_routed_down.weight"), &[n_embd, latent]);
            want(t("ffn_routed_up.weight"), &[latent, n_embd]);
            want(t("ffn_routed_norm.weight"), &[latent]);
            want(
                t("ffn_gate_exps.weight"),
                &[latent, s.n_ff_exp as u64, s.n_expert as u64],
            );
            want(
                t("ffn_up_exps.weight"),
                &[latent, s.n_ff_exp as u64, s.n_expert as u64],
            );
            want(
                t("ffn_down_exps.weight"),
                &[s.n_ff_exp as u64, latent, s.n_expert as u64],
            );
            for k in ["gate", "up", "down"] {
                want(t(&format!("ffn_{k}_shexp.weight")), &[]);
            }
        }
    }
    want("token_embd.weight".into(), &[n_embd, s.n_vocab as u64]);
    want("output_norm.weight".into(), &[n_embd]);
    want("output.weight".into(), &[n_embd, s.n_vocab as u64]);
    if s.attn_res_block > 0 {
        want("output_res_score.weight".into(), &[n_embd]);
    }

    println!("layers: {n_kda} KDA + {n_mla} MLA = {}", n_kda + n_mla);
    println!("missing: {}", missing.len());
    for m in missing.iter().take(20) {
        println!("  MISSING {m}");
    }
    println!("dim mismatches: {}", wrong.len());
    for w in wrong.iter().take(20) {
        println!("  WRONG {w}");
    }
    if missing.is_empty() && wrong.is_empty() {
        println!("OK: every tensor the K3 loader wants is present with the expected dims");
    } else {
        std::process::exit(1);
    }
}
