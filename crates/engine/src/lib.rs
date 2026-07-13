//! hy-v3 (Hy3) forward graph over the pulsar CUDA kernels.
//!
//! Op sequence is ds4's `hy3_forward_token`, the decode-parity reference:
//! embed -> per layer [rms-norm, qkv (q8_0), per-head q/k norm, neox rope,
//! kv append, gqa attention, out-proj, residual; rms-norm, dense FFN (layer
//! 0) or sigmoid-router MoE (shared expert + streamed routed experts)] ->
//! final norm -> lm head.
//!
//! Expert streaming here is v1: blocking pread of the selected experts'
//! slabs each layer, uploaded into a fixed device arena and handed to the
//! MoE kernels as explicit per-slot pointers. The StreamingStore (cache +
//! prefetch) replaces the fetch path later without touching the graph.

#[cfg(target_os = "linux")]
mod real {
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    use gguf::{Gguf, TensorInfo, TensorType, Value};
    use kernels::{DeviceBuf, ExpertPtrs};

    pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn meta_err(key: &str) -> Box<dyn std::error::Error> {
        format!("gguf metadata missing/bad: {key}").into()
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Shape {
        pub n_embd: u32,
        pub n_head: u32,
        pub n_head_kv: u32,
        pub head_dim: u32,
        pub n_layer: u32,
        pub n_exec_layer: u32,
        pub n_leading_dense: u32,
        pub n_expert: u32,
        pub n_expert_used: u32,
        pub n_ff_exp: u32,
        pub n_ff_dense: u32,
        pub n_vocab: u32,
        pub expert_weight_scale: f32,
        pub rope_freq_base: f32,
        pub rms_eps: f32,
    }

    impl Shape {
        fn from_gguf(g: &Gguf) -> Result<Shape> {
            let u = |k: &str| -> Result<u32> {
                Ok(g.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| meta_err(k))? as u32)
            };
            let f = |k: &str| -> Result<f32> {
                g.arch_meta(k).and_then(Value::as_f32).ok_or_else(|| meta_err(k))
            };
            let n_layer = u("block_count")?;
            let nextn = u("nextn_predict_layers").unwrap_or(0);
            let n_vocab = match g.metadata.get("tokenizer.ggml.tokens") {
                Some(Value::Array(a)) => a.len() as u32,
                _ => return Err(meta_err("tokenizer.ggml.tokens")),
            };
            Ok(Shape {
                n_embd: u("embedding_length")?,
                n_head: u("attention.head_count")?,
                n_head_kv: u("attention.head_count_kv")?,
                head_dim: u("attention.key_length")?,
                n_layer,
                n_exec_layer: n_layer - nextn,
                n_leading_dense: u("leading_dense_block_count")?,
                n_expert: u("expert_count")?,
                n_expert_used: u("expert_used_count")?,
                n_ff_exp: u("expert_feed_forward_length")?,
                n_ff_dense: u("feed_forward_length")?,
                n_vocab,
                expert_weight_scale: f("expert_weights_scale")?,
                rope_freq_base: f("rope.freq_base")?,
                rms_eps: f("attention.layer_norm_rms_epsilon")?,
            })
        }
    }

    /// File location of one routed expert tensor: uniform per-expert slabs.
    struct ExpertTensor {
        abs_offset: u64,
        expert_bytes: u64,
        row_bytes: u64,
        quant: u32,
    }

    impl ExpertTensor {
        fn new(g: &Gguf, t: &TensorInfo, n_expert: u32) -> Result<ExpertTensor> {
            let quant = match t.ty {
                TensorType::IQ2XXS => kernels::QUANT_IQ2_XXS,
                TensorType::Q2K => kernels::QUANT_Q2_K,
                other => return Err(format!("{}: unsupported expert type {other:?}", t.name).into()),
            };
            let row_elems = t.dims[0];
            let rows_per_expert = t.dims[1];
            let row_bytes = t.ty.row_bytes(row_elems).unwrap();
            Ok(ExpertTensor {
                abs_offset: g.data_offset + t.offset,
                expert_bytes: row_bytes * rows_per_expert,
                row_bytes,
                quant: {
                    debug_assert_eq!(t.dims[2], n_expert as u64);
                    quant
                },
            })
        }
    }

    enum Ffn {
        Dense {
            gate: DeviceBuf,
            up: DeviceBuf,
            down: DeviceBuf,
        },
        Moe {
            gate_inp: DeviceBuf,
            probs_b: DeviceBuf,
            shexp_gate: DeviceBuf,
            shexp_up: DeviceBuf,
            shexp_down: DeviceBuf,
            gate_exps: ExpertTensor,
            up_exps: ExpertTensor,
            down_exps: ExpertTensor,
        },
    }

    struct LayerW {
        attn_norm: DeviceBuf,
        attn_q: DeviceBuf,
        attn_k: DeviceBuf,
        attn_v: DeviceBuf,
        attn_output: DeviceBuf,
        q_norm: DeviceBuf,
        k_norm: DeviceBuf,
        ffn_norm: DeviceBuf,
        ffn: Ffn,
    }

    pub struct Model {
        path: std::path::PathBuf,
        pub shape: Shape,
        pub gguf: Gguf,
        token_embd: DeviceBuf,
        output_norm: DeviceBuf,
        output: DeviceBuf,
        layers: Vec<LayerW>,
    }

    /// v1 StreamingStore (DESIGN-expert-store.md): io_uring batch fetch of
    /// cache misses + LFU host cache of expert slabs, keyed by absolute
    /// file offset (unique per layer/tensor/expert).
    pub struct StreamingStore {
        fetcher: stream::fetch::Fetcher,
        cache: std::collections::HashMap<u64, CacheEntry>,
        used: usize,
        budget: usize,
        tick: u64,
        pub hits: u64,
        pub misses: u64,
    }

    struct CacheEntry {
        slab: stream::fetch::Slab,
        freq: u64,
        tick: u64,
    }

    impl StreamingStore {
        fn open(path: &Path, budget: usize) -> Result<StreamingStore> {
            Ok(StreamingStore {
                fetcher: stream::fetch::Fetcher::open(path, 32)?,
                cache: std::collections::HashMap::new(),
                used: 0,
                budget,
                tick: 0,
                hits: 0,
                misses: 0,
            })
        }

        /// Make every read resident in the cache and bump its LFU state.
        fn ensure(&mut self, wants: &[stream::Read]) -> Result {
            self.tick += 1;
            let missing: Vec<stream::Read> = wants
                .iter()
                .filter(|r| !self.cache.contains_key(&r.offset))
                .copied()
                .collect();
            self.hits += (wants.len() - missing.len()) as u64;
            self.misses += missing.len() as u64;
            if !missing.is_empty() {
                let slabs = self.fetcher.fetch(&missing)?;
                let incoming: usize = slabs.iter().map(|s| s.bytes()).sum();
                // evict lowest (freq, tick) not wanted right now
                // ponytail: O(n) scan per eviction; heap it if the cache
                // ever holds >100k entries
                while self.used + incoming > self.budget && !self.cache.is_empty() {
                    let victim = self
                        .cache
                        .iter()
                        .filter(|(k, _)| !wants.iter().any(|w| w.offset == **k))
                        .min_by_key(|(_, e)| (e.freq, e.tick))
                        .map(|(k, _)| *k);
                    let Some(k) = victim else { break };
                    if let Some(e) = self.cache.remove(&k) {
                        self.used -= e.slab.bytes();
                    }
                }
                for (r, slab) in missing.iter().zip(slabs) {
                    self.used += slab.bytes();
                    self.cache.insert(r.offset, CacheEntry { slab, freq: 0, tick: self.tick });
                }
            }
            for r in wants {
                if let Some(e) = self.cache.get_mut(&r.offset) {
                    e.freq += 1;
                    e.tick = self.tick;
                }
            }
            Ok(())
        }

        fn payload(&self, offset: u64) -> &[u8] {
            self.cache[&offset].slab.payload()
        }
    }

    /// How many header bytes to read before parsing; grows on Truncated.
    const HEAD_READ_START: usize = 32 << 20;

    pub fn parse_header(path: &Path) -> Result<(File, Gguf)> {
        let file = File::open(path)?;
        let mut n = HEAD_READ_START;
        loop {
            let mut head = vec![0u8; n];
            let got = file.read_at(&mut head, 0)?;
            head.truncate(got);
            match Gguf::parse(&head) {
                Ok(g) => return Ok((file, g)),
                Err(gguf::Error::Truncated { .. }) if got == n => n *= 2,
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn read_tensor_bytes(file: &File, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(buf)
    }

    fn upload(file: &File, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_bytes(&read_tensor_bytes(file, g, name)?)?)
    }

    impl Model {
        pub fn load(path: &Path) -> Result<Model> {
            let (file, gguf) = parse_header(path)?;
            let _ = &file;
            if gguf.architecture() != Some("hy-v3") {
                return Err(format!("not a hy-v3 gguf: {:?}", gguf.architecture()).into());
            }
            let shape = Shape::from_gguf(&gguf)?;

            let token_embd = upload(&file, &gguf, "token_embd.weight")?;
            let output_norm = upload(&file, &gguf, "output_norm.weight")?;
            let output = upload(&file, &gguf, "output.weight")?;

            let mut layers = Vec::with_capacity(shape.n_exec_layer as usize);
            for il in 0..shape.n_exec_layer {
                let t = |suffix: &str| format!("blk.{il}.{suffix}");
                let ffn = if il < shape.n_leading_dense {
                    Ffn::Dense {
                        gate: upload(&file, &gguf, &t("ffn_gate.weight"))?,
                        up: upload(&file, &gguf, &t("ffn_up.weight"))?,
                        down: upload(&file, &gguf, &t("ffn_down.weight"))?,
                    }
                } else {
                    let exps = |suffix: &str| -> Result<ExpertTensor> {
                        let name = t(suffix);
                        let ti = gguf.tensor(&name).ok_or_else(|| meta_err(&name))?;
                        ExpertTensor::new(&gguf, ti, shape.n_expert)
                    };
                    Ffn::Moe {
                        gate_inp: upload(&file, &gguf, &t("ffn_gate_inp.weight"))?,
                        probs_b: upload(&file, &gguf, &t("exp_probs_b"))?,
                        shexp_gate: upload(&file, &gguf, &t("ffn_gate_shexp.weight"))?,
                        shexp_up: upload(&file, &gguf, &t("ffn_up_shexp.weight"))?,
                        shexp_down: upload(&file, &gguf, &t("ffn_down_shexp.weight"))?,
                        gate_exps: exps("ffn_gate_exps.weight")?,
                        up_exps: exps("ffn_up_exps.weight")?,
                        down_exps: exps("ffn_down_exps.weight")?,
                    }
                };
                layers.push(LayerW {
                    attn_norm: upload(&file, &gguf, &t("attn_norm.weight"))?,
                    attn_q: upload(&file, &gguf, &t("attn_q.weight"))?,
                    attn_k: upload(&file, &gguf, &t("attn_k.weight"))?,
                    attn_v: upload(&file, &gguf, &t("attn_v.weight"))?,
                    attn_output: upload(&file, &gguf, &t("attn_output.weight"))?,
                    q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                    k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                    ffn_norm: upload(&file, &gguf, &t("ffn_norm.weight"))?,
                    ffn,
                });
            }
            Ok(Model {
                path: path.to_path_buf(),
                shape,
                gguf,
                token_embd,
                output_norm,
                output,
                layers,
            })
        }
    }

    /// Per-decode device state: activation buffers, KV caches, the routed
    /// expert staging arena, and reusable host staging.
    pub struct State {
        ctx: u32,
        tok: DeviceBuf,
        cur: DeviceBuf,
        normed: DeviceBuf,
        q: DeviceBuf,
        k: DeviceBuf,
        v: DeviceBuf,
        heads: DeviceBuf,
        attn_out: DeviceBuf,
        after_attn: DeviceBuf,
        gate_act: DeviceBuf,
        up_act: DeviceBuf,
        ffn_mid: DeviceBuf,
        ffn_out: DeviceBuf,
        shared_out: DeviceBuf,
        router_logits: DeviceBuf,
        router_selected: DeviceBuf,
        router_weights: DeviceBuf,
        moe_mid: DeviceBuf,
        moe_out: DeviceBuf,
        xq: DeviceBuf,
        midq: DeviceBuf,
        expert_arena: DeviceBuf,
        expert_ptrs: DeviceBuf,
        kcache: Vec<DeviceBuf>,
        vcache: Vec<DeviceBuf>,
        logits: DeviceBuf,
        pub store: StreamingStore,
    }

    impl State {
        pub fn new(m: &Model, ctx: u32) -> Result<State> {
            let gb = std::env::var("PULSAR_CACHE_GB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(12);
            Self::with_cache(m, ctx, gb << 30)
        }

        pub fn with_cache(m: &Model, ctx: u32, cache_bytes: usize) -> Result<State> {
            let s = m.shape;
            let f32s = |n: u32| DeviceBuf::alloc(n as usize * 4);
            let n_used = s.n_expert_used as usize;
            // uniform slab size across gate/up/down on this model; assert at fetch
            let max_slab = m
                .layers
                .iter()
                .filter_map(|l| match &l.ffn {
                    Ffn::Moe { gate_exps, up_exps, down_exps, .. } => {
                        Some(gate_exps.expert_bytes.max(up_exps.expert_bytes).max(down_exps.expert_bytes))
                    }
                    _ => None,
                })
                .max()
                .unwrap_or(0) as usize;

            let kv_bytes = s.n_head_kv as usize * ctx as usize * s.head_dim as usize * 4;
            let mut kcache = Vec::new();
            let mut vcache = Vec::new();
            for _ in 0..s.n_exec_layer {
                kcache.push(DeviceBuf::alloc(kv_bytes)?);
                vcache.push(DeviceBuf::alloc(kv_bytes)?);
            }

            Ok(State {
                ctx,
                tok: DeviceBuf::alloc(4)?,
                cur: f32s(s.n_embd)?,
                normed: f32s(s.n_embd)?,
                q: f32s(s.n_head * s.head_dim)?,
                k: f32s(s.n_head_kv * s.head_dim)?,
                v: f32s(s.n_head_kv * s.head_dim)?,
                heads: f32s(s.n_head * s.head_dim)?,
                attn_out: f32s(s.n_embd)?,
                after_attn: f32s(s.n_embd)?,
                gate_act: f32s(s.n_ff_dense.max(s.n_ff_exp))?,
                up_act: f32s(s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_mid: f32s(s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_out: f32s(s.n_embd)?,
                shared_out: f32s(s.n_embd)?,
                router_logits: f32s(s.n_expert)?,
                router_selected: DeviceBuf::alloc(n_used * 4)?,
                router_weights: f32s(s.n_expert_used)?,
                moe_mid: f32s(s.n_expert_used * s.n_ff_exp)?,
                moe_out: f32s(s.n_embd)?,
                xq: DeviceBuf::alloc(
                    s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS * kernels::Q8_K_BLOCK_BYTES,
                )?,
                midq: DeviceBuf::alloc(
                    n_used * s.n_ff_exp as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                expert_arena: DeviceBuf::alloc(n_used * 3 * max_slab)?,
                expert_ptrs: DeviceBuf::alloc(n_used * std::mem::size_of::<ExpertPtrs>())?,
                kcache,
                vcache,
                logits: f32s(s.n_vocab)?,
                store: StreamingStore::open(&m.path, cache_bytes)?,
            })
        }
    }

    impl Model {
        /// One full forward for one token at absolute position `pos`.
        /// Returns host logits when `want_logits`.
        pub fn forward_token(
            &self,
            st: &mut State,
            token: u32,
            pos: u32,
            want_logits: bool,
        ) -> Result<Option<Vec<f32>>> {
            let s = self.shape;
            if pos >= st.ctx {
                return Err("position exceeds context".into());
            }
            let eps = s.rms_eps;
            st.tok.write(0, kernels::as_bytes(&[token as i32]))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, 1)?;

            for (il, l) in self.layers.iter().enumerate() {
                // attention
                kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, 1, eps)?;
                kernels::matmul_q8_0(&mut st.q, &l.attn_q, &st.normed, s.n_embd, s.n_head * s.head_dim, 1)?;
                kernels::matmul_q8_0(&mut st.k, &l.attn_k, &st.normed, s.n_embd, s.n_head_kv * s.head_dim, 1)?;
                kernels::matmul_q8_0(&mut st.v, &l.attn_v, &st.normed, s.n_embd, s.n_head_kv * s.head_dim, 1)?;
                kernels::gqa_head_rms_norm(&mut st.q, &l.q_norm, s.n_head, s.head_dim, eps)?;
                kernels::gqa_head_rms_norm(&mut st.k, &l.k_norm, s.n_head_kv, s.head_dim, eps)?;
                kernels::gqa_rope(&mut st.q, 1, s.n_head, s.head_dim, pos, s.rope_freq_base)?;
                kernels::gqa_rope(&mut st.k, 1, s.n_head_kv, s.head_dim, pos, s.rope_freq_base)?;
                kernels::gqa_kv_append(&mut st.kcache[il], &st.k, 1, s.n_head_kv, s.head_dim, st.ctx, pos)?;
                kernels::gqa_kv_append(&mut st.vcache[il], &st.v, 1, s.n_head_kv, s.head_dim, st.ctx, pos)?;
                kernels::gqa_attention(&mut st.heads, &st.q, &st.kcache[il], &st.vcache[il], 1, s.n_head, s.n_head_kv, s.head_dim, st.ctx, pos)?;
                kernels::matmul_q8_0(&mut st.attn_out, &l.attn_output, &st.heads, s.n_head * s.head_dim, s.n_embd, 1)?;
                kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, s.n_embd)?;

                // ffn
                kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, 1, eps)?;
                match &l.ffn {
                    Ffn::Dense { gate, up, down } => {
                        kernels::matmul_q8_0(&mut st.gate_act, gate, &st.normed, s.n_embd, s.n_ff_dense, 1)?;
                        kernels::matmul_q8_0(&mut st.up_act, up, &st.normed, s.n_embd, s.n_ff_dense, 1)?;
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, s.n_ff_dense, 0.0, 1.0)?;
                        kernels::matmul_q8_0(&mut st.ffn_out, down, &st.ffn_mid, s.n_ff_dense, s.n_embd, 1)?;
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, s.n_embd)?;
                    }
                    Ffn::Moe { gate_inp, probs_b, shexp_gate, shexp_up, shexp_down, gate_exps, up_exps, down_exps } => {
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
                        )?;

                        // StreamingStore: LFU host cache + one io_uring
                        // batch for the misses, then upload to the arena.
                        kernels::sync()?;
                        let selected = st.router_selected.read_i32(s.n_expert_used as usize)?;
                        let slab = gate_exps.expert_bytes as usize;
                        debug_assert_eq!(up_exps.expert_bytes, gate_exps.expert_bytes);
                        debug_assert_eq!(down_exps.expert_bytes, gate_exps.expert_bytes);
                        let mut wants = Vec::with_capacity(3 * s.n_expert_used as usize);
                        for &e in &selected {
                            if e < 0 || e as u32 >= s.n_expert {
                                continue;
                            }
                            for t in [gate_exps, up_exps, down_exps] {
                                wants.push(stream::Read {
                                    offset: t.abs_offset + e as u64 * t.expert_bytes,
                                    len: t.expert_bytes,
                                });
                            }
                        }
                        st.store.ensure(&wants)?;
                        let mut ptrs = Vec::with_capacity(s.n_expert_used as usize);
                        for (slot, &e) in selected.iter().enumerate() {
                            if e < 0 || e as u32 >= s.n_expert {
                                ptrs.push(ExpertPtrs::NULL);
                                continue;
                            }
                            let base = slot * 3 * slab;
                            for (j, t) in [gate_exps, up_exps, down_exps].iter().enumerate() {
                                let off = t.abs_offset + e as u64 * t.expert_bytes;
                                st.expert_arena.write(base + j * slab, st.store.payload(off))?;
                            }
                            ptrs.push(ExpertPtrs {
                                gate: st.expert_arena.ptr_at(base),
                                up: st.expert_arena.ptr_at(base + slab),
                                down: st.expert_arena.ptr_at(base + 2 * slab),
                            });
                        }
                        st.expert_ptrs.write(0, kernels::as_bytes(&ptrs))?;

                        // shared expert
                        kernels::matmul_q8_0(&mut st.gate_act, shexp_gate, &st.normed, s.n_embd, s.n_ff_exp, 1)?;
                        kernels::matmul_q8_0(&mut st.up_act, shexp_up, &st.normed, s.n_embd, s.n_ff_exp, 1)?;
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, s.n_ff_exp, 0.0, 1.0)?;
                        kernels::matmul_q8_0(&mut st.shared_out, shexp_down, &st.ffn_mid, s.n_ff_exp, s.n_embd, 1)?;

                        // routed experts: activations quantized to q8_K,
                        // integer dp4a dots (ds4's exact math)
                        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, 1)?;
                        kernels::moe_pair_swiglu(
                            &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
                            s.n_embd, s.n_ff_exp, s.n_expert_used, 1, gate_exps.row_bytes, gate_exps.quant,
                        )?;
                        kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, s.n_expert_used)?;
                        kernels::moe_down(
                            &mut st.moe_out, &st.expert_ptrs, &st.midq,
                            s.n_ff_exp, s.n_embd, s.n_expert_used, 1, down_exps.row_bytes, down_exps.quant,
                        )?;

                        // cur = after_attn + routed + shared (ds4's add3)
                        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, s.n_embd)?;
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, s.n_embd)?;
                    }
                }
            }

            if !want_logits {
                return Ok(None);
            }
            kernels::rms_norm(&mut st.normed, &st.cur, &self.output_norm, s.n_embd, 1, eps)?;
            kernels::matmul_q8_0(&mut st.logits, &self.output, &st.normed, s.n_embd, s.n_vocab, 1)?;
            kernels::sync()?;
            Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
        }
    }

    /// First-max argmax, matching ds4's sample_argmax.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }
}

#[cfg(target_os = "linux")]
pub use real::*;
