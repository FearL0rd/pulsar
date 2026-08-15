//! pulsar-cli: Hy3 generation and diagnostics.
//!
//!   pulsar-cli -m model.gguf -p "text" -n 32 [--ctx 2048] [--no-bos]
//!   pulsar-cli -m model.gguf --chat [--system "..."] [--temp 0.9]
//!   pulsar-cli -m model.gguf --chat --jinja-chat [--system "..."]
//!   pulsar-cli -m model.gguf --tokens 120000,16883,11 -n 32
//!
//! -p tokenizes raw text (BOS prepended unless --no-bos); --tokens feeds
//! exact ids, which is how A/B runs align with ds4 --dump-tokens output.
//! --chat is an interactive multi-turn loop with the KV cache retained
//! across turns; sampling defaults come from the gguf's
//! general.sampling.* metadata unless --temp/--top-p are given.
//! --jinja-chat (or PULSAR_JINJA_CHAT=1) opts into Jinja encoding for
//! --chat: resolve embed → cache → HF → llama.cpp catalog (network
//! blocked only by PULSAR_OFFLINE).

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("pulsar-cli requires Linux + CUDA");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
const HELP: &str = "\
pulsar-cli: generation and diagnostics for the pulsar engine.

usage: pulsar-cli -m MODEL.gguf [-p TEXT | -f FILE | --tokens IDS] [options]

  -m PATH              model gguf (first shard of a split set)
  -p TEXT              prompt text (BOS prepended unless --no-bos)
  -f, --prompt-file P  read the prompt from a file (long prompts)
  --tokens A,B,C       feed exact token ids instead of text
  -n N                 tokens to generate (default 16)
  --ctx N              context length (default 2048)
  --bos / --no-bos     force BOS on/off (default: the gguf's add_bos)

  --chat               interactive multi-turn loop, KV retained per turn
  --system TEXT        system prompt for --chat
  --jinja-chat         encode with the Jinja chat template instead of the
                       built-in markers (network blocked by PULSAR_OFFLINE)
  --temp F, --top-p F, --min-p F, --seed N
                       sampling (defaults from general.sampling.* metadata)

diagnostics:
  --teacher-force      per-position top-5 along the given tokens; with
                       --dump-logits writes full logit rows instead
  --dump-logits PATH   write logits (see scripts/kld-ab.sh)
  --decode-consistency N
                       compare incremental decode against a fresh prefill
  --rows-consistency N
                       check multi-row (speculative verify) logits against
                       single-token steps at the same positions
  --dspark-capture A,B,C
                       check the DSpark hidden-state ring for those
                       target_layers (dsv4 only)

environment:
  PULSAR_KV=f32|int8|fp8|fp16|q8_0|q4_0|turbo8|turbo4
                       KV cache format (default: f32, auto-quantizes when
                       a big context would starve the expert cache)
  PULSAR_MTP=1         enable the gguf's nextn head, when it has one
  PULSAR_DFLASH=PATH   dflash/dspark draft gguf for speculative decode
  PULSAR_PROFILE=1     per-stage timing report
  PULSAR_OFFLINE=1     never touch the network

  -V, --version        print version and git sha
  -h, --help           this text
";

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-cli: {e}");
        std::process::exit(1);
    }
}

/// Flush the longest valid UTF-8 prefix of `buf` to stdout, keeping any
/// incomplete trailing multi-byte sequence for the next token.
#[cfg(target_os = "linux")]
#[allow(dead_code)] // used by the streaming chat path; dead in --chat-less builds
fn print_utf8_prefix(buf: &mut Vec<u8>) {
    use std::io::Write;
    let valid_len = match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid_len > 0 {
        let out = std::io::stdout();
        let mut lock = out.lock();
        lock.write_all(&buf[..valid_len]).ok();
        lock.flush().ok();
        buf.drain(..valid_len);
    }
}

/// Encode a multi-turn history through a resolved Jinja template.
#[cfg(target_os = "linux")]
fn encode_chat_jinja(
    tok: &tokenizer::Tokenizer,
    template: &tokenizer::ResolvedChatTemplate,
    messages: &[tokenizer::ChatMessage],
) -> Result<Vec<u32>, String> {
    let bos = tok.bos_id.and_then(|id| tok.token_str(id));
    let eos = tok.eos_id.and_then(|id| tok.token_str(id));
    let rendered = tokenizer::apply_chat_template(
        &template.template,
        messages,
        true,
        bos,
        eos,
        None,
    )
    .map_err(|e| e.to_string())?;
    if std::env::var_os("PULSAR_DEBUG_CHAT").is_some() {
        eprintln!("pulsar chat: jinja prompt:\n{rendered}");
    }
    Ok(tok.encode_with_specials(&rendered))
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_chat(
    model: &engine::Model,
    tok: &tokenizer::Tokenizer,
    ctx: u32,
    system: Option<String>,
    temp: Option<f32>,
    top_p: Option<f32>,
    min_p: f32,
    seed: u64,
    max_tokens: usize,
    chat_template: Option<&tokenizer::ResolvedChatTemplate>,
) -> engine::Result {
    use std::io::BufRead;

    let markers = match tokenizer::ChatMarkers::resolve(tok) {
        Ok(m) => m,
        Err(e) => {
            if chat_template.is_some() {
                eprintln!(
                    "pulsar chat: ChatMarkers unresolved ({e}); Jinja encoding with fallback markers"
                );
                tokenizer::ChatMarkers::jinja_fallback(tok)?
            } else {
                return Err(format!(
                    "ChatMarkers unresolved ({e}); pass --jinja-chat if this model needs its HF/GGUF template"
                )
                .into());
            }
        }
    };
    // sampling defaults from the gguf's own metadata (Hy3 ships 0.9/1.0)
    let meta_f = |k: &str, d: f32| {
        model.gguf.metadata.get(k).and_then(gguf::Value::as_f32).unwrap_or(d)
    };
    let temp = temp.unwrap_or_else(|| meta_f("general.sampling.temp", 0.9));
    let top_p = top_p.unwrap_or_else(|| meta_f("general.sampling.top_p", 1.0));
    let mut sampler = engine::Sampler::new(temp, top_p, min_p, seed);

    let mut st = engine::State::new(model, ctx)?;
    let max_tokens = if max_tokens <= 16 { 1024 } else { max_tokens };
    let jinja = chat_template.is_some();
    eprintln!(
        "pulsar chat: temp {temp} top-p {top_p} seed {seed}; ctx {ctx}; {}encoding; empty line or Ctrl-D exits",
        if jinja { "Jinja " } else { "ChatMarkers " }
    );

    let stdin = std::io::stdin();
    let mut pos = 0u32;
    let mut first = true;
    // Full message history for Jinja re-render each turn (correct multi-turn).
    let mut history: Vec<tokenizer::ChatMessage> = Vec::new();
    if let Some(sys) = system.as_ref() {
        if jinja {
            history.push(tokenizer::ChatMessage {
                role: "system".into(),
                content: sys.clone(),
            });
        }
    } else if jinja {
        if let Some(dflt) = markers.default_system() {
            history.push(tokenizer::ChatMessage {
                role: "system".into(),
                content: dflt,
            });
        }
    }

    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }

        let ids = if let Some(tmpl) = chat_template {
            history.push(tokenizer::ChatMessage {
                role: "user".into(),
                content: line.to_string(),
            });
            match encode_chat_jinja(tok, tmpl, &history) {
                Ok(ids) => {
                    // Full re-prefill each turn so assistant history is
                    // template-faithful (token stream may not match a pure
                    // incremental append of the prior generation).
                    pos = 0;
                    ids
                }
                Err(e) => {
                    eprintln!(
                        "pulsar chat: jinja apply failed ({e}); falling back to ChatMarkers for this turn"
                    );
                    history.pop(); // drop the user we just pushed
                    let mut ids = Vec::new();
                    if first {
                        ids.extend(markers.prologue());
                        ids.extend(markers.prologue_effort(tok));
                        let dflt = markers.default_system();
                        if let Some(sys) = system.as_deref().or(dflt.as_deref()) {
                            ids.extend(markers.render_system(tok, sys));
                        }
                        first = false;
                    }
                    ids.extend(markers.render_user_turn(tok, line));
                    ids
                }
            }
        } else {
            let mut ids = Vec::new();
            if first {
                ids.extend(markers.prologue());
                ids.extend(markers.prologue_effort(tok));
                let dflt = markers.default_system();
                if let Some(sys) = system.as_deref().or(dflt.as_deref()) {
                    ids.extend(markers.render_system(tok, sys));
                }
                first = false;
            }
            ids.extend(markers.render_user_turn(tok, line));
            ids
        };
        if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
            eprintln!("pulsar chat: turn ids {ids:?}");
        }

        if pos + ids.len() as u32 + 2 >= ctx {
            eprintln!("pulsar chat: context full ({pos}/{ctx}), restart to continue");
            break;
        }

        let mut bytes = Vec::new();
        let mut reply = String::new();
        pos = engine::generate(
            model,
            &mut st,
            &ids,
            pos,
            &mut sampler,
            max_tokens,
            |id| {
                let stop = markers.is_stop(id);
                if stop && std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
                    eprintln!("pulsar chat: stop token {id} (eos {}, eot {:?})", markers.eos, markers.eot);
                }
                stop
            },
            |id| {
                if std::env::var_os("PULSAR_DEBUG_IDS").is_some() {
                    eprint!("[{id}]");
                }
                bytes.extend_from_slice(&tok.decode(&[id]));
                // Mirror print_utf8_prefix into `reply` for Jinja history.
                let valid_len = match std::str::from_utf8(&bytes) {
                    Ok(s) => {
                        if chat_template.is_some() {
                            reply.push_str(s);
                        }
                        bytes.len()
                    }
                    Err(e) => {
                        let n = e.valid_up_to();
                        if n > 0
                            && chat_template.is_some() {
                                reply.push_str(std::str::from_utf8(&bytes[..n]).unwrap_or(""));
                            }
                        n
                    }
                };
                if valid_len > 0 {
                    use std::io::Write;
                    let out = std::io::stdout();
                    let mut lock = out.lock();
                    lock.write_all(&bytes[..valid_len]).ok();
                    lock.flush().ok();
                    bytes.drain(..valid_len);
                }
            },
        )?;
        // Flush any remaining incomplete multi-byte sequence as lossy text.
        if !bytes.is_empty() {
            let tail = String::from_utf8_lossy(&bytes);
            if chat_template.is_some() {
                reply.push_str(&tail);
            }
            print!("{tail}");
            bytes.clear();
        }
        println!();
        if chat_template.is_some() {
            history.push(tokenizer::ChatMessage {
                role: "assistant".into(),
                content: reply,
            });
        }
    }
    st.save_warm(model)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    let mut model_path = None;
    let mut prompt = None;
    let mut tokens_arg = None;
    let mut n_predict = 16usize;
    let mut ctx = 2048u32;
    let mut bos: Option<bool> = None; // None = model default (add_bos KV)
    let mut dump_logits = None;
    let mut teacher_force = false;
    let mut decode_consistency = None;
    let mut rows_consistency = None;
    let mut dspark_capture: Option<String> = None;
    let mut chat = false;
    let mut system = None;
    let mut temp = None;
    let mut top_p = None;
    let mut min_p = 0.0f32;
    let mut seed = 42u64;
    // Jinja encoding is opt-in only (same policy as pulsar-serve).
    let mut jinja_chat = match std::env::var("PULSAR_JINJA_CHAT") {
        Ok(v) if v == "0"
            || v.is_empty()
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off") =>
        {
            false
        }
        Ok(_) => true,
        Err(_) => false,
    };
    let env_offline = std::env::var_os("PULSAR_OFFLINE").is_some();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "-p" => prompt = Some(need("-p")?),
            // long prompts exceed the OS single-arg limit (~128KB on Linux)
            "-f" | "--prompt-file" => {
                let path = need("--prompt-file")?;
                prompt = Some(std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?);
            }
            "--tokens" => tokens_arg = Some(need("--tokens")?),
            "-n" => n_predict = need("-n")?.parse()?,
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--no-bos" => bos = Some(false),
            "--bos" => bos = Some(true),
            "--dump-logits" => dump_logits = Some(need("--dump-logits")?),
            "--teacher-force" => teacher_force = true,
            "--decode-consistency" => decode_consistency = Some(need("--decode-consistency")?.parse::<usize>()?),
            "--rows-consistency" => rows_consistency = Some(need("--rows-consistency")?.parse::<usize>()?),
            "--dspark-capture" => dspark_capture = Some(need("--dspark-capture")?),
            "--chat" => chat = true,
            "--system" => system = Some(need("--system")?),
            "--temp" => temp = Some(need("--temp")?.parse::<f32>()?),
            "--top-p" => top_p = Some(need("--top-p")?.parse::<f32>()?),
            "--min-p" => min_p = need("--min-p")?.parse::<f32>()?,
            "--seed" => seed = need("--seed")?.parse::<u64>()?,
            "--jinja-chat" => jinja_chat = true,
            "-V" | "--version" => {
                println!("pulsar-cli {}", engine::VERSION);
                return Ok(());
            }
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            other => return Err(format!("unknown arg {other}, try --help").into()),
        }
    }
    let model_path = model_path.ok_or("missing -m MODEL.gguf")?;

    eprintln!("pulsar: loading {model_path}");
    let t0 = std::time::Instant::now();
    // load the dflash draft BEFORE the model: the dense-split solver
    // fills cards to capacity from measured free VRAM, so the draft's
    // buffers must already be resident for the split to leave room
    let mut dflash_draft: Option<engine::DraftModel> = None;
    let mut dspark_model: Option<engine::Model> = None;
    if let Ok(p) = std::env::var("PULSAR_DFLASH") {
        let path = std::path::Path::new(&p);
        // deepseek4 DSpark drafts are full models (own experts + heads,
        // scripts/convert-dspark-dsv4.py); dflash-draft ggufs load the
        // lean qwen35 DraftModel
        if engine::parse_header(path)?.1.architecture() == Some("deepseek4") {
            dspark_model = Some(engine::Model::load(path)?);
            eprintln!("pulsar: dspark draft model loaded ({p})");
        } else {
            dflash_draft = Some(engine::DraftModel::load(path)?);
            eprintln!("pulsar: dflash draft loaded ({p})");
        }
    }
    let model = engine::Model::load(std::path::Path::new(&model_path))?;
    let tok = {
        let (_, g) = engine::parse_header(std::path::Path::new(&model_path))?;
        tokenizer::Tokenizer::from_gguf(&g)?
    };
    eprintln!(
        "pulsar: loaded in {:.1}s ({} layers, {} experts x top-{})",
        t0.elapsed().as_secs_f32(),
        model.shape.n_exec_layer,
        model.shape.n_expert,
        model.shape.n_expert_used
    );
    // Template resolution: with --jinja-chat, full rollover (embed → cache →
    // HF → llama.cpp catalog) unless PULSAR_OFFLINE. Without Jinja, offline
    // peek only so a normal CLI load never phones home.
    let chat_template = {
        let opts = tokenizer::ChatTemplateOptions {
            offline: if jinja_chat { env_offline } else { true },
            ..Default::default()
        };
        match tokenizer::get_chat_template_from_gguf(
            &model.gguf,
            Some(std::path::Path::new(&model_path)),
            None,
            &opts,
        ) {
            Ok(r) => {
                if jinja_chat {
                    eprintln!(
                        "pulsar: chat template from {} ({} bytes{})",
                        r.source,
                        r.template.len(),
                        r.model_id
                            .as_ref()
                            .map(|id| format!(", model_id={id}"))
                            .unwrap_or_default()
                    );
                } else {
                    eprintln!(
                        "pulsar: chat template available offline from {} ({} bytes); pass --jinja-chat to use it",
                        r.source,
                        r.template.len()
                    );
                }
                Some(r)
            }
            Err(e) => {
                if jinja_chat {
                    eprintln!(
                        "pulsar: chat template not resolved ({e}){}",
                        if env_offline {
                            " (PULSAR_OFFLINE: embed or local cache only)"
                        } else {
                            ""
                        }
                    );
                }
                None
            }
        }
    };
    let chat_template = if jinja_chat {
        if chat_template.is_none() {
            eprintln!("pulsar: --jinja-chat set but no template available; ChatMarkers encoding");
        }
        chat_template
    } else {
        None
    };

    if chat {
        return run_chat(
            &model,
            &tok,
            ctx,
            system,
            temp,
            top_p,
            min_p,
            seed,
            n_predict,
            chat_template.as_ref(),
        );
    }

    let prompt_ids: Vec<u32> = match (tokens_arg, prompt) {
        (Some(t), _) => t.split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?,
        (None, Some(p)) => {
            let mut ids = Vec::new();
            if bos.unwrap_or(tok.add_bos) {
                ids.push(tok.bos_id.ok_or("model has no BOS id")?);
            }
            ids.extend(tok.encode(&p));
            ids
        }
        (None, None) => return Err("need -p TEXT or --tokens IDS".into()),
    };
    eprintln!("pulsar: prompt ids {prompt_ids:?}");

    // Long prompts want one big prefill chunk (each chunk costs a full
    // expert-corpus pass) and can trade VRAM pool for it - the pool barely
    // hits during prefill anyway. Explicit env vars win.
    if prompt_ids.len() > 384 {
        if std::env::var_os("PULSAR_BATCH").is_none() {
            std::env::set_var("PULSAR_BATCH", prompt_ids.len().min(768).to_string());
        }
        if std::env::var_os("PULSAR_DEV_CACHE_GB").is_none() {
            std::env::set_var("PULSAR_DEV_CACHE_GB", "2");
        }
    }

    // DSpark draft state FIRST, with hard-capped budgets (~2GB): the
    // target's measuring budget solver then adapts around it, which is
    // the only direction that works - target-first leaves the draft
    // nothing once the census matures (measured: cudaMalloc fail), and
    // an uncapped draft-first starves verify (239ms -> 518ms, a net
    // loss; a 4-6 row verify costs 3x a draft round). PULSAR_BATCH=8
    // caps draft staging (it only ever runs block_size rows).
    // PULSAR_DSPARK_CACHE_GB and PULSAR_DSPARK_RESIDENT=1 raise the
    // draft's share on boxes with VRAM to spare.
    let mut dspark_state = match dspark_model.as_ref() {
        Some(dm) => {
            let cache_gb = std::env::var("PULSAR_DSPARK_CACHE_GB").unwrap_or_else(|_| "1".into());
            let saved_cache = std::env::var("PULSAR_DEV_CACHE_GB").ok();
            let saved_batch = std::env::var("PULSAR_BATCH").ok();
            let saved_host = std::env::var("PULSAR_CACHE_GB").ok();
            let saved_tiers = std::env::var("PULSAR_TIERS").ok();
            std::env::set_var("PULSAR_DEV_CACHE_GB", cache_gb);
            std::env::set_var("PULSAR_BATCH", "8");
            // host LFU too: the auto sizing takes min(12GB, avail-6) and
            // the draft is created first, so without a cap it pins the
            // RAM the target's own host cache needs (measured: verify
            // 239ms -> 338ms from host-cache starvation alone)
            std::env::set_var("PULSAR_CACHE_GB", "2");
            // no tiers for the draft: a tier RESERVES a spare card's
            // whole free VRAM as its arena even when it places 97
            // triples, and the target's tier pass then finds 1GiB free
            // on both cards (measured; verify 328ms vs 90ms healthy).
            // PULSAR_DSPARK_RESIDENT flips the priority for boxes with
            // VRAM to spare.
            if std::env::var_os("PULSAR_DSPARK_RESIDENT").is_none() {
                std::env::set_var("PULSAR_TIERS", "off");
            }
            let dst = engine::State::new(dm, 512)?;
            match saved_tiers {
                Some(v) => std::env::set_var("PULSAR_TIERS", v),
                None => std::env::remove_var("PULSAR_TIERS"),
            }
            match saved_cache {
                Some(v) => std::env::set_var("PULSAR_DEV_CACHE_GB", v),
                None => std::env::remove_var("PULSAR_DEV_CACHE_GB"),
            }
            match saved_batch {
                Some(v) => std::env::set_var("PULSAR_BATCH", v),
                None => std::env::remove_var("PULSAR_BATCH"),
            }
            match saved_host {
                Some(v) => std::env::set_var("PULSAR_CACHE_GB", v),
                None => std::env::remove_var("PULSAR_CACHE_GB"),
            }
            Some(dst)
        }
        None => None,
    };
    let mut st = engine::State::new(&model, ctx)?;

    if teacher_force {
        // With --dump-logits: full-distribution rows for the KV-quant KLD
        // A/B (scripts/kld-ab.sh). Format: u32 LE n_vocab, then one
        // n_vocab f32 LE row per position.
        if let Some(path) = dump_logits.as_ref() {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
            let mut n_vocab = 0usize;
            for (i, &id) in prompt_ids.iter().enumerate() {
                let l = model.forward_token(&mut st, id, i as u32, true)?.unwrap();
                if i == 0 {
                    n_vocab = l.len();
                    f.write_all(&(n_vocab as u32).to_le_bytes())?;
                }
                for v in &l {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            f.flush()?;
            eprintln!(
                "pulsar: wrote {} x {n_vocab} logit rows to {path}",
                prompt_ids.len()
            );
            return Ok(());
        }
        // Per-position top-5 (id, logit) along the given token sequence,
        // one JSON line per position, for cross-engine agreement checks.
        for (i, &id) in prompt_ids.iter().enumerate() {
            let l = model.forward_token(&mut st, id, i as u32, true)?.unwrap();
            let mut top: Vec<u32> = (0..l.len() as u32).collect();
            top.sort_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]));
            let entries: Vec<String> = top[..5]
                .iter()
                .map(|&t| format!("[{},{}]", t, l[t as usize]))
                .collect();
            println!("{{\"pos\":{},\"after\":{},\"top\":[{}]}}", i, id, entries.join(","));
        }
        return Ok(());
    }

    if let Some(spec) = dspark_capture {
        // Check the DSpark feature ring the draft will read. The failure
        // modes here are all silent: a slot never written reads as zeros,
        // a mis-indexed slot duplicates its neighbour, and a mis-indexed
        // POSITION still looks like a plausible hidden state. So check for
        // all three rather than eyeballing magnitudes.
        let layer_ids: Vec<usize> = spec
            .split(',')
            .map(|x| x.trim().parse::<usize>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| format!("--dspark-capture wants a layer list like 41,42,43: {e}"))?;
        let n_cap = layer_ids.len();
        let n_embd = model.shape.n_embd as usize;
        st.enable_dspark_capture(&model, layer_ids.clone())?;
        model.forward_rows(&mut st, &prompt_ids, 0, 1)?;

        let last = prompt_ids.len() as u32 - 1;
        println!("dspark capture: layers {layer_ids:?}, {n_cap} x {n_embd} per position");
        let mut bad = 0;
        let row = st.dspark_feature_row(&model, last)?;
        for (i, id) in layer_ids.iter().enumerate() {
            let slot = &row[i * n_embd..(i + 1) * n_embd];
            let nz = slot.iter().filter(|v| **v != 0.0).count();
            let finite = slot.iter().all(|v| v.is_finite());
            let rms = (slot.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
                / n_embd as f64)
                .sqrt();
            if nz == 0 || !finite {
                bad += 1;
            }
            println!(
                "  hidden[{id}] @pos {last}: rms {rms:.4}, {nz}/{n_embd} nonzero{}",
                if finite { "" } else { ", NOT FINITE" }
            );
        }
        // slots must differ from each other: equal slots mean one capture
        // point overwrote another
        for i in 0..n_cap {
            for j in i + 1..n_cap {
                let (a, b) = (&row[i * n_embd..(i + 1) * n_embd], &row[j * n_embd..(j + 1) * n_embd]);
                if a == b {
                    println!("  slots {i} and {j} are IDENTICAL (capture points collided)");
                    bad += 1;
                }
            }
        }
        // and positions must differ from each other
        if prompt_ids.len() >= 2 {
            let prev = st.dspark_feature_row(&model, last - 1)?;
            if prev == row {
                println!("  positions {} and {last} are IDENTICAL (position indexing wrong)", last - 1);
                bad += 1;
            }
        }
        // re-running must reproduce the ring exactly
        drop(st);
        let mut st2 = engine::State::new(&model, ctx)?;
        st2.enable_dspark_capture(&model, layer_ids)?;
        model.forward_rows(&mut st2, &prompt_ids, 0, 1)?;
        let again = st2.dspark_feature_row(&model, last)?;
        // Not a bit-exactness check: with the expert tier active, summing
        // per-card partials reorders float adds, so two runs of the same
        // prompt differ slightly by design (PULSAR_TIERS=off restores
        // exact, and the ring does reproduce bit-for-bit there). What
        // would be a real bug is a capture that drifts on the order of the
        // signal itself.
        // Cosine, not max |d|. Two runs can pick DIFFERENT experts: the
        // tier reorders float adds, a router logit crosses a neighbour,
        // and top-6-of-256 selects a different set. That is a genuinely
        // different hidden state, not a rounding error, so an absolute
        // bound rejects healthy runs. Direction is what has to hold, and
        // it must hold exactly (cos 1.0) under PULSAR_TIERS=off.
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in row.iter().zip(&again) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64) * (*a as f64);
            nb += (*b as f64) * (*b as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        if cos < 0.99 {
            println!("  repeat cosine {cos:.6} is too low to be routing jitter");
            bad += 1;
        }
        println!(
            "  repeat run: cosine {cos:.6}{}\ndspark capture: {}",
            if again == row { " (bit-exact)" } else { " (expert routing jitter)" },
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        return Ok(());
    }

    if let Some(r) = rows_consistency {
        // Multi-row logits must be (a) identical to the single-row path on
        // the row they share, and (b) actually distinct positions. A tail
        // that returned the last row R times would pass (a) alone, so both
        // halves are load-bearing.
        if r < 2 || r > prompt_ids.len() {
            return Err(format!("--rows-consistency needs 2..={} rows", prompt_ids.len()).into());
        }
        let all = model
            .forward_rows(&mut st, &prompt_ids, 0, r as u32)?
            .ok_or("no logits")?;
        let nv = all.len() / r;

        // (a) same batching, one row: the shared row must be bit-exact
        drop(st);
        let mut st1 = engine::State::new(&model, ctx)?;
        let one = model
            .forward_rows(&mut st1, &prompt_ids, 0, 1)?
            .ok_or("no logits")?;
        let shared = all[(r - 1) * nv..].iter().zip(&one).fold(0f32, |m, (a, b)| m.max((a - b).abs()));

        // (b) each row against a single-token step at that position. The
        // batched and single-token matmul kernels accumulate in different
        // orders, so this one is a tolerance check, not bit-exactness.
        drop(st1);
        let mut st2 = engine::State::new(&model, ctx)?;
        let mut per_pos: Vec<Vec<f32>> = Vec::new();
        for (i, &id) in prompt_ids.iter().enumerate() {
            let l = model.forward_rows(&mut st2, &[id], i as u32, 1)?.ok_or("no logits")?;
            if i + r >= prompt_ids.len() {
                per_pos.push(l);
            }
        }
        // Bit-exactness is NOT the bar here: head_logits dispatches its
        // matmul by row count, so rows=1 and rows=R accumulate in
        // different orders. Every family drifts (Laguna, whose multi-row
        // path ships, drifts more than dsv4). What must hold is that each
        // row is its OWN position, and that any argmax flip is a near-tie
        // the drift can explain rather than a wrong row.
        println!("rows-consistency r={r} over {} tokens:", prompt_ids.len());
        println!("  shared row (rows=1 vs rows={r}): max |dlogit| {shared:.6}");
        let mut worst = 0f32;
        let mut bad = 0;
        for j in 0..r {
            let row = &all[j * nv..(j + 1) * nv];
            let ref_row = &per_pos[j];
            let d = row.iter().zip(ref_row).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let (ra, pa) = (engine::argmax(row), engine::argmax(ref_row));
            worst = worst.max(d);
            // top1-top2 on the reference row: a flip inside this gap is
            // the drift reordering a tie, a flip outside it is a bug
            let gap = {
                let (mut t1, mut t2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &v in ref_row {
                    if v > t1 {
                        t2 = t1;
                        t1 = v;
                    } else if v > t2 {
                        t2 = v;
                    }
                }
                t1 - t2
            };
            let verdict = if ra == pa {
                "match"
            } else if gap <= d {
                "flip within drift"
            } else {
                bad += 1;
                "FLIP OUTSIDE DRIFT"
            };
            println!(
                "  row {j} (pos {}): max |dlogit| {d:.4}, gap {gap:.4}, argmax {ra} vs {pa} ({verdict})",
                prompt_ids.len() - r + j,
            );
        }
        println!(
            "  worst |dlogit| {worst:.4} -> {}",
            if bad == 0 { "PASS" } else { "FAIL" }
        );
        return Ok(());
    }

    if let Some(nsteps) = decode_consistency {
        // Greedy-decode nsteps tokens through the incremental (n_tok=1)
        // path, then fresh-prefill the identical sequence batched and
        // compare the logits at the same position. Divergence here is the
        // reduction-order drift between the batch and decode matmul
        // kernels - the ds4 --decode-consistency analogue.
        let mut logits = None;
        let mut pos0 = 0u32;
        for chunk in prompt_ids.chunks(st.max_batch() as usize) {
            logits = model.forward_batch(&mut st, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let mut seq = prompt_ids.clone();
        for _ in 0..nsteps.saturating_sub(1) {
            let next = engine::argmax(logits.as_ref().ok_or("no logits")?);
            seq.push(next);
            logits = model.forward_batch(&mut st, &[next], seq.len() as u32 - 1, true)?;
        }
        let decode_logits = logits.ok_or("no logits")?;
        let decode_argmax = engine::argmax(&decode_logits);

        drop(st); // free VRAM before the fresh state
        let mut st2 = engine::State::new(&model, ctx)?;
        let mut fresh = None;
        let mut pos0 = 0u32;
        for chunk in seq.chunks(st2.max_batch() as usize) {
            fresh = model.forward_batch(&mut st2, chunk, pos0, true)?;
            pos0 += chunk.len() as u32;
        }
        let fresh_logits = fresh.ok_or("no logits")?;
        let fresh_argmax = engine::argmax(&fresh_logits);

        let mut maxd = 0f32;
        let mut sum = 0f64;
        for (a, b) in decode_logits.iter().zip(&fresh_logits) {
            let d = (a - b).abs();
            maxd = maxd.max(d);
            sum += d as f64;
        }
        let gap = {
            let mut top = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for &v in &decode_logits {
                if v > top {
                    second = top;
                    top = v;
                } else if v > second {
                    second = v;
                }
            }
            top - second
        };
        println!(
            "decode-consistency after {} steps ({} total tokens):\n  max |dlogit| {maxd:.4}, mean {:.5}\n  argmax decode={decode_argmax} fresh-prefill={fresh_argmax} ({}), decode top1-top2 gap {gap:.4}",
            nsteps,
            seq.len(),
            sum / decode_logits.len() as f64,
            if decode_argmax == fresh_argmax { "MATCH" } else { "FLIP" },
        );
        return Ok(());
    }

    // DSpark speculative decode (deepseek4 + the converted DSpark draft
    // gguf): PULSAR_DFLASH=/path/to/dspark-draft.gguf, greedy one-shot
    if let (Some(dm), Some(mut dst), None) =
        (dspark_model.take(), dspark_state.take(), dump_logits.as_ref())
    {
        let mut generated: Vec<u32> = Vec::new();
        let mut t_first: Option<std::time::Instant> = None;
        let out = std::io::stdout();
        engine::generate_dspark(
            &model,
            &mut st,
            &dm,
            &mut dst,
            &prompt_ids,
            0,
            n_predict,
            |t| tok.is_eog(t),
            |t| {
                t_first.get_or_insert_with(std::time::Instant::now);
                generated.push(t);
                use std::io::Write;
                let mut o = out.lock();
                o.write_all(&tok.decode(&[t])).ok();
                o.flush().ok();
            },
        )?;
        println!();
        st.save_warm(&model)?;
        let dt = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        eprintln!(
            "pulsar: {} tokens in {:.2}s ({:.2} tok/s), dspark {}/{} drafts accepted ({:.0}%)\npulsar: ids {generated:?}",
            generated.len(),
            dt,
            generated.len() as f32 / dt.max(1e-6),
            st.mtp_accepted,
            st.mtp_drafted,
            100.0 * st.mtp_accepted as f64 / st.mtp_drafted.max(1) as f64
        );
        return Ok(());
    }

    // DFlash speculative decode (qwen35moe + a matched block-diffusion
    // draft gguf): PULSAR_DFLASH=/path/to/draft.gguf, greedy one-shot
    if let (Some(mut draft), None) = (dflash_draft.take(), dump_logits.as_ref()) {
        let mut generated: Vec<u32> = Vec::new();
        let mut t_first: Option<std::time::Instant> = None;
        let out = std::io::stdout();
        engine::generate_dflash(
            &model,
            &mut draft,
            &mut st,
            &prompt_ids,
            0,
            n_predict,
            |t| tok.is_eog(t),
            |t| {
                t_first.get_or_insert_with(std::time::Instant::now);
                generated.push(t);
                use std::io::Write;
                let mut o = out.lock();
                o.write_all(&tok.decode(&[t])).ok();
                o.flush().ok();
            },
        )?;
        println!();
        st.save_warm(&model)?;
        let dt = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        eprintln!(
            "pulsar: {} tokens in {:.2}s ({:.2} tok/s), dflash {}/{} drafts accepted ({:.0}%)\npulsar: ids {generated:?}",
            generated.len(),
            dt,
            generated.len() as f32 / dt.max(1e-6),
            st.mtp_accepted,
            st.mtp_drafted,
            100.0 * st.mtp_accepted as f64 / st.mtp_drafted.max(1) as f64
        );
        return Ok(());
    }

    // MTP speculative decode routes through engine::generate (the spec
    // loop lives there); greedy-only, so the one-shot default applies
    if (std::env::var("PULSAR_MTP").ok().as_deref() == Some("1")
        || std::env::var("PULSAR_NGRAM").is_ok())
        && dump_logits.is_none() {
        let mut generated: Vec<u32> = Vec::new();
        let mut t_first: Option<std::time::Instant> = None;
        let mut sampler = engine::Sampler::new(0.0, 1.0, 0.0, 1);
        let out = std::io::stdout();
        engine::generate(
            &model,
            &mut st,
            &prompt_ids,
            0,
            &mut sampler,
            n_predict,
            |t| tok.is_eog(t),
            |t| {
                t_first.get_or_insert_with(std::time::Instant::now);
                generated.push(t);
                use std::io::Write;
                let mut o = out.lock();
                o.write_all(&tok.decode(&[t])).ok();
                o.flush().ok();
            },
        )?;
        println!();
        if std::env::var_os("PULSAR_PROFILE").is_some() {
            eprintln!("pulsar: profile: {}", st.prof.report());
        }
        st.save_warm(&model)?;
        let dt = t_first.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        eprintln!(
            "pulsar: {} tokens in {:.2}s ({:.2} tok/s), mtp {}/{} drafts accepted ({:.0}%)\npulsar: ids {generated:?}",
            generated.len(),
            dt,
            generated.len() as f32 / dt.max(1e-6),
            st.mtp_accepted,
            st.mtp_drafted,
            100.0 * st.mtp_accepted as f64 / st.mtp_drafted.max(1) as f64
        );
        return Ok(());
    }

    let t1 = std::time::Instant::now();
    let mut logits = None;
    let mut pos0 = 0u32;
    let prof_chunks = std::env::var_os("PULSAR_PROFILE").is_some();
    for chunk in prompt_ids.chunks(st.max_batch() as usize) {
        let last = pos0 as usize + chunk.len() == prompt_ids.len();
        let tc = std::time::Instant::now();
        logits = model.forward_batch(&mut st, chunk, pos0, last)?;
        if prof_chunks {
            eprintln!("pulsar: prefill chunk @{pos0} len {} in {:.2}s", chunk.len(), tc.elapsed().as_secs_f64());
        }
        pos0 += chunk.len() as u32;
    }
    eprintln!(
        "pulsar: prefill {} tokens in {:.2}s",
        prompt_ids.len(),
        t1.elapsed().as_secs_f32()
    );

    if let Some(path) = dump_logits {
        let l = logits.as_ref().ok_or("no logits")?;
        let mut s = String::with_capacity(l.len() * 12);
        s.push('[');
        for (i, v) in l.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{v}"));
        }
        s.push(']');
        std::fs::write(&path, s)?;
        eprintln!("pulsar: wrote {} logits to {path}", l.len());
        return Ok(());
    }

    let pos0 = prompt_ids.len() as u32;
    let mut generated = Vec::new();
    let t2 = std::time::Instant::now();
    // greedy one-shot on qwen35: argmax-only rows (device argmax + the
    // TP vocab-split head) instead of a ~1MB logits readback per token
    let amax_fast = model.shape.family == engine::Family::Qwen35;
    let mut next_amax = logits.as_ref().map(|l| engine::argmax(l));
    for pos in pos0..pos0.saturating_add(n_predict as u32) {
        let next = if amax_fast {
            next_amax.ok_or("no argmax")?
        } else {
            engine::argmax(logits.as_ref().ok_or("no logits")?)
        };
        if tok.is_eog(next) {
            break;
        }
        generated.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode(&[next])));
        use std::io::Write;
        std::io::stdout().flush().ok();
        if pos >= ctx {
            break;
        }
        if amax_fast {
            st.skip_logit_read = true;
            let r = model.forward_token(&mut st, next, pos, true);
            st.skip_logit_read = false;
            r?;
            next_amax = st.last_argmax.first().copied();
        } else {
            logits = model.forward_token(&mut st, next, pos, true)?;
        }
    }
    println!();
    if std::env::var_os("PULSAR_PROFILE").is_some() {
        eprintln!("pulsar: profile: {}", st.prof.report());
    }
    st.save_warm(&model)?;
    let dt = t2.elapsed().as_secs_f32();
    let tier_note = {
        let hits: u64 = st.tiers.iter().map(|t| t.hits).sum();
        let mut s = if hits > 0 {
            format!(", tier {hits} resident slots")
        } else {
            String::new()
        };
        if st.cpu_hits > 0 {
            s += &format!(", cpu lane {} experts", st.cpu_hits);
        }
        if st.mtp_drafted > 0 {
            s += &format!(
                ", mtp {}/{} drafts accepted ({:.0}%)",
                st.mtp_accepted,
                st.mtp_drafted,
                100.0 * st.mtp_accepted as f64 / st.mtp_drafted as f64
            );
        }
        s
    };
    eprintln!(
        "pulsar: {} tokens in {:.2}s ({:.2} tok/s), vram cache {:.0}% hits, host cache {:.0}% of remainder{tier_note}\npulsar: ids {generated:?}",
        generated.len(),
        dt,
        generated.len() as f32 / dt.max(1e-6),
        100.0 * st.dev_cache.hits as f64 / (st.dev_cache.hits + st.dev_cache.misses).max(1) as f64,
        100.0 * st.store.hits as f64 / (st.store.hits + st.store.misses).max(1) as f64
    );
    Ok(())
}
