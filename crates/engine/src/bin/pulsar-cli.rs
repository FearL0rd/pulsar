//! pulsar-cli: greedy Hy3 decode for parity work.
//!
//!   pulsar-cli -m model.gguf -p "text" -n 32 [--ctx 2048] [--no-bos]
//!   pulsar-cli -m model.gguf --tokens 120000,16883,11 -n 32
//!
//! -p tokenizes raw text (BOS prepended unless --no-bos); --tokens feeds
//! exact ids, which is how A/B runs align with ds4 --dump-tokens output.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("pulsar-cli requires Linux + CUDA");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-cli: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
fn run() -> engine::Result {
    let mut model_path = None;
    let mut prompt = None;
    let mut tokens_arg = None;
    let mut n_predict = 16usize;
    let mut ctx = 2048u32;
    let mut bos = true;
    let mut dump_logits = None;
    let mut teacher_force = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-m" => model_path = Some(need("-m")?),
            "-p" => prompt = Some(need("-p")?),
            "--tokens" => tokens_arg = Some(need("--tokens")?),
            "-n" => n_predict = need("-n")?.parse()?,
            "--ctx" => ctx = need("--ctx")?.parse()?,
            "--no-bos" => bos = false,
            "--dump-logits" => dump_logits = Some(need("--dump-logits")?),
            "--teacher-force" => teacher_force = true,
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let model_path = model_path.ok_or("missing -m MODEL.gguf")?;

    eprintln!("pulsar: loading {model_path}");
    let t0 = std::time::Instant::now();
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

    let prompt_ids: Vec<u32> = match (tokens_arg, prompt) {
        (Some(t), _) => t.split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?,
        (None, Some(p)) => {
            let mut ids = Vec::new();
            if bos {
                ids.push(tok.bos_id.ok_or("model has no BOS id")?);
            }
            ids.extend(tok.encode(&p));
            ids
        }
        (None, None) => return Err("need -p TEXT or --tokens IDS".into()),
    };
    eprintln!("pulsar: prompt ids {prompt_ids:?}");

    let mut st = engine::State::new(&model, ctx)?;

    if teacher_force {
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

    let t1 = std::time::Instant::now();
    let mut logits = None;
    for (i, &id) in prompt_ids.iter().enumerate() {
        let last = i + 1 == prompt_ids.len();
        logits = model.forward_token(&mut st, id, i as u32, last)?;
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

    let mut pos = prompt_ids.len() as u32;
    let mut generated = Vec::new();
    let t2 = std::time::Instant::now();
    for _ in 0..n_predict {
        let l = logits.as_ref().ok_or("no logits")?;
        let next = engine::argmax(l);
        if Some(next) == tok.eos_id {
            break;
        }
        generated.push(next);
        print!("{}", String::from_utf8_lossy(&tok.decode(&[next])));
        use std::io::Write;
        std::io::stdout().flush().ok();
        if pos >= ctx {
            break;
        }
        logits = model.forward_token(&mut st, next, pos, true)?;
        pos += 1;
    }
    println!();
    let dt = t2.elapsed().as_secs_f32();
    eprintln!(
        "pulsar: {} tokens in {:.2}s ({:.2} tok/s), expert cache {}/{} hits ({:.0}%)\npulsar: ids {generated:?}",
        generated.len(),
        dt,
        generated.len() as f32 / dt.max(1e-6),
        st.store.hits,
        st.store.hits + st.store.misses,
        100.0 * st.store.hits as f64 / (st.store.hits + st.store.misses).max(1) as f64
    );
    Ok(())
}
