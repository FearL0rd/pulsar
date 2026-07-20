//! Offline hotlist generator: inverts a model's census (.warm file)
//! into a quantization-independent {layer, expert} preload seed for
//! crates/engine/hotlists/. Run on a machine that has decoded with the
//! model at least once (so the census exists), then check the output in.
//! Usage: hotlist-gen <model.gguf> > crates/engine/hotlists/<family>.hotlist

#[cfg(not(target_os = "linux"))]
fn main() {}

#[cfg(target_os = "linux")]
fn main() {
    let path = std::env::args().nth(1).expect("usage: hotlist-gen <model.gguf>");
    let text = engine::hotlist_text(std::path::Path::new(&path)).expect("hotlist");
    print!("{text}");
}
