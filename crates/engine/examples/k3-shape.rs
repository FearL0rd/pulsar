//! Resolve a gguf header into a `Shape` and print it, so a new
//! architecture's config parsing can be checked against the real file
//! before any weights exist locally:
//!   cargo run --example k3-shape -- /mnt/models/<model>.gguf
//! Header-only, so shard 1 of a split model is enough.
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("k3-shape needs the linux engine build");
}

#[cfg(target_os = "linux")]
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: k3-shape <model.gguf>");
    let (_shards, g) = engine::parse_header(std::path::Path::new(&path)).expect("parse gguf");
    println!("arch = {:?}", g.architecture());
    let s = engine::Shape::from_gguf(&g).expect("shape");
    println!("{s:#?}");
}
