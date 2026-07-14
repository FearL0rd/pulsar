fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return; // kernels are CUDA/Linux; other hosts get an empty crate
    }
    println!("cargo:rerun-if-changed=cuda/pulsar_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/gqa_kernels.inc");
    println!("cargo:rerun-if-changed=cuda/iq2_tables.inc");
    println!("cargo:rerun-if-changed=cuda/mla_kernels.inc");
    // PULSAR_CUDA_ARCH overrides the target (e.g. "89", or "89,120" once
    // the toolkit is new enough for native Blackwell SASS). Default emits
    // sm_89 SASS + compute_89 PTX in one fatbin: Ada runs the SASS, and a
    // Blackwell-aware driver JIT-forwards the PTX to sm_120 (5060 Ti) - so
    // one binary runs on both cards even on a CUDA 12.0 toolkit that can't
    // codegen sm_120 itself.
    let archs = std::env::var("PULSAR_CUDA_ARCH").unwrap_or_else(|_| "89".into());
    let mut build = cc::Build::new();
    build.cuda(true).flag("-O3").flag("--use_fast_math");
    for a in archs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        // code=[sm_XX,compute_XX] embeds native SASS and forward-JIT PTX
        build.flag("-gencode").flag(&format!("arch=compute_{a},code=[sm_{a},compute_{a}]"));
    }
    build.file("cuda/pulsar_kernels.cu").compile("pulsar_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
