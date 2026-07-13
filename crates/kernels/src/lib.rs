//! FFI to the pulsar CUDA kernel library. Linux + NVIDIA only; on other
//! hosts the crate compiles to nothing so the workspace still builds.

#[cfg(target_os = "linux")]
mod real {
    use std::ffi::c_void;

    pub type Result<T = ()> = std::result::Result<T, Error>;

    #[derive(Debug)]
    pub struct Error(pub &'static str);

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "cuda kernel op failed: {}", self.0)
        }
    }

    impl std::error::Error for Error {}

    /// Matches `pulsar_expert_ptrs` in pulsar_kernels.cu: explicit device
    /// pointers for one (token, slot); NULL means "not routed".
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ExpertPtrs {
        pub gate: *const c_void,
        pub up: *const c_void,
        pub down: *const c_void,
    }

    unsafe impl Send for ExpertPtrs {}

    impl ExpertPtrs {
        pub const NULL: ExpertPtrs = ExpertPtrs {
            gate: std::ptr::null(),
            up: std::ptr::null(),
            down: std::ptr::null(),
        };
    }

    pub const QUANT_Q2_K: u32 = 0;
    pub const QUANT_IQ2_XXS: u32 = 1;

    const H2D: i32 = 1;
    const D2H: i32 = 2;

    extern "C" {
        fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        fn cudaDeviceSynchronize() -> i32;

        fn pulsar_embed_q8_0(out: *mut c_void, w: *const c_void, tokens: *const c_void, n_embd: u32, n_vocab: u32, n_tok: u32) -> i32;
        fn pulsar_rms_norm(out: *mut c_void, x: *const c_void, w: *const c_void, n: u32, rows: u32, eps: f32) -> i32;
        fn pulsar_q8_0_matmul(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_f32(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_swiglu(out: *mut c_void, gate: *const c_void, up: *const c_void, n: u32, clamp: f32, weight: f32) -> i32;
        fn pulsar_add(out: *mut c_void, a: *const c_void, b: *const c_void, n: u32) -> i32;
        fn pulsar_router_select(selected: *mut c_void, weights: *mut c_void, logits: *const c_void, bias: *const c_void, n_expert: u32, k_used: u32, weight_scale: f32, n_tok: u32) -> i32;
        fn pulsar_quantize_q8_K(out: *mut c_void, x: *const c_void, in_dim: u32, n_rows: u32) -> i32;
        fn pulsar_moe_pair_swiglu(mid: *mut c_void, ptrs: *const c_void, weights: *const c_void, x: *const c_void, in_dim: u32, mid_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_moe_down(out: *mut c_void, ptrs: *const c_void, mid: *const c_void, mid_dim: u32, out_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_gqa_head_rms_norm(x: *mut c_void, w: *const c_void, rows: u32, head_dim: u32, eps: f32) -> i32;
        fn pulsar_gqa_rope(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, pos0: u32, theta: f32) -> i32;
        fn pulsar_gqa_kv_append(cache: *mut c_void, kv: *const c_void, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32) -> i32;
        fn pulsar_gqa_attention(out: *mut c_void, q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32) -> i32;

        fn pulsar_gqa_selftest() -> i32;
        fn pulsar_q8_0_matmul_selftest() -> i32;
        fn pulsar_router_selftest() -> i32;
        fn pulsar_moe_selftest() -> i32;
        fn pulsar_glue_selftest() -> i32;
    }

    fn check(ret: i32, op: &'static str) -> Result {
        if ret != 0 {
            Ok(())
        } else {
            Err(Error(op))
        }
    }

    fn check_rt(ret: i32, op: &'static str) -> Result {
        if ret == 0 {
            Ok(())
        } else {
            Err(Error(op))
        }
    }

    /// An owned device allocation. Byte-oriented; callers track element
    /// layout themselves (this engine's tensors are f32/i32/quant blobs).
    pub struct DeviceBuf {
        ptr: *mut c_void,
        bytes: usize,
    }

    unsafe impl Send for DeviceBuf {}

    impl DeviceBuf {
        pub fn alloc(bytes: usize) -> Result<Self> {
            let mut ptr = std::ptr::null_mut();
            check_rt(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc")?;
            Ok(DeviceBuf { ptr, bytes })
        }

        pub fn from_bytes(data: &[u8]) -> Result<Self> {
            let mut b = Self::alloc(data.len())?;
            b.write(0, data)?;
            Ok(b)
        }

        pub fn from_f32(data: &[f32]) -> Result<Self> {
            Self::from_bytes(as_bytes(data))
        }

        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn ptr(&self) -> *const c_void {
            self.ptr
        }

        pub fn ptr_mut(&mut self) -> *mut c_void {
            self.ptr
        }

        /// Device pointer at a byte offset (for slab arenas).
        pub fn ptr_at(&self, off: usize) -> *const c_void {
            debug_assert!(off <= self.bytes);
            unsafe { (self.ptr as *const u8).add(off) as *const c_void }
        }

        pub fn write(&mut self, off: usize, data: &[u8]) -> Result {
            assert!(off + data.len() <= self.bytes, "device write out of range");
            check_rt(
                unsafe {
                    cudaMemcpy(
                        (self.ptr as *mut u8).add(off) as *mut c_void,
                        data.as_ptr() as *const c_void,
                        data.len(),
                        H2D,
                    )
                },
                "cudaMemcpy h2d",
            )
        }

        pub fn read(&self, off: usize, out: &mut [u8]) -> Result {
            assert!(off + out.len() <= self.bytes, "device read out of range");
            check_rt(
                unsafe {
                    cudaMemcpy(
                        out.as_mut_ptr() as *mut c_void,
                        (self.ptr as *const u8).add(off) as *const c_void,
                        out.len(),
                        D2H,
                    )
                },
                "cudaMemcpy d2h",
            )
        }

        pub fn read_f32(&self, n: usize) -> Result<Vec<f32>> {
            let mut v = vec![0f32; n];
            self.read(0, as_bytes_mut(&mut v))?;
            Ok(v)
        }

        pub fn read_i32(&self, n: usize) -> Result<Vec<i32>> {
            let mut v = vec![0i32; n];
            self.read(0, as_bytes_mut(&mut v))?;
            Ok(v)
        }
    }

    impl Drop for DeviceBuf {
        fn drop(&mut self) {
            unsafe { cudaFree(self.ptr) };
        }
    }

    pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v))
        }
    }

    pub fn sync() -> Result {
        check_rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }

    pub fn embed_q8_0(out: &mut DeviceBuf, w: &DeviceBuf, tokens: &DeviceBuf, n_embd: u32, n_vocab: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_embed_q8_0(out.ptr_mut(), w.ptr(), tokens.ptr(), n_embd, n_vocab, n_tok) }, "embed_q8_0")
    }

    pub fn rms_norm(out: &mut DeviceBuf, x: &DeviceBuf, w: &DeviceBuf, n: u32, rows: u32, eps: f32) -> Result {
        check(unsafe { pulsar_rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), n, rows, eps) }, "rms_norm")
    }

    pub fn matmul_q8_0(out: &mut DeviceBuf, w: &DeviceBuf, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_q8_0_matmul(out.ptr_mut(), w.ptr(), x.ptr(), in_dim, out_dim, n_tok) }, "matmul_q8_0")
    }

    pub fn matmul_f32(out: &mut DeviceBuf, w: &DeviceBuf, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_matmul_f32(out.ptr_mut(), w.ptr(), x.ptr(), in_dim, out_dim, n_tok) }, "matmul_f32")
    }

    pub fn swiglu(out: &mut DeviceBuf, gate: &DeviceBuf, up: &DeviceBuf, n: u32, clamp: f32, weight: f32) -> Result {
        check(unsafe { pulsar_swiglu(out.ptr_mut(), gate.ptr(), up.ptr(), n, clamp, weight) }, "swiglu")
    }

    pub fn add(out: &mut DeviceBuf, a: &DeviceBuf, b: &DeviceBuf, n: u32) -> Result {
        check(unsafe { pulsar_add(out.ptr_mut(), a.ptr(), b.ptr(), n) }, "add")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router_select(selected: &mut DeviceBuf, weights: &mut DeviceBuf, logits: &DeviceBuf, bias: &DeviceBuf, n_expert: u32, k_used: u32, weight_scale: f32, n_tok: u32) -> Result {
        check(
            unsafe {
                pulsar_router_select(selected.ptr_mut(), weights.ptr_mut(), logits.ptr(), bias.ptr(), n_expert, k_used, weight_scale, n_tok)
            },
            "router_select",
        )
    }

    /// GGML q8_K block: f32 scale + 256 int8 + 16 i16 block sums.
    pub const Q8_K_BLOCK_BYTES: usize = 292;
    pub const Q8_K_BLOCK_ELEMS: usize = 256;

    /// Quantize f32 rows to q8_K (the activation side of the expert dots).
    pub fn quantize_q8_k(out: &mut DeviceBuf, x: &DeviceBuf, in_dim: u32, n_rows: u32) -> Result {
        check(unsafe { pulsar_quantize_q8_K(out.ptr_mut(), x.ptr(), in_dim, n_rows) }, "quantize_q8_k")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_pair_swiglu(mid: &mut DeviceBuf, ptrs: &DeviceBuf, weights: &DeviceBuf, x: &DeviceBuf, in_dim: u32, mid_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> Result {
        check(
            unsafe {
                pulsar_moe_pair_swiglu(mid.ptr_mut(), ptrs.ptr(), weights.ptr(), x.ptr(), in_dim, mid_dim, n_used, n_tok, row_bytes, quant)
            },
            "moe_pair_swiglu",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(out: &mut DeviceBuf, ptrs: &DeviceBuf, mid: &DeviceBuf, mid_dim: u32, out_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> Result {
        check(
            unsafe {
                pulsar_moe_down(out.ptr_mut(), ptrs.ptr(), mid.ptr(), mid_dim, out_dim, n_used, n_tok, row_bytes, quant)
            },
            "moe_down",
        )
    }

    pub fn gqa_head_rms_norm(x: &mut DeviceBuf, w: &DeviceBuf, rows: u32, head_dim: u32, eps: f32) -> Result {
        check(unsafe { pulsar_gqa_head_rms_norm(x.ptr_mut(), w.ptr(), rows, head_dim, eps) }, "gqa_head_rms_norm")
    }

    pub fn gqa_rope(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, pos0: u32, theta: f32) -> Result {
        check(unsafe { pulsar_gqa_rope(x.ptr_mut(), n_tok, n_head, head_dim, pos0, theta) }, "gqa_rope")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_kv_append(cache: &mut DeviceBuf, kv: &DeviceBuf, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32) -> Result {
        check(unsafe { pulsar_gqa_kv_append(cache.ptr_mut(), kv.ptr(), n_tok, n_kv_head, head_dim, cap, pos0) }, "gqa_kv_append")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention(out: &mut DeviceBuf, q: &DeviceBuf, k_cache: &DeviceBuf, v_cache: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32) -> Result {
        check(
            unsafe {
                pulsar_gqa_attention(out.ptr_mut(), q.ptr(), k_cache.ptr(), v_cache.ptr(), n_tok, n_head, n_kv_head, head_dim, cap, pos0)
            },
            "gqa_attention",
        )
    }

    pub fn gqa_selftest() -> bool {
        unsafe { pulsar_gqa_selftest() != 0 }
    }

    pub fn q8_0_matmul_selftest() -> bool {
        unsafe { pulsar_q8_0_matmul_selftest() != 0 }
    }

    pub fn router_selftest() -> bool {
        unsafe { pulsar_router_selftest() != 0 }
    }

    pub fn moe_selftest() -> bool {
        unsafe { pulsar_moe_selftest() != 0 }
    }

    pub fn glue_selftest() -> bool {
        unsafe { pulsar_glue_selftest() != 0 }
    }
}

#[cfg(target_os = "linux")]
pub use real::*;

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    /// GPU-required; run explicitly: cargo test -p kernels -- --ignored
    #[test]
    #[ignore = "requires a CUDA device"]
    fn gqa_kernels_match_cpu_reference() {
        assert!(super::gqa_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn q8_0_matmul_matches_cpu_reference() {
        assert!(super::q8_0_matmul_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn router_select_matches_cpu_reference() {
        assert!(super::router_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn moe_kernels_match_cpu_reference() {
        assert!(super::moe_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn glue_kernels_match_cpu_reference() {
        assert!(super::glue_selftest());
    }

    /// End-to-end DeviceBuf + rust-side wrapper smoke test: y = a + b.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_buf_roundtrip_and_add() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1024).map(|i| 2.0 * i as f32).collect();
        let da = super::DeviceBuf::from_f32(&a).unwrap();
        let db = super::DeviceBuf::from_f32(&b).unwrap();
        let mut dy = super::DeviceBuf::alloc(1024 * 4).unwrap();
        super::add(&mut dy, &da, &db, 1024).unwrap();
        super::sync().unwrap();
        let y = dy.read_f32(1024).unwrap();
        for i in 0..1024 {
            assert_eq!(y[i], 3.0 * i as f32);
        }
    }
}
