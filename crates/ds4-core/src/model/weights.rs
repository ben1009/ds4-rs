//! Typed weight accessors for [`WeightMap`].
//!
//! See rfcs/0002-forward-pass.md §2 / PR #8. Each function validates the
//! tensor's existence, GGML dtype, and shape, then returns a strongly-typed
//! view that the forward pass can feed directly into the matmul dispatch.
//!
//! ```ignore
//! let w = model.q8_0("blk.0.attn_q_a.weight")?; // WeightView::Q8_0
//! let r = model.f16("blk.0.ffn_gate_inp.weight")?; // WeightView::F16
//! let b = model.f32_1d("blk.0.attn_norm.weight", 4096)?; // &[f32]
//! ```

use anyhow::{Context, Result, bail};

use crate::{
    gguf::{GgmlType, TensorInfo},
    model::WeightMap,
    ops::matmul::WeightView,
};

impl WeightMap {
    // -----------------------------------------------------------------------
    // Quantised weight views (feed matmul dispatch)
    // -----------------------------------------------------------------------

    /// Get a Q8_0 weight matrix view.
    ///
    /// Expected GGML type: `Q8_0`. Byte layout validated by the matmul kernel.
    pub fn q8_0(&self, name: &str) -> Result<WeightView<'_>> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::Q8_0)?;
        let bytes = self.tensor_bytes(name)?;
        let out_features = info.dims[0] as usize;
        let in_features = info.dims.get(1).copied().unwrap_or(1) as usize;
        Ok(WeightView::Q8_0 {
            bytes,
            out_features,
            in_features,
        })
    }

    /// Get an F16 weight matrix view.
    ///
    /// Expected GGML type: `F16`. Each element is 2 bytes little-endian.
    pub fn f16(&self, name: &str) -> Result<WeightView<'_>> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::F16)?;
        let bytes = self.tensor_bytes(name)?;
        let out_features = info.dims[0] as usize;
        let in_features = info.dims.get(1).copied().unwrap_or(1) as usize;
        Ok(WeightView::F16 {
            bytes,
            out_features,
            in_features,
        })
    }

    /// Get a Q2_K weight matrix view.
    ///
    /// Expected GGML type: `Q2_K`. Byte layout validated by the matmul kernel.
    pub fn q2_k(&self, name: &str) -> Result<WeightView<'_>> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::Q2_K)?;
        let bytes = self.tensor_bytes(name)?;
        let out_features = info.dims[0] as usize;
        let in_features = info.dims.get(1).copied().unwrap_or(1) as usize;
        Ok(WeightView::Q2_K {
            bytes,
            out_features,
            in_features,
        })
    }

    /// Get an IQ2_XXS weight matrix view.
    ///
    /// Expected GGML type: `IQ2_XXS`. Byte layout validated by the matmul kernel.
    pub fn iq2_xxs(&self, name: &str) -> Result<WeightView<'_>> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::IQ2_XXS)?;
        let bytes = self.tensor_bytes(name)?;
        let out_features = info.dims[0] as usize;
        let in_features = info.dims.get(1).copied().unwrap_or(1) as usize;
        Ok(WeightView::IQ2_XXS {
            bytes,
            out_features,
            in_features,
        })
    }

    /// Get a quantized weight matrix view, auto-dispatching by the tensor's
    /// actual GGML dtype.
    ///
    /// This is useful for routed-expert tensors whose dtype may vary by model
    /// variant (e.g. `IQ2_XXS` vs `IQ4_K` for gate/up, `Q2_K` vs `Q4_K` for
    /// down). Only types with a matching [`WeightView`] variant are supported;
    /// unsupported dtypes return an error.
    pub fn quant_weight(&self, name: &str) -> Result<WeightView<'_>> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        let bytes = self.tensor_bytes(name)?;
        let out_features = info.dims[0] as usize;
        let in_features = info.dims.get(1).copied().unwrap_or(1) as usize;
        match info.dtype {
            GgmlType::Q8_0 => Ok(WeightView::Q8_0 {
                bytes,
                out_features,
                in_features,
            }),
            GgmlType::F16 => Ok(WeightView::F16 {
                bytes,
                out_features,
                in_features,
            }),
            GgmlType::Q2_K => Ok(WeightView::Q2_K {
                bytes,
                out_features,
                in_features,
            }),
            GgmlType::IQ2_XXS => Ok(WeightView::IQ2_XXS {
                bytes,
                out_features,
                in_features,
            }),
            other => bail!("{name}: unsupported GGML dtype for weight view: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Plain dtype direct slices
    // -----------------------------------------------------------------------

    /// Get an F32 tensor as a `&[f32]` slice.
    ///
    /// The caller must specify the expected element count so a shape mismatch
    /// is caught immediately.
    pub fn f32_1d(&self, name: &str, expect_elems: usize) -> Result<&[f32]> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::F32)?;
        let bytes = self.tensor_bytes(name)?;
        let expect_bytes = expect_elems
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("{name}: F32 element count overflow"))?;
        if bytes.len() != expect_bytes {
            bail!(
                "{name}: expected {expect_elems} F32 elems ({expect_bytes} B), got {} B",
                bytes.len()
            );
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            bail!("{name}: F32 data is not 4-byte aligned");
        }
        // SAFETY: we just validated length and alignment (u8 slice is 1-byte aligned,
        // and f32 requires 4-byte alignment which the GGUF file guarantees for F32
        // tensors via its alignment field, and mmap returns page-aligned memory).
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, expect_elems) })
    }

    /// Get an I32 tensor as a `&[i32]` slice.
    pub fn i32_1d(&self, name: &str, expect_elems: usize) -> Result<&[i32]> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::I32)?;
        let bytes = self.tensor_bytes(name)?;
        let expect_bytes = expect_elems
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("{name}: I32 element count overflow"))?;
        if bytes.len() != expect_bytes {
            bail!(
                "{name}: expected {expect_elems} I32 elems ({expect_bytes} B), got {} B",
                bytes.len()
            );
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            bail!("{name}: I32 data is not 4-byte aligned");
        }
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i32, expect_elems) })
    }

    /// Get an F32 tensor as a `&[f32]` slice without element-count validation.
    ///
    /// Only used when the caller already knows the shape (e.g. from `TensorInfo`).
    pub fn f32_slice(&self, name: &str) -> Result<&[f32]> {
        let info = self.tensor_info(name).with_context(|| name.to_string())?;
        check_dtype(info, name, GgmlType::F32)?;
        let bytes = self.tensor_bytes(name)?;
        if !bytes.len().is_multiple_of(4) {
            bail!("{name}: F32 bytes {} not multiple of 4", bytes.len());
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            bail!("{name}: F32 data is not 4-byte aligned");
        }
        let elems = bytes.len() / 4;
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, elems) })
    }
}

fn check_dtype(info: &TensorInfo, name: &str, expected: GgmlType) -> Result<()> {
    if info.dtype != expected {
        bail!(
            "{name}: expected GGML type {expected:?}, got {:?}",
            info.dtype
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn make_info(dtype: GgmlType) -> TensorInfo {
        TensorInfo {
            name: "x".to_string(),
            dtype,
            dims: vec![4, 1],
            offset: 0,
        }
    }

    #[test]
    fn check_dtype_matches() {
        let info = make_info(GgmlType::F32);
        assert!(check_dtype(&info, "x", GgmlType::F32).is_ok());
    }

    #[test]
    fn check_dtype_mismatch_errors_with_name_and_types() {
        let info = make_info(GgmlType::F16);
        let err = check_dtype(&info, "blk.0.foo", GgmlType::Q8_0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("blk.0.foo"));
        assert!(msg.contains("Q8_0"));
        assert!(msg.contains("F16"));
    }

    fn write_minimal_gguf(path: &std::path::Path) {
        let mut buf: Vec<u8> = Vec::new();
        let u32le = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let u64le = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
        let strle = |buf: &mut Vec<u8>, s: &str| {
            u64le(buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        };
        let kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            strle(buf, k);
            u32le(buf, 4);
            u32le(buf, v);
        };
        let kv_arr_string = |buf: &mut Vec<u8>, k: &str, values: &[String]| {
            strle(buf, k);
            u32le(buf, 9);
            u32le(buf, 8);
            u64le(buf, values.len() as u64);
            for v in values {
                strle(buf, v);
            }
        };
        u32le(&mut buf, crate::gguf::GGUF_MAGIC);
        u32le(&mut buf, 3);
        u64le(&mut buf, 0);
        let tokens: Vec<String> = (0u8..=255).map(|b| format!("<0x{b:02X}>")).collect();
        u64le(&mut buf, 7);
        kv_u32(&mut buf, "llama.vocab_size", 256);
        kv_u32(&mut buf, "llama.embedding_length", 16);
        kv_u32(&mut buf, "llama.attention.head_count", 4);
        kv_u32(&mut buf, "llama.attention.head_count_kv", 4);
        kv_u32(&mut buf, "llama.block_count", 2);
        kv_u32(&mut buf, "llama.feed_forward_length", 32);
        kv_arr_string(&mut buf, "tokenizer.ggml.tokens", &tokens);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn open_map() -> WeightMap {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ds4-weights-test-{}-{}.gguf",
            std::process::id(),
            seq,
        ));
        write_minimal_gguf(&path);
        let map = WeightMap::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        map
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn missing_tensor_q8_0_errors() {
        let m = open_map();
        let err = m.q8_0("does.not.exist").unwrap_err();
        assert!(err.to_string().contains("does.not.exist"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn missing_tensor_f16_errors() {
        let m = open_map();
        let err = m.f16("missing.f16").unwrap_err();
        assert!(err.to_string().contains("missing.f16"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn missing_tensor_f32_1d_errors() {
        let m = open_map();
        let err = m.f32_1d("missing.f32", 4).unwrap_err();
        assert!(err.to_string().contains("missing.f32"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn missing_tensor_i32_1d_errors() {
        let m = open_map();
        let err = m.i32_1d("missing.i32", 4).unwrap_err();
        assert!(err.to_string().contains("missing.i32"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn missing_tensor_f32_slice_errors() {
        let m = open_map();
        let err = m.f32_slice("missing.f32s").unwrap_err();
        assert!(err.to_string().contains("missing.f32s"));
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn tensor_info_returns_none_for_missing() {
        let m = open_map();
        assert!(m.tensor_info("nope").is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore = "uses mmap + filesystem")]
    fn tensor_names_empty_for_minimal_gguf() {
        let m = open_map();
        assert_eq!(m.tensor_names().len(), 0);
    }
}
