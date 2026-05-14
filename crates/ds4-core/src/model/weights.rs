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
