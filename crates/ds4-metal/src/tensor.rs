use anyhow::Result;

use crate::context::MetalContext;

/// Data type for GPU tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F16,
    I32,
    U32,
    Q8_0,
    Q2_K,
    Q4_K,
    IQ2_XXS,
}

impl DType {
    /// Number of elements per quantization block (1 for non-quantized types).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::I32 | Self::U32 => 1,
            Self::Q8_0 => 32,
            Self::Q2_K | Self::Q4_K | Self::IQ2_XXS => 256,
        }
    }

    /// Byte size of one quantization block.
    pub fn block_type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::Q8_0 => 34,
            Self::Q2_K => 84,
            Self::Q4_K => 144,
            Self::IQ2_XXS => 66,
        }
    }
}

/// Shape of a GPU tensor.
#[derive(Clone, Debug)]
pub struct TensorShape {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl TensorShape {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        let numel = self.numel();
        numel.div_ceil(self.dtype.block_size()) * self.dtype.block_type_size()
    }
}

/// A tensor on the GPU. On macOS this holds a MetalBuffer; on other platforms it's a stub.
pub struct GpuTensor {
    #[cfg(target_os = "macos")]
    buffer: crate::buffer::MetalBuffer,
    shape: TensorShape,
}

impl GpuTensor {
    pub fn alloc(ctx: &MetalContext, shape: TensorShape, label: &str) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let buffer = crate::buffer::MetalBuffer::alloc(ctx.device(), shape.nbytes(), label)?;
            Ok(Self { buffer, shape })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ctx, label);
            Ok(Self { shape })
        }
    }

    pub fn from_slice<T: Copy>(
        ctx: &MetalContext,
        data: &[T],
        shape: TensorShape,
        label: &str,
    ) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let buffer = crate::buffer::MetalBuffer::from_slice(ctx.device(), data, label)?;
            Ok(Self { buffer, shape })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ctx, data, label);
            Ok(Self { shape })
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn nbytes(&self) -> usize {
        self.shape.nbytes()
    }

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    #[cfg(target_os = "macos")]
    pub fn buffer(&self) -> &crate::buffer::MetalBuffer {
        &self.buffer
    }
}
