use anyhow::{bail, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF value types.
#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn to_string_val(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn to_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn to_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// GGML tensor type (quantization format).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            _ => None,
        }
    }
}

/// Number of elements per quantization block for each GGML type.
fn ggml_blck_size(dtype: GgmlType) -> usize {
    match dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::I8 | GgmlType::I16 | GgmlType::I32 => 1,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
        GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
        GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q8_K => 256,
        GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS
        | GgmlType::IQ1_S
        | GgmlType::IQ3_S
        | GgmlType::IQ2_S
        | GgmlType::IQ4_XS => 256,
        GgmlType::IQ4_NL => 32,
    }
}

/// Byte size of one quantization block.
fn ggml_type_size(dtype: GgmlType) -> usize {
    match dtype {
        GgmlType::F32 => 4,
        GgmlType::F16 => 2,
        GgmlType::Q4_0 => 18,
        GgmlType::Q4_1 => 20,
        GgmlType::Q5_0 => 22,
        GgmlType::Q5_1 => 24,
        GgmlType::Q8_0 => 34,
        GgmlType::Q8_1 => 36,
        GgmlType::Q2_K => 84,
        GgmlType::Q3_K => 110,
        GgmlType::Q4_K => 144,
        GgmlType::Q5_K => 176,
        GgmlType::Q6_K => 210,
        GgmlType::Q8_K => 304,
        GgmlType::IQ2_XXS => 66,
        GgmlType::IQ2_XS => 64,
        GgmlType::IQ3_XXS => 76,
        GgmlType::IQ1_S => 28,
        GgmlType::IQ4_NL => 18,
        GgmlType::IQ3_S => 84,
        GgmlType::IQ2_S => 64,
        GgmlType::IQ4_XS => 64,
        GgmlType::I8 => 1,
        GgmlType::I16 => 2,
        GgmlType::I32 => 4,
    }
}

/// Compute byte size for a tensor with the given element count and dtype.
/// For quantized types this accounts for block structure.
fn ggml_tensor_nbytes(elem_count: usize, dtype: GgmlType) -> usize {
    let blck = ggml_blck_size(dtype);
    let n_blocks = elem_count.div_ceil(blck);
    n_blocks * ggml_type_size(dtype)
}

/// Information about a tensor in the GGUF file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: GgmlType,
    pub dims: Vec<u64>,
    pub offset: u64,
}

/// Parsed GGUF file content.
pub struct GgufContent {
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: u64,
    pub file_len: u64,
}

impl GgufContent {
    /// Parse a GGUF file from a byte slice (the mmap).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);

        // Read magic
        let magic = read_u32(&mut cursor)?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: 0x{magic:08x}");
        }

        // Read version
        let version = read_u32(&mut cursor)?;
        if !(2..=3).contains(&version) {
            bail!("Unsupported GGUF version: {version}");
        }

        // Read tensor count and metadata KV count
        let tensor_count = read_u64(&mut cursor)? as usize;
        let metadata_kv_count = read_u64(&mut cursor)? as usize;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value_type = read_u32(&mut cursor)?;
            let value = read_value(&mut cursor, value_type)?;
            metadata.insert(key, value);
        }

        // Read alignment from metadata (default 32)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.to_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);

        // Parse tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut cursor)?);
            }
            let dtype_val = read_u32(&mut cursor)?;
            let dtype = GgmlType::from_u32(dtype_val)
                .ok_or_else(|| anyhow::anyhow!("Unknown GGML type: {dtype_val}"))?;
            let offset = read_u64(&mut cursor)?;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dtype,
                    dims,
                    offset,
                },
            );
        }

        // Align to the file's alignment (from metadata, default 32)
        let pos = cursor.position();
        let data_offset = pos.div_ceil(alignment) * alignment;

        let file_len = bytes.len() as u64;

        Ok(Self {
            metadata,
            tensors,
            data_offset,
            file_len,
        })
    }
}

/// Memory-mapped GGUF model file.
pub struct GgufMmap {
    pub content: GgufContent,
    pub mmap: Mmap,
}

impl GgufMmap {
    /// Open and memory-map a GGUF file. Uses mmap directly — no RAM copy.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we rely on the file not being modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        let content = GgufContent::parse(&mmap)?;
        Ok(Self { content, mmap })
    }

    /// Get raw bytes for a tensor by name.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .content
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found"))?;

        let start = (self.content.data_offset + info.offset) as usize;
        let elem_count = info
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' dimensions overflow"))?;
        let size = ggml_tensor_nbytes(elem_count, info.dtype);
        let end = start + size;

        if end > self.mmap.len() {
            bail!("Tensor '{name}' extends past end of file");
        }

        Ok(&self.mmap[start..end])
    }
}

fn read_u32(r: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(r: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u64(r)? as usize;
    if len > 1024 * 1024 {
        bail!("String too long: {len} bytes");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value(r: &mut Cursor<&[u8]>, value_type: u32) -> Result<Value> {
    match value_type {
        0 => Ok(Value::U8({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0]
        })),
        1 => Ok(Value::I8({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] as i8
        })),
        2 => Ok(Value::U16({
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b)
        })),
        3 => Ok(Value::I16({
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            i16::from_le_bytes(b)
        })),
        4 => Ok(Value::U32(read_u32(r)?)),
        5 => Ok(Value::I32(read_u32(r)? as i32)),
        6 => Ok(Value::U64(read_u64(r)?)),
        7 => Ok(Value::I64(read_u64(r)? as i64)),
        8 => Ok(Value::F32({
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            f32::from_le_bytes(b)
        })),
        9 => Ok(Value::F64({
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            f64::from_le_bytes(b)
        })),
        10 => Ok(Value::Bool({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] != 0
        })),
        11 => Ok(Value::String(read_string(r)?)),
        12 => {
            let arr_type = read_u32(r)?;
            let arr_len = read_u64(r)? as usize;
            if arr_len > 1024 * 1024 {
                bail!("Array too long: {arr_len} elements");
            }
            let mut arr = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                arr.push(read_value(r, arr_type)?);
            }
            Ok(Value::Array(arr))
        }
        _ => bail!("Unknown GGUF value type: {value_type}"),
    }
}
