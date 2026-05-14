use std::{
    collections::HashMap,
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

use anyhow::{Result, bail};
use memmap2::Mmap;

pub(crate) const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_GGUF_STRING_LEN: usize = 1_048_576; // 1 MiB
const MAX_GGUF_ARRAY_LEN: usize = 1_048_576; // 1 Mi entries
const MAX_GGUF_TENSOR_COUNT: usize = 1_048_576; // 1 Mi tensors
const MAX_GGUF_METADATA_KV_COUNT: usize = 1_048_576; // 1 Mi KV pairs

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
            Self::U64(v) => u32::try_from(*v).ok(),
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
        GgmlType::Q8_K => 292,
        GgmlType::IQ2_XXS => 66,
        GgmlType::IQ2_XS => 74,
        GgmlType::IQ3_XXS => 98,
        GgmlType::IQ1_S => 34,
        GgmlType::IQ4_NL => 18,
        GgmlType::IQ3_S => 110,
        GgmlType::IQ2_S => 82,
        GgmlType::IQ4_XS => 136,
        GgmlType::I8 => 1,
        GgmlType::I16 => 2,
        GgmlType::I32 => 4,
    }
}

/// Compute byte size for a tensor with the given element count and dtype.
/// For quantized types this accounts for block structure.
fn ggml_tensor_nbytes(elem_count: usize, dtype: GgmlType) -> Option<usize> {
    let blck = ggml_blck_size(dtype);
    // Overflow-safe ceil-div (and MSRV-safe vs usize::div_ceil).
    let n_blocks = if elem_count == 0 {
        0
    } else {
        (elem_count - 1) / blck + 1
    };
    n_blocks.checked_mul(ggml_type_size(dtype))
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
#[derive(Debug)]
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
        let tensor_count = usize::try_from(read_u64(&mut cursor)?)
            .map_err(|_| anyhow::anyhow!("tensor_count overflows usize"))?;
        if tensor_count > MAX_GGUF_TENSOR_COUNT {
            bail!("Tensor count too large: {tensor_count}");
        }
        let metadata_kv_count = usize::try_from(read_u64(&mut cursor)?)
            .map_err(|_| anyhow::anyhow!("metadata_kv_count overflows usize"))?;
        if metadata_kv_count > MAX_GGUF_METADATA_KV_COUNT {
            bail!("Metadata KV count too large: {metadata_kv_count}");
        }

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
        if alignment == 0 {
            bail!("Invalid alignment: 0");
        }

        // Parse tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = read_u32(&mut cursor)? as usize;
            if n_dims > 8 {
                bail!("Tensor '{name}' has {n_dims} dims, expected <=8");
            }
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

        // Align to the file's alignment (from metadata, default 32).
        // Overflow-safe ceil-div form.
        let pos = cursor.position();
        let data_offset = if pos == 0 {
            0
        } else {
            ((pos - 1) / alignment + 1) * alignment
        };

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

        let start = usize::try_from(
            self.content
                .data_offset
                .checked_add(info.offset)
                .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' offset overflow"))?,
        )
        .map_err(|_| anyhow::anyhow!("Tensor '{name}' offset overflows usize"))?;
        let elem_count = info
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| {
                let d_usize = usize::try_from(d).ok()?;
                acc.checked_mul(d_usize)
            })
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' dimensions overflow"))?;
        let size = ggml_tensor_nbytes(elem_count, info.dtype)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' byte size overflow"))?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' byte range overflow"))?;

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
    let len = usize::try_from(read_u64(r)?)
        .map_err(|_| anyhow::anyhow!("String length overflows usize"))?;
    if len > MAX_GGUF_STRING_LEN {
        bail!("String too long: {len} bytes");
    }
    // Slice the underlying mmap directly — no copy into a temporary Vec.
    let start = usize::try_from(r.position())
        .map_err(|_| anyhow::anyhow!("Cursor position overflows usize"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("String byte range overflow"))?;
    let buf = r.get_ref();
    if end > buf.len() {
        bail!("String extends past end of file");
    }
    let s = String::from_utf8_lossy(&buf[start..end]).into_owned();
    r.set_position(end as u64);
    Ok(s)
}

fn read_value(r: &mut Cursor<&[u8]>, value_type: u32) -> Result<Value> {
    read_value_inner(r, value_type, 0)
}

const MAX_ARRAY_DEPTH: usize = 16;

fn read_value_inner(r: &mut Cursor<&[u8]>, value_type: u32, depth: usize) -> Result<Value> {
    if depth > MAX_ARRAY_DEPTH {
        bail!("Array nesting too deep (>{MAX_ARRAY_DEPTH})");
    }
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
        6 => Ok(Value::F32({
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            f32::from_le_bytes(b)
        })),
        7 => Ok(Value::Bool({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] != 0
        })),
        8 => Ok(Value::String(read_string(r)?)),
        9 => {
            let inner_type = read_u32(r)?;
            let inner_len = usize::try_from(read_u64(r)?)
                .map_err(|_| anyhow::anyhow!("Array length overflows usize"))?;
            if inner_len > MAX_GGUF_ARRAY_LEN {
                bail!("Array too long: {inner_len} elements");
            }
            let mut arr = Vec::with_capacity(inner_len);
            for _ in 0..inner_len {
                arr.push(read_value_inner(r, inner_type, depth + 1)?);
            }
            Ok(Value::Array(arr))
        }
        10 => Ok(Value::U64(read_u64(r)?)),
        11 => Ok(Value::I64(read_u64(r)? as i64)),
        12 => Ok(Value::F64({
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            f64::from_le_bytes(b)
        })),
        _ => bail!("Unknown GGUF value type: {value_type}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- synthetic GGUF builder -------------------------------------------------

    struct GgufBuilder {
        buf: Vec<u8>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }

        fn raw_u32(&mut self, v: u32) -> &mut Self {
            self.buf.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn raw_u64(&mut self, v: u64) -> &mut Self {
            self.buf.extend_from_slice(&v.to_le_bytes());
            self
        }

        fn raw_str(&mut self, s: &str) -> &mut Self {
            self.raw_u64(s.len() as u64);
            self.buf.extend_from_slice(s.as_bytes());
            self
        }

        fn header(&mut self, tensor_count: u64, kv_count: u64) -> &mut Self {
            self.raw_u32(GGUF_MAGIC);
            self.raw_u32(3);
            self.raw_u64(tensor_count);
            self.raw_u64(kv_count);
            self
        }

        fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
            self.raw_str(key);
            self.raw_u32(4); // type U32
            self.raw_u32(value);
            self
        }

        fn kv_u64(&mut self, key: &str, value: u64) -> &mut Self {
            self.raw_str(key);
            self.raw_u32(10); // type U64
            self.raw_u64(value);
            self
        }

        fn kv_string(&mut self, key: &str, value: &str) -> &mut Self {
            self.raw_str(key);
            self.raw_u32(8); // type String
            self.raw_str(value);
            self
        }

        fn kv_bool(&mut self, key: &str, value: bool) -> &mut Self {
            self.raw_str(key);
            self.raw_u32(7); // type Bool
            self.buf.push(u8::from(value));
            self
        }

        fn tensor_info(&mut self, name: &str, dtype: u32, dims: &[u64], offset: u64) -> &mut Self {
            self.raw_str(name);
            self.raw_u32(dims.len() as u32);
            for d in dims {
                self.raw_u64(*d);
            }
            self.raw_u32(dtype);
            self.raw_u64(offset);
            self
        }

        fn align_to(&mut self, alignment: u64) -> &mut Self {
            let pos = self.buf.len() as u64;
            let padded = if pos == 0 {
                0
            } else {
                ((pos - 1) / alignment + 1) * alignment
            };
            self.buf.resize(padded as usize, 0);
            self
        }

        fn raw_bytes(&mut self, b: &[u8]) -> &mut Self {
            self.buf.extend_from_slice(b);
            self
        }

        fn build(self) -> Vec<u8> {
            self.buf
        }
    }

    // ---- parse() tests ----------------------------------------------------------

    #[test]
    fn parse_rejects_bad_magic() {
        let mut b = GgufBuilder::new();
        b.raw_u32(0xDEADBEEF).raw_u32(3).raw_u64(0).raw_u64(0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Invalid GGUF magic"));
    }

    #[test]
    fn parse_rejects_unsupported_version() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(1).raw_u64(0).raw_u64(0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Unsupported GGUF version"));
    }

    #[test]
    fn parse_rejects_too_many_tensors() {
        let mut b = GgufBuilder::new();
        b.header((MAX_GGUF_TENSOR_COUNT + 1) as u64, 0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Tensor count too large"));
    }

    #[test]
    fn parse_rejects_too_many_kv() {
        let mut b = GgufBuilder::new();
        b.header(0, (MAX_GGUF_METADATA_KV_COUNT + 1) as u64);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Metadata KV count too large"));
    }

    #[test]
    fn parse_rejects_zero_alignment() {
        let mut b = GgufBuilder::new();
        b.header(0, 1).kv_u64("general.alignment", 0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Invalid alignment"));
    }

    #[test]
    fn parse_rejects_bad_dtype() {
        let mut b = GgufBuilder::new();
        b.header(1, 0).tensor_info("bad", 9999, &[4], 0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Unknown GGML type"));
    }

    #[test]
    fn parse_rejects_too_many_dims() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(1).raw_u64(0);
        b.raw_str("t").raw_u32(9); // 9 dims — over the 8 limit
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("dims, expected <=8"));
    }

    #[test]
    fn parse_rejects_oversized_string() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_u64((MAX_GGUF_STRING_LEN + 1) as u64); // key length
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("String too long"));
    }

    #[test]
    fn parse_rejects_string_past_eof() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_u64(100); // claim 100 bytes for the key, but only a few follow
        b.raw_bytes(b"abc");
        let err = GgufContent::parse(&b.build()).unwrap_err();
        // Either the range check or read_exact can fire depending on whether
        // enough bytes are present to even try the slice. Accept both.
        let msg = err.to_string();
        assert!(
            msg.contains("extends past end of file") || msg.contains("failed to fill whole buffer")
        );
    }

    #[test]
    fn parse_reads_all_value_types() {
        let mut b = GgufBuilder::new();
        b.header(0, 4)
            .kv_u32("u32", 42)
            .kv_u64("u64", 0xFFFF_FFFF_FFFF)
            .kv_string("str", "hello")
            .kv_bool("flag", true);

        let c = GgufContent::parse(&b.build()).unwrap();
        assert_eq!(c.metadata.get("u32").unwrap().to_u32(), Some(42));
        assert_eq!(
            c.metadata.get("u64").unwrap().to_u64(),
            Some(0xFFFF_FFFF_FFFF)
        );
        assert_eq!(
            c.metadata.get("str").unwrap().to_string_val(),
            Some("hello")
        );
        assert_eq!(c.metadata.get("flag").unwrap().to_bool(), Some(true));
    }

    #[test]
    fn parse_with_tensor_and_read_data() {
        // Build a GGUF with one F32 tensor of shape [4] = 16 bytes.
        let mut b = GgufBuilder::new();
        b.header(1, 1)
            .kv_u32("general.alignment", 32)
            .tensor_info("t", 0, &[4], 0);
        b.align_to(32);
        // 4 × f32 values
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            b.raw_bytes(&v.to_le_bytes());
        }
        let bytes = b.build();

        let c = GgufContent::parse(&bytes).unwrap();
        assert_eq!(c.tensors.len(), 1);
        let info = &c.tensors["t"];
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dims, vec![4]);
        assert_eq!(info.offset, 0);

        // Wire up a fake GgufMmap via the public tensor_data path.
        // GgufMmap wraps an owned Mmap; to test tensor_data we need a real file.
        // Easier: validate data_offset + range math manually.
        let start = c.data_offset as usize;
        let slice = &bytes[start..start + 16];
        let mut vs = [0f32; 4];
        for (i, v) in vs.iter_mut().enumerate() {
            *v = f32::from_le_bytes(slice[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        assert_eq!(vs, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn parse_respects_custom_alignment() {
        let mut b = GgufBuilder::new();
        b.header(0, 1).kv_u64("general.alignment", 64);
        let bytes = b.build();
        let c = GgufContent::parse(&bytes).unwrap();
        assert_eq!(c.data_offset % 64, 0);
        assert!(c.data_offset >= bytes.len() as u64);
        assert!(c.data_offset < bytes.len() as u64 + 64);
    }

    // ---- helper & value-accessor tests -----------------------------------------

    #[test]
    fn value_accessors() {
        assert_eq!(Value::U32(7).to_u32(), Some(7));
        assert_eq!(Value::U64(7).to_u32(), Some(7));
        assert_eq!(Value::U64(u64::MAX).to_u32(), None); // truncation rejected
        assert_eq!(Value::U32(7).to_u64(), Some(7));
        assert_eq!(Value::U64(9).to_u64(), Some(9));
        assert_eq!(Value::F32(1.5).to_f32(), Some(1.5));
        assert_eq!(Value::F64(1.5).to_f32(), Some(1.5));
        assert_eq!(Value::String("x".into()).to_string_val(), Some("x"));
        assert_eq!(Value::Bool(false).to_bool(), Some(false));
        assert!(Value::Array(vec![Value::U32(1)]).to_array().is_some());
        // Mismatched types return None.
        assert_eq!(Value::String("x".into()).to_u32(), None);
        assert_eq!(Value::U32(1).to_string_val(), None);
        assert_eq!(Value::U32(1).to_bool(), None);
        assert!(Value::U32(1).to_array().is_none());
    }

    #[test]
    fn ggml_type_roundtrip() {
        for v in 0..=26u32 {
            if let Some(t) = GgmlType::from_u32(v) {
                assert_eq!(t as u32, v);
            }
        }
        assert!(GgmlType::from_u32(4).is_none()); // unused slot
        assert!(GgmlType::from_u32(999).is_none());
    }

    #[test]
    fn ggml_tensor_nbytes_block_math() {
        // F32: 1 byte/elem * 4 bytes/block = 4 bytes/elem effectively.
        assert_eq!(ggml_tensor_nbytes(4, GgmlType::F32), Some(16));
        // Q8_0: 34 bytes per 32-element block.
        assert_eq!(ggml_tensor_nbytes(32, GgmlType::Q8_0), Some(34));
        assert_eq!(ggml_tensor_nbytes(64, GgmlType::Q8_0), Some(68));
        // Partial block rounds up.
        assert_eq!(ggml_tensor_nbytes(33, GgmlType::Q8_0), Some(68));
        // Zero stays zero.
        assert_eq!(ggml_tensor_nbytes(0, GgmlType::Q8_0), Some(0));
        // Q8_K block is 256 elements × 292 bytes.
        assert_eq!(ggml_tensor_nbytes(256, GgmlType::Q8_K), Some(292));
    }

    // ---- GgufMmap tests ---------------------------------------------------------

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn ggufmmap_open_and_tensor_data_roundtrip() {
        use std::{
            io::Write,
            sync::atomic::{AtomicU64, Ordering},
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);

        // Build a GGUF with two F32 tensors back-to-back.
        let mut b = GgufBuilder::new();
        b.header(2, 0)
            .tensor_info("a", 0, &[2], 0)
            .tensor_info("b", 0, &[3], 8); // a is 8 bytes, so b starts at offset 8
        b.align_to(DEFAULT_ALIGNMENT);
        for v in [10.0f32, 20.0] {
            b.raw_bytes(&v.to_le_bytes());
        }
        for v in [1.5f32, 2.5, 3.5] {
            b.raw_bytes(&v.to_le_bytes());
        }
        let bytes = b.build();

        // Write to a tempfile and open via GgufMmap.
        let path =
            std::env::temp_dir().join(format!("ds4-gguf-test-{}-{}.bin", std::process::id(), seq,));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
        }

        let gguf = GgufMmap::open(&path).unwrap();
        assert_eq!(gguf.content.tensors.len(), 2);

        let a = gguf.tensor_data("a").unwrap();
        assert_eq!(a.len(), 8);
        let a0 = f32::from_le_bytes(a[0..4].try_into().unwrap());
        assert_eq!(a0, 10.0);

        let b_bytes = gguf.tensor_data("b").unwrap();
        assert_eq!(b_bytes.len(), 12);

        // Unknown tensor should error.
        assert!(gguf.tensor_data("missing").is_err());

        let _ = std::fs::remove_file(&path);
    }

    // ---- additional parse() tests ----------------------------------------------

    #[test]
    fn parse_rejects_truncated_header() {
        // Only magic, no version/counts.
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC);
        assert!(GgufContent::parse(&b.build()).is_err());
    }

    #[test]
    fn parse_rejects_empty_buffer() {
        assert!(GgufContent::parse(&[]).is_err());
    }

    #[test]
    fn parse_rejects_truncated_metadata_value() {
        // Header claims 1 KV pair, but the file ends mid-value (key + type only).
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_str("k").raw_u32(4); // type U32, no payload
        assert!(GgufContent::parse(&b.build()).is_err());
    }

    #[test]
    fn parse_rejects_unknown_value_type() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_str("k").raw_u32(99); // bogus value type
        b.raw_u32(0); // some payload
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Unknown GGUF value type"));
    }

    #[test]
    fn parse_rejects_oversized_array() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_str("k").raw_u32(9); // Array
        b.raw_u32(4); // inner type U32
        b.raw_u64((MAX_GGUF_ARRAY_LEN + 1) as u64);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Array too long"));
    }

    #[test]
    fn parse_rejects_deeply_nested_arrays() {
        let mut b = GgufBuilder::new();
        b.raw_u32(GGUF_MAGIC).raw_u32(3).raw_u64(0).raw_u64(1);
        b.raw_str("k").raw_u32(9); // Array (outer)
        // 17 levels of nested arrays each of length 1, exceeding MAX_ARRAY_DEPTH=16.
        for _ in 0..17 {
            b.raw_u32(9); // inner type Array
            b.raw_u64(1); // length
        }
        b.raw_u32(4); // innermost type U32
        b.raw_u64(1);
        b.raw_u32(0);
        let err = GgufContent::parse(&b.build()).unwrap_err();
        assert!(err.to_string().contains("Array nesting too deep"));
    }

    #[test]
    fn parse_duplicate_keys_keeps_last() {
        let mut b = GgufBuilder::new();
        b.header(0, 2).kv_u32("dup", 1).kv_u32("dup", 2);
        let c = GgufContent::parse(&b.build()).unwrap();
        assert_eq!(c.metadata.get("dup").unwrap().to_u32(), Some(2));
    }

    #[test]
    fn parse_empty_string_value() {
        let mut b = GgufBuilder::new();
        b.header(0, 1).kv_string("k", "");
        let c = GgufContent::parse(&b.build()).unwrap();
        assert_eq!(c.metadata.get("k").unwrap().to_string_val(), Some(""));
    }

    #[test]
    fn parse_empty_array() {
        let mut b = GgufBuilder::new();
        b.header(0, 1);
        b.raw_str("a").raw_u32(9); // Array
        b.raw_u32(4); // inner U32
        b.raw_u64(0); // length 0
        let c = GgufContent::parse(&b.build()).unwrap();
        let arr = c.metadata.get("a").unwrap().to_array().unwrap();
        assert!(arr.is_empty());
    }

    #[test]
    fn parse_array_of_strings() {
        let mut b = GgufBuilder::new();
        b.header(0, 1);
        b.raw_str("toks").raw_u32(9);
        b.raw_u32(8); // inner String
        b.raw_u64(3);
        b.raw_str("a").raw_str("bb").raw_str("ccc");
        let c = GgufContent::parse(&b.build()).unwrap();
        let arr = c.metadata.get("toks").unwrap().to_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].to_string_val(), Some("a"));
        assert_eq!(arr[2].to_string_val(), Some("ccc"));
    }

    #[test]
    fn parse_all_scalar_value_types() {
        let mut b = GgufBuilder::new();
        b.header(0, 10);
        // U8
        b.raw_str("u8").raw_u32(0).buf.push(7);
        // I8
        b.raw_str("i8").raw_u32(1).buf.push((-3i8) as u8);
        // U16
        b.raw_str("u16").raw_u32(2);
        b.buf.extend_from_slice(&12345u16.to_le_bytes());
        // I16
        b.raw_str("i16").raw_u32(3);
        b.buf.extend_from_slice(&(-1234i16).to_le_bytes());
        // I32
        b.raw_str("i32").raw_u32(5);
        b.buf.extend_from_slice(&(-42i32).to_le_bytes());
        // F32
        b.raw_str("f32").raw_u32(6);
        b.buf.extend_from_slice(&1.5f32.to_le_bytes());
        // I64
        b.raw_str("i64").raw_u32(11);
        b.buf.extend_from_slice(&(-99i64).to_le_bytes());
        // F64
        b.raw_str("f64").raw_u32(12);
        b.buf.extend_from_slice(&2.5f64.to_le_bytes());
        // U32 (already covered, but for sanity)
        b.kv_u32("u32", 100);
        // U64
        b.kv_u64("u64", 1234567890);

        let c = GgufContent::parse(&b.build()).unwrap();
        assert!(matches!(c.metadata["u8"], Value::U8(7)));
        assert!(matches!(c.metadata["i8"], Value::I8(-3)));
        assert!(matches!(c.metadata["u16"], Value::U16(12345)));
        assert!(matches!(c.metadata["i16"], Value::I16(-1234)));
        assert!(matches!(c.metadata["i32"], Value::I32(-42)));
        assert!(matches!(c.metadata["f32"], Value::F32(v) if (v - 1.5).abs() < 1e-6));
        assert!(matches!(c.metadata["i64"], Value::I64(-99)));
        assert!(matches!(c.metadata["f64"], Value::F64(v) if (v - 2.5).abs() < 1e-9));
        assert_eq!(c.metadata["u32"].to_u32(), Some(100));
        assert_eq!(c.metadata["u64"].to_u64(), Some(1234567890));
    }

    #[test]
    fn parse_default_alignment_when_metadata_absent() {
        let mut b = GgufBuilder::new();
        b.header(0, 0);
        let bytes = b.build();
        let c = GgufContent::parse(&bytes).unwrap();
        assert_eq!(c.data_offset % DEFAULT_ALIGNMENT, 0);
        assert!(c.data_offset >= bytes.len() as u64);
    }

    #[test]
    fn parse_zero_position_yields_zero_data_offset() {
        // The cursor-position-zero branch is unreachable in normal parsing
        // (header always advances), but exercise the math via an exact-aligned
        // header whose end is at a multiple of alignment.
        let mut b = GgufBuilder::new();
        b.header(0, 1).kv_u64("general.alignment", 8);
        let bytes = b.build();
        let c = GgufContent::parse(&bytes).unwrap();
        assert_eq!(c.data_offset % 8, 0);
    }

    #[test]
    fn parse_tensor_with_multiple_dims() {
        let mut b = GgufBuilder::new();
        b.header(1, 0).tensor_info("m", 0, &[2, 3], 0);
        b.align_to(DEFAULT_ALIGNMENT);
        for v in 0..6 {
            b.raw_bytes(&(v as f32).to_le_bytes());
        }
        let c = GgufContent::parse(&b.build()).unwrap();
        let info = &c.tensors["m"];
        assert_eq!(info.dims, vec![2, 3]);
        assert_eq!(info.dtype, GgmlType::F32);
    }

    #[test]
    fn parse_tensor_zero_dims_is_accepted() {
        // A 0-dim tensor (scalar) — n_dims=0 and offset=0.
        let mut b = GgufBuilder::new();
        b.header(1, 0).tensor_info("scalar", 0, &[], 0);
        b.align_to(DEFAULT_ALIGNMENT);
        let c = GgufContent::parse(&b.build()).unwrap();
        let info = &c.tensors["scalar"];
        assert!(info.dims.is_empty());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "uses mmap + real filesystem, unsupported under miri isolation"
    )]
    fn ggufmmap_tensor_data_rejects_past_eof() {
        use std::{
            io::Write,
            sync::atomic::{AtomicU64, Ordering},
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);

        // Tensor claims 16 bytes of F32 data but we write only 8.
        let mut b = GgufBuilder::new();
        b.header(1, 0).tensor_info("t", 0, &[4], 0);
        b.align_to(DEFAULT_ALIGNMENT);
        b.raw_bytes(&1.0f32.to_le_bytes());
        b.raw_bytes(&2.0f32.to_le_bytes());
        let bytes = b.build();

        let path =
            std::env::temp_dir().join(format!("ds4-gguf-eof-{}-{}.bin", std::process::id(), seq,));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&bytes).unwrap();
        }
        let gguf = GgufMmap::open(&path).unwrap();
        let err = gguf.tensor_data("t").unwrap_err();
        assert!(err.to_string().contains("extends past end of file"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ggml_tensor_nbytes_for_k_quants() {
        // Q4_K is 144 bytes per 256-element block.
        assert_eq!(ggml_tensor_nbytes(256, GgmlType::Q4_K), Some(144));
        assert_eq!(ggml_tensor_nbytes(512, GgmlType::Q4_K), Some(288));
        // Partial block rounds up.
        assert_eq!(ggml_tensor_nbytes(257, GgmlType::Q4_K), Some(288));
        // F16 is 2 bytes per element.
        assert_eq!(ggml_tensor_nbytes(10, GgmlType::F16), Some(20));
    }
}
