
#[cfg(target_os = "macos")]
mod inner {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

    pub struct MetalBuffer {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        size: usize,
    }

    impl MetalBuffer {
        pub fn alloc(
            device: &ProtocolObject<dyn MTLDevice>,
            size: usize,
            label: &str,
        ) -> Result<Self> {
            let buffer = device
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| anyhow::anyhow!("Failed to allocate {size} byte buffer"))?;
            if !label.is_empty() {
                buffer.setLabel(Some(&objc2_foundation::NSString::from_str(label)));
            }
            Ok(Self {
                buffer,
                offset: 0,
                size,
            })
        }

        pub fn from_slice<T: Copy>(
            device: &ProtocolObject<dyn MTLDevice>,
            data: &[T],
            label: &str,
        ) -> Result<Self> {
            let byte_len = std::mem::size_of_val(data);
            let ptr = data.as_ptr() as *const u8;
            let buffer = device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| anyhow::anyhow!("Failed to allocate {byte_len} byte buffer"))?;
            if !label.is_empty() {
                buffer.setLabel(Some(&objc2_foundation::NSString::from_str(label)));
            }
            unsafe {
                let dst = buffer.contents() as *mut u8;
                std::ptr::copy_nonoverlapping(ptr, dst, byte_len);
            }
            Ok(Self {
                buffer,
                offset: 0,
                size: byte_len,
            })
        }

        pub unsafe fn from_mmap_no_copy(
            device: &ProtocolObject<dyn MTLDevice>,
            ptr: *mut u8,
            size: usize,
            label: &str,
        ) -> Result<Self> {
            let buffer = device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    size,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or_else(|| anyhow::anyhow!("Failed to create no-copy buffer"))?;
            if !label.is_empty() {
                buffer.setLabel(Some(&objc2_foundation::NSString::from_str(label)));
            }
            Ok(Self {
                buffer,
                offset: 0,
                size,
            })
        }

        pub fn view(&self, offset: usize, size: usize) -> Self {
            debug_assert!(offset + size <= self.size);
            Self {
                buffer: self.buffer.clone(),
                offset: self.offset + offset,
                size,
            }
        }

        pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
            &self.buffer
        }

        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn size(&self) -> usize {
            self.size
        }

        pub fn write(&self, offset: usize, data: &[u8]) {
            debug_assert!(offset + data.len() <= self.size);
            unsafe {
                let base = (self.buffer.contents() as *mut u8).add(self.offset + offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), base, data.len());
            }
        }

        pub fn read(&self, offset: usize, dest: &mut [u8]) {
            debug_assert!(offset + dest.len() <= self.size);
            unsafe {
                let base = (self.buffer.contents() as *const u8).add(self.offset + offset);
                std::ptr::copy_nonoverlapping(base, dest.as_mut_ptr(), dest.len());
            }
        }

        pub unsafe fn contents_mut(&self) -> *mut u8 {
            (self.buffer.contents() as *mut u8).add(self.offset)
        }

        pub fn contents(&self) -> *const u8 {
            unsafe { (self.buffer.contents() as *const u8).add(self.offset) }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod inner {
    

    /// Stub MetalBuffer for non-macOS platforms.
    pub struct MetalBuffer {
        offset: usize,
        size: usize,
    }

    impl MetalBuffer {
        pub fn offset(&self) -> usize {
            self.offset
        }

        pub fn size(&self) -> usize {
            self.size
        }

        pub fn view(&self, offset: usize, size: usize) -> Self {
            Self {
                offset: self.offset + offset,
                size,
            }
        }
    }
}

pub use inner::MetalBuffer;
