#[cfg(target_os = "macos")]
mod inner {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLCommandBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
    };

    use crate::buffer::MetalBuffer;

    pub struct ComputeEncoder<'a> {
        encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        _phantom: std::marker::PhantomData<&'a ()>,
    }

    impl<'a> ComputeEncoder<'a> {
        pub fn new(cmd_buffer: &'a ProtocolObject<dyn MTLCommandBuffer>) -> Self {
            let encoder = cmd_buffer
                .newComputeCommandEncoder()
                .expect("Failed to create compute encoder");
            Self {
                encoder,
                _phantom: std::marker::PhantomData,
            }
        }

        pub fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
            unsafe {
                self.encoder.setComputePipelineState(pipeline);
            }
        }

        pub fn set_buffer(&self, index: usize, buffer: &MetalBuffer) {
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(buffer.raw()),
                    buffer.offset() as u64,
                    index as u64,
                );
            }
        }

        pub fn set_bytes<T: Copy>(&self, index: usize, value: &T) {
            unsafe {
                let ptr = std::ptr::NonNull::from(value).cast();
                self.encoder.setBytes_length_atIndex(
                    ptr,
                    std::mem::size_of::<T>() as u64,
                    index as u64,
                );
            }
        }

        pub fn dispatch(
            &self,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            n: usize,
        ) {
            let (groups, threads) = linear_split(pipeline, n);
            unsafe {
                self.encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
            }
        }

        pub fn dispatch_2d(
            &self,
            pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
            width: usize,
            height: usize,
            threads_per_group_x: usize,
            threads_per_group_y: usize,
        ) {
            let groups = MTLSize {
                width: (width + threads_per_group_x - 1) / threads_per_group_x,
                height: (height + threads_per_group_y - 1) / threads_per_group_y,
                depth: 1,
            };
            let threads = MTLSize {
                width: threads_per_group_x,
                height: threads_per_group_y,
                depth: 1,
            };
            unsafe {
                self.encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
            }
        }

        pub fn end_encoding(&self) {
            unsafe {
                self.encoder.endEncoding();
            }
        }
    }

    pub fn linear_split(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        length: usize,
    ) -> (MTLSize, MTLSize) {
        let tpg = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let groups = (length + tpg - 1) / tpg;
        (
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tpg,
                height: 1,
                depth: 1,
            },
        )
    }
}

#[cfg(not(target_os = "macos"))]
mod inner {
    // Encoder stubs — no-op on non-macOS
}

#[cfg(target_os = "macos")]
pub use inner::{ComputeEncoder, linear_split};

#[macro_export]
macro_rules! set_params {
    ($encoder:expr, $start:expr, $($buf:expr),+ $(,)?) => {
        let mut _idx = $start;
        $(
            $encoder.set_buffer(_idx, &$buf);
            _idx += 1;
        )+
    };
}
