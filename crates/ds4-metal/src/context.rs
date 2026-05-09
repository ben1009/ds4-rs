use anyhow::Result;
use std::sync::Arc;

#[cfg(target_os = "macos")]
mod inner {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary,
    };
    use std::collections::HashMap;

    use crate::shaders;

    pub struct MetalContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        library: Retained<ProtocolObject<dyn MTLLibrary>>,
        pipelines:
            HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    }

    impl MetalContext {
        pub fn new() -> Result<Arc<Self>> {
            let device = objc2_metal::MTLCreateSystemDefaultDevice()
                .ok_or_else(|| anyhow::anyhow!("No Metal device available"))?;

            let queue = device
                .newCommandQueue()
                .ok_or_else(|| anyhow::anyhow!("Failed to create command queue"))?;

            let source = shaders::combined_shader_source();
            let library = device
                .newLibraryWithSource_options_error(
                    &objc2_foundation::NSString::from_str(&source),
                    None,
                )
                .map_err(|e| anyhow::anyhow!("Failed to compile shaders: {e}"))?;

            let mut pipelines = HashMap::new();
            for name in shaders::KERNEL_NAMES {
                let function = library
                    .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                    .ok_or_else(|| {
                        anyhow::anyhow!("Shader function '{name}' not found")
                    })?;
                let pipeline = device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create pipeline for '{name}': {e}")
                    })?;
                pipelines.insert(*name, pipeline);
            }

            tracing::info!("Metal initialized: compiled {} pipelines", pipelines.len());

            Ok(Arc::new(Self {
                device,
                queue,
                library,
                pipelines,
            }))
        }

        pub fn pipeline(
            &self,
            name: &str,
        ) -> Option<&Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
            self.pipelines.get(name)
        }

        pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
            &self.device
        }

        pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
            &self.queue
        }

        pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
            &self.library
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod inner {
    use super::*;

    /// Stub MetalContext for non-macOS platforms.
    pub struct MetalContext;

    impl MetalContext {
        pub fn new() -> Result<Arc<Self>> {
            tracing::warn!("Metal not available on this platform; using stub context");
            Ok(Arc::new(Self))
        }
    }
}

pub use inner::MetalContext;
