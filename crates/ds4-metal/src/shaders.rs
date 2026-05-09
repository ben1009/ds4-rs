/// Embed all MSL shader source files at compile time.
#[cfg(target_os = "macos")]
mod embedded {
    const DSV4_HC: &str = include_str!("../../../metal/dsv4_hc.metal");
    const DSV4_KV: &str = include_str!("../../../metal/dsv4_kv.metal");
    const DSV4_MISC: &str = include_str!("../../../metal/dsv4_misc.metal");
    const DSV4_ROPE: &str = include_str!("../../../metal/dsv4_rope.metal");
    const FLASH_ATTN: &str = include_str!("../../../metal/flash_attn.metal");
    const MOE: &str = include_str!("../../../metal/moe.metal");
    const DENSE: &str = include_str!("../../../metal/dense.metal");
    const NORM: &str = include_str!("../../../metal/norm.metal");
    const ARGSORT: &str = include_str!("../../../metal/argsort.metal");
    const BIN: &str = include_str!("../../../metal/bin.metal");
    const CONCAT: &str = include_str!("../../../metal/concat.metal");
    const CPY: &str = include_str!("../../../metal/cpy.metal");
    const GET_ROWS: &str = include_str!("../../../metal/get_rows.metal");
    const GLU: &str = include_str!("../../../metal/glu.metal");
    const REPEAT: &str = include_str!("../../../metal/repeat.metal");
    const SET_ROWS: &str = include_str!("../../../metal/set_rows.metal");
    const SOFTMAX: &str = include_str!("../../../metal/softmax.metal");
    const SUM_ROWS: &str = include_str!("../../../metal/sum_rows.metal");
    const UNARY: &str = include_str!("../../../metal/unary.metal");

    pub fn combined_shader_source() -> String {
        [
            DSV4_HC, DSV4_KV, DSV4_MISC, DSV4_ROPE, FLASH_ATTN, MOE, DENSE, NORM,
            ARGSORT, BIN, CONCAT, CPY, GET_ROWS, GLU, REPEAT, SET_ROWS, SOFTMAX,
            SUM_ROWS, UNARY,
        ]
        .join("\n")
    }
}

#[cfg(target_os = "macos")]
pub use embedded::combined_shader_source;

#[cfg(not(target_os = "macos"))]
pub fn combined_shader_source() -> String {
    String::new()
}

/// Kernel function names to pre-compile into pipelines.
/// Populated as kernels are implemented.
pub const KERNEL_NAMES: &[&str] = &[];
