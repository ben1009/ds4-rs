/// Metal shader preamble — defines macros, enums, and structs used by all kernels.
/// This matches the preamble in ds4_metal.m (lines 1136-1178).
#[cfg(target_os = "macos")]
const PREAMBLE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define M_PI_F 3.14159265358979323846f

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

"#;

/// Embed all MSL shader source files at compile time.
#[cfg(target_os = "macos")]
mod embedded {
    use super::PREAMBLE;

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
            PREAMBLE,
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
