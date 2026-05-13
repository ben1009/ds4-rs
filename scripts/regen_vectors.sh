#!/usr/bin/env bash
# Regenerate cross-reference test vectors committed under
# crates/ds4-core/tests/vectors/.
#
# Usage:
#     scripts/regen_vectors.sh <op>
#
# Ops:
#     q8_0        Q8_0 dequant + matmul_row reference vectors
#     rms_norm    RMSNorm (weighted + no-weight) reference vectors
#
# After regeneration, update crates/ds4-core/tests/vectors/manifest.toml
# with the new SHA-256 sums:
#     sha256sum crates/ds4-core/tests/vectors/*.bin
# then run `cargo test -p ds4-core --test manifest` to check.
#
# The regen step is intentionally NOT in CI. Vectors are expected to stay
# frozen across commits; a manifest update must appear as a reviewed diff.
#
# Future ops that need cross-reference against antirez/ds4 (softmax sums,
# attention outputs, MoE routing) will add a subroutine that clones the
# upstream C build, applies scripts/patches/0001-disable-fp8-kv.patch to
# get bit-equal outputs, runs a dump binary, and writes to the same
# tests/vectors/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <op>

Available ops:
  q8_0        Regenerate Q8_0 dequant + matmul reference vectors.
  rms_norm    Regenerate RMSNorm (weighted + no-weight) reference vectors.
  rope        Regenerate partial RoPE + YaRN reference vectors.
EOF
    exit 1
}

summary() {
    echo
    echo "Done. SHAs:"
    ( cd "$REPO_ROOT/crates/ds4-core/tests/vectors" && sha256sum "$@" )
    echo
    echo "If they differ from the previous values, update"
    echo "  crates/ds4-core/tests/vectors/manifest.toml"
    echo "and run"
    echo "  cargo test -p ds4-core --test manifest"
}

regen_q8_0() {
    # Q8_0 dequant is pure block arithmetic (f16 scale * i8 quants) with a
    # fixed byte layout. No floating-point accumulation ordering to worry
    # about, so a Rust-generated vector is authoritative here.
    cd "$REPO_ROOT"
    cargo run --quiet -p ds4-core --example gen_vectors_q8_0
    summary q8_0_dequant.bin q8_0_matmul.bin
}

regen_rms_norm() {
    # RMSNorm uses a fixed left-to-right reduction in our impl, so a
    # Rust-generated vector is reproducible across platforms for the same
    # inputs. No C-side dump needed for this op.
    cd "$REPO_ROOT"
    cargo run --quiet -p ds4-core --example gen_vectors_rms_norm
    summary rms_norm.bin rms_norm_no_weight.bin
}

regen_rope() {
    # RoPE uses cos/sin on integer-derived inputs. The committed vectors are
    # whatever the Rust impl produces on the regen host; the manifest SHA
    # check enforces byte-stability on every CI run (not a cross-platform
    # bit-exact guarantee — libm trig can round differently on non-x86).
    # Regeneration stays manual, so any drift surfaces as a reviewed diff.
    cd "$REPO_ROOT"
    cargo run --quiet -p ds4-core --example gen_vectors_rope
    summary rope_plain_pos127.bin rope_yarn_pos0.bin rope_yarn_pos127.bin rope_yarn_inverse.bin
}

main() {
    if [ $# -ne 1 ]; then
        usage
    fi

    case "$1" in
        q8_0)
            regen_q8_0
            ;;
        rms_norm)
            regen_rms_norm
            ;;
        rope)
            regen_rope
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "error: unknown op '$1'" >&2
            usage
            ;;
    esac
}

main "$@"
