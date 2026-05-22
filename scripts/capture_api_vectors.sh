#!/usr/bin/env bash
# Capture reference outputs from the DeepSeek API for test vector validation.
#
# Requires:
#   - DEEPSEEK_API_KEY env var
#   - curl, jq
#
# Usage:
#   scripts/capture_api_vectors.sh
#
# Optional env:
#   DS4_API_MODEL   Model ID (default: deepseek-v3)
#   DS4_API_URL     API base URL (default: https://api.deepseek.com)
#
# Output:
#   crates/ds4-core/tests/vectors/api/<name>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/crates/ds4-core/tests/vectors/api"

MODEL="${DS4_API_MODEL:-deepseek-v3}"
BASE_URL="${DS4_API_URL:-https://api.deepseek.com}"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "error: DEEPSEEK_API_KEY not set" >&2
    exit 1
fi

for cmd in curl jq; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "error: $cmd is required but not found" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

call_api() {
    local prompt="$1"
    local max_tokens="${2:-64}"
    local payload
    payload=$(jq -n \
        --arg model "$MODEL" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        '{
            model: $model,
            messages: [{"role": "user", "content": $prompt}],
            temperature: 0,
            max_tokens: $max_tokens,
            stream: false
        }')

    curl -sS --connect-timeout 10 --max-time 60 \
        "$BASE_URL/v1/chat/completions" \
        -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$payload"
}

save_reference() {
    local name="$1"
    local prompt="$2"
    local max_tokens="${3:-64}"

    echo -n "Capturing '$name'... " >&2

    local response
    response=$(call_api "$prompt" "$max_tokens")

    # Check for API errors
    local error
    error=$(echo "$response" | jq -r '.error.message // empty')
    if [ -n "$error" ]; then
        echo "FAILED: $error" >&2
        return 1
    fi

    # Check finish reason — reject partial/filter responses
    local finish_reason
    finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason // "unknown"')
    if [ "$finish_reason" != "stop" ]; then
        echo "FAILED: finish_reason=$finish_reason (expected 'stop')" >&2
        return 1
    fi

    # Check content length via jq (not wc -c, which counts the trailing newline jq adds)
    local text_len
    text_len=$(echo "$response" | jq '(.choices[0].message.content // "") | length')
    if [ "$text_len" -eq 0 ]; then
        echo "FAILED: empty response content" >&2
        return 1
    fi

    # Use jq --arg for prompt to avoid shell injection via string interpolation.
    echo "$response" | jq \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        '{
            prompt: $prompt,
            expected_text: (.choices[0].message.content // ""),
            model: .model,
            max_tokens: $max_tokens
        }' > "$OUT_DIR/$name.json"

    echo "OK (${text_len} chars)" >&2
}

echo "Model: $MODEL" >&2
echo "Output: $OUT_DIR/" >&2
echo >&2

save_reference "factual"       "What is the capital of France?"
save_reference "arithmetic"    "Compute 17 * 23 + 5."
save_reference "code_gen"      "Write a Rust function that returns the maximum of two integers. Output only the code, no explanation." 256
save_reference "chinese"       "用一句话解释什么是量子计算。"

echo >&2
echo "Done. Reference files:" >&2
ls -la "$OUT_DIR"/*.json >&2
