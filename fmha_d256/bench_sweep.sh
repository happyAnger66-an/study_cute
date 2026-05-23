#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# bench_sweep.sh - automated benchmark sweep for fmha_d256/fmha_d256.py
#
# Runs the CUTLASS-official Blackwell d=256 mixed-input FMHA prefill kernel
# across a matrix of (B, H_q, H_k, S, causal) configurations, parses the
# "[fmha_d256] summary: ..." line, and produces both:
#   1) a machine-readable CSV (default: fmha_d256/bench_sweep.csv)
#   2) a human-readable markdown table to stdout
#
# Usage:
#   bash fmha_d256/bench_sweep.sh                  # default 8-case sweep
#   bash fmha_d256/bench_sweep.sh --quick          # 3-case smoke
#   bash fmha_d256/bench_sweep.sh --full           # 16-case extended sweep
#   bash fmha_d256/bench_sweep.sh --no-ref-check   # skip torch ref check (faster)
#   bash fmha_d256/bench_sweep.sh --out my.csv     # custom CSV path
#   bash fmha_d256/bench_sweep.sh --timeout 300    # per-case timeout (sec)
#   bash fmha_d256/bench_sweep.sh --warmup 50 --iters 500   # tighter timing
#
# Each case row prints PASS / REF_FAIL / FAIL_rc=N / FAIL_no_summary so a hang
# or accuracy regression is immediately visible (timeout in particular is
# treated as failure, not silently dropped).
# -----------------------------------------------------------------------------

set -uo pipefail

# ----------------------------- Argument parsing ------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_CSV="${SCRIPT_DIR}/bench_sweep.csv"
MODE="default"
SKIP_REF_CHECK=0
TIMEOUT_SEC=180
WARMUP=10
ITERS=100

usage() {
    sed -n '2,22p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)           OUT_CSV="$2"; shift 2;;
        --quick)         MODE="quick"; shift;;
        --full)          MODE="full"; shift;;
        --no-ref-check)  SKIP_REF_CHECK=1; shift;;
        --timeout)       TIMEOUT_SEC="$2"; shift 2;;
        --warmup)        WARMUP="$2"; shift 2;;
        --iters)         ITERS="$2"; shift 2;;
        -h|--help)       usage;;
        *) echo "Unknown arg: $1" >&2; echo "Try --help" >&2; exit 1;;
    esac
done

# ----------------------------- Sweep matrices --------------------------------
# Each entry is: "B H_q H_k S causal"   (D is always 256)

declare -a CASES_QUICK=(
    "1  8  8  1024  0"
    "1  8  8  2048  0"
    "1  32 8  1024  0"
)

declare -a CASES_DEFAULT=(
    "1  8  8  256   0"   # tiny — undersubscribed, exposes scheduling overhead
    "1  8  8  1024  0"   # mid baseline
    "1  8  8  1024  1"   # mid baseline + causal
    "1  8  8  2048  0"   # large, single batch
    "1  8  8  4096  0"   # very large, single batch (close to SOL)
    "1  32 8  1024  0"   # GQA Llama-3-ish, mid
    "1  32 8  2048  0"   # GQA Llama-3-ish, large
    "4  8  8  1024  0"   # batch sweep, mid
)

declare -a CASES_FULL=(
    "1  8  8  256   0"
    "1  8  8  256   1"
    "1  8  8  512   0"
    "1  8  8  512   1"
    "1  8  8  1024  0"
    "1  8  8  1024  1"
    "1  8  8  2048  0"
    "1  8  8  2048  1"
    "1  8  8  4096  0"
    "1  32 8  1024  0"
    "1  32 8  1024  1"
    "1  32 8  2048  0"
    "1  32 8  4096  0"
    "4  8  8  1024  0"
    "4  8  8  2048  0"
    "8  8  8  1024  0"
)

case "$MODE" in
    quick)   CASES=("${CASES_QUICK[@]}");;
    default) CASES=("${CASES_DEFAULT[@]}");;
    full)    CASES=("${CASES_FULL[@]}");;
esac

# ----------------------------- Setup -----------------------------------------

NUM_CASES=${#CASES[@]}

# Honor user-set CUTE_DSL_ARCH; otherwise the shim auto-detects.
ARCH_HINT="${CUTE_DSL_ARCH:-<auto>}"

echo "==============================================================="
echo " fmha_d256 benchmark sweep"
echo "---------------------------------------------------------------"
echo "  mode             : $MODE  ($NUM_CASES cases)"
echo "  CUTE_DSL_ARCH    : $ARCH_HINT"
echo "  warmup_iterations: $WARMUP"
echo "  iterations       : $ITERS"
echo "  ref check        : $([[ $SKIP_REF_CHECK -eq 1 ]] && echo skipped || echo enabled)"
echo "  per-case timeout : ${TIMEOUT_SEC}s"
echo "  output CSV       : $OUT_CSV"
echo "==============================================================="

# Make sure output dir exists.
mkdir -p "$(dirname "$OUT_CSV")"

CSV_HEADER="case_id,B,H_q,H_k,S_q,S_k,D,is_causal,is_persistent,warmup,iters,latency_us,tflops,io_bw_gbps,status"
echo "$CSV_HEADER" > "$OUT_CSV"

PASS_COUNT=0
FAIL_COUNT=0

# ----------------------------- Main loop -------------------------------------

for i in "${!CASES[@]}"; do
    read -r B H_q H_k S causal <<< "${CASES[$i]}"
    case_id=$((i + 1))

    # Pretty case banner.
    causal_tag=$([[ $causal -eq 1 ]] && echo causal || echo non-causal)
    echo
    echo "---------------------------------------------------------------"
    printf "Case %2d/%d  B=%-2d H_q=%-2d H_k=%-2d S=%-4d D=256 %s\n" \
        "$case_id" "$NUM_CASES" "$B" "$H_q" "$H_k" "$S" "$causal_tag"
    echo "---------------------------------------------------------------"

    EXTRA_ARGS=()
    [[ $causal -eq 1 ]]         && EXTRA_ARGS+=(--is_causal)
    [[ $SKIP_REF_CHECK -eq 1 ]] && EXTRA_ARGS+=(--skip_ref_check)

    LOG=$(mktemp -t fmha_d256_bench.XXXXXX.log)

    # Run the kernel under timeout. tee so user sees progress live + we keep
    # a log to grep. Capture python's exit code via PIPESTATUS.
    timeout --foreground --signal=KILL "${TIMEOUT_SEC}" \
        python3 "$REPO_ROOT/fmha_d256/fmha_d256.py" \
            --q_shape "$B,$H_q,$S,256" \
            --k_shape "$B,$H_k,$S,256" \
            --is_persistent \
            --warmup_iterations "$WARMUP" --iterations "$ITERS" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}

    # ----------------------- Parse + classify result -----------------------
    lat=""; tflops=""; bw=""; status=""

    if [[ $rc -ne 0 ]]; then
        if [[ $rc -eq 124 || $rc -eq 137 ]]; then
            status="FAIL_TIMEOUT"
        else
            status="FAIL_rc=$rc"
        fi
    else
        summary_line=$(grep -E '^\[fmha_d256\] summary:' "$LOG" | tail -n1 || true)
        if [[ -z "$summary_line" ]]; then
            status="FAIL_no_summary"
        else
            lat=$(   sed -n 's/.*latency=\([0-9.]\+\) us.*/\1/p'  <<< "$summary_line")
            tflops=$(sed -n 's/.*tflops=\([0-9.]\+\).*/\1/p'      <<< "$summary_line")
            bw=$(    sed -n 's/.*io_bw=\([0-9.]\+\) GB\/s.*/\1/p' <<< "$summary_line")

            if [[ $SKIP_REF_CHECK -eq 0 ]] && ! grep -q "Results verified successfully" "$LOG"; then
                status="REF_FAIL"
            elif [[ -z "$lat" || -z "$tflops" || -z "$bw" ]]; then
                status="FAIL_parse"
            else
                status="PASS"
            fi
        fi
    fi

    rm -f "$LOG"

    # ----------------------- Persist CSV row -------------------------------
    echo "$case_id,$B,$H_q,$H_k,$S,$S,256,$causal,1,$WARMUP,$ITERS,$lat,$tflops,$bw,$status" \
        >> "$OUT_CSV"

    if [[ "$status" == "PASS" ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo
    echo "[bench_sweep] case $case_id -> $status   (lat=${lat:-N/A} us, tflops=${tflops:-N/A})"
done

# ----------------------------- Final summary ---------------------------------

echo
echo "==============================================================="
echo " Sweep complete: $PASS_COUNT PASS / $FAIL_COUNT FAIL out of $NUM_CASES"
echo " CSV written to: $OUT_CSV"
echo "==============================================================="
echo

# Pretty markdown table from CSV (using awk so we don't depend on `column`).
awk -F',' '
    BEGIN {
        OFS=" | "
    }
    NR == 1 {
        n = NF
        for (i = 1; i <= n; i++) hdr[i] = $i
        printf "| %s |\n",  hdr[1]
        # Build header from CSV header
        line = "| " hdr[1]
        for (i = 2; i <= n; i++) line = line " | " hdr[i]
        line = line " |"
        # Already printed first column variant; reset, use proper builder.
    }
' "$OUT_CSV" >/dev/null  # no-op, we use python for the pretty table below

# Use python for a clean markdown render — bash arithmetic for column widths
# is ugly and we already require python3 to run the kernel anyway.
python3 - "$OUT_CSV" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.reader(f))
if not rows:
    sys.exit(0)
hdr, body = rows[0], rows[1:]
# compute column widths
widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
def fmt(row):
    return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
print(fmt(hdr))
print("| " + " | ".join("-" * widths[i] for i in range(len(hdr))) + " |")
for row in body:
    print(fmt(row))
PYEOF

# Exit non-zero if any case failed so this is CI-friendly.
[[ $FAIL_COUNT -eq 0 ]]
