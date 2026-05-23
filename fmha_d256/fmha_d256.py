#!/usr/bin/env python3
"""study_cute thin entry shim for CUTLASS official d=256 FMHA prefill kernel.

This file is a minimal wrapper around the (otherwise verbatim) official source
in ``mixed_input_fmha_prefill_d256.py``. It exposes the most common CLI flags
and delegates to the official ``run()`` function.

Quick usage (Blackwell sm_100a, e.g. Jetson Thor):
    cd <study_cute root>
    python3 fmha_d256/fmha_d256.py \\
        --q_shape 1,8,256,256 --k_shape 1,8,256,256 \\
        --is_persistent --skip_ref_check

The kernel's input contract (taken from CUTLASS upstream):
    Q  dtype : BFloat16            shape : (B, H_q, S_q, D)   D=256
    K  dtype : Int8                shape : (B, H_k, S_k, D)
    V  dtype : Int8                shape : (B, H_k, S_k, D)
    O  dtype : BFloat16            shape : (B, H_q, S_q, D)
    scale_k dtype : BFloat16       shape : (B, H_k, S_k, D//scale_granularity)
    scale_v dtype : BFloat16       shape : (B, H_k, S_k, D//scale_granularity)

NOTE: KV is *INT8 quantized*. study_cute's existing LLM prefill runner uses
FP16 KV (un-quantized) and packed cumulative seqlens, which is a different
interface from this kernel. Integration with the LLM prefill loop is a
follow-up task; see fmha_d256/README.md for the migration path.
"""

import argparse
import math
import os
import sys

# Make ``import fmha_d256`` resolve to the sibling directory both when invoked
# as a script (``python3 fmha_d256/fmha_d256.py``) and via ``python3 -m``.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_CURRENT_DIR))

import torch

import cutlass
from cutlass.cute.typing import Float32

from fmha_d256.mixed_input_fmha_prefill_d256 import run


def _parse_comma_separated_ints(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CUTLASS official Blackwell d=256 mixed-input FMHA prefill "
            "kernel (BF16 Q / INT8 K,V + scale / BF16 O)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shapes
    parser.add_argument(
        "--q_shape", type=_parse_comma_separated_ints,
        default=(1, 8, 256, 256),
        help="Q shape (B, H_q, S_q, D). D MUST be 256.",
    )
    parser.add_argument(
        "--k_shape", type=_parse_comma_separated_ints,
        default=(1, 8, 256, 256),
        help="K shape (B, H_k, S_k, D). Same B & D as Q; H_q must be divisible by H_k.",
    )

    # Dtypes (defaults match upstream)
    parser.add_argument(
        "--q_dtype", type=cutlass.dtype, default=cutlass.BFloat16,
        help="Q dtype (BFloat16 supported).",
    )
    parser.add_argument(
        "--kv_dtype", type=cutlass.dtype, default=cutlass.Int8,
        help="KV dtype (Int8 supported).",
    )
    parser.add_argument(
        "--o_dtype", type=cutlass.dtype, default=cutlass.BFloat16,
        help="O dtype (BFloat16 supported).",
    )
    parser.add_argument(
        "--scale_dtype", type=cutlass.dtype, default=cutlass.BFloat16,
        help="Scale dtype (BFloat16 supported).",
    )
    parser.add_argument(
        "--scale_granularity", type=int, default=256, choices=(128, 256),
        help="Per-D group size for the KV dequant scales.",
    )
    parser.add_argument(
        "--qk_acc_dtype", type=cutlass.dtype, default=Float32,
        help="QK accumulator dtype (Float32 supported).",
    )
    parser.add_argument(
        "--pv_acc_dtype", type=cutlass.dtype, default=Float32,
        help="PV accumulator dtype (Float32 supported).",
    )

    # Kernel options
    parser.add_argument(
        "--is_persistent", action="store_true",
        help="Enable persistent-kernel scheduling.",
    )
    parser.add_argument(
        "--is_causal", action="store_true",
        help="Apply causal mask.",
    )

    # Scaling
    parser.add_argument("--scale_q", type=float, default=1.0)
    parser.add_argument("--scale_k", type=float, default=1.0)
    parser.add_argument("--scale_v", type=float, default=1.0)
    parser.add_argument("--inv_scale_o", type=float, default=1.0)
    parser.add_argument(
        "--scale_softmax", type=float, default=0.0,
        help="Softmax scale; 0.0 -> default 1/sqrt(D).",
    )

    # Test driver
    parser.add_argument("--tolerance", type=float, default=1e-1)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--skip_ref_check", action="store_true",
        help="Skip torch reference check (faster smoke test).",
    )
    parser.add_argument(
        "--use_cold_l2", action="store_true", default=False,
        help="Use circular buffer tensor sets for L2-cold benchmarking.",
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.q_shape) != 4:
        parser.error("--q_shape must contain exactly 4 values (B, H_q, S_q, D)")
    if len(args.k_shape) != 4:
        parser.error("--k_shape must contain exactly 4 values (B, H_k, S_k, D)")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    run(
        args.q_shape,
        args.k_shape,
        args.q_dtype,
        args.kv_dtype,
        args.o_dtype,
        args.scale_dtype,
        args.scale_granularity,
        args.qk_acc_dtype,
        args.pv_acc_dtype,
        args.is_persistent,
        args.is_causal,
        args.scale_q,
        args.scale_k,
        args.scale_v,
        args.inv_scale_o,
        args.scale_softmax,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
