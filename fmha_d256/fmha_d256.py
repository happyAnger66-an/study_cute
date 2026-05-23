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


def _auto_set_cute_dsl_arch() -> str:
    """Detect the local GPU compute capability and set CUTE_DSL_ARCH.

    The CUTLASS DSL compiles the kernel for the arch given in the
    ``CUTE_DSL_ARCH`` env var, then launches on the current GPU. If the two
    don't match, the launch fails with cudaErrorNoKernelImageForDevice (209).

    On Jetson Thor the GPU reports compute capability (11, 0) which the DSL
    spells ``sm_110``; on a B200 it's ``sm_100a``. Hard-coding ``sm_100a``
    works for B200 only.

    This helper:
        * respects an explicit user override (env var already set);
        * otherwise queries the active CUDA device via the CUDA driver and
          sets a sensible value (Blackwell-family GPUs => ``sm_<MAJOR><MINOR>a``).

    Must be called BEFORE ``import cutlass`` (the DSL caches the value at
    import time).
    """
    user_override = os.environ.get("CUTE_DSL_ARCH")
    if user_override:
        print(f"[fmha_d256] using user-specified CUTE_DSL_ARCH={user_override}")
        return user_override
    def _last(ret):
        """cuda-python returns either (err, value) or (err,); normalize to last element."""
        if isinstance(ret, tuple):
            return ret[-1] if len(ret) > 1 else None
        return ret

    try:
        import cuda.bindings.driver as cuda

        _last(cuda.cuInit(0))
        dev = _last(cuda.cuDeviceGet(0))
        attr = cuda.CUdevice_attribute
        major = _last(
            cuda.cuDeviceGetAttribute(
                attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
            )
        )
        minor = _last(
            cuda.cuDeviceGetAttribute(
                attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
            )
        )
        if major is None or minor is None:
            raise RuntimeError("cuDeviceGetAttribute returned no value")
        major = int(major)
        minor = int(minor)
    except Exception as e:
        # Fall back to sm_100a (B200 default). User can still override.
        fallback = "sm_100a"
        print(
            f"[fmha_d256] WARNING: could not auto-detect GPU arch ({e}); "
            f"falling back to CUTE_DSL_ARCH={fallback}"
        )
        os.environ["CUTE_DSL_ARCH"] = fallback
        return fallback
    # Blackwell family => append the 'a' suffix the DSL expects for
    # arch-specific tcgen05.* instructions. Older arches fall through to plain
    # sm_<MM>.
    if major >= 10:
        arch = f"sm_{major}{minor}a"
    else:
        arch = f"sm_{major}{minor}"
    os.environ["CUTE_DSL_ARCH"] = arch
    print(
        f"[fmha_d256] auto-detected GPU CC ({major}, {minor}) -> "
        f"CUTE_DSL_ARCH={arch}"
    )
    return arch


_AUTO_ARCH = _auto_set_cute_dsl_arch()


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
    parser.add_argument(
        "--warmup_iterations", type=int, default=10,
        help="Number of warm-up kernel launches before timing.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of timed kernel launches. Set to 0 to skip benchmarking.",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true",
        help="Skip torch reference check (faster smoke test).",
    )
    parser.add_argument(
        "--use_cold_l2", action="store_true", default=False,
        help="(Reserved; kernel is L2-hot benchmarked unless this is set in the future.)",
    )
    parser.add_argument(
        "--no_benchmark", action="store_true",
        help="Disable benchmarking entirely (equivalent to --iterations 0).",
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

    iterations = 0 if args.no_benchmark else args.iterations

    latency_us = run(
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
        iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    # Already printed inside `run()` when iterations > 0; surface a compact
    # one-line summary as well so it's easy to grep from CI logs.
    if latency_us is not None:
        b, h_q, s_q, _ = args.q_shape
        _, h_k, s_k, d = args.k_shape
        flops = 4.0 * b * h_q * s_q * s_k * d
        if args.is_causal:
            flops *= 0.5
        tflops = flops / (latency_us * 1e-6) / 1e12
        # Approximate IO bytes: BF16 Q + Int8 K/V + BF16 O + BF16 scales.
        bytes_q = b * h_q * s_q * d * 2
        bytes_kv = 2 * b * h_k * s_k * d * 1
        bytes_o = b * h_q * s_q * d * 2
        bytes_scale = 2 * b * h_k * s_k * (d // args.scale_granularity) * 2
        gb = (bytes_q + bytes_kv + bytes_o + bytes_scale) / 1e9
        bw = gb / (latency_us * 1e-6)
        print(
            f"[fmha_d256] summary: latency={latency_us:.3f} us  "
            f"tflops={tflops:.2f}  io_bw={bw:.2f} GB/s  "
            f"shape=(B={b},H_q={h_q},H_k={h_k},S_q={s_q},S_k={s_k},D={d})"
        )

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
