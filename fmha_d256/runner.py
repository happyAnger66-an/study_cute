# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Test driver / benchmark entry for d=256 FMHA (mixed-input and homogeneous)."""

import math
import os
import time
from typing import Tuple, Type

import cupy as cp
import numpy as np
import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32
from cutlass.cute import testing as cute_testing
from cutlass.cute.runtime import from_dlpack

from fmha_d256 import fmha_helpers as fmha_utils
from fmha_d256.host.config import MixedInputFusedMultiHeadAttentionPrefillD256
from fmha_d256.host.tensor_layout import (
    mark_1d_dynamic,
    mark_bshd_dynamic,
    mark_kv_cache_dynamic,
)
from fmha_d256.host.torch_ref import (
    create_tensor,
    run_torch_fmha,
    run_torch_fmha_homo,
)


def _numpy_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def run(
    q_shape: Tuple[int, int, int, int],
    k_shape: Tuple[int, int, int, int],
    q_dtype: Type[cutlass.Numeric],
    kv_dtype: Type[cutlass.Numeric],
    o_dtype: Type[cutlass.Numeric],
    scale_dtype: Type[cutlass.Numeric],
    scale_granularity: int,
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    is_persistent: bool,
    is_causal: bool,
    scale_q: float,
    scale_k: float,
    scale_v: float,
    inv_scale_o: float,
    scale_softmax: float,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    print(f"Running Blackwell SM100 Mixed Input FMHA Prefill D256 test with:")
    print(f"  q_shape: {q_shape}")
    print(f"  k_shape: {k_shape}")
    print(f"  q_dtype: {q_dtype}")
    print(f"  kv_dtype: {kv_dtype}")
    print(f"  o_dtype: {o_dtype}")
    print(f"  scale_dtype: {scale_dtype}")
    print(f"  scale_granularity: {scale_granularity}")
    print(f"  qk_acc_dtype: {qk_acc_dtype}")
    print(f"  pv_acc_dtype: {pv_acc_dtype}")
    print(f"  is_persistent: {is_persistent}")
    print(f"  is_causal: {is_causal}")
    print(f"  scale_q: {scale_q}")
    print(f"  scale_k: {scale_k}")
    print(f"  scale_v: {scale_v}")
    print(f"  inv_scale_o: {inv_scale_o}")
    print(f"  scale_softmax: {scale_softmax}")
    print(f"  tolerance: {tolerance}")
    print(f"  warmup_iterations: {warmup_iterations}")
    print(f"  iterations: {iterations}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  use_cold_l2: {use_cold_l2}")
    import cutlass.torch as cutlass_torch

    # Unpack parameters
    b, h_q, s_q, d = q_shape
    b_, h_k, s_k, d_ = k_shape
    window_size_left, window_size_right = None, None
    if is_causal:
        window_size_right = 0

    if b != b_:
        raise ValueError("q & k must have the same batch size")

    if d != d_:
        raise ValueError("q & k must have the same head dimension")

    if d not in {256}:
        raise ValueError("head dimension must be 256")

    if d % scale_granularity != 0:
        raise ValueError("head dimension must be divisible by scale_granularity")

    if scale_granularity not in {128, 256}:
        raise ValueError("scale_granularity must be 128, 256")

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")

    if isinstance(s_q, tuple) and len(s_q) != b:
        raise ValueError("variable_seqlen s_q must have the length of batch size")
    if isinstance(s_k, tuple) and len(s_k) != b:
        raise ValueError("variable_seqlen s_k must have the length of batch size")

    if q_dtype not in {cutlass.BFloat16, cutlass.Float16}:
        raise ValueError("q_dtype must be BFloat16 or Float16")

    if o_dtype not in {cutlass.BFloat16}:
        raise ValueError("o_dtype must be BFloat16")

    if kv_dtype not in {cutlass.Int8, cutlass.BFloat16, cutlass.Float16}:
        raise ValueError("kv_dtype must be Int8, BFloat16, or Float16")

    is_mixed_input = kv_dtype == cutlass.Int8
    if not is_mixed_input and kv_dtype != q_dtype:
        raise ValueError("homogeneous path requires kv_dtype == q_dtype")

    if qk_acc_dtype not in {cutlass.Float32}:
        raise ValueError("qk_acc_dtype must be Float32")

    if pv_acc_dtype not in {cutlass.Float32}:
        raise ValueError("pv_acc_dtype must be Float32")

    h_r = h_q // h_k

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    scale_shape = (b, h_k, s_k, d // scale_granularity)

    q_ref, q_tensor, q_torch = create_tensor(q_shape, q_dtype)
    k_ref, k_tensor, k_torch = create_tensor(k_shape, kv_dtype)
    v_ref, v_tensor, v_torch = create_tensor(k_shape, kv_dtype)
    o_ref, o_tensor, o_torch = create_tensor(q_shape, o_dtype)
    scale_k_ref = scale_k_tensor = scale_k_torch = None
    scale_v_ref = scale_v_tensor = scale_v_torch = None
    if is_mixed_input:
        scale_k_ref, scale_k_tensor, scale_k_torch = create_tensor(
            scale_shape, scale_dtype
        )
        scale_v_ref, scale_v_tensor, scale_v_torch = create_tensor(
            scale_shape, scale_dtype
        )

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
    if is_causal:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
    else:
        if s_k % 128 != 0:
            mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK

    fmha = MixedInputFusedMultiHeadAttentionPrefillD256(
        scale_granularity,
        qk_acc_dtype,
        pv_acc_dtype,
        is_persistent,
        mask_type,
        is_mixed_input=is_mixed_input,
    )

    # Initialize Stream
    current_stream = cutlass_torch.default_stream()

    if scale_softmax == 0.0:  # default to 1/sqrt(d)
        scale_softmax = 1.0 / math.sqrt(d)
    log2_e = math.log2(
        math.exp(1.0)
    )  # gpu uses exp2 for perf concerns, we need an extra factor 'log2_e' here

    scale_softmax = scale_q * scale_k * scale_softmax
    scale_softmax_log2 = scale_softmax * log2_e
    scale_output = scale_v * inv_scale_o
    problem_size = (b, s_q, s_k, h_q, h_k, d)
    if is_mixed_input:
        compiled_fmha = cute.compile(
            fmha,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            scale_k_tensor.iterator,
            scale_v_tensor.iterator,
            problem_size,
            scale_softmax_log2,
            scale_output,
            window_size_left if window_size_left is None else Int32(window_size_left),
            (
                window_size_right
                if window_size_right is None
                else Int32(window_size_right)
            ),
            current_stream,
            options="--opt-level 2",
        )
    else:
        compiled_fmha = cute.compile(
            fmha.launch_homo,
            q_tensor.iterator,
            k_tensor.iterator,
            v_tensor.iterator,
            o_tensor.iterator,
            problem_size,
            scale_softmax_log2,
            scale_output,
            window_size_left if window_size_left is None else Int32(window_size_left),
            (
                window_size_right
                if window_size_right is None
                else Int32(window_size_right)
            ),
            current_stream,
            None,
            options="--opt-level 2",
        )
    if not skip_ref_check:
        if is_mixed_input:
            compiled_fmha(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                scale_k_tensor.iterator,
                scale_v_tensor.iterator,
                problem_size,
                scale_softmax_log2,
                scale_output,
                window_size_left if window_size_left is None else Int32(window_size_left),
                (
                    window_size_right
                    if window_size_right is None
                    else Int32(window_size_right)
                ),
                current_stream,
            )
        else:
            compiled_fmha(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                problem_size,
                scale_softmax_log2,
                scale_output,
                window_size_left if window_size_left is None else Int32(window_size_left),
                (
                    window_size_right
                    if window_size_right is None
                    else Int32(window_size_right)
                ),
                current_stream,
                None,
            )
        print("Verifying results...")
        if is_mixed_input:
            o_ref = run_torch_fmha(
                q_ref,
                k_ref,
                v_ref,
                scale_k_ref,
                scale_v_ref,
                scale_softmax,
                scale_output,
                is_causal,
            )
        else:
            o_ref = run_torch_fmha_homo(
                q_ref,
                k_ref,
                v_ref,
                scale_softmax,
                scale_output,
                is_causal,
            )

        # convert o back to f32 for comparison
        o_fp32, o_fp32_torch = cutlass_torch.cute_tensor_like(
            torch.empty(*o_torch.shape, dtype=torch.float32),
            Float32,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        cute.testing.convert(o_tensor, o_fp32)
        o_result = o_fp32_torch.cpu()
        torch.testing.assert_close(o_ref, o_result, atol=tolerance, rtol=1e-05)

        print("Results verified successfully!")

    # ------------------------------------------------------------------
    # Benchmark (study_cute addition, NOT in upstream).
    # Only runs when iterations > 0; warmup_iterations may be 0.
    # Returns avg latency in microseconds so the CLI shim can print a
    # summary. Pure no-op when iterations <= 0 (return None).
    # ------------------------------------------------------------------
    if iterations > 0:
        if is_mixed_input:
            kernel_args = cute_testing.JitArguments(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                scale_k_tensor.iterator,
                scale_v_tensor.iterator,
                problem_size,
                scale_softmax_log2,
                scale_output,
                window_size_left if window_size_left is None else Int32(window_size_left),
                (
                    window_size_right
                    if window_size_right is None
                    else Int32(window_size_right)
                ),
                current_stream,
            )
        else:
            kernel_args = cute_testing.JitArguments(
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                problem_size,
                scale_softmax_log2,
                scale_output,
                window_size_left if window_size_left is None else Int32(window_size_left),
                (
                    window_size_right
                    if window_size_right is None
                    else Int32(window_size_right)
                ),
                current_stream,
                None,
            )
        avg_time_us = cute_testing.benchmark(
            compiled_fmha,
            kernel_arguments=kernel_args,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            stream=current_stream,
        )
        # FMHA FLOPs: QK is 2*B*H_q*S_q*S_k*D, PV is 2*B*H_q*S_q*S_k*D
        # (softmax exp/div is treated as non-MAC, dominant cost is the two GEMMs).
        # Causal halves the effective work; we conservatively use the same 0.5
        # factor as the cutlass reference benchmarks.
        flops = 4.0 * b * h_q * s_q * s_k * d
        if is_causal:
            flops *= 0.5
        tflops = flops / (avg_time_us * 1e-6) / 1e12
        print(
            f"[benchmark] avg latency: {avg_time_us:.3f} us "
            f"(warmup={warmup_iterations}, iterations={iterations})"
        )
        print(
            f"[benchmark] throughput: {tflops:.2f} TFLOPS  "
            f"(B={b}, H_q={h_q}, H_k={h_k}, S_q={s_q}, S_k={s_k}, "
            f"D={d}, is_causal={is_causal}, is_persistent={is_persistent})"
        )
        return avg_time_us
    return None


def run_llm_multi_round_prefill_test_d256(
    batch_size: int = 4,
    seq_len: int = 8,
    num_rounds: int = 3,
    h_q: int = 8,
    h_k: int = 8,
    d: int = 256,
    kv_cache_capacity: int = 64,
    is_persistent: bool = True,
    is_causal: bool = True,
    bottom_right_align: bool = True,
    tolerance: float = 0.1,
    q_dtype= cutlass.BFloat16,
):
    """Multi-round LLM prefill test for d=256 homogeneous FMHA (BSHD + packed KV)."""
    _tag = "[llm_prefill_d256]"
    b = batch_size
    cap = kv_cache_capacity
    h_r = h_q // h_k
    window_size_left = None
    window_size_right = 0 if is_causal else None

    print(f"{_tag} Running multi-round prefill accuracy test:")
    print(
        f"{_tag}   b={b}, seq_len={seq_len}, rounds={num_rounds}, "
        f"cap={cap}, h_q={h_q}, h_k={h_k}, d={d}, is_causal={is_causal}"
    )

    if d != 256:
        raise ValueError("d must be 256 for fmha_d256 LLM test")
    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")
    if num_rounds * seq_len > cap:
        raise ValueError(
            f"total tokens ({num_rounds * seq_len}) exceeds capacity ({cap})"
        )

    cp.random.seed(42)
    np.random.seed(42)

    _scale_q = 1.0
    _scale_k = 1.0
    _scale_v = 1.0
    _inv_scale_o = 1.0
    ref_scale_softmax = 1.0 / math.sqrt(d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    fmha_op = MixedInputFusedMultiHeadAttentionPrefillD256(
        scale_granularity=256,
        qk_acc_dtype=Float32,
        pv_acc_dtype=Float32,
        is_persistent=is_persistent,
        mask_type=mask_type,
        is_mixed_input=False,
    )
    import cutlass.torch as cutlass_torch

    current_stream = cutlass_torch.default_stream()
    _wsl = Int32(0)

    def _to_cute(arr, element_type):
        t = from_dlpack(arr, assumed_align=16)
        t.element_type = element_type
        return t

    cp_dtype = cp.float16 if q_dtype == cutlass.Float16 else cp.float16
    if q_dtype == cutlass.BFloat16:
        cp_dtype = cp.float16  # cupy has no bf16; use fp16 storage for test data

    kv_np = np.zeros((b, 2, h_k, cap, d), dtype=np.float32)
    compiled_fmha = None
    all_pass = True
    current_pos = 0

    for round_idx in range(num_rounds):
        effective_kv_len = current_pos + seq_len
        print(
            f"\n--- Round {round_idx + 1}/{num_rounds} "
            f"(pos={current_pos}, s_k={effective_kv_len}, cap={cap}) ---"
        )

        q_np = np.random.randint(-2, 2, (b, seq_len, h_q, d)).astype(np.float32)
        new_k_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(np.float32)
        new_v_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(np.float32)

        kv_np[:, 0, :, current_pos:current_pos + seq_len, :] = new_k_np
        kv_np[:, 1, :, current_pos:current_pos + seq_len, :] = new_v_np

        q_cp = cp.asarray(q_np.astype(cp_dtype))
        kv_cp = cp.asarray(kv_np.astype(cp_dtype))
        o_cp = cp.zeros((b, seq_len, h_q, d), dtype=cp_dtype)

        q_t = mark_bshd_dynamic(_to_cute(q_cp, q_dtype))
        kv_t = mark_kv_cache_dynamic(_to_cute(kv_cp, q_dtype))
        o_t = mark_bshd_dynamic(_to_cute(o_cp, q_dtype))

        cu_kv_np = np.arange(b + 1, dtype=np.int32) * effective_kv_len
        cu_kv_cp = cp.asarray(cu_kv_np)
        cu_kv = mark_1d_dynamic(from_dlpack(cu_kv_cp, assumed_align=16))

        if compiled_fmha is None:
            start_time = time.time()
            compiled_fmha = cute.compile(
                fmha_op.call_llm,
                q_t,
                kv_t,
                o_t,
                cu_kv,
                _wsl,
                _scale_q,
                _scale_k,
                _scale_v,
                _inv_scale_o,
                current_stream,
                options="--opt-level 2",
            )
            print(f"{_tag} Compilation time: {time.time() - start_time:.4f}s")

        compiled_fmha(
            q_t,
            kv_t,
            o_t,
            cu_kv,
            _wsl,
            _scale_q,
            _scale_k,
            _scale_v,
            _inv_scale_o,
            current_stream,
        )

        o_f32_cp = cp.empty(o_cp.shape, dtype=cp.float32)
        o_f32_cute = from_dlpack(o_f32_cp, assumed_align=16)
        o_f32_cute.element_type = Float32
        o_f32_cute = o_f32_cute.mark_layout_dynamic(leading_dim=3)
        cute.testing.convert(o_t, o_f32_cute)
        o_result = o_f32_cp.get()

        for bi in range(b):
            q_b = q_np[bi].transpose(1, 0, 2)
            k_b = kv_np[bi, 0, :, :effective_kv_len]
            v_b = kv_np[bi, 1, :, :effective_kv_len]
            if h_q != h_k:
                k_b = np.repeat(k_b, h_r, axis=0)
                v_b = np.repeat(v_b, h_r, axis=0)
            scores = np.einsum("hqd,hkd->hqk", q_b, k_b) * ref_scale_softmax
            s_k_len = effective_kv_len
            if is_causal:
                q_coords = np.arange(seq_len).reshape(-1, 1)
                k_coords = np.arange(s_k_len).reshape(1, -1)
                offset = (s_k_len - seq_len) if bottom_right_align else 0
                mask = k_coords > q_coords + offset
                scores = np.where(mask, -np.inf, scores)
            probs = _numpy_softmax(scores, axis=-1)
            o_ref = np.einsum("hqk,hkd->hqd", probs, v_b)
            o_ref = o_ref.transpose(1, 0, 2)
            o_actual = o_result[bi]
            max_diff = np.max(np.abs(o_actual - o_ref))
            mean_diff = np.mean(np.abs(o_actual - o_ref))
            if max_diff > tolerance:
                print(
                    f"  batch {bi}: FAIL  max_diff={max_diff:.6f}  "
                    f"mean_diff={mean_diff:.6f}"
                )
                all_pass = False
            else:
                print(
                    f"  batch {bi}: PASS  max_diff={max_diff:.6f}  "
                    f"mean_diff={mean_diff:.6f}"
                )
        current_pos += seq_len

    if all_pass:
        print(f"\n{_tag} All {num_rounds} rounds passed.")
    else:
        raise AssertionError(f"{_tag} Some rounds failed accuracy check!")
    return all_pass
