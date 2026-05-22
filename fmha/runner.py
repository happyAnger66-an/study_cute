"""Top-level CLI / driver for the Blackwell SM100 FMHA kernel.

Pulls together the host config (``fmha.host.config``), launchers
(``fmha.host.launcher``), tensor helpers (``fmha.host.tensor_layout``), and
numpy reference (``fmha.host.numpy_ref``) into:

- :func:`run` -- one-shot single-config compile + benchmark + ref check
- :func:`run_llm_multi_round_prefill_test` -- multi-round LLM prefill regression
- :func:`main` -- argparse + ``run`` entry point (invoked from ``fmha.py``)
"""

import argparse
import math
import os
import time
from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

from fmha import fmha_helpers as fmha_utils
from fmha import BlackwellFusedMultiHeadAttentionForward
from fmha.host.numpy_ref import (
    maybe_quantize_ref_for_narrow_out,
    numpy_softmax,
    run_numpy_single_shot_reference_packed,
)
from fmha.host.tensor_layout import (
    create_and_pad_tensor,
    get_leading_dim,
    mark_1d_dynamic,
    mark_bshd_dynamic,
    mark_kv_cache_dynamic,
    mark_shd_dynamic,
)


def run(
    q_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]],
    k_shape: Union[Tuple[int, int, int, int], Tuple[int, Tuple[int, ...], int, int]],
    in_dtype: Type[cutlass.Numeric],
    out_dtype: Type[cutlass.Numeric],
    qk_acc_dtype: Type[cutlass.Numeric],
    pv_acc_dtype: Type[cutlass.Numeric],
    mma_tiler_mn: Tuple[int, int],
    is_persistent: bool,
    is_causal: bool,
    bottom_right_align: bool,
    lse_calculation: bool,
    window_size: Tuple[int, int],
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
    output_dir: str = "./fmha_aot_artifacts",
    export_only: bool = False,
    file_name: str = "fmha",
    function_prefix: str = "fmha",
    vit_mode: bool = False,
    **kwargs,
):
    """Compile + benchmark + (optionally) validate FMHA for one configuration.

    See ``fmha.py`` docstrings for the full parameter contract; in short:
    builds Q/K/V/O CuPy tensors (with padding), wraps them as CuTe tensors,
    constructs :class:`BlackwellFusedMultiHeadAttentionForward`, JIT-compiles
    the launcher, and runs benchmarks. When ``skip_ref_check=False`` it also
    runs a numpy reference (single-shot for ViT, multi-round prefill for LLM).
    """
    _tag = f"[{file_name}]"

    if export_only:
        print(
            f"{_tag} Compiling: head_dim={q_shape[-1]}, in_dtype={in_dtype}, "
            f"causal={is_causal}, window={window_size}, "
            f"mma_tiler_mn={mma_tiler_mn}, persistent={is_persistent}, "
            f"bottom_right_align={bottom_right_align}, "
            f"sliding_window={window_size[0] != -1}"
        )
    else:
        print(f"{_tag} Running Blackwell SM100 FMHA test with:")
        print(f"{_tag}   q_shape={q_shape}, k_shape={k_shape}")
        print(f"{_tag}   in_dtype={in_dtype}, out_dtype={out_dtype}")
        print(f"{_tag}   qk_acc_dtype={qk_acc_dtype}, pv_acc_dtype={pv_acc_dtype}")
        print(f"{_tag}   mma_tiler_mn={mma_tiler_mn}, is_persistent={is_persistent}")
        print(f"{_tag}   is_causal={is_causal}, window_size={window_size}")
        print(
            f"{_tag}   tolerance={tolerance}, warmup={warmup_iterations}, "
            f"iterations={iterations}"
        )

    b, s_q, h_q, d = q_shape
    b_, s_k, h_k, d_ = k_shape
    window_size_left, window_size_right = window_size
    if window_size_left == -1:
        window_size_left = None
    if window_size_right == -1:
        window_size_right = None
    if is_causal:
        window_size_right = 0

    if b != b_:
        raise ValueError("q & k must have the same batch size")
    if d != d_:
        raise ValueError("q & k must have the same head dimension")
    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")
    if isinstance(s_q, tuple) and len(s_q) != b:
        raise ValueError("variable_seqlen s_q must have the length of batch size")
    if isinstance(s_k, tuple) and len(s_k) != b:
        raise ValueError("variable_seqlen s_k must have the length of batch size")
    if in_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("in_dtype must be Float8E4M3FN or Float16")
    if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16}:
        raise ValueError("out_dtype must be Float8E4M3FN or Float16")
    if qk_acc_dtype not in {Float32}:
        raise ValueError("qk_acc_dtype must be Float32")
    if pv_acc_dtype not in {Float32}:
        raise ValueError("pv_acc_dtype must be Float32")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    h_r = h_q // h_k

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this example!")

    if not export_only:
        cp.random.seed(1111)
    np.random.seed(1111)

    if isinstance(s_q, tuple) or isinstance(s_k, tuple):
        raise NotImplementedError(
            "Variable-length sequences (nested tensors) require PyTorch. "
            "Use fmha_runtimeargs_kvcache.py for variable-length support."
        )

    qo_shape = (b, s_q, h_r * h_k, d)
    kvcache_shape = (b, 2, h_k, s_k, d)
    lse_shape = (b, h_r * h_k, s_q)
    qo_padding = (0, 0, 0, 0, 0)
    kvcache_padding = (0, 0, 0, 0, 0, 0)
    lse_padding = (0, 0, 0, 0)

    q_ref, q_tensor, q_cp, *_q_keep = create_and_pad_tensor(
        qo_shape, qo_padding, in_dtype,
        is_dynamic_layout=True, export_only=export_only,
    )
    kvcache_ref, kvcache_tensor, kvcache_cp, *_kv_keep = create_and_pad_tensor(
        kvcache_shape, kvcache_padding, in_dtype,
        is_dynamic_layout=True, export_only=export_only,
    )
    # BHSD -> BSHD for numpy comparison
    k_ref = np.ascontiguousarray(kvcache_ref[:, 0].transpose(0, 2, 1, 3))
    v_ref = np.ascontiguousarray(kvcache_ref[:, 1].transpose(0, 2, 1, 3))
    _, o_tensor, o_cp, *_o_keep = create_and_pad_tensor(
        qo_shape, qo_padding, out_dtype,
        is_dynamic_layout=True, export_only=export_only,
    )
    if lse_calculation:
        _, lse_tensor, lse_cp, *_lse_keep = create_and_pad_tensor(
            lse_shape, lse_padding, cutlass.Float32,
            is_dynamic_layout=True, export_only=export_only,
        )
    else:
        lse_cp = None

    # P1 D-chunk: MMA/SMEM/TMEM use d_chunk_k-wide tiles; full head_dim via num_d_chunks.
    _MMA_K_ATOM = 256 // in_dtype.width
    d_chunk_k = BlackwellFusedMultiHeadAttentionForward.D_CHUNK
    mma_tiler = (*mma_tiler_mn, d_chunk_k)
    actual_head_dim = d
    if d % _MMA_K_ATOM != 0:
        print(
            f"[fmha] head_dim {d} is not a multiple of MMA K atom {_MMA_K_ATOM}; "
            f"TMA ZFILL/OOB-drop apply within each {d_chunk_k}-wide chunk"
        )

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    s_q_list = s_q if isinstance(s_q, tuple) else [s_q] * b
    s_k_list = s_k if isinstance(s_k, tuple) else [s_k] * b

    def _check_seqlen_valid(
        s_q, s_k, window_size_left, window_size_right, bottom_right_align
    ):
        """Ensure no row is fully masked (would yield NaN in softmax)."""
        for i in range(s_q):
            offset = 0 if not bottom_right_align else s_k - s_q
            s_q_start = (
                0 if window_size_left is None else i + offset - window_size_left
            )
            s_q_end = (
                s_q if window_size_right is None else i + offset + window_size_right
            )
            s_q_min = max(s_q_start, 0)
            s_q_max = min(s_q_end, s_k)
            if s_q_max - s_q_min == 0 and (i != 0 and i != s_q - 1):
                return False
        return True

    need_check_seqlen_valid = (
        window_size_left is not None or window_size_right is not None
    )
    for i in range(b):
        if need_check_seqlen_valid and not _check_seqlen_valid(
            s_q_list[i], s_k_list[i],
            window_size_left, window_size_right, bottom_right_align,
        ):
            raise ValueError("sliding window doesn't support current setting")

    use_sliding_window = window_size_left is not None
    if vit_mode:
        mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype, pv_acc_dtype, mma_tiler, is_persistent, mask_type,
        is_causal=(is_causal and not vit_mode),
        use_sliding_window=(use_sliding_window and not vit_mode),
        actual_head_dim=actual_head_dim,
    )
    fmha.validate_config_host(in_dtype)

    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    # Reference / ViT path needs pre-folded scales; LLM __call__ does this internally.
    if scale_softmax == 0.0:
        scale_softmax = 1.0 / math.sqrt(d)
    log2_e = math.log2(math.exp(1.0))
    ref_scale_softmax = scale_q * scale_k * scale_softmax
    ref_scale_softmax_log2 = ref_scale_softmax * log2_e
    ref_scale_output = scale_v * inv_scale_o

    if vit_mode:
        _s = s_q if not isinstance(s_q, tuple) else max(s_q)
        total_S = b * _s
        q_vit_shape = (total_S, h_r * h_k, d)
        k_vit_shape = (total_S, h_k, d)
        q_vit_ref, q_vit_tensor, q_vit_cp, *_qv = create_and_pad_tensor(
            q_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True,
            export_only=export_only,
        )
        k_vit_ref, k_vit_tensor, k_vit_cp, *_kv = create_and_pad_tensor(
            k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True,
            export_only=export_only,
        )
        v_vit_ref, v_vit_tensor, v_vit_cp, *_vv = create_and_pad_tensor(
            k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True,
            export_only=export_only,
        )
        _, o_vit_tensor, o_vit_cp, *_ov = create_and_pad_tensor(
            q_vit_shape, (0, 0, 0, 0), out_dtype, is_dynamic_layout=True,
            export_only=export_only,
        )

        cu_seqlens_np = np.arange(b + 1, dtype=np.int32) * _s
        cu_seqlens_cp = cp.asarray(cu_seqlens_np)
        cu_seqlens = from_dlpack(cu_seqlens_cp, assumed_align=16)

        q_dyn = mark_shd_dynamic(q_vit_tensor)
        k_dyn = mark_shd_dynamic(k_vit_tensor)
        v_dyn = mark_shd_dynamic(v_vit_tensor)
        o_dyn = mark_shd_dynamic(o_vit_tensor)
        cu_dyn = mark_1d_dynamic(cu_seqlens)

        _max_seqlen = Int32(_s)

        start_time = time.time()
        compiled_fmha = cute.compile(
            fmha.__call_vit__,
            q_dyn, k_dyn, v_dyn, o_dyn, cu_dyn, _max_seqlen,
            ref_scale_softmax_log2, ref_scale_softmax, ref_scale_output,
            current_stream,
        )
    else:
        q_dyn = mark_bshd_dynamic(q_tensor)
        kv_dyn = mark_kv_cache_dynamic(kvcache_tensor)
        o_dyn = mark_bshd_dynamic(o_tensor)

        _wsl = (
            Int32(window_size_left) if window_size_left is not None else Int32(0)
        )

        _s_k = s_k if not isinstance(s_k, tuple) else max(s_k)
        cu_kv_seqlens_np = np.arange(b + 1, dtype=np.int32) * _s_k
        cu_kv_seqlens_cp = cp.asarray(cu_kv_seqlens_np)
        cu_kv_seqlens = from_dlpack(cu_kv_seqlens_cp, assumed_align=16)
        cu_kv_seqlens = mark_1d_dynamic(cu_kv_seqlens)

        start_time = time.time()
        compiled_fmha = cute.compile(
            fmha,
            q_dyn, kv_dyn, o_dyn, cu_kv_seqlens, _wsl,
            scale_q, scale_k, scale_v, inv_scale_o,
            current_stream,
        )

    compilation_time = time.time() - start_time
    print(f"{_tag} Compilation time: {compilation_time:.4f}s")

    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_fmha.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None

    if vit_mode:
        _vit_test_tag = "[vit_single_shot_test]"
        if not skip_ref_check:
            print(f"{_vit_test_tag} Running single-shot packed accuracy test:")
            print(
                f"{_vit_test_tag}   b={b}, seq_len={_s}, total_s={total_S}, "
                f"h_q={h_q}, h_k={h_k}, d={d}, is_causal=False"
            )
            print(
                f"{_vit_test_tag}   layout=[total_S,H,D], "
                f"uniform cu_seqlens, max_seqlen={_s}"
            )
            compiled_fmha(
                q_vit_tensor, k_vit_tensor, v_vit_tensor, o_vit_tensor,
                cu_seqlens, _max_seqlen,
                ref_scale_softmax_log2, ref_scale_softmax, ref_scale_output,
                current_stream,
            )

            o_fp32_cp = cp.empty(o_vit_cp.shape, dtype=cp.float32)
            o_fp32_cute = from_dlpack(o_fp32_cp, assumed_align=16)
            o_fp32_cute.element_type = Float32
            o_fp32_cute = o_fp32_cute.mark_layout_dynamic(leading_dim=2)
            cute.testing.convert(o_vit_tensor, o_fp32_cute)
            o_result = o_fp32_cp.get()

            cu_q_np = np.arange(b + 1, dtype=np.int32) * _s
            cu_k_np = np.arange(b + 1, dtype=np.int32) * _s
            o_ref, _ = run_numpy_single_shot_reference_packed(
                q_vit_ref, k_vit_ref, v_vit_ref,
                cu_q_np, cu_k_np,
                scale_softmax=ref_scale_softmax,
                scale_output=ref_scale_output,
                is_causal=False,
                bottom_right_align=False,
                lse_calculation=False,
                window_size_left=None,
                window_size_right=None,
            )
            o_ref, tol_for_check = maybe_quantize_ref_for_narrow_out(
                o_ref, out_dtype, tolerance
            )
            np.testing.assert_allclose(
                o_result, o_ref, atol=tol_for_check, rtol=1e-05
            )
            print(f"{_vit_test_tag} ViT single-shot accuracy check passed.")

        def generate_vit_tensors():
            _, q_ws, *_gq = create_and_pad_tensor(
                q_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True
            )
            _, k_ws, *_gk = create_and_pad_tensor(
                k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True
            )
            _, v_ws, *_gv = create_and_pad_tensor(
                k_vit_shape, (0, 0, 0, 0), in_dtype, is_dynamic_layout=True
            )
            _, o_ws, *_go = create_and_pad_tensor(
                q_vit_shape, (0, 0, 0, 0), out_dtype, is_dynamic_layout=True
            )
            return testing.JitArguments(
                mark_shd_dynamic(q_ws), mark_shd_dynamic(k_ws),
                mark_shd_dynamic(v_ws), mark_shd_dynamic(o_ws),
                cu_dyn, _max_seqlen,
                ref_scale_softmax_log2, ref_scale_softmax, ref_scale_output,
                current_stream,
            )

        exec_time = testing.benchmark(
            compiled_fmha,
            workspace_generator=generate_vit_tensors,
            workspace_count=1,
            stream=current_stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )
        return exec_time

    # LLM path: multi-round prefill regression.
    if not skip_ref_check:
        llm_prefill_tolerance = (
            0.13 if (out_dtype.is_float and out_dtype.width <= 8) else tolerance
        )
        if not isinstance(s_q, tuple) and not isinstance(s_k, tuple):
            # FMHA_DEBUG_ROUNDS env var caps num_rounds for diagnostics
            # (e.g. set to 1 to skip later rounds that may deadlock).
            _num_rounds = int(os.environ.get("FMHA_DEBUG_ROUNDS", "3"))
            _prefill_seq = min(s_q, s_k)
            _cap = _prefill_seq * _num_rounds
            print(
                f"{_tag} Running LLM multi-round prefill test "
                f"(cap={_cap}, seq_len={_prefill_seq}, "
                f"rounds={_num_rounds}) ..."
            )
            run_llm_multi_round_prefill_test(
                batch_size=b, seq_len=_prefill_seq, num_rounds=_num_rounds,
                h_q=h_q, h_k=h_k, d=d, kv_cache_capacity=_cap,
                mma_tiler_mn=mma_tiler_mn,
                is_persistent=is_persistent, is_causal=is_causal,
                bottom_right_align=bottom_right_align,
                use_sliding_window=use_sliding_window,
                window_size_left_val=(
                    window_size_left if window_size_left is not None else -1
                ),
                tolerance=llm_prefill_tolerance,
            )
            print(f"{_tag} LLM multi-round prefill test passed.")

    def generate_tensors():
        _, q_tensor_workspace, *_gq = create_and_pad_tensor(
            qo_shape, qo_padding, in_dtype, is_dynamic_layout=True
        )
        _, kvcache_tensor_workspace, *_gkv = create_and_pad_tensor(
            kvcache_shape, kvcache_padding, in_dtype, is_dynamic_layout=True
        )
        _, o_tensor_workspace, *_go = create_and_pad_tensor(
            qo_shape, qo_padding, out_dtype, is_dynamic_layout=True
        )
        q_ws = mark_bshd_dynamic(q_tensor_workspace)
        kv_ws = mark_kv_cache_dynamic(kvcache_tensor_workspace)
        o_ws = mark_bshd_dynamic(o_tensor_workspace)
        return testing.JitArguments(
            q_ws, kv_ws, o_ws, cu_kv_seqlens, _wsl,
            scale_q, scale_k, scale_v, inv_scale_o,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            q_cp.size * q_cp.itemsize
            + kvcache_cp.size * kvcache_cp.itemsize
            + o_cp.size * o_cp.itemsize
            + (lse_cp.size * lse_cp.itemsize if lse_cp is not None else 0)
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_fmha,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return exec_time


def run_llm_multi_round_prefill_test(
    batch_size: int = 4,
    seq_len: int = 8,
    num_rounds: int = 3,
    h_q: int = 8,
    h_k: int = 8,
    d: int = 128,
    kv_cache_capacity: int = 64,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    is_persistent: bool = True,
    is_causal: bool = True,
    bottom_right_align: bool = True,
    use_sliding_window: bool = False,
    window_size_left_val: int = -1,
    tolerance: float = 0.1,
):
    """Multi-round prefill regression aligned with the attention-plugin unit test.

    Each round appends ``seq_len`` new tokens to a KV cache with physical
    capacity ``kv_cache_capacity >> effective_kv_len``, exercising the
    ``cap != s_k`` stride path. Validates against a numpy reference per round.
    """
    _tag = "[llm_prefill_test]"
    b = batch_size
    cap = kv_cache_capacity
    h_r = h_q // h_k
    window_size_left = window_size_left_val if use_sliding_window else None
    window_size_right = 0 if is_causal else None

    print(f"{_tag} Running multi-round prefill accuracy test:")
    print(
        f"{_tag}   b={b}, seq_len={seq_len}, rounds={num_rounds}, "
        f"cap={cap}, h_q={h_q}, h_k={h_k}, d={d}, is_causal={is_causal}"
    )

    d_chunk_k = BlackwellFusedMultiHeadAttentionForward.D_CHUNK
    actual_head_dim = d

    if h_q % h_k != 0:
        raise ValueError("h_q must be divisible by h_k")
    if num_rounds * seq_len > cap:
        raise ValueError(
            f"total tokens ({num_rounds * seq_len}) exceeds capacity ({cap})"
        )

    cp.random.seed(42)
    np.random.seed(42)

    # FP16 test: per-tensor scales are all 1.0; kernel folds 1/sqrt(d) internally.
    _scale_q = 1.0
    _scale_k = 1.0
    _scale_v = 1.0
    _inv_scale_o = 1.0
    ref_scale_softmax = 1.0 / math.sqrt(d)

    mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    if bottom_right_align:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    fmha_op = BlackwellFusedMultiHeadAttentionForward(
        Float32, Float32, (*mma_tiler_mn, d_chunk_k),
        is_persistent, mask_type, use_sliding_window=use_sliding_window,
        is_causal=is_causal, actual_head_dim=actual_head_dim,
    )
    fmha_op.validate_config_host(cutlass.Float16)
    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    _wsl = Int32(window_size_left_val) if use_sliding_window else Int32(0)

    def _to_cute(arr, element_type):
        t = from_dlpack(arr, assumed_align=16)
        t.element_type = element_type
        return t

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
        new_k_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(
            np.float32
        )
        new_v_np = np.random.randint(-2, 2, (b, h_k, seq_len, d)).astype(
            np.float32
        )

        kv_np[:, 0, :, current_pos:current_pos + seq_len, :] = new_k_np
        kv_np[:, 1, :, current_pos:current_pos + seq_len, :] = new_v_np

        q_cp = cp.asarray(q_np.astype(np.float16))
        kv_cp = cp.asarray(kv_np.astype(np.float16))
        o_cp = cp.zeros((b, seq_len, h_q, d), dtype=cp.float16)

        q_t = mark_bshd_dynamic(_to_cute(q_cp, cutlass.Float16))
        kv_t = mark_kv_cache_dynamic(_to_cute(kv_cp, cutlass.Float16))
        o_t = mark_bshd_dynamic(_to_cute(o_cp, cutlass.Float16))

        cu_kv_np = np.arange(b + 1, dtype=np.int32) * effective_kv_len
        cu_kv_cp = cp.asarray(cu_kv_np)
        cu_kv = from_dlpack(cu_kv_cp, assumed_align=16)
        cu_kv = mark_1d_dynamic(cu_kv)

        if compiled_fmha is None:
            start_time = time.time()
            compiled_fmha = cute.compile(
                fmha_op, q_t, kv_t, o_t, cu_kv, _wsl,
                _scale_q, _scale_k, _scale_v, _inv_scale_o,
                current_stream,
            )
            print(
                f"{_tag} Compilation time: {time.time() - start_time:.4f}s"
            )

        compiled_fmha(
            q_t, kv_t, o_t, cu_kv, _wsl,
            _scale_q, _scale_k, _scale_v, _inv_scale_o,
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
            scores = (
                np.einsum("hqd,hkd->hqk", q_b, k_b) * ref_scale_softmax
            )
            s_k = effective_kv_len
            if window_size_left is not None or window_size_right is not None:
                q_coords = np.arange(seq_len).reshape(-1, 1)
                k_coords = np.arange(s_k).reshape(1, -1)
                offset = (s_k - seq_len) if bottom_right_align else 0
                if window_size_left is None:
                    mask = k_coords > q_coords + offset + window_size_right
                elif window_size_right is None:
                    mask = k_coords < q_coords + offset - window_size_left
                else:
                    mask = (
                        (k_coords > q_coords + offset + window_size_right)
                        | (k_coords < q_coords + offset - window_size_left)
                    )
                scores = np.where(mask, -np.inf, scores)
            probs = numpy_softmax(scores, axis=-1)
            o_ref = np.einsum("hqk,hkd->hqd", probs, v_b)
            o_ref = o_ref.transpose(1, 0, 2) * _scale_v * _inv_scale_o
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

            # FMHA_DEBUG_DCHUNK: per-d_chunk breakdown to detect if D>128
            # PV gemm is wrongly accumulating across d_chunks. Prints, for
            # head=0 token=0, the GPU output / numpy ref / abs(diff) summed
            # over each 128-wide d_chunk slice. If GPU O matches ref on
            # d_chunk_0 but differs on d_chunk_1 (or all chunks look like
            # a sum of the per-chunk truths), the bug is in the PV
            # accumulation / epilogue replication.
            if os.environ.get("FMHA_DEBUG_DCHUNK"):
                head_dim = o_actual.shape[-1]
                d_chunk_k = 128
                if head_dim > d_chunk_k:
                    num_d_chunks = (head_dim + d_chunk_k - 1) // d_chunk_k
                    print(
                        f"    [DCHUNK] head_dim={head_dim} "
                        f"num_d_chunks={num_d_chunks}"
                    )
                    for ci in range(num_d_chunks):
                        lo = ci * d_chunk_k
                        hi = min(lo + d_chunk_k, head_dim)
                        a_chunk = o_actual[..., lo:hi]
                        r_chunk = o_ref[..., lo:hi]
                        d_chunk = np.abs(a_chunk - r_chunk)
                        print(
                            f"    [DCHUNK chunk={ci} d=[{lo}:{hi})] "
                            f"actual: mean={a_chunk.mean():.4f} "
                            f"max={np.abs(a_chunk).max():.4f}  "
                            f"ref: mean={r_chunk.mean():.4f} "
                            f"max={np.abs(r_chunk).max():.4f}  "
                            f"diff: mean={d_chunk.mean():.4f} "
                            f"max={d_chunk.max():.4f}"
                        )
                    # Token 0, head 0: print first few values of each chunk
                    print(
                        f"    [DCHUNK token=0 head=0] (first 8 values "
                        f"per chunk)"
                    )
                    for ci in range(num_d_chunks):
                        lo = ci * d_chunk_k
                        a_slice = o_actual[0, 0, lo:lo + 8]
                        r_slice = o_ref[0, 0, lo:lo + 8]
                        print(
                            f"      chunk={ci} actual={a_slice.tolist()}"
                        )
                        print(
                            f"      chunk={ci}    ref={r_slice.tolist()}"
                        )
        current_pos += seq_len

    if all_pass:
        print(f"\n{_tag} All {num_rounds} rounds passed.")
    else:
        raise AssertionError(f"{_tag} Some rounds failed accuracy check!")
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_comma_separated_ints(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def _parse_nested_comma_separated_ints(s: str):
    try:
        s = s.strip()
        if "(" not in s:
            return tuple(int(x.strip()) for x in s.split(","))

        start = s.find("(")
        end = s.find(")")
        if start == -1 or end == -1:
            raise ValueError("Mismatched parentheses")

        before = s[:start].strip().rstrip(",")
        middle = s[start + 1:end].strip()
        after = s[end + 1:].strip().lstrip(",")

        result = []
        if before:
            result.extend(int(x.strip()) for x in before.split(","))
        if middle:
            nested_tuple = tuple(int(x.strip()) for x in middle.split(","))
            result.append(nested_tuple)
        if after:
            result.extend(int(x.strip()) for x in after.split(","))
        return tuple(result)
    except ValueError as e:
        if str(e) == "Mismatched parentheses":
            raise argparse.ArgumentTypeError("Mismatched parentheses in input")
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers with optional "
            "parentheses for nested tuple."
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example of FMHA on Blackwell.")
    parser.add_argument("--in_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--out_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--qk_acc_dtype", type=cutlass.dtype, default=Float32)
    parser.add_argument("--pv_acc_dtype", type=cutlass.dtype, default=Float32)
    parser.add_argument(
        "--mma_tiler_mn", type=_parse_comma_separated_ints, default=(128, 128)
    )
    parser.add_argument("--is_persistent", action="store_true")
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--bottom_right_align", action="store_true")
    parser.add_argument("--lse_calculation", action="store_true")
    parser.add_argument(
        "--window_size", type=_parse_comma_separated_ints, default=(-1, -1)
    )
    parser.add_argument(
        "--q_shape",
        type=_parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of Q (B, S_q, H, D)",
    )
    parser.add_argument(
        "--k_shape",
        type=_parse_nested_comma_separated_ints,
        default=(1, 256, 8, 128),
        help="Shape of K (B, S_k, H_k, D)",
    )
    parser.add_argument("--scale_q", type=float, default=1.0)
    parser.add_argument("--scale_k", type=float, default=1.0)
    parser.add_argument("--scale_v", type=float, default=1.0)
    parser.add_argument("--inv_scale_o", type=float, default=1.0)
    parser.add_argument("--scale_softmax", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1e-1)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./fmha_aot_artifacts")
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--file_name", type=str, default="fmha")
    parser.add_argument("--function_prefix", type=str, default="fmha")
    parser.add_argument(
        "--vit_mode",
        action="store_true",
        help="Compile ViT FMHA variant: packed varlen with separate Q/K/V, "
        "bidirectional (no causal mask). Produces a different ABI.",
    )
    return parser


def main(argv=None):
    """CLI entry point. Parses ``--q_shape`` etc. then calls :func:`run`."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this example!")

    if len(args.q_shape) != 4:
        parser.error("--q_shape must contain exactly 4 values")
    if len(args.k_shape) != 4:
        parser.error("--k_shape must contain exactly 4 values")
    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if args.vit_mode:
        assert args.k_shape == args.q_shape, (
            f"vit_mode requires k_shape == q_shape; got k_shape={args.k_shape}, "
            f"q_shape={args.q_shape}"
        )
        assert not args.is_causal, (
            "vit_mode is bidirectional; --is_causal must not be set"
        )
        assert args.window_size == (-1, -1), (
            f"vit_mode does not support sliding window; "
            f"got --window_size={args.window_size}"
        )

    latency = run(
        args.q_shape, args.k_shape, args.in_dtype, args.out_dtype,
        args.qk_acc_dtype, args.pv_acc_dtype, args.mma_tiler_mn,
        args.is_persistent, args.is_causal, args.bottom_right_align,
        args.lse_calculation, args.window_size,
        args.scale_q, args.scale_k, args.scale_v, args.inv_scale_o,
        args.scale_softmax, args.tolerance,
        args.warmup_iterations, args.iterations,
        args.skip_ref_check, args.use_cold_l2,
        output_dir=args.output_dir, export_only=args.export_only,
        file_name=args.file_name, function_prefix=args.function_prefix,
        vit_mode=args.vit_mode,
    )
    if latency is not None:
        print(f"[fmha] kernel latency: {latency:.2f} us")
    return latency
