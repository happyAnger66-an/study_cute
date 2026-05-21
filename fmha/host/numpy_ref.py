"""NumPy reference implementations for FMHA.

Used by :mod:`fmha.runner` to validate kernel output. None of this code runs
on the GPU; all routines operate on plain ``numpy.ndarray``.
"""

import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack

from fmha.host.tensor_layout import get_leading_dim


def numpy_softmax(x, axis=-1):
    """Numerically stable softmax with handling for all-``-inf`` rows."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    e_x = np.exp(x - x_max)
    s = np.sum(e_x, axis=axis, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e_x / s


def numpy_logsumexp(x, axis=-1):
    """Numerically stable log-sum-exp; treats ``-inf`` rows as 0."""
    x_max = np.max(x, axis=axis)
    x_max_safe = np.where(np.isfinite(x_max), x_max, 0.0)
    return np.log(np.sum(np.exp(x - x_max_safe[..., np.newaxis]), axis=axis)) + x_max


def run_numpy_single_shot_reference_packed(
    q_packed,
    k_packed,
    v_packed,
    cu_seqlens_q,
    cu_seqlens_k,
    scale_softmax: float = 1.0,
    scale_output: float = 1.0,
    is_causal: bool = False,
    bottom_right_align: bool = False,
    lse_calculation: bool = False,
    window_size_left=None,
    window_size_right=None,
):
    """Single-shot packed (ViT-style) numpy attention reference.

    ``q_packed`` is shaped ``[total_q, H_q, D]``; ``k_packed`` / ``v_packed`` are
    ``[total_k, H_k, D]``. ``cu_seqlens_q`` / ``cu_seqlens_k`` give per-batch
    offsets of length ``B+1``.

    GQA is supported transparently: when ``H_q > H_k`` the K/V are broadcast
    via ``np.repeat`` along the head axis.
    """
    h_q_local = q_packed.shape[1]
    h_k_local = k_packed.shape[1]
    if h_q_local % h_k_local != 0:
        raise ValueError("H_q must be divisible by H_k in packed reference")
    repeat_factor = h_q_local // h_k_local
    _wsr = 0 if is_causal else window_size_right

    ref_list = []
    lse_list = []
    batch_size = len(cu_seqlens_q) - 1
    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx]
        q_end = cu_seqlens_q[batch_idx + 1]
        k_start = cu_seqlens_k[batch_idx]
        k_end = cu_seqlens_k[batch_idx + 1]

        q_i = q_packed[q_start:q_end].transpose(1, 0, 2)
        k_i = k_packed[k_start:k_end].transpose(1, 0, 2)
        v_i = v_packed[k_start:k_end].transpose(1, 0, 2)

        if repeat_factor > 1:
            k_i = np.repeat(k_i, repeat_factor, axis=0)
            v_i = np.repeat(v_i, repeat_factor, axis=0)

        s_i = np.einsum("hqd,hkd->hqk", q_i, k_i) * scale_softmax
        s_q_local = q_i.shape[1]
        s_k_local = k_i.shape[1]

        if window_size_left is not None or _wsr is not None:
            q_coords = np.arange(s_q_local).reshape(-1, 1)
            k_coords = np.arange(s_k_local).reshape(1, -1)
            offset = 0 if not bottom_right_align else s_k_local - s_q_local
            if window_size_left is None:
                _mask = k_coords > q_coords + offset + _wsr
            elif _wsr is None:
                _mask = k_coords < q_coords + offset - window_size_left
            else:
                _mask = (k_coords > q_coords + offset + _wsr) | (
                    k_coords < q_coords + offset - window_size_left
                )
            s_i = np.where(_mask, -np.inf, s_i)

        if lse_calculation:
            lse_i = numpy_logsumexp(s_i, axis=-1)
        else:
            lse_i = None

        p_i = numpy_softmax(s_i, axis=-1)
        ref_i = np.einsum("hqk,hkd->hqd", p_i, v_i)
        ref_i = ref_i.transpose(1, 0, 2) * scale_output
        ref_list.append(ref_i)
        if lse_calculation:
            lse_list.append(lse_i.transpose(1, 0))

    ref = np.concatenate(ref_list, axis=0)
    lse = np.concatenate(lse_list, axis=0) if lse_calculation else None
    return ref, lse


def maybe_quantize_ref_for_narrow_out(o_ref_np, out_dtype, tolerance):
    """Round-trip a fp32 reference through the kernel's narrow output dtype.

    For FP8 / Int4 outputs, the kernel quantizes its final result. To make the
    numerical comparison fair, we quantize-then-dequantize the reference through
    the same dtype so both sides incur identical rounding error.

    Returns ``(adjusted_ref_np, adjusted_tolerance)``. For non-narrow dtypes the
    reference and tolerance are returned unchanged.
    """
    if not (out_dtype.is_float and out_dtype.width <= 8):
        return o_ref_np, tolerance

    ref_narrow_cp = cp.empty(o_ref_np.shape, dtype=cp.uint8)
    ref_narrow_cute = from_dlpack(ref_narrow_cp, assumed_align=16)
    ref_narrow_cute.element_type = out_dtype
    ref_narrow_cute = ref_narrow_cute.mark_layout_dynamic(
        leading_dim=get_leading_dim(ref_narrow_cp)
    )

    ref_o_f32_cp = cp.asarray(o_ref_np)
    ref_o_f32_cute = from_dlpack(ref_o_f32_cp, assumed_align=16)
    ref_o_f32_cute.element_type = cutlass.Float32
    ref_o_f32_cute = ref_o_f32_cute.mark_layout_dynamic(
        leading_dim=get_leading_dim(ref_o_f32_cp)
    )

    cute.testing.convert(ref_o_f32_cute, ref_narrow_cute)
    cute.testing.convert(ref_narrow_cute, ref_o_f32_cute)
    return ref_o_f32_cp.get(), 0.13
