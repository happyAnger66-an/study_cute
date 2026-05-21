"""``@cute.jit`` host launchers for the FMHA kernel.

This module defines the two entry points :func:`_call_llm` and :func:`_call_vit`
that get bound onto :class:`BlackwellFusedMultiHeadAttentionForward` as
``__call__`` / ``__call_vit__`` from :mod:`fmha.__init__`.

The two entry points only differ in how raw user tensors are reshaped into the
canonical ``(s, d, ((h_r, h_k), b))`` cute layouts. All MMA / SMEM / TMA / launch
plumbing is in the shared helper :func:`_build_kernel_inputs_and_launch`.
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64

from fmha import fmha_helpers as fmha_utils


# ---------------------------------------------------------------------------
# Shared builder + launcher
# ---------------------------------------------------------------------------
@cute.jit
def _build_kernel_inputs_and_launch(
    self,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    lse: Optional[cute.Tensor],
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    scale_output: Float32,
    window_size_left: Optional[Int32],
    stream: cuda.CUstream,
):
    """Shared body: from canonical Q/K/V/O tensors, build MMA / SMEM / TMA
    objects, declare SharedStorage, and launch ``self.kernel``.

    Both LLM and ViT entry points massage their raw inputs into the canonical
    ``(s, d, ((h_r, h_k), b))`` layout and then delegate here.
    """
    # ------------------------------------------------------------------
    # static attribute setup
    # ------------------------------------------------------------------
    self.q_dtype = q.element_type
    self.k_dtype = k.element_type
    self.v_dtype = v.element_type
    self.o_dtype = o.element_type

    s_q = q.layout.shape[0]
    d = self.head_dim
    h_r, h_k = q.layout.shape[2][0]
    b = q.layout.shape[2][1]

    self.tile_sched_params, grid = fmha_utils.compute_grid(
        cute.shape((s_q, d, ((h_r, h_k), b))),
        self.cta_tiler,
        self.is_persistent,
    )

    self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
    self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
    self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
    self.o_layout = utils.LayoutEnum.from_tensor(o)

    if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
        raise RuntimeError("The layout of q is not supported")
    if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
        raise RuntimeError("The layout of k is not supported")
    if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
        raise RuntimeError("The layout of v is not supported")

    if cutlass.const_expr(self.q_dtype != self.k_dtype):
        raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
    if cutlass.const_expr(self.q_dtype != self.v_dtype):
        raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
    self._setup_attributes()

    # ------------------------------------------------------------------
    # MMA atoms (one per gemm: QK^T and P V)
    # ------------------------------------------------------------------
    cta_group = tcgen05.CtaGroup.ONE
    # The intermediate tensor P is from tmem & k-major.
    p_source = tcgen05.OperandSource.TMEM
    p_major_mode = tcgen05.OperandMajorMode.K

    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.q_dtype,
        self.q_major_mode,
        self.k_major_mode,
        self.qk_acc_dtype,
        cta_group,
        self.qk_mma_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.v_dtype,
        p_major_mode,
        self.v_major_mode,
        self.pv_acc_dtype,
        cta_group,
        self.pv_mma_tiler[:2],
        p_source,
    )

    self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
    self.cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(self.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )

    self.epi_tile = self.pv_mma_tiler[:2]

    # ------------------------------------------------------------------
    # SMEM layouts (staged: each has an outer pipeline mode)
    # ------------------------------------------------------------------
    q_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage,
    )
    k_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, self.qk_mma_tiler, self.k_dtype, self.kv_stage,
    )
    p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.acc_stage,
    )
    v_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.kv_stage,
    )
    o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
        self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage,
    )

    # ------------------------------------------------------------------
    # TMA atoms (G->S for Q/K/V; S->G for O)
    # ------------------------------------------------------------------
    tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
    tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op, q, q_smem_layout, self.qk_mma_tiler,
        qk_tiled_mma, self.cluster_layout_vmnk.shape,
    )

    k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, k, k_smem_layout, self.qk_mma_tiler,
        qk_tiled_mma, self.cluster_layout_vmnk.shape,
    )

    v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_op, v, v_smem_layout, self.pv_mma_tiler,
        pv_tiled_mma, self.cluster_layout_vmnk.shape,
    )

    o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
    tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
        tma_store_op, o, o_smem_layout, self.epi_tile,
    )

    self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
    self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)

    # ------------------------------------------------------------------
    # SharedStorage struct (pipeline barriers + SMEM tiles)
    # ------------------------------------------------------------------
    self.shared_storage = _make_shared_storage(
        self,
        q_smem_layout_staged,
        k_smem_layout_staged,
        o_smem_layout_staged,
    )

    # ------------------------------------------------------------------
    # Compile-time dispatch for causal / sliding-window mask
    # ------------------------------------------------------------------
    if cutlass.const_expr(self.use_sliding_window):
        _wsl = window_size_left
    else:
        _wsl = None
    _wsr = Int32(0) if cutlass.const_expr(self.is_causal) else None

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    self.kernel(
        qk_tiled_mma, pv_tiled_mma,
        tma_atom_q, tma_tensor_q,
        tma_atom_k, tma_tensor_k,
        tma_atom_v, tma_tensor_v,
        tma_atom_o, tma_tensor_o,
        cum_seqlen_q, cum_seqlen_k, lse,
        scale_softmax_log2, scale_softmax, scale_output,
        _wsl, _wsr,
        q_smem_layout_staged, k_smem_layout_staged,
        p_tmem_layout_staged, v_smem_layout_staged,
        o_smem_layout_staged, self.tile_sched_params,
    ).launch(
        grid=grid,
        block=[self.threads_per_cta, 1, 1],
        cluster=self.cluster_shape_mnk,
        stream=stream,
        min_blocks_per_mp=1,
    )


def _make_shared_storage(
    cfg,
    q_smem_layout_staged,
    k_smem_layout_staged,
    o_smem_layout_staged,
):
    """Build the SharedStorage struct used by the kernel.

    Defined as a local class (instead of file-scope) so each ``cute.compile``
    can specialize the struct to the current dtypes / staging counts. The two
    launchers share this so the layout is identical between LLM and ViT.
    """

    @cute.struct
    class SharedStorage:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, cfg.q_stage * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, cfg.kv_stage * 2]
        mma_s0_mbar_ptr: cute.struct.MemRange[Int64, cfg.mma_softmax_stage * 2]
        mma_s1_mbar_ptr: cute.struct.MemRange[Int64, cfg.mma_softmax_stage * 2]
        s0_corr_mbar_ptr: cute.struct.MemRange[Int64, cfg.softmax_corr_stage * 2]
        s1_corr_mbar_ptr: cute.struct.MemRange[Int64, cfg.softmax_corr_stage * 2]
        s0_s1_sequence_mbar_ptr: cute.struct.MemRange[
            Int64, cfg.softmax_warpgroup_count
        ]
        corr_epi_mbar_ptr: cute.struct.MemRange[Int64, cfg.epi_stage * 2]
        mma_corr_mbar_ptr: cute.struct.MemRange[Int64, cfg.mma_corr_stage * 2]
        tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
        # Tmem holding buffer (single i32, holds the dynamic tmem base ptr)
        tmem_holding_buf: Int32
        # Smem tensors
        sO: cute.struct.Align[
            cute.struct.MemRange[cfg.o_dtype, cute.cosize(o_smem_layout_staged)],
            cfg.buffer_align_bytes,
        ]
        sQ: cute.struct.Align[
            cute.struct.MemRange[cfg.q_dtype, cute.cosize(q_smem_layout_staged)],
            cfg.buffer_align_bytes,
        ]
        sK: cute.struct.Align[
            cute.struct.MemRange[cfg.k_dtype, cute.cosize(k_smem_layout_staged)],
            cfg.buffer_align_bytes,
        ]

    return SharedStorage


# ---------------------------------------------------------------------------
# LLM entry point: __call__
# ---------------------------------------------------------------------------
@cute.jit
def _call_llm(
    self,
    q_tensor: cute.Tensor,           # (B, S_q, H_q, D)
    kv_cache: cute.Tensor,           # (B, 2, H_kv, S_k, D)
    o_tensor: cute.Tensor,           # (B, S_q, H_q, D)
    cum_seqlen_k: cute.Tensor,       # (B+1,) Int32
    window_size_left: Int32,
    scale_q: Float32,
    scale_k: Float32,
    scale_v: Float32,
    inv_scale_o: Float32,
    stream: cuda.CUstream,
):
    """LLM entry point: batched [B, S, H, D] Q + KV cache [B, 2, H, cap, D].

    softmax_scale = scale_q * scale_k * (1 / sqrt(head_dim))
    For FP16 (no quantization), pass scale_q = scale_k = scale_v = inv_scale_o = 1.0.
    For FP8, pass the dequant scales so they're folded into softmax/output scaling.
    """
    scale_softmax = scale_q * scale_k * self.inv_sqrt_head_dim
    scale_softmax_log2 = scale_softmax * self.log2_e
    scale_output = scale_v * inv_scale_o

    b = q_tensor.layout.shape[0]
    s_q = q_tensor.layout.shape[1]
    h_q = q_tensor.layout.shape[2]
    h_k = kv_cache.layout.shape[2]
    cap = kv_cache.layout.shape[3]
    d = self.head_dim

    q_iter = q_tensor.iterator
    # KV cache shape[3] = cap (physical capacity, for stride computation).
    # Per-batch actual KV lengths come from cum_seqlen_k at runtime.
    kv_base = kv_cache.iterator
    stride_kv_head = cap * d
    stride_kv_select = h_k * stride_kv_head
    k_iter = kv_base
    v_iter = kv_base + stride_kv_select
    o_iter = o_tensor.iterator

    cum_seqlen_q = None
    lse = None
    h_r = h_q // h_k

    qo_offset = 0
    kv_offset = 0
    stride_b_qo = h_r * h_k * s_q * d
    stride_b_kv = 2 * stride_kv_select

    # (s, d, ((h_r, h_k), b))
    q_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    q = cute.make_tensor(q_iter + qo_offset, q_layout)
    # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    k_layout = cute.make_layout(
        (cap, d, ((h_r, h_k), b)),
        stride=(d, 1, ((0, stride_kv_head), stride_b_kv)),
    )
    k = cute.make_tensor(k_iter + kv_offset, k_layout)
    # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
    v_layout = cute.make_layout(
        (d, cap, ((h_r, h_k), b)),
        stride=(1, d, ((0, stride_kv_head), stride_b_kv)),
    )
    v = cute.make_tensor(v_iter + kv_offset, v_layout)
    # (s, d, ((h_r, h_k), b))
    o_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    o = cute.make_tensor(o_iter + qo_offset, o_layout)

    _build_kernel_inputs_and_launch(
        self, q, k, v, o, lse, cum_seqlen_q, cum_seqlen_k,
        scale_softmax_log2, scale_softmax, scale_output,
        window_size_left, stream,
    )


# ---------------------------------------------------------------------------
# ViT entry point: __call_vit__
# ---------------------------------------------------------------------------
@cute.jit
def _call_vit(
    self,
    q_tensor: cute.Tensor,   # (total_S, H_q, D)
    k_tensor: cute.Tensor,   # (total_S, H_kv, D)
    v_tensor: cute.Tensor,   # (total_S, H_kv, D)
    o_tensor: cute.Tensor,   # (total_S, H_q, D)
    cu_seqlens: cute.Tensor, # (B+1,) Int32
    max_seqlen: Int32,
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    scale_output: Float32,
    stream: cuda.CUstream,
):
    """ViT entry point: packed varlen Q/K/V, bidirectional, no causal mask.

    All sequences are packed into flat [total_S, H, D] tensors with boundaries
    defined by ``cu_seqlens``. ``max_seqlen`` is the longest individual sequence
    length (not total_S); it controls grid size and tile counts. Per-batch
    boundaries enter the kernel via ``domain_offset`` at run time.
    """
    s_q = max_seqlen
    h_q = q_tensor.layout.shape[1]
    h_k = k_tensor.layout.shape[1]
    s_k = max_seqlen
    d = self.head_dim

    b = cu_seqlens.layout.shape[0] - 1
    h_r = h_q // h_k

    q_iter = q_tensor.iterator
    k_iter = k_tensor.iterator
    v_iter = v_tensor.iterator
    o_iter = o_tensor.iterator

    cum_seqlen_q = cu_seqlens
    cum_seqlen_k = cu_seqlens
    lse = None

    qo_offset = -s_q * d * h_r * h_k
    kv_offset = -s_k * d * h_k
    b_qo = s_q * (1 + b)
    b_kv = s_k * (1 + b)
    stride_b_qo = d * h_r * h_k
    stride_b_kv = d * h_k

    q_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b_qo)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    q = cute.make_tensor(q_iter + qo_offset, q_layout)
    k_layout = cute.make_layout(
        (s_k, d, ((h_r, h_k), b_kv)),
        stride=(d * h_k, 1, ((0, d), stride_b_kv)),
    )
    k = cute.make_tensor(k_iter + kv_offset, k_layout)
    v_layout = cute.make_layout(
        (d, s_k, ((h_r, h_k), b_kv)),
        stride=(1, d * h_k, ((0, d), stride_b_kv)),
    )
    v = cute.make_tensor(v_iter + kv_offset, v_layout)
    o_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b_qo)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    o = cute.make_tensor(o_iter + qo_offset, o_layout)

    _build_kernel_inputs_and_launch(
        self, q, k, v, o, lse, cum_seqlen_q, cum_seqlen_k,
        scale_softmax_log2, scale_softmax, scale_output,
        Int32(0), stream,
    )
