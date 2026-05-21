"""Top-level ``@cute.kernel`` for FMHA.

This module contains the kernel entry point and the warp-role dispatcher. Each
warp role's body lives in its own ``fmha.device.warp_*`` module and is invoked
here once the kernel has built the shared resources (pipelines, SMEM tensors,
TMEM fragments).

Bound onto :class:`BlackwellFusedMultiHeadAttentionForward` as ``self.kernel``
from :mod:`fmha.__init__`.
"""

from typing import Optional, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.typing import Float32, Int32

from fmha import fmha_helpers as fmha_utils

from fmha.host.config import make_thread_cooperative_group


@cute.kernel
def kernel(
    self,
    qk_tiled_mma: cute.TiledMma,
    pv_tiled_mma: cute.TiledMma,
    tma_atom_q: cute.CopyAtom,
    mQ_qdl: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK_kdl: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV_dkl: cute.Tensor,
    tma_atom_o: cute.CopyAtom,
    mO_qdl: cute.Tensor,
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    mLSE: Optional[cute.Tensor],
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    scale_output: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    q_smem_layout_staged: cute.ComposedLayout,
    k_smem_layout_staged: cute.ComposedLayout,
    p_tmem_layout_staged: cute.ComposedLayout,
    v_smem_layout_staged: cute.ComposedLayout,
    o_smem_layout_staged: cute.ComposedLayout,
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Warp-specialized FMHA kernel: dispatches to per-role bodies.

    Warp roles (see fmha.md for the pipeline diagram):

    - ``empty_warp``        idle, only used to init the tmem dealloc mbarrier
    - ``load_warp``         TMA G->S for Q/K/V (one warp)
    - ``mma_warp``          QK and PV ``cute.gemm`` (one warp, allocates TMEM)
    - ``softmax0/1 warps``  online softmax on S0 / S1 (two warpgroups, 4 warps each)
    - ``correction warps``  rescale O0 / O1, write LSE, dump O to SMEM (one warpgroup)
    - ``epilogue warp``     TMA S->G for O (one warp)
    """
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    # ------------------------------------------------------------------
    # 1. TMA descriptor prefetch (only the load warp needs them in L1)
    # ------------------------------------------------------------------
    if warp_idx == self.load_warp_id:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

    # ------------------------------------------------------------------
    # 2. SMEM allocation
    # ------------------------------------------------------------------
    smem = utils.SmemAllocator()
    storage = smem.allocate(self.shared_storage)

    # ------------------------------------------------------------------
    # 3. Build pipelines (one per producer/consumer pair)
    # ------------------------------------------------------------------
    load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=self.q_stage,
        producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
        consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        tx_count=self.tma_copy_q_bytes,
        barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
    ).make_participants()
    load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=self.kv_stage,
        producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
        consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        tx_count=self.tma_copy_kv_bytes,
        barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
    ).make_participants()
    mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=self.mma_softmax_stage,
        producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.softmax0_warp_ids)
        ),
        barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
    ).make_participants()
    mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=self.mma_softmax_stage,
        producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.softmax1_warp_ids)
        ),
        barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
    ).make_participants()
    s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
        num_stages=self.softmax_corr_stage,
        producer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.softmax0_warp_ids)
        ),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.correction_warp_ids)
        ),
        barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
    ).make_participants()
    s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
        num_stages=self.softmax_corr_stage,
        producer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.softmax1_warp_ids)
        ),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.correction_warp_ids)
        ),
        barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
    ).make_participants()
    corr_epi_producer, corr_epi_consumer = pipeline.PipelineAsync.create(
        num_stages=self.epi_stage,
        producer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.correction_warp_ids)
        ),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len([self.epilogue_warp_id])
        ),
        barrier_storage=storage.corr_epi_mbar_ptr.data_ptr(),
    ).make_participants()
    mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=self.mma_corr_stage,
        producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        consumer_group=make_thread_cooperative_group(
            self.threads_per_warp * len(self.correction_warp_ids)
        ),
        barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
    ).make_participants()
    s0_s1_sequence_producer, s0_s1_sequence_consumer = (
        pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s0_s1_sequence_mbar_ptr.data_ptr(),
        ).make_participants()
    )
    tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

    # ------------------------------------------------------------------
    # 4. TMEM dealloc mbarrier init (only one thread does this)
    # ------------------------------------------------------------------
    if warp_idx == self.empty_warp_id:
        cute.arch.mbarrier_init(
            tmem_dealloc_mbar_ptr,
            self.threads_per_warp
            * len(
                (
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
    cute.arch.mbarrier_init_fence()

    # ------------------------------------------------------------------
    # 5. SMEM tensors (Q, K, V, O); V aliases K's SMEM buffer
    # ------------------------------------------------------------------
    sQ = storage.sQ.get_tensor(
        q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
    )
    sK = storage.sK.get_tensor(
        k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
    )
    # Strip swizzle info to reuse smem
    sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
    sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
    sO = storage.sO.get_tensor(
        o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
    )

    # ------------------------------------------------------------------
    # 6. MMA fragments (operand registers) + TMEM accumulators (S/O/P)
    # ------------------------------------------------------------------
    qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
    pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
    tSrQ = qk_thr_mma.make_fragment_A(sQ)
    tSrK = qk_thr_mma.make_fragment_B(sK)
    tOrV = pv_thr_mma.make_fragment_B(sV)
    qk_acc_shape = qk_thr_mma.partition_shape_C(
        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
    )
    tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)
    pv_acc_shape = pv_thr_mma.partition_shape_C(
        (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
    )
    tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

    # Two Q-tile views of the same S/O TMEM region (offsets defined in config)
    tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
    tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
    tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
    tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

    tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
    tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
    tOrP0 = cute.make_tensor(
        tOrP.iterator
        + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
        tOrP.layout,
    )
    tOrP1 = cute.make_tensor(
        tOrP.iterator
        + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
        tOrP.layout,
    )

    # CTA-wide rendezvous: all warps must finish setup before any can start.
    self.cta_sync_barrier.arrive_and_wait()

    # ------------------------------------------------------------------
    # 7. Warp dispatch
    # ------------------------------------------------------------------
    if warp_idx == self.empty_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_other)

    if warp_idx == self.load_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        self.load_warp_body(
            tma_atom_q, mQ_qdl,
            tma_atom_k, mK_kdl,
            tma_atom_v, mV_dkl,
            sQ, sK, sV,
            qk_thr_mma, pv_thr_mma,
            load_q_producer, load_kv_producer,
            cum_seqlen_q, cum_seqlen_k,
            window_size_left, window_size_right,
            tile_sched_params,
        )

    if warp_idx == self.mma_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        self.mma_warp_body(
            qk_tiled_mma, pv_tiled_mma,
            tSrQ, tSrK, tOrV,
            tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1,
            load_q_consumer, load_kv_consumer,
            mma_s0_producer, mma_s1_producer, mma_corr_producer,
            mQ_qdl, mK_kdl,
            cum_seqlen_q, cum_seqlen_k,
            window_size_left, window_size_right,
            storage,
            tmem_dealloc_mbar_ptr,
            tile_sched_params,
        )

    if warp_idx == self.epilogue_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_other)
        self.epilogue_warp_body(
            tma_atom_o, mO_qdl, sO,
            corr_epi_consumer,
            cum_seqlen_q, mQ_qdl,
            tile_sched_params,
        )

    if warp_idx < self.softmax1_warp_ids[0]:
        # softmax0 warpgroup (warps 0..3)
        cute.arch.setmaxregister_increase(self.num_regs_softmax)
        self.softmax(
            stage=0,
            seqlen_k=mK_kdl.shape[0],
            seqlen_q=mQ_qdl.shape[0],
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            scale_softmax_log2=scale_softmax_log2,
            qk_thr_mma=qk_thr_mma,
            tStS=tStS,
            tStSi=tStS0,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mma_si_consumer=mma_s0_consumer,
            si_corr_producer=s0_corr_producer,
            s0_s1_sequence_consumer=s0_s1_sequence_consumer,
            s0_s1_sequence_producer=s0_s1_sequence_producer,
            tile_sched_params=tile_sched_params,
        )
        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    if (
        warp_idx < self.correction_warp_ids[0]
        and warp_idx >= self.softmax1_warp_ids[0]
    ):
        # softmax1 warpgroup (warps 4..7)
        cute.arch.setmaxregister_increase(self.num_regs_softmax)
        self.softmax(
            stage=1,
            seqlen_k=mK_kdl.shape[0],
            seqlen_q=mQ_qdl.shape[0],
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            scale_softmax_log2=scale_softmax_log2,
            qk_thr_mma=qk_thr_mma,
            tStS=tStS,
            tStSi=tStS1,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mma_si_consumer=mma_s1_consumer,
            si_corr_producer=s1_corr_producer,
            s0_s1_sequence_consumer=s0_s1_sequence_consumer,
            s0_s1_sequence_producer=s0_s1_sequence_producer,
            tile_sched_params=tile_sched_params,
        )
        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
        cute.arch.setmaxregister_decrease(self.num_regs_correction)
        self.correction_warp_body(
            pv_thr_mma, qk_thr_mma,
            tStS, tOtO0, tOtO1,
            mLSE, mQ_qdl, mK_kdl,
            sO,
            scale_softmax_log2, scale_softmax, scale_output,
            window_size_left, window_size_right,
            cum_seqlen_q, cum_seqlen_k,
            s0_corr_consumer, s1_corr_consumer,
            mma_corr_consumer, corr_epi_producer,
            tile_sched_params,
        )
        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    return
