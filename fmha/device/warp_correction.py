"""Correction warpgroup: O0 / O1 rescaling, LSE write, SMEM O dump.

The correction warpgroup (4 warps) sits between softmax and the epilogue. Each
KV iter it receives ``(old_max, new_max)`` vectors from the two softmax
warpgroups; it then rescales the running O accumulators in TMEM by
``exp(scale_softmax_log2 * (old_max - new_max))`` so the partial PV results
stay numerically consistent with the latest softmax.

After all KV iters it loads the final ``(row_sum, row_max)`` vector and:
- Optionally writes the per-row LSE = ln(row_sum) + scale*row_max
- Multiplies O by ``scale_output / row_sum`` and dumps it to SMEM for the
  epilogue warp to TMA-store.

Module-level helpers :func:`correction_rescale` and :func:`correction_epilog`
are bound onto :class:`BlackwellFusedMultiHeadAttentionForward` so they remain
callable as ``self.correction_*`` from within the kernel.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32

from fmha import fmha_helpers as fmha_utils


@cute.jit
def correction_warp_body(
    self,
    pv_thr_mma: cute.ThrMma,
    qk_thr_mma: cute.ThrMma,
    tStS: cute.Tensor,
    tOtO0: cute.Tensor,
    tOtO1: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    mQ_qdl: cute.Tensor,
    mK_kdl: cute.Tensor,
    sO: cute.Tensor,
    scale_softmax_log2: Float32,
    scale_softmax: Float32,
    scale_output: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    s0_corr_consumer,
    s1_corr_consumer,
    mma_corr_consumer,
    corr_epi_producer,
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Body of the correction warpgroup. See module docstring."""
    tidx, _, _ = cute.arch.thread_idx()

    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
    tScS = qk_thr_mma.partition_C(cS)

    tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
    tStS_vec0 = cute.make_tensor(
        tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout
    )
    tStS_vec1 = cute.make_tensor(
        tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout
    )

    tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
    tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

    tmem_load_v_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
        self.qk_acc_dtype,
    )

    tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
    thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
    thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)

    tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
    tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
    tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)

    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        curr_block_coord_lse = curr_block_coord
        batch_coord = curr_block_coord[2][1]
        seqlen_k = mK_kdl.shape[0]
        continue_cond = False
        cuseqlen_q = Int32(0)
        seqlen_q = mQ_qdl.shape[0]

        if cutlass.const_expr(cum_seqlen_q is not None):
            cuseqlen_q = cum_seqlen_q[batch_coord]
            seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
            # for varlen LSE, batch == 1
            curr_block_coord_lse = (
                curr_block_coord[0],
                curr_block_coord[1],
                (curr_block_coord[2][0], 0),
            )
            continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                self.cta_tiler[0],
                curr_block_coord[0],
                seqlen_q,
            )

        if not continue_cond:
            row_idx = (
                curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
            )
            if cutlass.const_expr(cum_seqlen_k is not None):
                cuseqlen_k = cum_seqlen_k[batch_coord]
                seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

            seqlen_kv_loop_steps = (
                fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                - 1
            )

            # D-chunking outer loop: the entire KV correction loop + final
            # epilog runs ``num_d_chunks`` times per kv tile, mirroring the
            # MMA / LOAD outer loop. Final correction_epilog writes
            # ``sO[..., d_chunk_outer]`` for the epilogue warp to TMA-store
            # to gO[..., d_chunk_outer, ...]. For D<=128 (num_d_chunks=1)
            # this collapses to the legacy single-iteration path.
            for d_chunk_outer in cutlass.range_constexpr(self.num_d_chunks):
                # Ignore first signal from softmax (no correction needed
                # for the prologue's first PV).
                vec0_handle = s0_corr_consumer.wait_and_advance()
                vec0_handle.release()
                vec1_handle = s1_corr_consumer.wait_and_advance()

                for i in cutlass.range(
                    0, seqlen_kv_loop_steps, 1, unroll=1
                ):
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                        tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                    )
                    cute.copy(
                        tiled_tmem_load_vec,
                        tTMEM_LOAD_VECtS0,
                        tTMEM_LOAD_VECrS,
                    )
                    scale_ = scale_softmax_log2 * (
                        tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                    )
                    scale = cute.math.exp2(scale_, fastmath=True)
                    o0_handle = mma_corr_consumer.wait_and_advance()
                    self.correction_rescale(pv_thr_mma, tOtO0, scale)
                    vec1_handle.release()
                    cute.arch.fence_view_async_tmem_store()
                    o0_handle.release()

                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    cute.copy(
                        tiled_tmem_load_vec,
                        tTMEM_LOAD_VECtS1,
                        tTMEM_LOAD_VECrS,
                    )
                    scale_ = scale_softmax_log2 * (
                        tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                    )
                    scale = cute.math.exp2(scale_, fastmath=True)
                    o1_handle = mma_corr_consumer.wait_and_advance()
                    self.correction_rescale(pv_thr_mma, tOtO1, scale)
                    vec0_handle.release()
                    cute.arch.fence_view_async_tmem_store()
                    o1_handle.release()
                # End of seqlen_corr_loop_steps
                vec1_handle.release()

                vec0_handle = s0_corr_consumer.wait_and_advance()
                tTMEM_LOAD_VECrS = cute.make_rmem_tensor(
                    tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                )
                cute.copy(
                    tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS
                )
                cute.arch.fence_view_async_tmem_load()
                vec0_handle.release()
                o0_handle = mma_corr_consumer.wait_and_advance()
                o0_final_handle = corr_epi_producer.acquire_and_advance()
                self.correction_epilog(
                    pv_thr_mma,
                    tOtO0,
                    mLSE,
                    tTMEM_LOAD_VECrS,
                    row_idx,
                    cuseqlen_q,
                    seqlen_q,
                    curr_block_coord_lse,
                    scale_softmax,
                    scale_output / tTMEM_LOAD_VECrS[0],
                    sO[None, None, 0],
                )
                o0_handle.release()
                o0_final_handle.commit()

                vec1_handle = s1_corr_consumer.wait_and_advance()
                cute.copy(
                    tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS
                )
                cute.arch.fence_view_async_tmem_load()
                vec1_handle.release()
                o1_handle = mma_corr_consumer.wait_and_advance()
                o1_final_handle = corr_epi_producer.acquire_and_advance()
                row_idx_o1 = row_idx + self.qk_mma_tiler[0]
                self.correction_epilog(
                    pv_thr_mma,
                    tOtO1,
                    mLSE,
                    tTMEM_LOAD_VECrS,
                    row_idx_o1,
                    cuseqlen_q,
                    seqlen_q,
                    curr_block_coord_lse,
                    scale_softmax,
                    scale_output / tTMEM_LOAD_VECrS[0],
                    sO[None, None, 1],
                )
                o1_handle.release()
                o1_final_handle.commit()
                # Cross-d_outer sync (only required for num_d_chunks > 1).
                if cutlass.const_expr(self.num_d_chunks > 1):
                    self.d_outer_sync_barrier.arrive_and_wait()
            # End of d_chunk_outer loop
        # Advance to next tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    # End of persistent scheduler loop


@cute.jit
def correction_rescale(
    self,
    thr_mma: cute.ThrMma,
    tOtO: cute.Tensor,
    scale: Float32,
):
    """Rescale a partial O accumulator in TMEM by ``scale``.

    Tiled by ``corr_tile_size=16`` columns so we issue exactly
    ``cta_tiler[2] / 16`` tmem load+mul+store rounds per call. ``scale`` is
    typically ``exp(scale_softmax_log2 * (old_max - new_max))``.
    """
    pv_tiled_mma_shape = (
        self.pv_mma_tiler[0],
        self.pv_mma_tiler[1],
    )
    cO = cute.make_identity_tensor(pv_tiled_mma_shape)
    tOcO = thr_mma.partition_C(cO)

    corr_tile_size = 16  # tuneable parameter
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
        self.pv_acc_dtype,
    )
    tmem_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
        self.pv_acc_dtype,
    )

    tOtO_i_layout = cute.composition(
        tOtO.layout, cute.make_layout((128, corr_tile_size))
    )
    tOcO_i_layout = cute.composition(
        tOcO.layout, cute.make_layout((128, corr_tile_size))
    )

    tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
    tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

    tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
    tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
    thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
    thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

    tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
    tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

    tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

    tTMrO = cute.make_rmem_tensor(
        (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype
    )
    for i in range(self.cta_tiler[2] // corr_tile_size):
        tTMrO_i_ = tTMrO[None, i]
        tTMrO_i_layout = cute.composition(
            tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
        )
        tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
        tTMEM_LOADtO_i = cute.make_tensor(
            tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
        )
        tTMEM_STOREtO_i = cute.make_tensor(
            tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
        )

        cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
        for j in cutlass.range(cute.size(tTMrO_i), vectorize=True):
            tTMrO_i[j] = tTMrO_i[j] * scale
        cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)


@cute.jit
def correction_epilog(
    self,
    thr_mma: cute.ThrMma,
    tOtO: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    tTMEM_LOAD_VECrS: cute.Tensor,
    row_idx: Int32,
    cuseqlen_q: Int32,
    seqlen_q: Int32,
    blk_coord: Int32,
    scale_softmax: Float32,
    scale: Float32,
    sO: cute.Tensor,
):
    """Final O scaling + SMEM dump (+ optional LSE write).

    Reads the final O TMEM block, multiplies by ``scale = scale_output / row_sum``,
    casts to ``o_dtype``, stores into the staged ``sO`` SMEM buffer for the
    epilogue warp's TMA store. When ``mLSE`` is provided, also writes one LSE
    value per row.
    """

    pv_tiled_mma_shape = (
        self.pv_mma_tiler[0],
        self.pv_mma_tiler[1],
    )
    cO = cute.make_identity_tensor(pv_tiled_mma_shape)

    corr_tile_size = 32 * 8 // self.o_dtype.width
    tOsO = thr_mma.partition_C(sO)
    tOcO = thr_mma.partition_C(cO)

    tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
    tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
    tOsO_i = cute.logical_divide(tOsO, cute.make_layout((128, corr_tile_size)))
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))

    epi_subtile = (self.epi_tile[0], corr_tile_size)
    tmem_copy_atom = sm100_utils.get_tmem_load_op(
        self.pv_mma_tiler,
        self.o_layout,
        self.o_dtype,
        self.pv_acc_dtype,
        epi_subtile,
        use_2cta_instrs=False,
    )

    tiled_tmem_load = tcgen05.make_tmem_copy(
        tmem_copy_atom, tOtO_i[(None, None), 0]
    )

    thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
    smem_copy_atom = sm100_utils.get_smem_store_op(
        self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
    )
    tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

    tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
    tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
    tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

    for i in range(self.cta_tiler[2] // corr_tile_size):
        tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
        tTMEM_LOADsO_i = tTMEM_LOADsO[None, 0, 0, i]
        tTMrO = cute.make_rmem_tensor(
            tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype
        )
        cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
        for j in range(cute.size(tTMrO), vectorize=True):
            tTMrO[j] = tTMrO[j] * scale
        tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
        o_vec = tTMrO.load()
        tSMrO.store(o_vec.to(self.o_dtype))
        cute.copy(tiled_smem_store, tSMrO, tTMEM_LOADsO_i)

    if cutlass.const_expr(mLSE is not None):
        scaled_tmp = scale_softmax * tTMEM_LOAD_VECrS[1]
        lse = (
            cute.math.log(tTMEM_LOAD_VECrS[0], fastmath=True)
            + scaled_tmp
            - self.softmax_prescale_ln
        )
        if row_idx < seqlen_q:
            mLSE[row_idx + cuseqlen_q, blk_coord[2]] = lse

    cute.arch.fence_proxy(
        "async.shared",
        space="cta",
    )
