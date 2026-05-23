"""Softmax warps (stage 0 and stage 1) and the per-step inner loop.

There are two warpgroups (4 warps each) dedicated to online softmax:
- stage 0 processes ``S0`` produced by ``QK0`` (Q0 * K^T)
- stage 1 processes ``S1`` produced by ``QK1`` (Q1 * K^T)

Both warpgroups share the same algorithmic implementation
(:func:`softmax`); a sequence barrier (``s0_s1_sequence_*``) gates them so
their TMEM stores don't race.

Each invocation of :func:`softmax_step` walks one KV tile:
1. wait for ``Si`` produced by MMA (``mma_si_consumer.wait_and_advance``)
2. load Si from TMEM into registers, apply mask, compute running max/sum
3. write the new row_max / row_max_safe vector to TMEM for the correction warp
4. exp2(x*scale - max*scale), store the resulting P_i (fp16/fp8) back to TMEM
5. notify correction (``si_corr_producer.commit``) and tensor core (``si_handle.release``)

Bound onto :class:`BlackwellFusedMultiHeadAttentionForward` as
``self.softmax`` / ``self.softmax_step`` from :mod:`fmha.__init__`.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
from cutlass.cute.typing import Float32, Int32

from fmha import fmha_helpers as fmha_utils


@cute.jit
def softmax_step(
    self,
    stage: int,
    need_apply_mask: bool,
    iter_args: tuple,
    value_args: tuple,
    pipeline_args: tuple,
    atom_args: tuple,
    tensor_args: tuple,
) -> Tuple[
    Float32,
    Float32,
    pipeline.PipelineProducer.ImmutableResourceHandle,
    pipeline.PipelineConsumer,
    pipeline.PipelineProducer,
    pipeline.PipelineConsumer,
    pipeline.PipelineProducer,
]:
    """Process a single KV block of attention scores through online softmax.

    See module docstring for the data-flow; arguments are bundled into tuples
    to keep the per-iteration call site short.

    Returns updated ``(row_max, row_sum, vec_i_handle, mma_si_consumer,
    si_corr_producer, s0_s1_sequence_consumer, s0_s1_sequence_producer)``.
    """
    cS, row_max, row_sum, vec_i_handle = iter_args
    seqlen_k, seqlen_q, scale_softmax_log2, window_size_left, window_size_right = (
        value_args
    )
    (
        mma_si_consumer,
        si_corr_producer,
        s0_s1_sequence_consumer,
        s0_s1_sequence_producer,
    ) = pipeline_args
    (
        qk_thr_mma,
        tiled_tmem_load,
        tiled_tmem_store,
        tiled_tmem_store_vec,
        thr_tmem_load,
        thr_tmem_store,
        thr_tmem_store_vec,
    ) = atom_args
    (
        tTMEM_LOADtS,
        tTMEM_STORE_VECtS,
        tTMEM_STOREtS_x4,
    ) = tensor_args

    tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width
    tScS = qk_thr_mma.partition_C(cS)
    tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
    tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

    tScS_P_layout = cute.composition(
        tScS.layout, cute.make_layout((128, tilePlikeFP32))
    )
    tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
    tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
    tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
    tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

    # Wait for Si produced by MMA
    si_handle = mma_si_consumer.wait_and_advance()
    tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
    cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
    if need_apply_mask:
        fmha_utils.FusedMask.apply_mask(
            self.mask_type,
            tTMEM_LOADrS,
            tTMEM_LOADcS,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

    old_row_max = row_max
    row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
    row_max_safe = row_max
    if row_max == -cutlass.Float32.inf:
        row_max_safe = 0.0
    tTMEM_STORE_VECrS = cute.make_rmem_tensor(
        tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
    )
    tTMEM_STORE_VECrS[0] = old_row_max
    tTMEM_STORE_VECrS[1] = row_max_safe
    cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
    cute.arch.fence_view_async_tmem_store()
    # Notify correction warp that row_max is ready
    vec_i_handle.commit()

    tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.qk_acc_dtype)
    tTMEM_STORErS_x4_e = cute.make_tensor(
        cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.q_dtype),
        tTMEM_LOADrS.layout,
    )

    scale = scale_softmax_log2
    minus_row_max_scale = (0.0 - row_max_safe) * scale + self.softmax_prescale_log2

    # Sequence barrier wait (gates stage 0 -> stage 1 ordering)
    if cutlass.const_expr(stage == 0):
        sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
    else:
        sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()
    frg_cnt = 4
    frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
    tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
    tTMEM_STORErS_x4_e_frg = cute.logical_divide(
        tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
    )
    for j in range(frg_cnt):
        for k in cutlass.range(
            cute.size(tTMEM_LOADrS_frg, mode=[0]), vectorize=True
        ):
            tTMEM_LOADrS_frg[k, j] = (
                tTMEM_LOADrS_frg[k, j] * scale + minus_row_max_scale
            )
            tTMEM_LOADrS_frg[k, j] = cute.math.exp2(
                tTMEM_LOADrS_frg[k, j], fastmath=True
            )

        s_vec = tTMEM_LOADrS_frg[None, j].load()
        tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))
    if cutlass.const_expr(stage == 0):
        sequence_producer_handle.commit()
    else:
        sequence_consumer_handle.release()
    cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
    cute.arch.fence_view_async_tmem_store()
    # Notify tensor core warp that softmax(S->P) is ready
    si_handle.release()

    vec_i_handle = si_corr_producer.acquire_and_advance()
    acc_scale_ = scale * (old_row_max - row_max_safe)
    acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
    row_sum *= acc_scale
    local_row_sum_0 = (row_sum, row_sum)
    local_row_sum_1 = (0.0, 0.0)
    local_row_sum_2 = (0.0, 0.0)
    local_row_sum_3 = (0.0, 0.0)

    reduction_unroll = 4
    frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
    tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))

    for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
        local_row_sum_0 = cute.arch.add_packed_f32x2(
            local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
        )
        local_row_sum_1 = cute.arch.add_packed_f32x2(
            local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
        )
        local_row_sum_2 = cute.arch.add_packed_f32x2(
            local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
        )
        local_row_sum_3 = cute.arch.add_packed_f32x2(
            local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
        )

    local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
    local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
    local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
    row_sum = local_row_sum_0[0] + local_row_sum_0[1]

    return (
        row_max,
        row_sum,
        vec_i_handle,
        mma_si_consumer,
        si_corr_producer,
        s0_s1_sequence_consumer,
        s0_s1_sequence_producer,
    )


@cute.jit
def softmax(
    self,
    stage: int,
    seqlen_k: Int32,
    seqlen_q: Int32,
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    scale_softmax_log2: Float32,
    qk_thr_mma: cute.ThrMma,
    tStS: cute.Tensor,
    tStSi: cute.Tensor,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    mma_si_consumer: pipeline.PipelineConsumer,
    si_corr_producer: pipeline.PipelineProducer,
    s0_s1_sequence_consumer: pipeline.PipelineConsumer,
    s0_s1_sequence_producer: pipeline.PipelineProducer,
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Softmax warp body (one warpgroup runs this with ``stage=0`` or ``stage=1``).

    For each persistent work tile this loops over KV iterations and dispatches
    to :func:`softmax_step`. The leading / unmasked / trailing iteration ranges
    come from :class:`fmha_helpers.FusedMask` so masking work is bounded.
    """
    tidx, _, _ = cute.arch.thread_idx()
    thread_idx = tidx % (
        self.threads_per_warp
        * (
            len(self.softmax0_warp_ids)
            if stage == 0
            else len(self.softmax1_warp_ids)
        )
    )

    cS_base = cute.make_identity_tensor(
        (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
    )
    tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width
    tScS = qk_thr_mma.partition_C(cS_base)
    tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
    tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
    tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
    tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
    tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
    tStS_P_layout = cute.composition(
        tStS.layout, cute.make_layout((128, tilePlikeFP32))
    )
    tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
    tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
        self.qk_acc_dtype,
    )
    tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
    thread_idx = tidx % (
        self.threads_per_warp
        * (
            len(self.softmax0_warp_ids)
            if stage == 0
            else len(self.softmax1_warp_ids)
        )
    )
    thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
    tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
    tmem_store_vec_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
        self.qk_acc_dtype,
    )
    tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
    thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
    tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
    tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
    tmem_store_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
        self.qk_acc_dtype,
    )
    tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
    thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
    tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        batch_coord = curr_block_coord[2][1]
        seqlen_k_ = seqlen_k
        seqlen_q_ = seqlen_q
        continue_cond = False
        cuseqlen_q = Int32(0)
        seqlen_q_ = seqlen_q
        if cutlass.const_expr(cum_seqlen_q is not None):
            cuseqlen_q = cum_seqlen_q[batch_coord]
            seqlen_q_ = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
            continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                self.cta_tiler[0],
                curr_block_coord[0],
                seqlen_q_,
            )

        if not continue_cond:
            if cutlass.const_expr(cum_seqlen_k is not None):
                cuseqlen_k = cum_seqlen_k[batch_coord]
                seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
            value_args = (
                seqlen_k_,
                seqlen_q_,
                scale_softmax_log2,
                window_size_left,
                window_size_right,
            )
            atom_args = (
                qk_thr_mma,
                tiled_tmem_load,
                tiled_tmem_store,
                tiled_tmem_store_vec,
                thr_tmem_load,
                thr_tmem_store,
                thr_tmem_store_vec,
            )
            tensor_args = (
                tTMEM_LOADtS,
                tTMEM_STORE_VECtS,
                tTMEM_STOREtS_x4,
            )

            logical_offset = (
                curr_block_coord[0] * self.cta_tiler[0]
                + stage * self.qk_mma_tiler[0],
                0,
            )
            cS = cute.domain_offset(logical_offset, cS_base)

            start_count = fmha_utils.FusedMask.get_trip_start(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q_,
                seqlen_k_,
                window_size_left,
            )
            leading_mask_count = fmha_utils.FusedMask.get_masked_leading_count(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q_,
                seqlen_k_,
                window_size_left,
                window_size_right,
            )
            unmask_count = fmha_utils.FusedMask.get_unmasked_trip_count(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q_,
                seqlen_k_,
                window_size_left,
                window_size_right,
            )
            trailing_mask_count = fmha_utils.FusedMask.get_masked_trailing_count(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q_,
                seqlen_k_,
                window_size_left,
                window_size_right,
            )

            # D-chunking outer loop: softmax mirrors MMA / correction --
            # it runs the full leading + unmask + trailing KV loop
            # ``num_d_chunks`` times per work_tile, each iteration
            # resetting (row_max, row_sum) and acquiring a fresh
            # vec_i_handle. For D<=128 (num_d_chunks=1) this is a single
            # iteration -> identical to the legacy single-chunk path.
            for d_chunk_outer in cutlass.range_constexpr(self.num_d_chunks):
                row_max = -Float32.inf
                row_sum = 0.0
                vec_i_handle = si_corr_producer.acquire_and_advance()

                # ---- masked-leading region ----
                for i in cutlass.range(
                    start_count, start_count + leading_mask_count, 1, unroll=1
                ):
                    cS_iter = cute.domain_offset(
                        (0, i * self.qk_mma_tiler[1]), cS
                    )
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )

                # ---- unmasked region ----
                for i in cutlass.range(
                    start_count + leading_mask_count,
                    start_count + leading_mask_count + unmask_count,
                    1,
                    unroll=1,
                ):
                    cS_iter = cute.domain_offset(
                        (0, i * self.qk_mma_tiler[1]), cS
                    )
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        False,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )

                # ---- masked-trailing region ----
                for i in cutlass.range(
                    start_count + leading_mask_count + unmask_count,
                    start_count
                    + leading_mask_count
                    + unmask_count
                    + trailing_mask_count,
                    1,
                    unroll=1,
                ):
                    cS_iter = cute.domain_offset(
                        (0, i * self.qk_mma_tiler[1]), cS
                    )
                    iter_args = (cS_iter, row_max, row_sum, vec_i_handle)
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.softmax_step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                    )

                # ---- final: dump (row_sum, row_max) to correction ----
                si_handle = mma_si_consumer.wait_and_advance()
                tTMEM_STORE_VECrS = cute.make_rmem_tensor(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                cute.copy(
                    tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS
                )
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
                si_corr_producer.acquire()
                si_handle.release()
                # Cross-d_outer sync (only required for num_d_chunks > 1).
                if cutlass.const_expr(self.num_d_chunks > 1):
                    self.d_outer_sync_barrier.arrive_and_wait()
            # End of d_chunk_outer loop

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    # End of persistent scheduler loop
