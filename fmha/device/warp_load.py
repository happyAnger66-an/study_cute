"""Producer warp: TMA G->S loads of Q, K, V.

One warp issues all TMA bulk-tensor loads for the CTA. For each persistent
work tile it walks the KV iteration axis; within each KV step it walks every
D-chunk so the MMA warp can accumulate QK^T / PV across all chunks.

Per KV step it produces, in order:
- ``num_d_chunks`` Q0 + K + Q1 tiles (Q pipeline + KV pipeline)
- ``num_d_chunks`` V tiles (KV pipeline)

The Q tile load is double-buffered because each CTA owns two Q tiles
(``cta_tiler[0] = 2 * qk_mma_tiler[0]``).
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from fmha import fmha_helpers as fmha_utils


@cute.jit
def load_warp_body(
    self,
    tma_atom_q: cute.CopyAtom,
    mQ_qdl: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    mK_kdl: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    mV_dkl: cute.Tensor,
    sQ: cute.Tensor,
    sK: cute.Tensor,
    sV: cute.Tensor,
    qk_thr_mma: cute.ThrMma,
    pv_thr_mma: cute.ThrMma,
    load_q_producer,
    load_kv_producer,
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Body of the load warp. See module docstring."""
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        batch_coord = curr_block_coord[2][1]
        continue_cond = False
        cuseqlen_q = Int32(0)
        seqlen_q = mQ_qdl.shape[0]
        if cutlass.const_expr(cum_seqlen_q is not None):
            cuseqlen_q = cum_seqlen_q[batch_coord]
            seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
            continue_cond = not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                self.cta_tiler[0],
                curr_block_coord[0],
                seqlen_q,
            )
        if not continue_cond:
            mQ_qdl_ = mQ_qdl
            mK_kdl_ = mK_kdl
            mV_dkl_ = mV_dkl
            seqlen_k = mK_kdl.shape[0]
            curr_block_coord_q = curr_block_coord
            curr_block_coord_kv = curr_block_coord

            # Re-base packed varlen tensors so per-batch slices line up
            # with the global tile coords.
            if cutlass.const_expr(cum_seqlen_q is not None):
                logical_offset_mQ = (
                    mQ_qdl.shape[0] - seqlen_q,
                    0,
                    (0, cuseqlen_q + seqlen_q),
                )
                mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                curr_block_coord_q = (
                    curr_block_coord[0],
                    curr_block_coord[1],
                    (curr_block_coord[2][0], Int32(0)),
                )

            if cutlass.const_expr(cum_seqlen_k is not None):
                cuseqlen_k = cum_seqlen_k[batch_coord]
                seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                if cutlass.const_expr(cum_seqlen_q is not None):
                    logical_offset_mK = (
                        mK_kdl.shape[0] - seqlen_k,
                        0,
                        (0, cuseqlen_k + seqlen_k),
                    )
                    logical_offset_mV = (
                        0,
                        mK_kdl.shape[0] - seqlen_k,
                        (0, cuseqlen_k + seqlen_k),
                    )
                    mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                    mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                    curr_block_coord_kv = (
                        curr_block_coord[0],
                        curr_block_coord[1],
                        (curr_block_coord[2][0], Int32(0)),
                    )

            # ------------------------------------------------------------------
            # Local tile partition of global tensors (bM, bK, loopM, loopK, loopL)
            # ------------------------------------------------------------------
            gQ_qdl = cute.flat_divide(
                mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
            )
            tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
            tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ_qdl, 0, 3),
            )
            gK_kdl = cute.flat_divide(
                mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
            )
            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK_kdl, 0, 3),
            )

            gV_dkl = cute.flat_divide(
                mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
            )
            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tSgV_dkl, 0, 3),
            )

            q0_coord = 2 * curr_block_coord_q[0]
            q1_coord = q0_coord + 1

            seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q,
                seqlen_k,
                window_size_left,
            )
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

            # ------------------------------------------------------------------
            # D-chunking outer loop: for D>128 we run the full attention
            # pipeline ``num_d_chunks`` times per kv tile, each iteration
            # loading the same Q+K (across inner d_chunks, so QK can
            # accumulate S over the full d) but only ONE V slice
            # (``V[d_chunk_outer]``). MMA / correction / epilogue mirror
            # this outer loop; PV degenerates to single-V mode.
            # See docs/d_chunk_redesign.md.
            #
            # For D<=128 (num_d_chunks=1) this outer loop collapses to a
            # single iteration -> identical to the legacy D=128 path.
            # ------------------------------------------------------------------
            for d_chunk_outer in cutlass.range_constexpr(self.num_d_chunks):
                kv_coord = seqlen_kv_loop_start
                # ----- prologue: load Q0/K/Q1 (all inner d_chunks)
                # then V[d_chunk_outer] for the first KV tile -----
                if cutlass.const_expr(self.debug_pipeline):
                    cute.printf(
                        "LOAD prologue d_outer=%d kv=%d num_d_chunks=%d\n",
                        d_chunk_outer, kv_coord, self.num_d_chunks)
                for d_chunk_idx in cutlass.range_constexpr(self.num_d_chunks):
                    tQgQ = tQgQ_qdl[
                        None, None, d_chunk_idx, curr_block_coord_q[2]
                    ]
                    tKgK = tKgK_kdl[
                        None, None, d_chunk_idx, curr_block_coord_kv[2]
                    ]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                # Produce a single V slice (V[d_chunk_outer]).
                tVgV = tVgV_dkl[
                    None, d_chunk_outer, None, curr_block_coord_kv[2]
                ]
                v_handle = load_kv_producer.acquire_and_advance()
                if cutlass.const_expr(self.debug_pipeline):
                    cute.printf(
                        "LOAD pro V d_outer=%d slot=%d\n",
                        d_chunk_outer, v_handle.index)
                cute.copy(
                    tma_atom_v,
                    tVgV[None, kv_coord],
                    tVsV[None, v_handle.index],
                    tma_bar_ptr=v_handle.barrier,
                )
                kv_coord += 1

                # ----- inner loop: per remaining KV step, same shape -----
                for i in cutlass.range(
                    0, seqlen_kv_loop_steps, 1, unroll=1
                ):
                    if cutlass.const_expr(self.debug_pipeline):
                        cute.printf(
                            "LOAD main d_outer=%d iter=%d start\n",
                            d_chunk_outer, i,
                        )
                    for d_chunk_idx in cutlass.range_constexpr(
                        self.num_d_chunks
                    ):
                        tQgQ = tQgQ_qdl[
                            None, None, d_chunk_idx, curr_block_coord_q[2]
                        ]
                        tKgK = tKgK_kdl[
                            None, None, d_chunk_idx, curr_block_coord_kv[2]
                        ]
                        q0_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, q0_coord],
                            tQsQ[None, q0_handle.index],
                            tma_bar_ptr=q0_handle.barrier,
                        )
                        k_handle = load_kv_producer.acquire_and_advance()
                        if cutlass.const_expr(self.debug_pipeline):
                            cute.printf(
                                "  LOAD K d_outer=%d iter=%d d_inner=%d slot=%d\n",
                                d_chunk_outer, i, d_chunk_idx, k_handle.index,
                            )
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        q1_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, q1_coord],
                            tQsQ[None, q1_handle.index],
                            tma_bar_ptr=q1_handle.barrier,
                        )
                    tVgV = tVgV_dkl[
                        None, d_chunk_outer, None, curr_block_coord_kv[2]
                    ]
                    v_handle = load_kv_producer.acquire_and_advance()
                    if cutlass.const_expr(self.debug_pipeline):
                        cute.printf(
                            "  LOAD V d_outer=%d iter=%d slot=%d\n",
                            d_chunk_outer, i, v_handle.index,
                        )
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                    kv_coord += 1
                # End of seqlen_kv loop for this d_chunk_outer
                if cutlass.const_expr(self.debug_pipeline):
                    cute.printf(
                        "LOAD d_outer=%d done (kv loop end)\n",
                        d_chunk_outer,
                    )
                # Cross-d_outer sync (only required for num_d_chunks > 1).
                if cutlass.const_expr(self.num_d_chunks > 1):
                    self.d_outer_sync_barrier.arrive_and_wait()

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    # End of persistent scheduler loop
