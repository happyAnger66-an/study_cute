"""Tensor-core warp: ``cute.gemm`` of Q*K^T and P*V via tcgen05.

One warp issues all MMA instructions for the CTA. It also allocates and frees
the TMEM region used by S/O/P/vec buffers.

Per persistent KV iteration this warp performs, in order:

1. ``QK0`` (Q0 * K^T) and ``QK1`` (Q1 * K^T) for every D-chunk, accumulating
   into the same ``S0`` / ``S1`` TMEM accumulators. Commits ``s0`` / ``s1``
   once after all D-chunks finish.
2. ``PV1`` (P1 * V_prev) and then ``PV0`` (P0 * V_curr) for every D-chunk.

The D-chunk inner loop is keyed to :attr:`self.num_d_chunks`; for D <= 128 this
is 1 (legacy single-chunk path) and for D = 256 it is 2.

Pipeline contracts (one ``acquire``/``commit`` per object per KV step):

- ``mma_s0_producer`` / ``mma_s1_producer``  -> softmax warps consume S0 / S1
- ``mma_corr_producer``                      -> correction warp consumes O0 / O1
- ``load_q_consumer`` / ``load_kv_consumer`` -> we consume Q / K / V

V-handle release policy: see :func:`mma_warp_body` body comments. This module
preserves the legacy D=128 pattern; the D-chunk variant of the release policy
is the locus of the D=256 hang documented in ``docs/d_256.md``.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.typing import Float32, Int32

from fmha import fmha_helpers as fmha_utils


@cute.jit
def mma_warp_body(
    self,
    qk_tiled_mma: cute.TiledMma,
    pv_tiled_mma: cute.TiledMma,
    tSrQ: cute.Tensor,
    tSrK: cute.Tensor,
    tOrV: cute.Tensor,
    tStS0: cute.Tensor,
    tStS1: cute.Tensor,
    tOtO0: cute.Tensor,
    tOtO1: cute.Tensor,
    tOrP0: cute.Tensor,
    tOrP1: cute.Tensor,
    load_q_consumer,
    load_kv_consumer,
    mma_s0_producer,
    mma_s1_producer,
    mma_corr_producer,
    mQ_qdl: cute.Tensor,
    mK_kdl: cute.Tensor,
    cum_seqlen_q: Optional[cute.Tensor],
    cum_seqlen_k: Optional[cute.Tensor],
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    storage,
    tmem_dealloc_mbar_ptr,
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Body of the MMA warp. See module docstring for the pipeline contract."""
    # ------------------------------------------------------------------
    # Allocate TMEM (S/O/P/vec all live in this 512-column block)
    # ------------------------------------------------------------------
    tmem_alloc_cols = Int32(self.tmem_alloc_cols)
    cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
    self.tmem_alloc_barrier.arrive_and_wait()

    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        batch_coord = curr_block_coord[2][1]
        continue_cond = False
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
            seqlen_k = mK_kdl.shape[0]
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

            # ============================================================
            # D-chunking OUTER loop: for D>128 the full attention pipeline
            # (prologue + main loop + tail) runs ``num_d_chunks`` times per
            # kv tile, each iteration consuming a single V[d_chunk_outer]
            # slice and writing its own gO[..., d_chunk_outer, ...].
            #
            # QK still walks the inner d_chunk loop and accumulates S over
            # the full d (S depends on every d slice). PV degenerates to
            # the D=128 single-V mode (no inner d_chunk loop in PV) so
            # tOtO0 / tOtO1 fit within the 512-col TMEM budget.
            #
            # For D<=128 (num_d_chunks=1) this outer loop collapses to a
            # single iteration -> identical to the legacy D=128 path.
            # See docs/d_chunk_redesign.md.
            # ============================================================
            for d_chunk_outer in cutlass.range_constexpr(self.num_d_chunks):

                # ============================================================
                # Prologue: GEMM_QK00 / QK10 + GEMM_PV00 (first KV tile)
                # ============================================================
                # QK00 / QK10: walk every D-chunk (inner), accumulate into
                # S0/S1, commit once.
                if cutlass.const_expr(self.debug_pipeline):
                    cute.printf(
                        "MMA prologue d_outer=%d trip=%d num_d_chunks=%d\n",
                        d_chunk_outer, seqlen_kv_loop_steps,
                        self.num_d_chunks,
                    )
                s0_handle = mma_s0_producer.acquire_and_advance()
                s1_handle = mma_s1_producer.acquire_and_advance()
                for d_chunk_idx in cutlass.range_constexpr(self.num_d_chunks):
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    num_kphases = cute.size(tSrQ0, mode=[2])
                    for kphase_idx in cutlass.range(
                        num_kphases, unroll_full=True
                    ):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            d_chunk_idx != 0 or kphase_idx != 0,
                        )
                        cute.gemm(
                            qk_tiled_mma,
                            tStS0,
                            tSrQ0[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS0,
                        )
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    num_kphases = cute.size(tSrQ1, mode=[2])
                    for kphase_idx in cutlass.range(
                        num_kphases, unroll_full=True
                    ):
                        kphase_coord = (None, None, kphase_idx)
                        qk_tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            d_chunk_idx != 0 or kphase_idx != 0,
                        )
                        cute.gemm(
                            qk_tiled_mma,
                            tStS1,
                            tSrQ1[kphase_coord],
                            tSrK0[kphase_coord],
                            tStS1,
                        )
                    k_handle.release()
                    q0_handle.release()
                    q1_handle.release()
                s0_handle.commit()
                s1_handle.commit()

                # GEMM_PV00 (single V mode): only one V is produced per
                # kv tile for this d_chunk_outer. Defer the release until
                # the first main-loop PV1 (or tail PV1).
                o0_handle = mma_corr_producer.acquire_and_advance()
                s0_handle = mma_s0_producer.acquire_and_advance()
                v_handle = load_kv_consumer.wait_and_advance()
                if cutlass.const_expr(self.debug_pipeline):
                    cute.printf(
                        "MMA PV00 d_outer=%d slot=%d\n",
                        d_chunk_outer, v_handle.index,
                    )
                tOrVi = tOrV[None, None, None, v_handle.index]
                num_kphases = cute.size(tOrP0, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    pv_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE, kphase_idx != 0
                    )
                    cute.gemm(
                        pv_tiled_mma,
                        tOtO0,
                        tOrP0[kphase_coord],
                        tOrVi[kphase_coord],
                        tOtO0,
                    )
                o0_handle.commit()
                # v_handle is deferred to the first main-loop PV1 (or
                # the tail PV1 if loop_steps == 0).

                # ============================================================
                # Main loop: 4 phases per KV step (single-V PV mode):
                #   QK0i  -> PV1(i-1) -> QK1i -> PV0i
                #
                # Pipeline invariants (same as the D=128 path):
                #   - s0_handle: acquired in PV0/PV00, committed at QK0i
                #   - s1_handle: acquired in PV1, committed at QK1i
                #   - Q0/Q1/K handles cross QK0i/QK1i; released at QK1i tail
                #   - v_handle: acquired in PV0/PV00, released at next PV1
                # ============================================================
                pv_whether_acc = False
                for i in cutlass.range(
                    0, seqlen_kv_loop_steps, 1, unroll=1
                ):
                    # --- Phase 1: QK0i (write S0, commit s0) ---
                    k_handles = []
                    q0_handles = []
                    for d_chunk_idx in cutlass.range_constexpr(
                        self.num_d_chunks
                    ):
                        q0_handle = load_q_consumer.wait_and_advance()
                        tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                        q0_handles.append(q0_handle)
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        k_handles.append((k_handle, tSrKi))
                        inner_num_kphases = cute.size(tSrQ0, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                d_chunk_idx != 0 or kphase_idx != 0,
                            )
                            cute.gemm(
                                qk_tiled_mma,
                                tStS0,
                                tSrQ0[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS0,
                            )
                    s0_handle.commit()

                    # --- Phase 2: PV1(i-1) (read P1, gemm O1, release V_prev)
                    o1_handle = mma_corr_producer.acquire_and_advance()
                    s1_handle = mma_s1_producer.acquire_and_advance()
                    # Single V mode: tOrVi from outer scope (= V_{i-1}).
                    inner_num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(
                        inner_num_kphases, unroll_full=True
                    ):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(
                            tcgen05.Field.ACCUMULATE, pv_whether_acc
                        )
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO1,
                            tOrP1[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO1,
                        )
                        pv_whether_acc = True
                    o1_handle.commit()
                    v_handle.release()
                    # s1_handle NOT committed here -- QK1i writes into it.

                    # --- Phase 3: QK1i (write S1, commit s1)
                    for d_chunk_idx in cutlass.range_constexpr(
                        self.num_d_chunks
                    ):
                        q1_handle = load_q_consumer.wait_and_advance()
                        tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                        _, tSrKi = k_handles[d_chunk_idx]
                        inner_num_kphases = cute.size(tSrQ1, mode=[2])
                        for kphase_idx in cutlass.range(
                            inner_num_kphases, unroll_full=True
                        ):
                            kphase_coord = (None, None, kphase_idx)
                            qk_tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                d_chunk_idx != 0 or kphase_idx != 0,
                            )
                            cute.gemm(
                                qk_tiled_mma,
                                tStS1,
                                tSrQ1[kphase_coord],
                                tSrKi[kphase_coord],
                                tStS1,
                            )
                        q0_handles[d_chunk_idx].release()
                        q1_handle.release()
                        k_handles[d_chunk_idx][0].release()
                    s1_handle.commit()

                    # --- Phase 4: PV0i (acquire V_i, gemm O0, defer release)
                    o0_handle = mma_corr_producer.acquire_and_advance()
                    s0_handle = mma_s0_producer.acquire_and_advance()
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    inner_num_kphases = cute.size(tOrP0, mode=[2])
                    for kphase_idx in cutlass.range(
                        inner_num_kphases, unroll_full=True
                    ):
                        kphase_coord = (None, None, kphase_idx)
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(
                            pv_tiled_mma,
                            tOtO0,
                            tOrP0[kphase_coord],
                            tOrVi[kphase_coord],
                            tOtO0,
                        )
                    o0_handle.commit()

                # ============================================================
                # Tail PV1 final: P1 @ V_{N-1}, single V mode.
                # ============================================================
                o1_handle = mma_corr_producer.acquire_and_advance()
                s1_handle = mma_s1_producer.acquire_and_advance()
                num_kphases = cute.size(tOrP1, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    pv_tiled_mma.set(
                        tcgen05.Field.ACCUMULATE, pv_whether_acc
                    )
                    cute.gemm(
                        pv_tiled_mma,
                        tOtO1,
                        tOrP1[kphase_coord],
                        tOrVi[kphase_coord],
                        tOtO1,
                    )
                    pv_whether_acc = True
                o1_handle.commit()
                v_handle.release()

                s0_handle.commit()
                s1_handle.commit()
            # End of d_chunk_outer loop

        # Advance to next tile
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    # End of persistent scheduler loop

    # ------------------------------------------------------------------
    # Dealloc TMEM
    # ------------------------------------------------------------------
    cute.arch.relinquish_tmem_alloc_permit()
    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
    tmem_alloc_cols = Int32(self.tmem_alloc_cols)
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        Float32,
        alignment=16,
        ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
    )
    cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
