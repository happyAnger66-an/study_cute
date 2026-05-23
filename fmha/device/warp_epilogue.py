"""Epilogue warp: TMA S->G store of O.

One warp issues all TMA stores for the final O tensor. The correction warp
writes O0 / O1 into SMEM (``sO``); this warp picks them up via the
``corr_epi`` pipeline and issues bulk-tensor stores back to global memory.

For D > 128, the SMEM ``sO`` only holds a 128-wide chunk of D at a time. To
TMA-store all chunks, the warp emits the remaining D-chunk strides as
follow-up bulk stores after the first. (The SMEM buffer is single-buffered
across the D-chunk axis; only the chunk index changes.)
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from fmha import fmha_helpers as fmha_utils


@cute.jit
def epilogue_warp_body(
    self,
    tma_atom_o: cute.CopyAtom,
    mO_qdl: cute.Tensor,
    sO: cute.Tensor,
    corr_epi_consumer,
    cum_seqlen_q: Optional[cute.Tensor],
    mQ_qdl: cute.Tensor,
    tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
):
    """Body of the epilogue warp. See module docstring."""
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
            curr_block_coord_o = curr_block_coord
            mO_qdl_ = mO_qdl
            if cutlass.const_expr(cum_seqlen_q is not None):
                logical_offset_mO = (
                    mO_qdl_.shape[0] - seqlen_q,
                    0,
                    (0, cuseqlen_q + seqlen_q),
                )
                mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                curr_block_coord_o = (
                    curr_block_coord[0],
                    curr_block_coord[1],
                    (curr_block_coord[2][0], 0),
                )

            o0_coord = 2 * curr_block_coord_o[0]
            o1_coord = o0_coord + 1
            gO_qdl = cute.flat_divide(
                mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1])
            )

            # D-chunking outer loop: store the per-d_chunk sO produced by
            # the correction warp to gO[..., d_chunk_outer, ...]. Each
            # iteration consumes ONE pair of corr_epi handles (o0, o1)
            # that the correction warp just committed. For D<=128
            # (num_d_chunks=1) this is a single iteration -> identical to
            # the legacy single-chunk path.
            for d_chunk_outer in cutlass.range_constexpr(self.num_d_chunks):
                gO = gO_qdl[
                    None, None, None, d_chunk_outer, curr_block_coord_o[2]
                ]
                tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sO, 0, 2),
                    cute.group_modes(gO, 0, 2),
                )
                o0_handle = corr_epi_consumer.wait_and_advance()
                cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                cute.arch.cp_async_bulk_commit_group()
                o1_handle = corr_epi_consumer.wait_and_advance()
                cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(1, read=True)
                o0_handle.release()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                o1_handle.release()
                # Cross-d_outer sync (only required for num_d_chunks > 1).
                if cutlass.const_expr(self.num_d_chunks > 1):
                    self.d_outer_sync_barrier.arrive_and_wait()

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    # End of persistent scheduler loop
