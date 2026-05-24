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
"""Load warp body."""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.typing import Float32
from typing import Optional, Tuple

from fmha_d256 import fmha_helpers as fmha_utils
from fmha_d256 import prefill_helpers as prefill_utils

@cute.jit
def load_warp_body(self,
    qk_tiled_mma, pv_tiled_mma,
    tQgQ_qdl, tKgK_kdl, tKgScaleK_kdl, tVgV_dkl, tVgScaleV_dkl,
    tQsQ, tKsK, tKsScaleK, tVsV, tVsScaleV,
    tma_atom_q, tma_atom_k, tma_atom_v, tma_atom_scale_k, tma_atom_scale_v,
    load_q_producer, load_kv_producer, load_scale_k_producer, load_scale_v_producer,
    seqlen_q, seqlen_k, window_size_left, window_size_right,
    tile_sched_params,
):
    cute.arch.setmaxregister_decrease(self.num_regs_other)
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        mma_block_coord = (
            curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
            curr_block_coord[1],
            curr_block_coord[2],
        )
        seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
            self.mask_type,
            mma_block_coord,
            self.qk_mma_tiler,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        # ((atom_v, rest_v), RestK)
        tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
        # ((atom_v, rest_v), RestN, RestK)
        tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
        tKgScaleK = tKgScaleK_kdl[None, None, mma_block_coord[2]]
        # ((atom_v, rest_v), RestN, RestK)
        tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
        tVgScaleV = tVgScaleV_dkl[None, None, mma_block_coord[2]]
        load_kv_producer, load_scale_k_producer, load_q_producer = (
            prefill_utils.load_qk(  # Q & K0 & ScaleK0
                self.iterations_qk,
                kv_step=0,
                k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                scale_k_args=(
                    tKgScaleK,
                    tKsScaleK,
                    tma_atom_scale_k,
                    load_scale_k_producer,
                ),
                q_args=(tQgQ, tQsQ, tma_atom_q, load_q_producer),
            )
        )
        for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
            load_kv_producer, load_scale_k_producer = (
                prefill_utils.load_qk(  # Ki & ScaleKi
                    self.iterations_qk,
                    kv_step=step,
                    k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                    scale_k_args=(
                        tKgScaleK,
                        tKsScaleK,
                        tma_atom_scale_k,
                        load_scale_k_producer,
                    ),
                )
            )
            load_kv_producer, load_scale_v_producer = (
                prefill_utils.load_v(  # Vi-1 & ScaleVi-1
                    self.iterations_pv,
                    kv_step=step - 1,
                    v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                    scale_v_args=(
                        tVgScaleV,
                        tVsScaleV,
                        tma_atom_scale_v,
                        load_scale_v_producer,
                    ),
                )
            )
        load_kv_producer, load_scale_v_producer = (
            prefill_utils.load_v(  # Vend & ScaleVend
                self.iterations_pv,
                kv_step=seqlen_kv_loop_steps - 1,
                v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                scale_v_args=(
                    tVgScaleV,
                    tVsScaleV,
                    tma_atom_scale_v,
                    load_scale_v_producer,
                ),
            )
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    load_kv_producer.tail()
    load_scale_k_producer.tail()
    load_scale_v_producer.tail()
    load_q_producer.tail()
