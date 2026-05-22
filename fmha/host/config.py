"""Configuration class for the Blackwell SM100 FMHA kernel.

This module holds the *host* part of :class:`BlackwellFusedMultiHeadAttentionForward`:

- constructor (problem shape / mma tiler / pipeline stages / TMEM offsets ...)
- pipeline-stage selection and staged-SMEM budget estimation
- pre-launch validation (``validate_config_host``)
- ``_setup_attributes`` populating dtype-dependent scaling factors

The ``@cute.jit`` host launchers (``__call__`` / ``__call_vit__``) and all
``@cute.kernel`` / ``@cute.jit`` device methods live in :mod:`fmha.host.launcher`
and :mod:`fmha.device.*` respectively, and are bound onto this class from
:mod:`fmha.__init__`.
"""

import math
import os
from typing import Optional, Tuple, Type

import cutlass
import cutlass.pipeline as pipeline

from fmha import fmha_helpers as fmha_utils


def make_thread_cooperative_group(size: int):
    """Shorthand for a thread-level ``pipeline.CooperativeGroup``."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


class BlackwellFusedMultiHeadAttentionForward:
    """Host-side configuration for the Blackwell SM100 fused MHA kernel.

    See module docstring for what's *not* here (launchers / kernel bodies).
    """

    WINDOW_NO_LIMIT = 1 << 30
    # Blackwell per-CTA dynamic SMEM ~227KB; P1 uses D_chunk=128 so staged SMEM
    # matches D=128.
    SMEM_BUDGET_BYTES = 227 * 1024
    D_CHUNK = 128

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
        is_causal: bool = False,
        use_sliding_window: bool = False,
        actual_head_dim: Optional[int] = None,
    ):
        """Initialize FMHA configuration.

        :param qk_acc_dtype:    accumulator dtype for Q*K^T (currently fp32 only)
        :param pv_acc_dtype:    accumulator dtype for P*V (currently fp32 only)
        :param mma_tiler:       (M, N, K) shape of the MMA instruction unit;
                                ``K`` becomes the per-chunk D and is always 128 in P1.
        :param is_persistent:   whether the launcher uses a persistent grid scheduler
        :param mask_type:       :class:`fmha_helpers.MaskEnum` variant
        :param is_causal:       if True, ``window_size_right`` is compile-time pinned to 0
        :param use_sliding_window: if True, ``window_size_left`` is wired through to the
                                kernel; otherwise it is treated as ``None`` at compile time
        :param actual_head_dim: real head dim (may not be divisible by 128); D-chunking
                                bridges the gap via TMA ZFILL on load / OOB-drop on store
        """
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.head_dim = actual_head_dim if actual_head_dim is not None else mma_tiler[2]
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.log2_e = math.log2(math.e)
        # MMA / SMEM / TMEM use d_chunk_k-wide tiles; full head_dim via num_d_chunks.
        self.d_chunk_k = mma_tiler[2]
        self.num_d_chunks = max(
            1, (self.head_dim + self.d_chunk_k - 1) // self.d_chunk_k
        )
        self.cta_tiler = (
            2 * mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.qk_mma_tiler = mma_tiler
        self.pv_mma_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1])
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.is_causal = is_causal
        self.use_sliding_window = use_sliding_window

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15

        # tcgen05 hardware constants
        import cutlass.cute as cute

        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        # P1 D-chunk: serialize Load vs MMA via PipelineTmaUmma acquire/release only
        # (no NamedBarrier between warps -- persistent scheduling desyncs tile phases).
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )

        # TMEM map (column offsets; QK S is 128 wide, PV O is D_chunk_k wide)
        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        self.tmem_p0_offset = 32
        self.tmem_p1_offset = 160

        # vec buffers for row_max & row_sum
        self.tmem_vec0_offset = 0
        self.tmem_vec1_offset = 128

        self.num_regs_softmax = 192
        self.num_regs_correction = 96
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids))
            // num_warps_per_warpgroup
        )

    # ------------------------------------------------------------------
    # Pipeline-stage selection / SMEM budgeting
    # ------------------------------------------------------------------
    def _pick_pipeline_stages(self, d_eff: int) -> None:
        """Choose ``q_stage`` / ``kv_stage`` / ``epi_stage``.

        ``q_stage`` must stay at 2 (dual Q-tile per CTA). For D > 128 we still
        run D-chunked, so SMEM per-chunk matches D=128 and we keep the same
        staging budget.
        """
        is_fp8 = self.q_dtype.width == 8
        self.q_stage = 2
        self.kv_stage = 4 if is_fp8 else 3
        self.epi_stage = 2

    @staticmethod
    def _estimate_staged_smem_bytes(
        d_eff: int,
        elem_bytes: int,
        q_stage: int,
        kv_stage: int,
        epi_stage: int,
        m_tile: int,
        n_tile: int,
        epi_m: int,
    ) -> int:
        """Conservative staged Q+K(V)+O SMEM per MMA tile (not full ``cta_tiler`` M)."""
        return elem_bytes * (
            q_stage * m_tile * d_eff
            + kv_stage * n_tile * d_eff
            + epi_stage * epi_m * d_eff
        )

    def validate_config_host(self, q_dtype: Type[cutlass.Numeric]) -> None:
        """Pre-``cute.compile`` sanity check: fail fast if SMEM clearly won't fit.

        Picks pipeline stages for the requested input dtype, then computes a
        conservative staged-SMEM estimate. Raises if the estimate exceeds
        :attr:`SMEM_BUDGET_BYTES` (227 KB on Blackwell).
        Also prints a notice when D-chunking is active (num_d_chunks > 1).
        """
        self.q_dtype = q_dtype
        d_eff = self.qk_mma_tiler[2]
        self._pick_pipeline_stages(d_eff)
        elem_bytes = max(1, q_dtype.width // 8)
        m_tile = self.qk_mma_tiler[0]
        n_tile = self.qk_mma_tiler[1]
        est = self._estimate_staged_smem_bytes(
            d_eff,
            elem_bytes,
            self.q_stage,
            self.kv_stage,
            self.epi_stage,
            m_tile,
            n_tile,
            m_tile,
        )
        if est > self.SMEM_BUDGET_BYTES:
            raise ValueError(
                f"FMHA estimated staged SMEM {est} bytes exceeds budget "
                f"{self.SMEM_BUDGET_BYTES} for D_chunk={d_eff} "
                f"(head_dim={self.head_dim}, num_d_chunks={self.num_d_chunks}). "
                f"see study_cute/docs/support_d256.md"
            )
        if self.num_d_chunks > 1:
            # D>128 is implemented via an OUTER d_chunk loop in every
            # warp: the full attention pipeline runs num_d_chunks times
            # per kv tile, each iteration using a single V[d_chunk_outer]
            # slice and writing to gO[..., d_chunk_outer, ...].
            #   - QK still walks the inner d_chunk loop and accumulates S
            #     across ALL d_chunks (S depends on the full d).
            #   - PV degenerates to the D=128 single-V mode (no inner
            #     d_chunk loop), so TMEM stays within 512 cols.
            #   - Cost: QK / softmax are recomputed num_d_chunks times
            #     (D=256 -> ~1.5x latency, D=128 unaffected).
            # See docs/d_chunk_redesign.md for details.
            print(
                f"[fmha] D-chunking: head_dim={self.head_dim}, "
                f"d_chunk_k={self.d_chunk_k}, "
                f"num_d_chunks={self.num_d_chunks} (outer-loop mode), "
                f"stages q={self.q_stage} kv={self.kv_stage} "
                f"epi={self.epi_stage}, estimated staged SMEM={est} bytes"
            )

    def _setup_attributes(self):
        """Populate dtype-dependent pipeline / scaling parameters.

        Sets staging counts (``acc_stage`` / ``softmax_corr_stage`` /
        ``mma_corr_stage`` / ``mma_softmax_stage``) and softmax pre-scale used
        to spread FP8 E4M3's [0, 448] dynamic range across the typical [0, 1]
        probability range. Idempotent.
        """
        self._pick_pipeline_stages(self.qk_mma_tiler[2])
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1

        # Pre-scale softmax output by 2^FP8_E4M3_PRESCALE_LOG2=256 for FP8 to
        # maximize E4M3 dynamic range. P in [0,1] -> P*256 in [0,256], utilizing
        # more of the [0,448] FP8 range. Fused into exp2:
        # exp2(x + 8) = exp2(x) * 256, zero extra per-element ops.
        FP8_E4M3_PRESCALE_LOG2 = 8.0
        self.softmax_prescale_log2 = (
            FP8_E4M3_PRESCALE_LOG2 if self.q_dtype.width == 8 else 0.0
        )
        self.softmax_prescale_ln = self.softmax_prescale_log2 * 0.6931471805599453

    # ------------------------------------------------------------------
    # Debug trace gate (env-gated)
    # ------------------------------------------------------------------
    # Setting this to True enables cute.printf traces in the load/mma warps
    # so the producer/consumer pipeline can be observed step by step. The
    # printf is const_expr-gated, so when False it has zero runtime cost.
    # Toggle via FMHA_DEBUG_PIPELINE=1 in the environment.
    debug_pipeline = bool(int(os.environ.get("FMHA_DEBUG_PIPELINE", "0")))
