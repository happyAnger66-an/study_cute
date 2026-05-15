#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# 基于 cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py 的变体：
# - 保留 Blackwell tcgen05：SMEM 操作数、TMEM 累加、tcgen05 TMEM→RMEM→GMEM epilogue。
# - 去掉 TMA：GMEM→SMEM 使用 cp.async `CopyG2SOp` + `make_tiled_copy_tv`（与 Ampere TensorOpGemm 同类路径，
#   见 cute/ampere/kernel/dense_gemm/tensorop_gemm.py）。
# - 为降低复杂度，A/B 仅 **单 stage** SMEM（无 multi-stage AB 流水），K 维按 tile 顺序同步搬移+MMA。
#
# 运行（需在含 CuTeDSL 的 CUTLASS 环境、Blackwell SM100 设备上）::
#
#   python3 fp16_gemm_no_tma.py --mnk 8192,8192,8192
#
# 约束：M、N 可被 (128,256) 整除；K 可被 64 整除（与 mma_tiler 的 K 维一致）。
# G2S 的 tiled copy 固定为 **ROW_MAJOR（K-major）** 布局，与下方 `run_dense_gemm` 中
# `mark_layout_dynamic(leading_dim=1)` 一致；若需列主存储，应另写 `const_expr` 分支或单独 JIT。

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack


def _ensure_cutedsl_examples_on_path() -> None:
    env = os.environ.get("CUTLASS_CUTEDSL_PATH")
    if env:
        p = Path(env).resolve()
        if (p / "cute").is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return
    here = Path(__file__).resolve().parent
    for p in (
        here.parent.parent / "cutlass" / "examples" / "python" / "CuTeDSL",
        here.parent.parent.parent / "cutlass" / "examples" / "python" / "CuTeDSL",
    ):
        if (p / "cute").is_dir():
            s = str(p.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_ensure_cutedsl_examples_on_path()

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128

# 单 stage AB SMEM（非 TMA 路径下简化同步与索引）
ab_stages = 1
acc_stage = 1


@cute.struct
class SharedStorage:
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        getattr(storage.tmem_holding_buf, "ptr", storage.tmem_holding_buf),
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(512)

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()

    # make_tiled_copy_tv 的首维是嵌套 (Thr,Val)（如 (8,1),(8,64)），需用嵌套 None 与 Ampere
    # tensorop_gemm 中 tAgA[None, None, None, k] 对齐；单 stage 时 SMEM 管道下标恒为 0。
    _tv = ((None, None), (None, None))
    for k_tile_idx in cutlass.range(num_k_tiles):
        cute.copy(
            tiled_copy_A,
            tAgA[_tv, None, k_tile_idx],
            tAsA[_tv, None, 0],
        )
        cute.copy(
            tiled_copy_B,
            tBgB[_tv, None, k_tile_idx],
            tBsB[_tv, None, 0],
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if warp_idx == 0:
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

    if warp_idx == 0:
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()

    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op_mma = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op_mma)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )

    atom_g2s = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        a.element_type,
        num_bits_per_copy=128,
    )
    copy_bits = 128
    copy_elems = copy_bits // a.element_type.width
    bK = mma_tiler_mnk[2]
    # 与 run_dense_gemm 中 mark_layout_dynamic(leading_dim=1) 一致：K-major / ROW_MAJOR。
    # 不在此用运行时 if 切换 thread_layout（DSL 要求分支两侧类型结构一致，需 const_expr）。
    shape_dim_1 = bK // copy_elems
    thread_layout_copy = cute.make_layout(
        (threads_per_cta // shape_dim_1, shape_dim_1),
        stride=(shape_dim_1, 1),
    )
    value_layout_copy = cute.make_layout((1, copy_elems))
    tiled_copy_A = cute.make_tiled_copy_tv(atom_g2s, thread_layout_copy, value_layout_copy)
    tiled_copy_B = cute.make_tiled_copy_tv(atom_g2s, thread_layout_copy, value_layout_copy)

    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        tiled_copy_A,
        tiled_copy_B,
        a,
        b,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )


def run_dense_gemm(mnk: Tuple[int, int, int], tolerance: float) -> None:
    import torch
    import cutlass.torch as cutlass_torch

    print("===================================================================")
    print("Blackwell fp16 GEMM（无 TMA，cp.async G2S + tcgen05）")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")

    m, n, k = mnk
    torch.manual_seed(1111)

    def make_tensors(mn: int, kk: int, dtype):
        return (
            torch.empty(mn, kk, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    host_function(a_tensor, b_tensor, c_tensor, no_cache=True)

    ref = torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32)).cpu()
    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-5
    )


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            ) from e

    from cuda.bindings import driver as cu_driver

    cu_driver.cuInit(0)
    err, device_count = cu_driver.cuDeviceGetCount()
    if err != cu_driver.CUresult.CUDA_SUCCESS or device_count < 1:
        raise RuntimeError("需要 CUDA GPU 运行本示例")

    parser = argparse.ArgumentParser(
        description="Blackwell fp16 GEMM：cp.async G2S（无 TMA）+ tcgen05"
    )
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=[8192, 8192, 8192],
        help="MNK，逗号分隔",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.1, help="校验容差"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk 必须为三个整数")
    m, n, k = args.mnk
    if m % mma_tiler_mnk[0] != 0 or n % mma_tiler_mnk[1] != 0:
        parser.error("m、n 必须分别被 mma_tiler 的 M、N 整除")
    if k % mma_tiler_mnk[2] != 0:
        parser.error("k 必须被 mma_tiler 的 K（64）整除")

    run_dense_gemm(tuple(args.mnk), args.tolerance)
    print("PASS")
