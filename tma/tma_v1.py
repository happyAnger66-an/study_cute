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

"""
================================================================================
tma_v1.py — Blackwell (SM100) 上使用 TMA 的矩阵转置示例
================================================================================

【功能】
  对 GPU 上的矩阵做转置：输入 src 形状 (M, N)，输出 dst 形状 (N, M)，
  满足 dst[j, i] = src[i, j]（与 PyTorch 的 input.t() 一致）。

【数据通路（单 CTA 处理一个 128×128 tile）】
  Global(src)  --TMA Load-->  sA (SMEM, 行主序 swizzle)
                --S2R + R2S-->  sB (SMEM, 列主序 swizzle，逻辑上为转置后的 tile)
  Global(dst)  <--TMA Store--  sB

【Warp 分工（生产者-消费者，共 6 个 warp = 192 线程）】
  Warp 0–3 : 消费者/生产者 — 从 sA 读到寄存器，写到 sB（转置由 layout 完成）
  Warp 4   : 生产者 — TMA 把全局 src tile 载入 sA
  Warp 5   : 消费者 — TMA 把 sB 写回全局 dst

【两类同步】
  1. mbarrier (load_mbar_ptr)
     - 生产者: TMA Load warp（+ 硬件 TMA 事务字节）
     - 消费者: 转置 warp 0–3（mbarrier_wait 后才读 sA）
  2. mbarrier (store_mbar_ptr)
     - 生产者: 转置 warp（elect_one 发起 arrive）
     - 消费者: TMA Store warp
  3. NamedBarrier (trans_sync_barrier)
     - 仅转置 warp 0–3 内部：确保 sB 的 SMEM store 对同 CTA 内其它线程可见，
       再向 store_mbar 发出 arrive

【与 tma_v0 的区别】
  v0 通常只有 TMA load/store + mbarrier；v1 在中间插入多 warp 的 S2R/R2S 转置阶段，
  演示「TMA + 普通 copy + 多种 barrier」的组合。

【依赖】
  CUTLASS CuTeDSL、SM100 (Blackwell) 上的 TMA (cpasync)、pipeline.NamedBarrier。

详见同目录 barrier.md。
"""

import argparse
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import torch


# =============================================================================
# Sm100MatrixTransposeKernelV1
# =============================================================================
class Sm100MatrixTransposeKernelV1:
    """
    SM100 矩阵转置内核的 Host 侧封装。

    职责：
      - 在 __call__ 中根据 src/dst 构建 TMA descriptor、SMEM layout、grid/block；
      - 在 kernel 中按 warp 角色执行 load → transpose → store 流水线。

    默认 tile：128×128 float16，每个 CTA 处理一个 tile。
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # Tile 与 Cluster
        # ---------------------------------------------------------------------
        # 每个 CTA 处理的子块大小 (tile_m, tile_n)。与 TMA atom 的 tile 一致。
        self.tile_shape = (128, 128)
        self.tile_m, self.tile_n = self.tile_shape

        # CTA cluster 形状 (M_cluster, N_cluster, K_cluster)。 (1,1,1) 表示无 cluster。
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)

        # ---------------------------------------------------------------------
        # Warp 角色分配（warp_idx 从 0 递增）
        # ---------------------------------------------------------------------
        # 转置 warp 数量：把 128 行均分给 4 个 warp，每 warp 负责 32 行相关的 copy。
        self.num_trans_warps = 4
        self.trans_warp_id = tuple(range(self.num_trans_warps))  # warp 0,1,2,3

        self.tma_load_warp_id = self.num_trans_warps       # warp 4：TMA Load
        self.tma_store_warp_id = self.num_trans_warps + 1  # warp 5：TMA Store

        # 每个 CTA 总线程数 = 6 warps × 32 = 192
        self.threads_per_cta = 32 * len(
            (self.tma_store_warp_id, self.tma_load_warp_id, *self.trans_warp_id)
        )
        # 参与转置的线程数（仅 warp 0–3）
        self.num_trans_threads = 32 * len(self.trans_warp_id)

        # 每个转置 warp 在 tile 内处理的子块形状（用于理解分块，copy 仍以 TV layout 映射）
        self.trans_tile = (self.tile_shape[0] // self.num_trans_warps, 8)

        # ---------------------------------------------------------------------
        # Named Barrier（CTA 内线程到齐，不跟踪 TMA 字节）
        # ---------------------------------------------------------------------
        # barrier_id=1：转置 warp 0–3 共 128 线程，在写 sB 后 fence 再 arrive_and_wait
        self.trans_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.trans_warp_id),
        )
        # barrier_id=2：预留给 store warp 的 NamedBarrier（本 kernel 主要用 mbarrier 等 store）
        self.store_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32,
        )
        # SMEM 缓冲区对齐（字节），满足 TMA / swizzle 要求
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor):
        """
        Host 可调用的 JIT 入口：配置 TMA + SMEM layout 并 launch kernel。

        参数:
            src: CuTe 张量，逻辑形状 (M, N)，行主序，dtype 与 dst 相同（如 float16）
            dst: CuTe 张量，逻辑形状 (N, M)，即转置后的全局输出

        计算逻辑（Host 侧）:
            1. 为 dst 构造「转置视图」transed_dst，形状 (M, N)，供 TMA store 描述符使用
            2. 分配 sA/sB 的 swizzled SMEM layout
            3. make_tiled_tma_atom：G2S 读 src→sA，S2G 写 sB→transed_dst
            4. grid = ceil_div(M,128) × ceil_div(N,128)，block = (192,1,1)

        返回:
            无；通过 launch 在 GPU 上执行 kernel
        """
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("Source and destination element types must match")

        self.dtype: Type[cutlass.Numeric] = src.element_type

        # -----------------------------------------------------------------
        # transed_dst：与 dst 共享底层存储，但 layout 为 (M,N) 且 stride 交换
        # 这样 TMA store 按「转置后的 (M,N) 行主序 tile」写回时，对应全局 dst(N,M)
        # -----------------------------------------------------------------
        transed_dst = cute.make_tensor(
            dst.iterator,
            cute.make_layout(
                (dst.shape[1], dst.shape[0]), stride=(dst.stride[1], dst.stride[0])
            ),
        )

        # sA：与 src 同 major 的 SMEM layout，形状 (tile_m, tile_n) = (128,128)
        smem_layout_sA = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(src).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,  # stages（单缓冲）
        )

        # sB：与 transed_dst 同 major 的 layout；列主序视角实现转置后的存法
        smem_layout_sB = sm100_utils.make_smem_layout(
            utils.LayoutEnum.from_tensor(transed_dst).mma_major_mode(),
            (self.tile_m, self.tile_n),
            self.dtype,
            1,
        )

        @cute.struct
        class SharedStorage:
            """CTA 级共享内存布局（由 SmemAllocator 在 kernel 内实例化）。"""

            # mbarrier：TMA load 完成 ↔ 转置 warp 可读 sA
            load_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # mbarrier：转置写 sB 完成 ↔ TMA store 可读 sB
            store_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            # 源 tile 缓冲区（TMA 目标）
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sA)], 128
            ]
            # 转置后 tile 缓冲区（TMA store 源）
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_sB)], 128
            ]

        self.shared_storage = SharedStorage
        # TMA load 期望完成的字节数（用于 mbarrier_expect_tx）
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout_sA)

        # -----------------------------------------------------------------
        # TMA Atom：描述「如何从全局张量搬一个 tile 到 SMEM」或反向
        # -----------------------------------------------------------------
        # CopyBulkTensorTileG2SOp：Global → Shared（Load）
        tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            src,
            smem_layout_sA,
            (self.tile_m, self.tile_n),
        )

        # CopyBulkTensorTileS2GOp：Shared → Global（Store）
        tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            transed_dst,
            smem_layout_sB,
            (self.tile_m, self.tile_n),
        )

        # Grid：每个 block 负责 src 上的一个 128×128 tile
        grid_shape = cute.ceil_div((*src.layout.shape, 1), self.tile_shape)
        self.kernel(
            tma_atom_src,
            tma_tensor_src,
            tma_atom_dst,
            tma_tensor_dst,
            smem_layout_sA,
            smem_layout_sB,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_load: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,
        tma_atom_store: cute.CopyAtom,
        tma_tensor_dst: cute.Tensor,
        smem_layout_sA: cute.ComposedLayout,
        smem_layout_sB: cute.ComposedLayout,
    ):
        """
        设备端内核：单 CTA 内 6 个 warp 协作完成一个 tile 的转置。

        参数:
            tma_atom_load   : TMA G2S copy atom（预编码 swizzle、tile 形状等）
            tma_tensor_src  : 带 TMA 信息的源张量视图（全局）
            tma_atom_store  : TMA S2G copy atom
            tma_tensor_dst  : 带 TMA 信息的目标张量视图（全局，对应 transed_dst）
            smem_layout_sA  : sA 的 composed layout（含 swizzle）
            smem_layout_sB  : sB 的 composed layout（含 swizzle）

        计算逻辑（按 warp 分支）:

          [初始化] 线程 0 初始化两个 mbarrier；全 CTA barrier 同步

          [Warp 4 - TMA Load]
            tma_partition → cute.copy(G→sA)，绑定 load_mbar_ptr
            elect_one → mbarrier_arrive（与 expect_tx 配合表示「发起方已到」）

          [Warp 0–3 - Transpose]
            mbarrier_wait(load) → 读 sA 到寄存器 → 写寄存器到 sB
            （转置：partition_S 用 sA layout，partition_D 用 sB layout，同一线程 TV 映射）
            fence_proxy(async.shared) → trans_sync_barrier → mbarrier_arrive(store)

          [Warp 5 - TMA Store]
            mbarrier_wait(store) → cute.copy(sB→G)

        CTA 坐标 (bidx, bidy) 选择全局 tile：gA[(None, bidx, bidy)] 等。
        """
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ------------------------------------------------------------------
        # 分配共享内存：sA、sB 两片 buffer + 两个 mbarrier 对象
        # ------------------------------------------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(smem_layout_sA.outer, swizzle=smem_layout_sA.inner)
        sB = storage.sB.get_tensor(smem_layout_sB.outer, swizzle=smem_layout_sB.inner)
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout_sA)
        load_mbar_ptr = storage.load_mbar_ptr.data_ptr()
        store_mbar_ptr = storage.store_mbar_ptr.data_ptr()

        # ------------------------------------------------------------------
        # 初始化 mbarrier（所有 warp 参与；仅 tidx==0 写入 SMEM）
        # ------------------------------------------------------------------
        if tidx == 0:
            # Load 屏障：需要 1 次 arrive（TMA warp 在 copy 后 arrive）
            # expect_tx：硬件在 TMA 搬完 num_tma_load_bytes 后才会满足 wait
            cute.arch.mbarrier_init(load_mbar_ptr, 1)
            cute.arch.mbarrier_expect_tx(load_mbar_ptr, self.num_tma_load_bytes)

            # Store 屏障：需要 len(trans_warp_id)=4 次 arrive（每个转置 warp elect_one 一次）
            cute.arch.mbarrier_init(store_mbar_ptr, len(self.trans_warp_id))
        cute.arch.mbarrier_init_fence()

        # 确保 init 对所有线程可见后再分工
        cute.arch.barrier()

        # ------------------------------------------------------------------
        # PRODUCER: TMA Load Warp — Global → sA
        # ------------------------------------------------------------------
        if warp_idx == self.tma_load_warp_id:
            # local_tile：把全局张量按 tile_shape 切分为 ((128,128), grid_m, grid_n)
            gA = cute.local_tile(tma_tensor_src, self.tile_shape, (None, None))
            # tma_partition：把 TMA atom 绑定到 SMEM tile 与全局 tile 的对应切片
            #   tAsA: SMEM 侧视图  tAgA: Global 侧视图
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )

            # 异步 TMA：把 (bidx,bidy) 对应的全局 tile 载入 sA[(None,0)]
            # tma_bar_ptr：copy 与 load_mbar 关联，字节到位由硬件计入 expect_tx
            cute.copy(
                tma_atom_load,
                tAgA[(None, bidx, bidy)],
                tAsA[(None, 0)],
                tma_bar_ptr=load_mbar_ptr,
            )

            # 生产者 arrive：满足 init 的 arrive 计数（与 TMA 事务一起完成 wait 条件）
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(load_mbar_ptr)

        # ------------------------------------------------------------------
        # CONSUMER + PRODUCER: Transpose Warps — sA → Reg → sB
        # ------------------------------------------------------------------
        if warp_idx < self.tma_load_warp_id:
            # 在 128 个转置线程中编号 0..127（跨 warp 0–3）
            trans_tid = tidx % self.num_trans_threads

            # 消费者：等待 TMA load 完成（phase=0）
            cute.arch.mbarrier_wait(load_mbar_ptr, 0)

            # Universal copy atom：每次拷贝 1 个元素（width = dtype 位宽）
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=self.dtype.width,
            )

            copy_elems = 1

            # -----------------------------------------------------------------
            # Tiled Copy TV 布局（T=thread, V=value lane）
            # -----------------------------------------------------------------
            # thread_layout: (128, 1)，stride (1, 128) → 线性下标 idx = trans_tid
            # value_layout: (1, 1) → 每线程 1 个元素
            #
            # 转置如何发生：
            #   - 读：thr_copy.partition_S(sA) 按 sA 的 SMEM layout 划分
            #   - 写：thr_copy.partition_D(sB) 按 sB 的 SMEM layout 划分
            #   - 同一线程 TV 索引读写的「逻辑坐标」在 sA/sB 中对应转置关系
            #     （sB layout 按 transed_dst 列主序/swizzle 构建）
            # -----------------------------------------------------------------
            thread_layout = cute.make_layout(
                (self.num_trans_threads, 1),
                stride=(1, self.num_trans_threads),
            )
            value_layout = cute.make_layout((1, copy_elems))
            tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
            thr_copy = tiled_copy.get_slice(trans_tid)

            # Step 1: SMEM sA → Register（按 sA layout 分片）
            tCsA = thr_copy.partition_S(sA)
            tCrA = cute.make_fragment_like(tCsA)
            cute.copy(tiled_copy, tCsA, tCrA)

            # Step 2: Register → SMEM sB（按 sB layout 分片，实现转置写入）
            tCsB = thr_copy.partition_D(sB)
            cute.copy(tiled_copy, tCrA, tCsB)

            # 保证 sB 的 SMEM 写对 CTA 内其它线程（及后续 TMA）可见
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # 转置 warp 内部到齐（NamedBarrier，不涉及 TMA 字节）
            self.trans_sync_barrier.arrive_and_wait()

            # 每个转置 warp 选 1 个线程 arrive store_mbar（共 4 次，匹配 init 计数）
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(store_mbar_ptr)

        # ------------------------------------------------------------------
        # CONSUMER: TMA Store Warp — sB → Global
        # ------------------------------------------------------------------
        if warp_idx == self.tma_store_warp_id:
            # 等待所有转置 warp 完成 sB 写入
            cute.arch.mbarrier_wait(store_mbar_ptr, 0)

            gDst_cta = cute.local_tile(
                tma_tensor_dst, (self.tile_m, self.tile_n), (None, None)
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_store,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gDst_cta, 0, 2),
            )
            # 从 sB 搬回全局 (bidx, bidy) 对应 tile
            cute.copy(tma_atom_store, tBsB[(None, 0)], tBgB[(None, bidx, bidy)])


# =============================================================================
# Host 测试与性能
# =============================================================================
def run_transpose(M, N, num_warmup=5, num_iters=20):
    """
    编译并运行转置 kernel，测带宽并与 PyTorch 参考结果比对。

    参数:
        M: 源矩阵行数（input shape 第一维）
        N: 源矩阵列数（input shape 第二维）
        num_warmup: 计时前预热次数（触发 JIT 后稳定 GPU 状态）
        num_iters: 计时迭代次数

    数据布局:
        input_data  : torch (M, N) float16 CUDA
        output_data : torch (N, M) float16 CUDA，kernel 写入转置结果

    性能模型:
        总流量 ≈ 2 * M * N * sizeof(dtype)（读 src 一遍 + 写 dst 一遍）
        Throughput = total_bytes / avg_time
        Theoretical BW：按 Blackwell 文档 2048 B/clk @ 4000 MHz 估算峰值

    验证:
        expected = input_data.t()，atol=1e-2（float16）
    """
    torch.manual_seed(1111)
    input_data = torch.randn((M, N), device="cuda", dtype=torch.float16)
    output_data = torch.zeros((N, M), device="cuda", dtype=torch.float16)

    # from_dlpack：把 PyTorch 张量零拷贝包装为 CuTe 张量
    # mark_layout_dynamic(leading_dim=1)：leading dimension 运行时可变
    # mark_compact_shape_dynamic(mode=1, divisibility=16)：第 1 维紧凑且 16 对齐（TMA 要求）
    tensor_src = (
        from_dlpack(input_data, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )
    tensor_dst = (
        from_dlpack(output_data, assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )

    transpose_kernel = Sm100MatrixTransposeKernelV1()

    print("Start kernel compilation...")
    # cute.compile：JIT 编译 __call__，生成可重复调用的 compiled_kernel
    compiled_kernel = cute.compile(
        transpose_kernel, tensor_src, tensor_dst, options="--generate-line-info"
    )

    print("Start kernel warmup...")
    for _ in range(num_warmup):
        compiled_kernel(tensor_src, tensor_dst)
    torch.cuda.synchronize()
    print("Kernel warmup completed.")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        compiled_kernel(tensor_src, tensor_dst)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iters

    bytes_per_element = input_data.element_size()
    total_bytes = 2 * M * N * bytes_per_element
    throughput_gb_s = (total_bytes / 1e9) / (avg_time_ms / 1000)

    bytes_per_clk = 2048
    freq_mhz = 4000
    theoretical_bw_gb_s = bytes_per_clk * freq_mhz * 1e6 / 1e9
    theoretical_bw_tb_s = theoretical_bw_gb_s / 1000
    bandwidth_efficiency = (throughput_gb_s / theoretical_bw_gb_s) * 100

    print(f"Matrix size: {M}×{N}")
    print(f"Tile shape: {transpose_kernel.tile_shape}")
    print(f"Average time: {avg_time_ms:.4f} ms")
    print(f"Throughput: {throughput_gb_s:.2f} GB/s")
    print(
        f"Theoretical BW: {theoretical_bw_tb_s:.2f} TB/s ({theoretical_bw_gb_s:.2f} GB/s)"
    )
    print(f"Bandwidth Efficiency: {bandwidth_efficiency:.2f}%")

    expected = input_data.t()
    if torch.allclose(output_data, expected, atol=1e-2):
        print("Verification: PASSED ✓")
    else:
        print("Verification: FAILED ✗")
        print(f"Max diff: {(output_data - expected).abs().max()}")


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="TMA Matrix Transpose with Producer-Consumer Pattern (SM100)"
    )
    parser.add_argument("--M", type=int, default=128, help="源矩阵行数 M")
    parser.add_argument("--N", type=int, default=128, help="源矩阵列数 N")
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="预热迭代次数"
    )
    parser.add_argument(
        "--num_iters", type=int, default=20, help="计时迭代次数"
    )
    args = parser.parse_args()

    run_transpose(
        args.M,
        args.N,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
