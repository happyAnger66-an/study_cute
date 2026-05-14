#!/usr/bin/env python3
"""
CuTeDSL 最小矩阵乘法示例：D = A @ B（A: M×K, B: K×N，行主序展平为一维）。

用法概要（与 NVIDIA CUTLASS CuTeDSL 文档/示例一致）：
- `cutlass.cute`：用 `@cute.kernel` / `@cute.jit` 编写设备与主机侧 DSL。
- `from_dlpack`：把 `torch.Tensor` 零拷贝包装为 `cute.Tensor`；`mark_layout_dynamic` 便于按运行时形状 JIT。
- `cute.compile`：对 `@cute.jit` 入口编译，得到可反复调用的 callable。

官方示例与教程路径（本地 CUTLASS 树内）：
- 环境：`python/CuTeDSL/setup.sh`（或 `--editable` / `--cu13`）。
- Torch + DLPack：`examples/python/CuTeDSL/dsl_tutorials/tvm_ffi/jit_and_use_in_torch.py`
- 同类 tiled GEMM（JAX 侧）：`examples/python/CuTeDSL/dsl_tutorials/jax/cute_dsl_jax_kernels.py` 中 `gemm_kernel` / `launch_gemm`
- Notebook：`examples/python/CuTeDSL/cute/notebooks/hello_world.ipynb`

运行：

    python3 hello_cutedsl_matmul.py --m 128 --n 128 --k 64
"""

from __future__ import annotations

import argparse
import time

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    D: cute.Tensor,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    BLOCK_M: cutlass.Constexpr[int],
    BLOCK_N: cutlass.Constexpr[int],
):
    """GEMM 核函数：按输出分块（tile），块内线程协作覆盖 tile 内多个 (m,n)。

    数据布局：A 展平为长度 M*K（行主，第 m 行从 m*K 起连续 K 个元素）；
    B 展平为长度 K*N（行主，第 k 行从 k*N 起连续 N 个元素）；
    D 展平为长度 M*N。公式 D[m,n] = sum_k A[m,k]*B[k,n] 对应一维下标如上。
    """
    # 当前线程在 CTA（block）内的线性下标，以及当前 CTA 在二维 grid 上的 (bm, bn)。
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    # 本 CTA 的线程总数；循环步长为 bdx，实现“线程 i 处理 i, i+bdx, i+2bdx, ...”的条带划分。
    bdx, _, _ = cute.arch.block_dim()

    for i in cutlass.range(tidx, BLOCK_M * BLOCK_N, bdx):
        # 把线性下标 i 映射到本 tile 内的局部 (row, col)，tile 形状为 BLOCK_M × BLOCK_N。
        row = i // BLOCK_N
        col = i % BLOCK_N
        # 全局输出坐标：第 (bm,bn) 个 tile 的左上角加上局部偏移。
        m_idx = bm * BLOCK_M + row
        n_idx = bn * BLOCK_N + col
        # 边界：最后一行/列 tile 可能越界，仅合法 (m,n) 才写回。
        if m_idx < M and n_idx < N:
            acc = cutlass.Float32(0.0)
            # 朴素 K 维点积；每个输出点由单线程独立完成（无共享内存/Warp MMA）。
            for k in cutlass.range(K):
                acc += A[m_idx * K + k] * B[k * N + n_idx]
            D[m_idx * N + n_idx] = acc


@cute.jit
def gemm_entry(
    A: cute.Tensor,
    B: cute.Tensor,
    D: cute.Tensor,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    BLOCK_M: cutlass.Constexpr[int] = 64,
    BLOCK_N: cutlass.Constexpr[int] = 64,
):
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    gemm_kernel(A, B, D, M, N, K, BLOCK_M, BLOCK_N).launch(
        grid=[grid_m, grid_n, 1],
        block=[256, 1, 1],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CuTeDSL hello: D = A @ B (FP32)")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=64)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU。")

    m, n, k = args.m, args.n, args.k
    a2 = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b2 = torch.randn(k, n, device="cuda", dtype=torch.float32)
    d2 = torch.zeros(m, n, device="cuda", dtype=torch.float32)

    a1 = a2.reshape(-1).contiguous()
    b1 = b2.reshape(-1).contiguous()
    d1 = d2.reshape(-1).contiguous()

    a_t = from_dlpack(a1).mark_layout_dynamic()
    b_t = from_dlpack(b1).mark_layout_dynamic()
    d_t = from_dlpack(d1).mark_layout_dynamic()

    t0 = time.time()
    compiled = cute.compile(
        gemm_entry,
        a_t,
        b_t,
        d_t,
        m,
        n,
        k,
        options="--generate-line-info",
    )
    t1 = time.time()
    # compile 与 invoke 的实参列表需一致（含 M,N,K）；形状变化需重新 compile。
    compiled(a_t, b_t, d_t, m, n, k)
    torch.cuda.synchronize()
    t2 = time.time()

    ref = a2 @ b2
    torch.testing.assert_close(d2, ref, rtol=1e-3, atol=1e-3)
    print(f"[OK] D = A@B 校验通过 M={m} N={n} K={k}")
    print(f"[INFO] compile: {(t1 - t0):.3f}s, 首次执行: {(t2 - t1):.3f}s")
    print("[INFO] D[0,0:4] =", d2[0, 0:4].tolist())


if __name__ == "__main__":
    main()
