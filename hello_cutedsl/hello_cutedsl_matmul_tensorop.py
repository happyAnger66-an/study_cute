#!/usr/bin/env python3
"""
高性能 GEMM（CuTeDSL）：D = A @ B，基于 Ampere Tensor Core 的 `TensorOpGemm`。

与 `fmha.py` 中 `cutlass.cute` 用法的关系
----------------------------------------
`fmha.py` 里在 MMA warp 上调用 `cute.gemm(tiled_mma, tStC, tSrA, tSrB, tStC)`，配合
`tcgen05`、`pipeline`、`tiled_mma` 与多级流水线，属于 Blackwell 上高度定制的融合算子。

本脚本走同一 DSL 家族里**更通用、可复用的 dense GEMM 路径**：官方实现的
`TensorOpGemm`（见 CUTLASS 树内 `cute/ampere/kernel/dense_gemm/tensorop_gemm.py`），内部同样通过
`tiled_mma` + `cp.async` G2S + 多 stage SMEM 实现高吞吐，但面向经典 C = A×B，无需 SM100。

依赖与硬件
----------
- **GPU**：Ampere（SM80）及以上（Tensor Core FP16 路径；Ada/Hopper 通常也可跑该 Ampere 示例）。
- **Python**：需能 `import cute`（CuTeDSL 示例包）。若未安装到 site-packages，请设置环境变量
  `CUTLASS_CUTEDSL_PATH` 指向 `.../cutlass/examples/python/CuTeDSL`，或把本仓库与 `cutlass` 放在
  同级目录（脚本会尝试 `../../../cutlass/examples/python/CuTeDSL`）。

精度与形状
----------
- 计算为 **FP16 × FP16 → FP32 累加 → FP16 输出**（与官方 `tensorop_gemm` 默认一致）。
- `TensorOpGemm` 的 CTA tile 为 128×128×32；动态编译要求 K、N 维满足可整除约束（与官方
  `compile_bmm_dynamic_layout` 一致）。本示例对 M、N、K **向上 pad** 到 128/128/32 的倍数，
  再截取 `D[:M,:N]` 与 `torch.matmul` 对比。

运行示例::

    python3 hello_cutedsl_matmul_tensorop.py --m 512 --n 256 --k 128
    python3 hello_cutedsl_matmul_tensorop.py --m 512 --n 256 --k 128 --bench
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch


def _ensure_cutedsl_examples_on_path() -> str:
    env = os.environ.get("CUTLASS_CUTEDSL_PATH")
    if env:
        p = Path(env).resolve()
        if (p / "cute").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return s
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "cutlass" / "examples" / "python" / "CuTeDSL",
        here.parent.parent.parent / "cutlass" / "examples" / "python" / "CuTeDSL",
    ]
    for p in candidates:
        if (p / "cute").is_dir():
            s = str(p.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return s
    raise RuntimeError(
        "找不到 CuTeDSL 示例目录（需包含 `cute` 包）。请设置 CUTLASS_CUTEDSL_PATH 或把 cutlass "
        "clone 到与 study_cute 同级的 codes/ 目录下。"
    )


_ensure_cutedsl_examples_on_path()
from cute.ampere.kernel.dense_gemm.tensorop_gemm import TensorOpGemm  # noqa: E402


def _pad(x: int, align: int) -> int:
    return (x + align - 1) // align * align


@cute.jit
def bmm_tensorop(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    """与 `dsl_tutorials/tvm_ffi/ampere_gemm_with_fake_tensor.py` 中 `bmm` 相同：L 维 batch GEMM。"""
    # atom_layout_mnk=(M_atom, N_atom, K_atom)：在 CTA 内如何把「单条 WMMA 指令」在逻辑 MNK 上铺成 tiled_mma。
    # 单条 Ampere FP16 MMA 形状为 16×8×16（见 tensorop_gemm.mma_inst_shape）；(2,2,1) 表示在 M、N 方向各复制
    # 2 份、K 方向 1 份，共 2×2×1=4 个 MMA 原子；每个原子对应一整个 warp（32 线程），故 MMA 相关 warps=128 线程。
    # K_atom 在本示例中必须为 1（源码 assert）；CTA tile 128×128×32 需能被 (16*M_atom, 8*N_atom*调整因子, 16*K_atom) 整除。
    gemm_op = TensorOpGemm(
        cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1)
    )
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c)


def _compile_bmm(m: int, n: int, k: int, batch: int = 1):
    from cutlass.cute.runtime import make_fake_compact_tensor

    fake_a = make_fake_compact_tensor(
        cutlass.Float16, (batch, m, k), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_b = make_fake_compact_tensor(
        cutlass.Float16, (batch, k, n), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_c = make_fake_compact_tensor(
        cutlass.Float16, (batch, m, n), stride_order=(2, 1, 0), assumed_align=16
    )
    return cute.compile(
        bmm_tensorop, fake_a, fake_b, fake_c, options="--enable-tvm-ffi"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="CuTeDSL TensorOp GEMM (Ampere+)")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=128)
    p.add_argument(
        "--bench",
        action="store_true",
        help="与 torch.matmul (fp16) 粗略对比耗时（仅作数量级参考）",
    )
    p.add_argument("--bench-iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU。")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(
            f"TensorOpGemm 本示例针对 SM80+ Tensor Core，当前设备 capability {major}.{minor}。"
        )

    m, n, k = args.m, args.n, args.k
    mp, np, kp = _pad(m, 128), _pad(n, 128), _pad(k, 32)

    torch.manual_seed(0)
    a = torch.randn(1, mp, kp, dtype=torch.float16, device="cuda")
    b = torch.randn(1, kp, np, dtype=torch.float16, device="cuda")
    c = torch.zeros(1, mp, np, dtype=torch.float16, device="cuda")

    t0 = time.time()
    compiled = _compile_bmm(mp, np, kp, 1)
    t1 = time.time()

    compiled(a, b, c)
    torch.cuda.synchronize()
    t2 = time.time()

    a_sub = a[:, :m, :k]
    b_sub = b[:, :k, :n]
    ref = torch.matmul(a_sub.float(), b_sub.float()).half()
    torch.testing.assert_close(c[:, :m, :n], ref, rtol=2e-3, atol=2e-3)
    print(f"[OK] TensorOpGemm 校验通过 (逻辑形状 M={m} N={n} K={k}, pad 至 {mp}×{np}×{kp})")
    print(f"[INFO] compile: {(t1 - t0):.3f}s, 首次执行: {(t2 - t1):.3f}s")

    if args.bench:
        it = args.bench_iters
        warm = 5
        for _ in range(warm):
            compiled(a, b, c)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(it):
            compiled(a, b, c)
        torch.cuda.synchronize()
        t_cutlass = (time.time() - t0) / it

        a2 = a_sub.contiguous()
        b2 = b_sub.contiguous()
        c2 = torch.zeros(1, m, n, dtype=torch.float16, device="cuda")
        for _ in range(warm):
            torch.matmul(a2, b2, out=c2)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(it):
            torch.matmul(a2, b2, out=c2)
        torch.cuda.synchronize()
        t_torch = (time.time() - t0) / it
        print(f"[BENCH] TensorOp GEMM (pad): {t_cutlass * 1e6:.2f} us/iter")
        print(f"[BENCH] torch.matmul:       {t_torch * 1e6:.2f} us/iter")


if __name__ == "__main__":
    main()
