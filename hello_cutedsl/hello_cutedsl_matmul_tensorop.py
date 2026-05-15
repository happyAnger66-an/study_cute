#!/usr/bin/env python3
"""
高性能 GEMM（CuTeDSL）：D = A @ B，按 GPU 架构选择官方 dense GEMM 实现。

与 `fmha.py` 中 `cutlass.cute` 用法的关系
----------------------------------------
`fmha.py` 里在 MMA warp 上调用 `cute.gemm(tiled_mma, tStC, tSrA, tSrB, tStC)`，配合
`tcgen05`、`pipeline`、`tiled_mma` 与多级流水线，属于 Blackwell 上高度定制的融合算子。

本脚本走 DSL 里**可复用的 dense GEMM 封装**，根据 `torch.cuda.get_device_capability()` 分支：

- **SM8.x / 9.x（Ampere、Ada、Hopper 等）**：`TensorOpGemm`
  （`cute/ampere/kernel/dense_gemm/tensorop_gemm.py`），`tiled_mma` + `cp.async` G2S + 多 stage SMEM。
- **SM11.x / 12.x+（Jetson Thor `sm_110`、GB10 等 Blackwell GeForce）**：`Sm120GemmKernel`
  （`cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py`）。PyTorch 对 Thor 通常报告
  `get_device_capability() == (11, 0)`（与 CUDA 架构名 sm_110 对应），与桌面 SM12x 一样走该示例核。

数据中心 Blackwell **仅 SM10x**（`DenseGemmKernel` / tcgen05）未在此脚本接入；请直接用 CUTLASS 树内
`cute/blackwell/kernel/dense_gemm` 示例。

依赖与硬件
----------
- **GPU**：SM80–9x 走 Ampere 路径；**SM11+**（含 Jetson Thor + CUDA 13）走 Blackwell GeForce `Sm120GemmKernel`
  路径（需与当前 CUTLASS CuTeDSL 版本匹配）。
- **Python**：需能 `import cute`（CuTeDSL 示例包）。若未安装到 site-packages，请设置环境变量
  `CUTLASS_CUTEDSL_PATH` 指向 `.../cutlass/examples/python/CuTeDSL`，或把本仓库与 `cutlass` 放在
  同级目录（脚本会尝试 `../../../cutlass/examples/python/CuTeDSL`）。

精度与形状
----------
- 计算为 **FP16 × FP16 → FP32 累加 → FP16 输出**。
- **Ampere 路径**：CTA tile 128×128×32；M、N、K 向上 pad 到 128/128/**32** 的倍数。
- **Blackwell GeForce 路径（SM110 Thor / SM12x 等）**：默认 tile 128×128×**64**；M、N、K 向上 pad 到
  128/128/**64** 的倍数。
- 再截取逻辑 `M×N` 与 `torch.matmul`（FP32 累加参考）对比。

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
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import from_dlpack


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

try:
    from cute.blackwell_geforce.kernel.dense_gemm.dense_gemm import (  # noqa: E402
        Sm120GemmKernel,
    )
except ImportError:
    Sm120GemmKernel = None  # type: ignore[misc, assignment]


def _pad(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _tensorop_backend(major: int, minor: int) -> str:
    """Ampere 示例核 vs Blackwell GeForce 示例核（Jetson Thor SM110=cap 11.0、桌面 SM12x 等）。"""
    # Thor：`sm_110` → PyTorch 常见为 (11, 0)；与 GB10 等共用 `cute.blackwell_geforce` 的 Sm120GemmKernel。
    if major >= 11:
        if Sm120GemmKernel is None:
            raise RuntimeError(
                "当前 GPU capability >= 11（含 Jetson Thor SM110），需要 `cute.blackwell_geforce` 中的 "
                "Sm120GemmKernel，但导入失败。请升级 CUTLASS/CuTeDSL 或检查 CUTLASS_CUTEDSL_PATH。"
            )
        return "sm120"
    if major == 10:
        raise RuntimeError(
            f"当前 GPU capability {major}.{minor} 为数据中心 Blackwell SM10x（tcgen05 路径）；"
            "本脚本未封装 `DenseGemmKernel`。请使用 CUTLASS 树内 `cute/blackwell/kernel/dense_gemm` 示例。"
        )
    if major >= 8:
        return "ampere"
    raise RuntimeError(
        f"需要 SM80+ Tensor Core，或 SM11+ Blackwell GeForce（Jetson Thor 等），当前设备 "
        f"capability {major}.{minor}。"
    )


@cute.jit
def bmm_tensorop_ampere(
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


@cute.jit
def bmm_tensorop_sm120(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr[int],
    stream,
):
    """与 `cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py` 中 `run()` 一致：M×K×L / N×K×L / M×N×L。"""
    gemm_op = Sm120GemmKernel(cutlass.Float32, (128, 128, 64))
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, max_active_clusters, stream)


def _make_fake_gemm_tensors(batch: int, m: int, n: int, k: int):
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
    return fake_a, fake_b, fake_c


def _compile_bmm_ampere(m: int, n: int, k: int, batch: int = 1):
    fake_a, fake_b, fake_c = _make_fake_gemm_tensors(batch, m, n, k)
    return cute.compile(
        bmm_tensorop_ampere, fake_a, fake_b, fake_c, options="--enable-tvm-ffi"
    )


def _compile_bmm_sm120(m: int, n: int, k: int, batch: int = 1):
    fake_a, fake_b, fake_c = _make_fake_gemm_tensors(batch, m, n, k)
    hw = cutlass_utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    return cute.compile(
        bmm_tensorop_sm120,
        fake_a,
        fake_b,
        fake_c,
        max_active_clusters,
        stream,
    )


def _torch_to_cute_fp16(t: torch.Tensor) -> cute.Tensor:
    ct = from_dlpack(t.contiguous(), assumed_align=16)
    ct = ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))
    ct.element_type = cutlass.Float16
    return ct


def _invoke_gemm(
    backend: str,
    compiled,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> None:
    if backend == "ampere":
        compiled(a, b, c)
        return
    stream = cutlass_torch.default_stream()
    compiled(_torch_to_cute_fp16(a), _torch_to_cute_fp16(b), _torch_to_cute_fp16(c), stream)


def main() -> None:
    p = argparse.ArgumentParser(
        description="CuTeDSL TensorOp GEMM（Ampere / Jetson Thor SM110+ Sm120GemmKernel）"
    )
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
    backend = _tensorop_backend(major, minor)
    k_align = 64 if backend == "sm120" else 32

    m, n, k = args.m, args.n, args.k
    mp, np, kp = _pad(m, 128), _pad(n, 128), _pad(k, k_align)

    torch.manual_seed(0)
    a = torch.randn(1, mp, kp, dtype=torch.float16, device="cuda")
    b = torch.randn(1, kp, np, dtype=torch.float16, device="cuda")
    c = torch.zeros(1, mp, np, dtype=torch.float16, device="cuda")

    t0 = time.time()
    if backend == "sm120":
        compiled = _compile_bmm_sm120(mp, np, kp, 1)
    else:
        compiled = _compile_bmm_ampere(mp, np, kp, 1)
    t1 = time.time()

    _invoke_gemm(backend, compiled, a, b, c)
    torch.cuda.synchronize()
    t2 = time.time()

    a_sub = a[:, :m, :k]
    b_sub = b[:, :k, :n]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = torch.matmul(a_sub.float(), b_sub.float()).half()
    torch.testing.assert_close(c[:, :m, :n], ref, rtol=2e-3, atol=2e-3)
    label = (
        "Sm120GemmKernel (Blackwell GeForce, SM11 Thor / SM12x+)"
        if backend == "sm120"
        else "TensorOpGemm"
    )
    print(
        f"[OK] {label} 校验通过 (逻辑形状 M={m} N={n} K={k}, pad 至 {mp}×{np}×{kp}, "
        f"capability={major}.{minor})"
    )
    print(f"[INFO] compile: {(t1 - t0):.3f}s, 首次执行: {(t2 - t1):.3f}s")

    if args.bench:
        it = args.bench_iters
        warm = 5
        for _ in range(warm):
            _invoke_gemm(backend, compiled, a, b, c)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(it):
            _invoke_gemm(backend, compiled, a, b, c)
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
        print(f"[BENCH] CuTeDSL GEMM (pad): {t_cutlass * 1e6:.2f} us/iter")
        print(f"[BENCH] torch.matmul:       {t_torch * 1e6:.2f} us/iter")


if __name__ == "__main__":
    main()
