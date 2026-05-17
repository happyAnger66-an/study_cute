# TMA（Tensor Memory Accelerator）

本文结合 [`gemm/fp16_gemm_0.py`](../gemm/fp16_gemm_0.py) 说明 TMA 是什么、在 GEMM 里怎么用，以及指令发射占用的硬件资源。

---

## TMA 是什么

**TMA（Tensor Memory Accelerator）** 是 NVIDIA 从 Hopper（SM90）起在 GPU 上提供的 **专用硬件单元**，用来把 **多维张量的一块 tile** 在 **不同地址空间之间异步搬运**，最常见是：

```text
GMEM（全局内存）  ──TMA──►  SMEM（共享内存）
SMEM             ──TMA──►  GMEM（写回，视架构/指令而定）
```

在 `fp16_gemm_0` 里对应：

```python
op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(...)
# kernel 内:
cute.copy(tma_atom_a, tAgA[...], tAsA[...], tma_bar_ptr=...)
```

它不是「每个线程各 load 一个数」，而是：**用一张描述符（descriptor）说明「源张量哪块、目标 SMEM 哪块、layout 如何」**，硬件按 **tile** 做 **bulk 异步拷贝**。

---

## 和「普通 CUDA load」的差别

| | 普通全局 load（线程 load） | TMA |
|---|---------------------------|-----|
| 谁搬数据 | 大量线程用 LDG，经 L1/L2 | **TMA 引擎**按描述符搬 |
| 编程模型 | 每线程算地址、寄存器接数据 | **描述符 + 少量发射线程** |
| 与计算重叠 | 要自己管 async copy / pipeline | 原生 **异步**，用 **mbarrier** 同步 |
| 适合 | 不规则、小规模 | **大块、规则 layout 的 tile**（GEMM） |

CuTe 里 `make_tiled_tma_atom_A` / `make_tiled_tma_atom_B` 在 host 上 **生成符合硬件要求的描述符和 GMEM 坐标映射**。

---

## 在本仓库 GEMM 示例里的数据流

```text
主循环每个 K-tile:
  1. warp0: 发 TMA，把 GMEM 里 A/B 的 tile 搬进 sA/sB 的某个 stage
  2. TMA 在后台跑（异步）
  3. MMA 消费「上一 stage 已满」的 sA/sB
  4. mbarrier：TMA 写完 → MMA 才能读；MMA 读完 → stage 才能给下一次 TMA
```

相关 host 代码链：

```text
make_smem_layout_a/b(ab_stages=4)     →  SMEM 含 4 个 stage
select(layout, mode=[0,1,2])        →  单 stage layout（给 TMA 描述符）
make_tiled_tma_atom_A/B             →  CopyAtom + GMEM 的 TMA 张量
kernel: prefetch_descriptor → tma_partition → cute.copy
```

`prefetch_descriptor(tma_atom_a)`：预取 TMA 描述符，减少首次发射延迟。

---

## 4 段 pipeline 与 TMA 的关系

`ab_stages = 4` 表示 **sA/sB 在 SMEM 里各 4 份 K-tile buffer**（生产者–消费者流水线）：

```text
Stage 0 │ Stage 1 │ Stage 2 │ Stage 3
   ↑         ↑         ↑         ↑
  TMA 写入  /  MMA 读取  轮转复用
```

| 概念 | 说明 |
|------|------|
| **4 段** | SMEM layout 多一维；不是 TMA 硬件固定为 4 |
| **单 stage** | `select(mode=[0,1,2])` 给 **一次** TMA 的目标 layout |
| **为何常选 4** | SMEM 占用 vs 掩盖 TMA/MMA 延迟的经验默认；可改，需 benchmark |

`prefetch_stages = ab_stages - 2`：K 循环里最多提前发 2 次 TMA，与多段 buffer 配合。

---

## 指令发射占用哪些资源

分 **「谁发指令」** 和 **「谁干重活」** 两部分。

### 发射侧（warp / 线程）

- 通常 **极少数线程** 发 TMA（`fp16_gemm_0` 里 **warp 0** 在 `if warp_idx == 0:` 中 `cute.copy`）。
- 发射占用：
  - 少量 **指令发射槽**
  - 少量 **标量/地址寄存器**（描述符指针、mbarrier 指针、坐标）
- **不会** 让整块 CTA 的线程都用寄存器接一整片 tile。

### 执行侧（真正搬数据）

| 资源 | 是否占用 | 说明 |
|------|----------|------|
| **CUDA Core（FP 算力）** | 基本不用于搬数 | 矩阵乘用 MMA；TMA 不走算术管道搬 tile |
| **TMA / Async Copy 引擎** | **是** | 专用逻辑，与 core 并行 |
| **HBM / L2 带宽** | **是** | 从全局内存读 A/B |
| **SMEM 带宽 + 容量** | **是** | 写入 sA/sB；多 stage 成倍占用 |
| **mbarrier** | **是** | `tma_bar_ptr`、`PipelineTmaUmma` |
| **TMA 描述符存储** | **是** | constant / 专用路径；可 prefetch |
| **寄存器文件（数据）** | **很少** | 数据进 SMEM，非逐元素 load 灌寄存器 |

### 与 MMA 的重叠（Blackwell）

```text
时间轴 ─────────────────────────────────────►

TMA:  ████████░░░░████████░░░░   （异步 GMEM→SMEM）
MMA:  ░░░░████████░░░░████████   （SMEM→TMEM，tcgen05）
```

TMA 与 **tcgen05 UMMA** 是 **不同硬件通路**，通过 **pipeline + mbarrier** 协调，而不是用 CUDA Core 做 bulk load。

---

## 相关 API 速查（`fp16_gemm_0`）

| API | 作用 |
|-----|------|
| `CopyBulkTensorTileG2SOp` | 选定 GMEM→SMEM 的 bulk tensor 拷贝指令类 |
| `make_tiled_tma_atom_A/B` | 生成 `CopyAtom` + GMEM 的 `tma_tensor` |
| `cpasync.prefetch_descriptor` | 预取描述符 |
| `cpasync.tma_partition` | 对齐 GMEM 片、SMEM stage、MMA 视图 |
| `cute.copy(tma_atom, ...)` | 发射一次异步 TMA |
| `PipelineTmaUmma` | TMA（producer）与 MMA（consumer）握手 |

---

## 一句话

**TMA 是专用异步张量搬运引擎，用描述符把 GMEM 的 tile bulk 拷到 SMEM；发射只占很少线程和标量资源，真正占用的是 TMA 引擎、HBM/L2→SMEM 带宽、mbarrier 和多段 SMEM buffer，几乎不占 CUDA Core 算力和大块寄存器来搬数据。**

---

## 延伸阅读

- [NVIDIA PTX — bulk tensor async copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)（`cp.async.bulk.tensor` 等）
- [`gemm/fp16_gemm_0.py`](../gemm/fp16_gemm_0.py)
- CUTLASS 教程：[`examples/.../fp16_gemm_0.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py)
