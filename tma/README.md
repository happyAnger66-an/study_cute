# TMA 学习笔记

| 文档 | 内容 |
|------|------|
| **[`tma_v0.md`](tma_v0.md)** | `tma_v0.py` **完整逐段分析**（host、mbarrier、partition、load/store、时间线图） |
| **[`barrier.md`](barrier.md)** | **Named Barrier vs mbarrier**（区别、作用范围、使用场景、`tma_v1` 对照） |
| [`docs/tma.md`](../docs/tma.md) | TMA 通用概念 + GEMM 多 stage pipeline |

下文为速查摘要；细节以 [`tma_v0.md`](tma_v0.md) 为准。

---

## 参考代码位置

| 文件 | 说明 |
|------|------|
| CUTLASS `tma_v0.py` | 最小 TMA copy kernel，讲 `tma_partition` |
| 本仓库 `docs/tma.md` | TMA 概念 + `fp16_gemm_0` 多 stage pipeline |
| 本仓库 `gemm/fp16_gemm_0.py` | 带 MMA 重叠的生产级用法 |

---

## 1. `product_each(shape)` vs `cosize(layout)`

`tma_v0` host 侧两处都用到 `smem_layout`，职责不同：

```python
# SMEM 分配：要开多少个元素槽（标量）
smem_data: cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)]

# TMA 配置：每个 CTA 的逻辑 tile 是几×几（元组）
cta_tiler = cute.product_each(smem_layout.shape)
```

| | `cosize(smem_layout)` | `product_each(smem_layout.shape)` |
|---|----------------------|----------------------------------|
| **回答的问题** | SMEM **一共要预留多少元素**？ | 每个 CTA 的 tile **逻辑尺寸 (M, N)**？ |
| **返回值** | 标量，如 `16384` | 元组，如 `(128, 128)` |
| **输入** | 整个 layout（含 stride） | 只看 `shape` |
| **是否考虑 stride / swizzle** | 是（codomain：最大 offset + 1） | 否（各 mode 内子维相乘） |
| **用途** | `MemRange` 分配 raw SMEM | `make_tiled_tma_atom(..., cta_tiler)` |

### 本例数值（`tile_shape = (128, 128)`，`stride = (128, 1)`）

```text
cosize(smem_layout)              →  16384        # 128 × 128 个元素槽
product_each(smem_layout.shape)  →  (128, 128)  # TMA 逻辑 tile
```

### swizzle 时二者会「分叉」

```text
逻辑 tile（TMA 关心）     product_each  →  (128, 128)
物理 SMEM footprint       cosize        →  可能 > 128×128（swizzle 空洞/填充）
```

**一句话**：`cosize` = 物理上要开多少 SMEM；`product_each(shape)` = TMA 一次搬多大逻辑块。

---

## 2. 一个 CTA 只处理 (128, 128)，线程不是很多吗？

### 配置

```python
self.tile_shape = (128, 128)      # 每个 CTA 负责的 tile
self.threads_per_cta = 32         # block 只有 32 线程
```

128×128 = **16,384 元素**，只有 **32 线程**——若按「一线程搬若干元素」理解，确实对不上。

### 实际分工

| 工作 | 谁在做 |
|------|--------|
| GMEM → SMEM 搬 128×128 | **TMA 硬件**（`cute.copy` + TMA atom） |
| barrier 初始化 / `mbarrier_arrive` | **`elect_one()` 选出的 1 个线程** |
| 等 TMA 完成 | 所有线程 `mbarrier_wait` |
| SMEM → GMEM |  again **TMA** |

```text
传统 copy:  N 个线程 × 每人 LDG 若干次  →  tile 大小与线程分工绑定
TMA copy:   发 TMA 描述符  →  硬件 bulk 搬整块  →  线程数几乎不决定 tile 大小
```

### 并行来自 Grid，不是单 CTA 内线程分工

默认 `M=512, N=128`：

```text
grid = (512/128, 128/128) = (4, 1)   →  4 个 CTA，各管一块 128×128

  CTA(0,0)     CTA(1,0)     CTA(2,0)     CTA(3,0)
 [128×128]    [128×128]    [128×128]    [128×128]
     ↓ TMA        ↓ TMA        ↓ TMA        ↓ TMA
  各自 SMEM    各自 SMEM    各自 SMEM    各自 SMEM
```

**总吞吐 ∝ 同时运行的 CTA 数**，不是单 block 里 32 线程各搬一点。

### 为何教程用 128×128 + 32 线程？

- `tma_v0` 目标是讲 **`tma_partition`**，不是高性能 copy
- 线程主要做同步，真正搬运者是 TMA
- 真实 GEMM：TMA 搬 tile → **大量线程做 MMA**；本例只有 load/store，没有中间计算

---

## 3. 全员 `mbarrier_wait`：异步好处去哪了？

### TMA「异步」指什么？

> 发射 `cute.copy` 之后，**不需要这些线程自己用 LDG 逐元素搬**；搬运由 **TMA 引擎在后台**完成。

不等于：「发完指令后本 kernel 里所有线程立刻去干别的且从不等待」。

### `tma_v0` 里的实际流程

```text
cute.copy(TMA load)          # GMEM tile → SMEM
elect_one: mbarrier_arrive   # 上报「TMA 已发射」
全员 mbarrier_wait           # 等字节真正写完
cute.copy(TMA store)         # 依赖 SMEM 已就绪
```

**必须 wait**：store 前 SMEM 数据必须完整；且 load 与 store 之间 **没有任何计算** 可重叠。

因此在 **v0 这个文件里**，确实 **几乎看不出异步重叠收益**——这是故意的最小闭环，不是 TMA 的极限。

### 时间线图 A：`tma_v0`（无重叠）

```text
时间 ──────────────────────────────────────────────────────►

TMA 引擎:  [======== GMEM→SMEM 128×128 ========]
线程(32):  发copy │········ 全员 mbarrier_wait ········│ 发 store
           ↑ elect_one arrive              ↑ 全部阻塞等数据

MMA/计算:  (无)
```

特点：等待期间线程 **不参与搬运**，也 **没有别的活**；异步只体现在「搬运不占 LDG 带宽、不占大量寄存器」。

### 时间线图 B：GEMM 多 stage pipeline（有重叠）

见 [`docs/tma.md`](../docs/tma.md)。核心模式：

```text
主循环每个 K-tile:
  1. 发 TMA → 下一 tile 搬进 stage (k+1)     ← TMA 后台跑
  2. MMA 消费 stage k 已就绪的数据             ← 与 TMA 重叠
  3. mbarrier：TMA 写完才能读；MMA 读完才能复用 stage
```

```text
时间 ──────────────────────────────────────────────────────►

Stage 0 SMEM:  [==== TMA load A0 ====][======== MMA on A0 ========]
Stage 1 SMEM:        [==== TMA load A1 ====][======== MMA on A1 ========]
Stage 2 SMEM:              [==== TMA load A2 ====][======== MMA on A2 ========]
                             ↑ 重叠区间 ↑
```

`ab_stages = 4` 时 SMEM 里轮转 4 份 buffer，把「等 TMA」藏进流水线。

### 时间线图 C：warp 分工（`fp16_gemm_0` 思路）

```text
时间 ──────────────────────────────────────────────────────►

Warp 0 (producer):  [发TMA][发TMA][发TMA] ...     mbarrier 握手
Warp 1..N (MMA):         [==== MMA ====][==== MMA ====] ...
TMA 引擎:           [==搬==][==搬==][==搬==] ...
```

- 不必全员参与发 TMA
- consumer warp 在 **需要读 SMEM 前** 才 `mbarrier_wait`
- TMA 与 **tcgen05 UMMA** 是不同硬件通路，靠 pipeline + mbarrier 协调

### 对比表

| 方面 | `tma_v0` | GEMM (`fp16_gemm_0`) |
|------|----------|----------------------|
| load 后立刻 store | 是 | 否，中间有大段 MMA |
| 多 stage SMEM | 无 | 通常 2~4 stage |
| 线程分工 | 32 线程主要同步 | producer / consumer warp |
| 能否看出 overlap | **不能** | **能** |
| TMA 仍可能更快的原因 | 专用引擎、规则 tile 访问、少占寄存器 | 同上 + 与 MMA 并行 |

### 即便没有 overlap，TMA 也可能更快

- 专用 **TMA 引擎**，不占大量 LDG 发射
- **按 tile** 的规则访问，对 GMEM/SMEM 更友好
- 数据进 SMEM，不经过每线程寄存器灌数

**快 ≠ 一定有异步重叠**；重叠是 pipeline 再叠一层。

---

## 4. 概念速查

| 概念 | 含义 |
|------|------|
| `cta_tiler` | 每个 CTA 一次 TMA 搬的逻辑 (M, N) |
| `cosize` | layout 在 SMEM 上占用的元素槽总数 |
| TMA 异步 | 搬运在 TMA 硬件后台，不靠线程 LDG |
| `mbarrier_wait` | 等「SMEM 数据就绪」，不是否定异步 |
| v0 全员 wait | 无计算、无多 stage → 故意同步外观 |
| 异步收益 | 多 stage：TMA 搬下一块时 MMA 算上一块 |

---

## 5. 一句话总结

**`tma_v0` 用最小 kernel 讲清 TMA + partition + barrier；`(128,128)` 是每 CTA 的逻辑 tile，由 TMA 整块搬运，与 32 线程数无关；全员 `mbarrier_wait` 在 v0 里看不出重叠，真正的异步收益在 GEMM 多 stage pipeline 的时间线里才显现。**

---

## 延伸阅读

- [`docs/tma.md`](../docs/tma.md) — TMA 概念、4-stage pipeline、资源占用
- CUTLASS [`tutorial_tma/tma_v0.py`](../../cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py)
- CUTLASS 后续：`tma_v1` … `tma_v4`（cluster、multicast 等）
