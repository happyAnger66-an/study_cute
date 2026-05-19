# `tma_v0.py` 代码精读

对 CUTLASS 教程 [`tma_v0.py`](../../cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py) 的完整分析：host 配置、mbarrier、`tma_partition`、TMA load/store，以及 CTA tile / 异步 / barrier 等常见问题。

- 索引与其他 TMA 笔记：[`README.md`](README.md)
- TMA 通用概念与 GEMM pipeline：[`docs/tma.md`](../docs/tma.md)

---

## 0. 这个文件在教什么？

**TMA V0**：用最小的 copy kernel 讲清三件事：

1. **TMA Load**：GMEM → SMEM（`CopyBulkTensorTileG2SOp`）
2. **TMA Store**：SMEM → GMEM（`CopyBulkTensorTileS2GOp`）
3. **`tma_partition`**：把 GMEM/SMEM 张量切成 TMA 硬件能 `copy` 的视图

默认配置：`tile_shape = (128, 128)`，`threads_per_cta = 32`，`cluster = (1,1)`，默认矩阵 `M=512, N=128`。

---

## 1. 代码结构总览

```text
Sm100SimpleCopyKernel
├── __init__          # tile、线程数、对齐
├── __call__ (host)   # smem_layout、TMA atom、grid launch
└── kernel (device)   # barrier、partition、TMA copy

run_tma_copy()        # PyTorch 张量、compile、benchmark、校验
```

### 执行流水线（device kernel）

```text
allocate SMEM + barrier
    ↓
mbarrier_init + expect_tx + fence + barrier()
    ↓
local_tile (GMEM) + smem_tensor
    ↓
group_modes + tma_partition (load / store)
    ↓
tAgA_cta / tBgB_cta = 本 CTA 的 GMEM tile
    ↓
TMA load (cute.copy G2S) + mbarrier_arrive + mbarrier_wait
    ↓
TMA store (cute.copy S2G)
```

---

## 2. 类配置 `__init__`（73–82）

```python
self.tile_shape = (128, 128)
self.cluster_shape_mn = (1, 1)
self.threads_per_cta = 32
self.buffer_align_bytes = 1024
```

| 字段 | 含义 |
|------|------|
| `tile_shape` | 每个 CTA 负责的逻辑块大小 (M, N) |
| `cluster_shape_mn` | CTA cluster 形状；v0 为 1×1（无 multicast） |
| `threads_per_cta` | block 大小；本例主要用于 barrier 同步，不参与逐元素搬运 |
| `buffer_align_bytes` | SMEM 缓冲区对齐（1024 字节） |

---

## 3. Host 侧 `__call__`（84–132）

### 3.1 `smem_layout`

```python
smem_layout = cute.make_layout(
    (self.tile_m, self.tile_n), stride=(self.tile_n, 1)
)
```

每个 CTA 一块 **128×128** SMEM，行主序 stride `(128, 1)`（N 为 leading dim 时对应 `(tile_n, 1)`）。

### 3.2 `SharedStorage`：SMEM 里放什么

```python
@cute.struct
class SharedStorage:
    barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]   # mbarrier
    smem_data: cute.struct.Align[
        cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)],
        self.buffer_align_bytes,
    ]
```

| 成员 | 作用 |
|------|------|
| `barrier_storage` | 1 个 `Int64`，作 `mbarrier` 对象 |
| `smem_data` | 实际数据缓冲区，大小由 `cosize` 决定 |

```python
self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout)
```

一次 TMA load 的字节数，用于 `mbarrier_expect_tx`。

### 3.3 `cosize` vs `product_each(shape)`

两处都依赖 `smem_layout`，**职责不同**：

```python
# SMEM 分配：元素槽总数（标量）
MemRange[self.dtype, cute.cosize(smem_layout)]

# TMA 配置：每 CTA 逻辑 tile 尺寸（元组）
cta_tiler = cute.product_each(smem_layout.shape)
```

| | `cosize(smem_layout)` | `product_each(smem_layout.shape)` |
|---|----------------------|----------------------------------|
| 回答问题 | SMEM **开多少元素**？ | 每 CTA **逻辑 (M,N)**？ |
| 返回值 | 标量 `16384` | 元组 `(128, 128)` |
| 输入 | layout + stride | 仅 `shape` |
| stride/swizzle | **考虑**（codomain） | **不考虑** |
| 用途 | `MemRange` 分配 | `make_tiled_tma_atom(..., cta_tiler)` |

本例：`cosize → 16384`，`product_each → (128, 128)`。

swizzle 时可能分叉：逻辑 tile 仍 `(128,128)`，但 `cosize` 可能更大（物理 footprint）。

另有 `size_in_bytes` → `num_tma_load_bytes`，给 barrier 的 `expect_tx` 用（与 `cosize` 元素数×元素大小一致）。

### 3.4 `make_tiled_tma_atom`

```python
tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), src, smem_layout, cta_tiler
)
tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(), dst, smem_layout, cta_tiler
)
```

| 返回值 | 含义 |
|--------|------|
| `tma_atom_*` | TMA CopyAtom（描述符 + 指令类型） |
| `tma_tensor_*` | GMEM 张量的 TMA 坐标映射（供 kernel 里 `local_tile` / `partition`） |

在 host 上生成符合硬件要求的 descriptor，kernel 只负责发射。

### 3.5 Launch

```python
grid_shape = cute.ceil_div((*src.layout.shape, 1), self.tile_shape)
# 默认 M=512,N=128 → grid (4, 1)

self.kernel(...).launch(
    grid=grid_shape,
    block=(32, 1, 1),
    cluster=(1, 1, 1),
)
```

**并行在 grid**：4 个 CTA 各处理一块 128×128，合起来覆盖整矩阵。

---

## 4. 一个 CTA 只处理 (128,128)，和 32 线程什么关系？

128×128 = **16,384 元素**，block 只有 **32 线程**——这不是「32 线程分工搬完 16K 元素」。

| 工作 | 执行者 |
|------|--------|
| GMEM → SMEM | **TMA 硬件** |
| barrier init / arrive | **`elect_one()` 的 1 线程** |
| 等 TMA | 全员 `mbarrier_wait` |
| SMEM → GMEM | **TMA 硬件** |

```text
传统:  N 线程 × LDG  →  tile 大小 ∝ 线程分工
TMA:   一条 bulk 描述符 → 硬件搬整块 → 线程数几乎不决定 tile
```

tile 大小由 **SMEM 容量、TMA 描述符、对齐** 决定，不是 `tile_elements = num_threads`。

---

## 5. Kernel：`mbarrier` 初始化（149–163）

```python
barrier_ptr = storage.barrier_storage.data_ptr()

with cute.arch.elect_one():
    cute.arch.mbarrier_init(barrier_ptr, 1)
    cute.arch.mbarrier_expect_tx(barrier_ptr, self.num_tma_load_bytes)

cute.arch.mbarrier_init_fence()
cute.arch.barrier()
```

### 作用（一句话）

**在发 TMA load 之前**，配置 SMEM 里的 `mbarrier`：等 **1 次软件 arrive** + **TMA 搬完 `num_tma_load_bytes` 字节**。

### 逐行

| 代码 | 含义 |
|------|------|
| `barrier_ptr` | SMEM 中 mbarrier 地址 |
| `elect_one` + `mbarrier_init(..., 1)` | **仅 1 线程** init；期望 **1 次** `mbarrier_arrive` |
| `mbarrier_expect_tx(..., N)` | 登记：TMA 硬件应完成 **N 字节**异步事务 |
| `mbarrier_init_fence` | init / expect_tx 对全 CTA 可见 |
| `barrier()` | `__syncthreads()`，全员汇合后再发 TMA |

### 为何 `elect_one` 而非 `if tid == 0`？

- PTX 要求 `mbarrier.init` 等由 **单线程** 执行
- `elect_one` 提供正确的 block 内同步与内存序，避免 race

### 与后面 load 的握手

```text
【149-163】init(期望 1 arrive) + expect_tx(期望 N 字节) + fence + barrier

【298-311】
  cute.copy(..., tma_bar_ptr=barrier_ptr)   # TMA 绑定 barrier
  elect_one: mbarrier_arrive()              # 软件 +1 arrive
  全员 mbarrier_wait(phase=0)               # 两条件都满足才返回

【313-318】TMA store（SMEM 已就绪）
```

`mbarrier_wait` 返回条件：

| 条件 | 满足方 |
|------|--------|
| arrive 计数 = 1 | `mbarrier_arrive`（elect_one） |
| 事务字节 = `num_tma_load_bytes` | TMA 硬件完成（`expect_tx`） |

**这段是「配置同步器」，不是「等待搬运」本身**；等待在 311 行的 `mbarrier_wait`。

### `mbarrier` vs `barrier()`

| | `barrier()` | `mbarrier` |
|---|-------------|------------|
| 等待 | 线程到齐 | **异步事务字节** + arrive |
| 用途 | init 后全员汇合 | **TMA 搬完再读 SMEM** |

---

## 6. Kernel：`local_tile` 与 SMEM 视图（165–173）

```python
gSrc_tiled = cute.local_tile(tma_tensor_src, (128, 128), (None, None))
gDst_tiled = cute.local_tile(tma_tensor_dst, (128, 128), (None, None))
smem_tensor = storage.smem_data.get_tensor(smem_layout)
```

### `local_tile`

把 GMEM 逻辑 shape 从 `(M, N)` 变为 **`((128,128), M/128, N/128)`**。

- `coord=(None, None)`：**不取单块**，保留全部 grid 维
- 默认 `M=512, N=128` → `gSrc_tiled` shape **`(128, 128, 4, 1)`**

| mode | 含义 |
|------|------|
| 0, 1 | tile 内 M、N |
| 2, 3 | grid：`bidx`、`bidy` |

### `smem_tensor`

本 CTA 的 128×128 SMEM 视图（load 目标、store 源）。

---

## 7. Kernel：`group_modes` + `tma_partition`（269–286）

TMA 要求：**mode 0 = 整个 TMA atom**（一次 bulk copy 的区域）。

### `group_modes(tensor, 0, 2)`

合并前 2 个 mode（tile 的 M、N）：

```text
(128, 128, 4, 1)  →  ((128, 128), 4, 1)    # GMEM
(128, 128)        →  ((128, 128),)         # SMEM
```

### `tma_partition(atom, cta_coord, cta_layout, smem, gmem)`

```python
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_src, 0, cute.make_layout(1),
    cute.group_modes(smem_tensor, 0, 2),
    cute.group_modes(gSrc_tiled, 0, 2),
)
_, tBgB = cpasync.tma_partition(
    tma_atom_dst, 0, cute.make_layout(1),
    cute.group_modes(smem_tensor, 0, 2),
    cute.group_modes(gDst_tiled, 0, 2),
)
```

| 参数 | v0 取值 | 含义 |
|------|---------|------|
| `cta_coord` | `0` | 1×1 cluster |
| `cta_layout` | `make_layout(1)` | cluster 仅 1 CTA |
| `tAsA` | SMEM TMA 视图 | mode0 = TMA 内部 layout（可 swizzle） |
| `tAgA` | GMEM TMA 视图 | `((TMA_Layout), 4, 1)` |
| `_` / `tBgB` | store 的 GMEM 视图 | load 的 `tAsA` 可复用 |

按 TMA atom 的硬件 layout 重排张量，使 `cute.copy` 可直接发射。

### 文件内注释示意图（另一组 tile 尺寸）

源码 175–263 行用 **128×64、grid 4×2** 做教学示意；**本文件实际**为 **128×128、默认 grid 4×1**。理解步骤时以 `self.tile_m/n` 为准。

```text
GMEM (512×128)                    SMEM (128×128)
┌───┬───┬───┬───┐                 ┌─────────┐
│0,0│1,0│2,0│3,0│  4×1 tiles      │ tAsA    │
└───┴───┴───┴───┘                 └─────────┘
     ↑ bidx=0..3                        ↑
     │ tAgA[None,bidx,bidy]             │ TMA load / store
     └──────────────────────────────────┘
```

---

## 8. Kernel：选本 CTA 的 tile（288–296）

```python
tAgA_cta = tAgA[(None, bidx, bidy)]
tBgB_cta = tBgB[(None, bidx, bidy)]
```

| 索引 | 含义 |
|------|------|
| `None` | 保留 mode 0（整块 TMA atom） |
| `bidx` | M 方向第几块 |
| `bidy` | N 方向第几块 |

```text
tAgA:     ((TMA_Layout), 4, 1)
tAgA_cta: ((TMA_Layout),)     # 本 block 唯一一块
```

---

## 9. Kernel：TMA Load / Store（298–318）

### Load（G2S）

```python
cute.copy(tma_atom_src, tAgA_cta, tAsA, tma_bar_ptr=barrier_ptr)
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(barrier_ptr)
cute.arch.mbarrier_wait(barrier_ptr, 0)
```

| 步骤 | 作用 |
|------|------|
| `cute.copy` + `tma_bar_ptr` | 异步 TMA load，绑定 mbarrier |
| `mbarrier_arrive` | 满足 init 时的 arrive 计数 |
| `mbarrier_wait` | 等 TMA 字节搬完 |

### Store（S2G）

```python
cute.copy(tma_atom_dst, tAsA, tBgB_cta)
```

- 无 barrier：load 已 wait，SMEM 就绪
- 源 `tAsA`（SMEM），目的 `tBgB_cta`（本 CTA 的 GMEM tile）

整 kernel 效果：每个 CTA 把 `src` 的一块 128×128 拷到 SMEM 再写到 `dst` 对应位置。

---

## 10. 异步与 `mbarrier_wait`

### TMA「异步」指什么？

> 发射 `cute.copy` 后，线程 **不必 LDG 逐元素搬**；**TMA 引擎在后台**完成。

≠ 「发完指令后本 kernel 里线程从不等待」。

### v0 里为何「看不出重叠」？

load 与 store 之间 **无 MMA、无多 stage**，全员 `mbarrier_wait` 后立刻 store——**故意做成最小同步闭环**。

### 时间线图 A：`tma_v0`（无重叠）

```text
时间 ──────────────────────────────────────────────────────►

TMA 引擎:  [======== GMEM→SMEM 128×128 ========]
线程(32):  发copy │········ 全员 mbarrier_wait ········│ 发 store
           ↑ elect_one arrive              ↑ 全部阻塞

MMA/计算:  (无)
```

### 时间线图 B：GEMM 多 stage（有重叠）

```text
时间 ──────────────────────────────────────────────────────►

Stage 0:  [==== TMA load ====][======== MMA ==========]
Stage 1:        [==== TMA load ====][======== MMA ========]
                 ↑ TMA 与 MMA 重叠 ↑
```

详见 [`docs/tma.md`](../docs/tma.md) 与 [`gemm/fp16_gemm_0.py`](../gemm/fp16_gemm_0.py)。

### 时间线图 C：warp 分工

```text
Warp 0 (producer):  [发TMA][发TMA]...
Warp 1..N (MMA):         [==== MMA ====]...
TMA 引擎:           [==搬==][==搬==]...
```

### 对比

| 方面 | `tma_v0` | GEMM |
|------|----------|------|
| load 后立刻 store | 是 | 否 |
| 多 stage SMEM | 无 | 2~4 |
| 能否看出 overlap | **否** | **是** |
| 仍可能更快 | 专用 TMA 引擎、少占寄存器 | + 与 MMA 并行 |

---

## 11. Host 驱动 `run_tma_copy`（321–387）

| 步骤 | 代码要点 |
|------|----------|
| 建张量 | `torch` fp16，`a` 随机，`b` 零 |
| CuTe 包装 | `from_dlpack`，`mark_layout_dynamic`，N 维 `% 16` |
| 编译 | `cute.compile(copy_kernel, a_cute, b_cute)` |
| 运行 | warmup + CUDA Event 计时 |
| 校验 | `torch.allclose(a, b)` |

吞吐按 **读+写** `2×M×N×elem_size` 计算。

---

## 12. 概念速查

| 概念 | 含义 |
|------|------|
| `cta_tiler` | 每 CTA 一次 TMA 的逻辑 (M,N) |
| `cosize` | SMEM 元素槽总数（物理 footprint） |
| `num_tma_load_bytes` | 一次 TMA load 字节数 → `expect_tx` |
| `tma_tensor` | GMEM 的 TMA 坐标映射 |
| `local_tile` | `(M,N)` → `((tileM,tileN), grid...)` |
| `group_modes` | 合并 tile 两维为 mode 0（atom） |
| `tma_partition` | 生成 TMA 可用的 GMEM/SMEM 视图 |
| `[None,bidx,bidy]` | 选本 CTA 的 GMEM 块 |
| TMA 异步 | 硬件后台搬，不靠线程 LDG |
| `mbarrier_wait` | 等 SMEM 就绪，非否定异步 |

---

## 13. 一句话总结

**`tma_v0` 在 host 上配好 SMEM layout、TMA descriptor 与 grid；在 device 上 init mbarrier → `local_tile` + `tma_partition` → 每 CTA 用 TMA 搬一块 128×128（与 32 线程数无关）→ barrier 等 load 完成 → TMA 写回。异步重叠不在 v0 体现，在 GEMM 多 stage pipeline 中才显现。**

---

## 延伸阅读

- [`README.md`](README.md) — TMA 学习索引
- [`docs/tma.md`](../docs/tma.md) — TMA 概念、资源占用、4-stage pipeline
- CUTLASS `tma_v1` … `tma_v4` — cluster、multicast 等
