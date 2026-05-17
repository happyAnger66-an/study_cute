# fp16_gemm_0 中的 Shared Memory（SMEM）

以 `gemm/fp16_gemm_0.py` 为例，说明 kernel 内 SMEM 分配、4-stage 流水线与 occupancy 的关系。

## Kernel 内的 SMEM 布局

```python
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
sA = smem.allocate_tensor(..., layout=a_smem_layout.outer, swizzle=a_smem_layout.inner)
sB = smem.allocate_tensor(..., layout=b_smem_layout.outer, swizzle=b_smem_layout.inner)
```

`SmemAllocator` 在 CTA 内做 **顺序 bump 分配**：从动态 SMEM 基址（1024 字节对齐）向后划区，JIT 时自动累计总用量，launch 时一般无需手写 `shared_mem_bytes`。

### 物理排列（概念）

```
[ SharedStorage: mbarriers + tmem buf ]
[ sA: ab_stages 段 FP16 A tile buffer   ]
[ sB: ab_stages 段 FP16 B tile buffer   ]
```

| 对象 | 内容 | 后续用途 |
|------|------|----------|
| `storage` | `ab_mbar_ptr`、`acc_mbar_ptr`、`tmem_holding_buf` | TMA/UMMA 流水线 barrier、TMEM 分配握手 |
| `sA` / `sB` | 多 stage 的 FP16 tile buffer | TMA 写入、UMMA `make_fragment_A/B` |

`SharedStorage` 不存矩阵数据，只存同步与 TMEM 元数据。

### allocate_tensor 参数

| 参数 | 含义 |
|------|------|
| `layout=...outer` | `ComposedLayout` 外层：tile 形状 + **stage 维** |
| `swizzle=...inner` | 内层 swizzle，缓解 bank conflict，匹配 TMA/UMMA |
| `byte_alignment=128` | TMA bulk copy 与 SMEM 访问对齐要求 |

Host 侧用 `sm100_utils.make_smem_layout_a/b(..., ab_stages)` 规划 layout，再传入 kernel 实例化到 SMEM。

---

## 4 stages 与 SMEM 大小

`ab_stages = 4` 时，`make_smem_layout_a/b` 会在 tile shape 后 **追加长度为 4 的 stage 维**：

- **完整 layout**（`a_smem_layout`）→ kernel 里 `sA` 的总分配（4 个 stage 槽位）
- **单 stage**（`cute.select(layout, mode=[0,1,2])`）→ TMA atom、`num_tma_copy_bytes` 等单次搬运大小

因此：

```
size(sA) ≈ 4 × size(单 stage A tile)
size(sB) ≈ 4 × size(单 stage B tile)
```

要点：

- **4 倍只针对 A 或 B 各自**；AB 合计约为 `4×(A_tile + B_tile)`，不是把 `[sA|sB]` 再整体乘 4。
- 4 stages 是 **流水线环形缓冲**（TMA 写 stage i、MMA 读 stage j），用空间换 TMA/UMMA 重叠，不是同一份数据的 4 份备份。

本例 `mma_tiler_mnk = (128, 256, 64)`、FP16、K-major 时，单 stage 量级约为：

- A：\(128 \times 64 \times 2\) B ≈ **16 KiB**
- B：\(256 \times 64 \times 2\) B ≈ **32 KiB**
- 4 stage 的 AB buffer 合计约 **192 KiB**（再加对齐与 `SharedStorage`，每 CTA 常约 **~200 KiB**）

---

## 每个 CTA 一份 vs occupancy

### 每个 CTA 独立一份 SMEM

SMEM 是 **block 级私有**。Grid 可 launch 大量 CTA，但 **每个正在执行的 CTA 都占有同样大的一块 SMEM**，CTA 之间不共享 `sA`/`sB`。

### 影响什么

| 概念 | 是否受 per-CTA SMEM 影响 |
|------|-------------------------|
| Grid 中 CTA **总数** | 基本不影响（由问题规模与 `mma_tiler` 决定） |
| 每 SM **同时驻留**的 CTA 数 | **会受影响**（occupancy） |
| Kernel 能否 launch | 单 CTA SMEM 超过架构 per-block 上限会失败 |

每个 SM 的 shared memory 池有限。`sm_100` 在 CuTeDSL 中约 **227 KiB/SM**（可配置 carveout）。若每 CTA ~200 KiB，SMEM 维度上往往 **每 SM 只能挂约 1 个 block**；还需与线程数（本例 128）、寄存器、**TMEM**（如 `tmem.allocate(512 columns)`）一并算 occupancy。

### 与「最大 CTA 数」的区分

- **Launch 的 CTA 总数**：可以很大；超出 SM 容量的 CTA 会排队。
- **每 SM 同时活跃的 CTA**：才主要由 per-CTA SMEM 等资源限制。

### 为何仍用 4 stage

| 更少 stage | 更多 stage（如 4） |
|------------|-------------------|
| 每 CTA SMEM 更小，每 SM 可能多挂 CTA | 每 CTA SMEM 更大，并发 CTA 更少 |
| TMA/UMMA 更难重叠 | 单 CTA 流水线更满，吞吐常更好 |

GEMM 常刻意让 SMEM 接近吃满，接受「每 SM 1 个 heavy CTA」以换取单 CTA 性能。

---

## 相关代码位置

| 内容 | 文件 |
|------|------|
| `ab_stages`、`SharedStorage`、kernel 内分配 | `gemm/fp16_gemm_0.py` |
| `make_smem_layout_a/b` | CUTLASS `blackwell_helpers.py` |
| `SmemAllocator` | CUTLASS `cutlass/utils/smem_allocator.py` |
