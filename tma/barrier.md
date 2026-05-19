# Named Barrier 与 mbarrier

本文说明 GPU 上两类同步原语的**区别、作用范围、使用场景**，并对照本仓库 [`tma_v0.py`](../../cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py)（仅 mbarrier）与 [`tma_v1.py`](tma_v1.py)（两者配合）。

- TMA 与 pipeline 总览：[`docs/tma.md`](../docs/tma.md)
- `tma_v0` 精读：[`tma_v0.md`](tma_v0.md)

---

## 一句话区分

| | **Named Barrier** | **mbarrier** |
|---|-------------------|--------------|
| 本质 | CTA 里一批线程的 **「人到齐」** 同步（PTX `bar.sync`） | SMEM 里的 **异步屏障**：等 **线程 arrive** + 可选等 **异步事务字节完成**（TMA 等） |
| 典型问题 | 「这 128 个线程是不是都执行到这里了？」 | 「TMA 把 N 字节搬进 SMEM 了吗？生产者可以走了吗？」 |

---

## 1. 硬件与 API 层

```text
Named Barrier (bar.sync / bar.arrive)
  - 固定 barrier_id：0–15（每 CTA）
  - 创建时声明：num_threads（参与这次同步的线程数）
  - 典型 API：arrive_and_wait() → 到齐才继续

mbarrier (mbarrier.init / expect_tx / arrive / wait)
  - 对象在 SMEM（如 MemRange[Int64, 1]）
  - init：期望几次 arrive
  - expect_tx：期望多少字节由异步引擎完成（TMA）
  - TMA copy 可绑定 tma_bar_ptr，搬完由硬件记账
  - wait(phase)：等条件满足
```

**Named Barrier** 不跟踪「搬了多少字节」；**mbarrier** 专为 **异步内存操作 + 生产者–消费者** 设计（Hopper SM90+）。

CuTe 封装：

| 机制 | CuTe API |
|------|----------|
| Named Barrier | `cutlass.pipeline.NamedBarrier` → `cute.arch.barrier` / `barrier_arrive` |
| mbarrier | `cute.arch.mbarrier_init` / `expect_tx` / `arrive` / `wait` |

---

## 2. 作用范围（谁参与、管多远）

### Named Barrier

| 维度 | 说明 |
|------|------|
| **空间** | 单个 **CTA** 内（不跨 block） |
| **参与者** | 由你指定：`num_threads` + 实际调用 `arrive_and_wait` 的线程子集 |
| **ID** | `barrier_id` 1–15（0 常被 `sync_threads` 占用，慎用） |
| **约束** | 参与同步的线程数必须和 `num_threads` **一致**；通常 **同一批线程** 在相近路径上调用 |

`tma_v1` 示例：

```python
trans_sync_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=32 * 4,  # 仅转置 warp 0–3，共 128 线程
)
```

只有转置 warp 参与；TMA Load / Store warp **不进** 这个 barrier。

### mbarrier

| 维度 | 说明 |
|------|------|
| **空间** | 对象在 **SMEM**；默认 **本 CTA**；cluster 场景可配 peer CTA |
| **参与者** | **不必全员**在同一时刻 `wait` |
| **计数** | `init(arrive_count)`：需要多少次 `mbarrier_arrive` |
| **事务** | `expect_tx(bytes)`：TMA 等异步通路搬完多少字节 |

`tma_v1` 示例：

```python
# Load：TMA warp 发拷贝 + 1 次 arrive，硬件搬 N 字节
mbarrier_init(load_mbar_ptr, 1)
mbarrier_expect_tx(load_mbar_ptr, num_tma_load_bytes)

# Store：4 个转置 warp 各 1 次 arrive（无 expect_tx）
mbarrier_init(store_mbar_ptr, 4)
```

Load warp 发 TMA 后**不必**和 128 个转置线程一起 `bar.sync`；转置线程 `mbarrier_wait(load_mbar)`，Store warp 另等 `store_mbar`。

---

## 3. 「在等什么」—— 核心语义差异

### Named Barrier：只等「人」

```text
K 个线程都执行了 arrive_and_wait
    → 大家一起继续
```

- 不区分谁写了 SMEM、搬了多少字节  
- 若要先保证 SMEM 写对 TMA 可见，需自己加 `fence_proxy("async.shared")` 等（见 `tma_v1` 转置段）

### mbarrier：等「人」+ 可选等「事务」

`mbarrier_wait` 返回通常要同时满足：

| 条件 | 来源 |
|------|------|
| arrive 次数达到 `init` 时的计数 | 软件 `mbarrier_arrive`（常包在 `elect_one` 里） |
| 事务字节达到 `expect_tx` 登记的量 | **TMA 硬件**在 `cute.copy(..., tma_bar_ptr=...)` 完成后 |

适合：**一方发异步搬运，另一方稍后消费 SMEM**。

---

## 4. 与 `barrier()` / `__syncthreads` 的关系

| API | 近似 | 作用 |
|-----|------|------|
| `cute.arch.barrier()` | `__syncthreads` | **整个 CTA** 所有线程到齐 |
| `NamedBarrier(..., num_threads=K)` | `bar.sync` | **指定 K 个线程** 到齐（可多组 barrier 并存） |
| `mbarrier_*` | PTX mbarrier | **异步事务** + 可选部分线程 arrive/wait |

`tma_v0`：mbarrier init 后用 `barrier()` 保证全员看到初始化；TMA 完成用 `mbarrier_wait`。

---

## 5. `tma_v1` 中的分工（两套机制如何配合）

### Warp 角色

| Warp | 角色 | 线程数 |
|------|------|--------|
| 0–3 | 转置 sA → sB | 128 |
| 4 | TMA Load | 32 |
| 5 | TMA Store | 32 |

### 同步流水线

```text
Warp4 TMA Load ──[load_mbar]──► Warp0–3 转置 ──[trans_sync NB]──► (4×arrive store_mbar) ──[store_mbar]──► Warp5 TMA Store
         mbarrier                    Named Barrier                      mbarrier
      (字节 + arrive)               (128 线程到齐)                    (4×arrive)
```

| 阶段 | 机制 | 原因 |
|------|------|------|
| TMA Load → 转置 | **load_mbar** | TMA **异步**；需等 **N 字节进 sA** |
| 转置 warp 之间 | **trans_sync_barrier** | 4 warp **并行写 sB**；128 线程都写完 + fence 后再通知下游 |
| 转置 → TMA Store | **store_mbar** | Store warp 不必参与转置 barrier；只等 4 个生产者 warp |

### 转置段代码要点（`tma_v1.py`）

```python
cute.arch.fence_proxy("async.shared", space="cta")
self.trans_sync_barrier.arrive_and_wait()

with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(store_mbar_ptr)
```

顺序：**fence → Named Barrier（128 人到齐）→ 每 warp elect_one 各 arrive 一次 store_mbar（共 4 次）**。

### CUTLASS 文档中的计数差异

`NamedBarrier.wait` 注释指出：两 warp 用 **mbarrier** 做生产者–消费者时，arrive 可能只需 **32**（单侧）；若两 warp 都进 **Named Barrier**，往往要按 **64** 线程计数——**参与者集合不同**。

### 为何不用 Named Barrier 等 TMA？

TMA 在独立引擎上跑，完成由 **字节数** 决定，不是「多少线程执行到某行」。`bar.sync` **无法**表达「等 32KB 异步拷贝完成」，必须用 **mbarrier + expect_tx**。

### 为何转置阶段不用 mbarrier 互等？

转置是线程 **主动写 SMEM**（S2R/R2S），无 TMA 事务计数；只需 **「128 人都写完」** → Named Barrier 更直接。跨到 Store 仍是 **warp 级生产者–消费者** → **store_mbar**。

---

## 6. 使用场景速查

### 优先用 **mbarrier**

- TMA / `cp.async` **GMEM↔SMEM** 异步拷贝  
- 需要 **`expect_tx(字节数)`**  
- **生产者 warp 发搬运，消费者 warp 稍后读 SMEM**（不必同一同步点）  
- 多 **stage pipeline**（TMA 写 stage k+1，MMA 读 stage k）  
- 例：`tma_v0` 全程；`tma_v1` 的 `load_mbar` / `store_mbar`

### 优先用 **Named Barrier**

- **同一阶段、固定子集线程** 必须全部完成（如多 warp 一起写 SMEM）  
- **无** 硬件异步事务计数，只需「执行到同一点」  
- 同一 CTA 内 **多组 warp 分工**，各用不同 `barrier_id`  
- 例：`tma_v1` 的 `trans_sync_barrier`（128 转置线程）

### 两者配合

```text
mbarrier     ：跨阶段、跨 warp 角色、绑异步引擎
Named Barrier：阶段内部、固定线程集合、保证 SMEM 写齐
fence_proxy  ：Named Barrier 前保证 SMEM 写对 TMA 可见
```

---

## 7. 对比总表

| 维度 | Named Barrier | mbarrier |
|------|---------------|----------|
| PTX | `bar.sync` / `bar.arrive` | `mbarrier.*` |
| 存储 | 无 SMEM 对象，仅 barrier id | SMEM 中 barrier 对象 |
| 等待内容 | 线程 arrive 计数 | arrive + **可选异步字节** |
| 与 TMA | 不直接关联 | `tma_bar_ptr` + `expect_tx` |
| 参与者 | 声明 `num_threads`，调用者子集 | arrive/wait 可 **不同线程、不同时刻** |
| 典型范围 | CTA 内指定线程 | CTA（可扩展 cluster） |
| `tma_v0` | 未用 | 全程 |
| `tma_v1` | 转置 128 线程互 sync | Load/Store 两阶段 |

---

## 8. `tma_v1` 中 `store_barrier` 的补充

```python
store_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32)
```

注释意图是「Store warp 在转置后等待」，但当前 kernel **未调用** `store_barrier`；Trans→Store 实际由 **`store_mbar_ptr`** 完成（与 CUTLASS 上游一致）。

若将来要在 Store warp **内部** 再做 32 线程同步，才会用到；**跨 warp 的「转置完成」仍应走 mbarrier**。

---

## 9. 选型口诀

- **等 TMA / 异步拷贝的字节** → **mbarrier**（+ `expect_tx`）  
- **等一批线程都跑到某点 / 都写完 SMEM** → **Named Barrier**（+ 必要时 fence）  
- **整个 block 所有人** → `barrier()` / `__syncthreads`  
- **跨 warp 生产者–消费者 + 异步搬运** → **mbarrier**；阶段内多 warp 齐步走 → **Named Barrier**

---

## 延伸阅读

- [NVIDIA PTX — mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
- [NVIDIA PTX — bar.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
- [`tma_v0.md`](tma_v0.md) — mbarrier init / wait 逐行说明
- [`tma_v1.py`](tma_v1.py) — 生产者–消费者 + 双 barrier 实例
