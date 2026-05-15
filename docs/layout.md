# CuTe / CUTLASS 中的 Layout 概念小结

本文整理 CuTeDSL（Python）与 CUTLASS 中 **Layout** 的含义：**形状（shape）与步长（stride）**如何共同决定下标到线性存储的映射，以及 **为何在算子库里要把 Layout 做成一等抽象**。

---

## 1. Layout 在记什么？

在 CuTe 里，张量常与一个 **Layout** 绑定。最熟悉的写法是：

```text
shape : stride
```

- **Shape（形状）**：每一逻辑维有多少个合法下标。
- **Stride（步长）**：在该维上索引 **+1** 时，**线性偏移（以元素个数计）** 增加多少。

对简单的 \(D\) 维矩形布局，坐标 \((i_0,\ldots,i_{D-1})\) 的线性偏移为：

\[
\text{offset} = i_0 s_0 + i_1 s_1 + \cdots + i_{D-1} s_{D-1}
\]

**Shape 定义「能取哪些下标」；stride 定义「每维走一步偏移加多少」。** 二者一起定义从逻辑坐标到存储位置的函数，而不是「先 shape 后 stride」的运行时顺序问题。

---

## 2. 嵌套 Shape / Stride（分层布局）

CuTe 常用**分层**（hierarchical）记法，例如 `elementwise_add` 教程里对 `2048×2048` 做向量化分块：

```text
((1, 8), (2048, 256)) : ((0, 1), (2048, 8))
```

可理解为两级：

| 层级 | Shape | Stride | 含义（直觉） |
|------|--------|--------|----------------|
| 内层（tile / 向量块） | `(1, 8)` | `(0, 1)` | 每块 **1 行 × 8 列**（FP16 共 128 bit，适合向量访存）；列方向 stride 1；第一维长度为 1，stride 常为 0 |
| 外层（块网格 / Rest） | `(2048, 256)` | `(2048, 8)` | 共 **2048 行块 × 256 列块**（每列块宽度 8）；下行块 +2048（一行元素数），下列块 +8 |

合成后与行主序 `2048×2048 : (2048, 1)` 一致：若网格坐标为 \((g_0,g_1)\)，块内列为 \(t_1\)，则

\[
\text{offset} = g_0 \cdot 2048 + g_1 \cdot 8 + t_1
\]

对应原矩阵行 \(g_0\)、列 \(8 g_1 + t_1\)。

**Stride 不必小于「另一维」的 shape**：例如列块数 `256` 与行 stride `2048` 同时出现是正常的—— stride 只描述「本维索引 +1」时的增量。

---

## 3. `zipped_divide` 与向量化 elementwise

```python
gA = cute.zipped_divide(mA, (1, 8))
```

- **`zipped_divide(target, tiler)`**：对张量做 **逻辑分块**，结果呈 **`((Tiler…), (Rest…))`** 两层结构；**不搬数据**，通常与 `mA` 共享同一底层指针。
- **`tiler = (1, 8)`**：每个逻辑小块是 **一行上连续 8 个元素**，便于 `gA[(None, (mi, ni))].load()` 一类 **向量化加载**。

教程示例（`2048×2048`）给出的布局即上一节的

`((1, 8), (2048, 256)) : ((0, 1), (2048, 8))`。

使用时注意：Rest 维是 `(M, N/8)`，grid 要按「向量块个数」计算；并配合 **`from_dlpack(..., assumed_align=16)`** 等对齐假设，否则编译器无法安全生成宽向量指令。

---

## 4. 为何 CUTLASS / CuTe 要引入 Layout？

1. **分离语义与存储展开**  
   同一数学对象可落在 GMEM、带 swizzle 的 SMEM、TMEM、寄存器 fragment 上；Layout 描述「坐标 → 偏移」，算法（如何切 tile、如何流水）不必到处手写 `row * lda + col`。

2. **显式表达切块、流水、向量化**  
   `zipped_divide`、`local_tile`、`logical_divide` 等在不复制数据的前提下切换索引视角。**「当前线程看到的是哪一块子张量」** 成为可组合的结构，便于 `partition_*`、`make_tiled_copy` 等与 MMA 对齐。

3. **统一多地址空间的同一种语言**  
   GMEM / SMEM / TMEM / RMEM 的最优 stride 不同，但都希望用同一套 shape/stride 与分层规则描述，使 `cute.copy`、`cute.gemm` 等能在不同层级复用组合规律。

4. **利于 DSL / 编译器验证与 codegen**  
   Layout 进入 MLIR 后可检查切片是否与视图 **弱同余**、向量访存是否满足对齐等，减少「运行期才暴露或静默错误」的指针运算。

5. **可重用、可维护**  
   GEMM、Attention、Norm、elementwise 中重复的 K-major、CTA tile、swizzle atom 等可沉淀为 **布局构造函数**（如 `make_smem_layout_a`、`tile_to_shape`），改指令形状或 swizzle 时常改 layout 构造而非整核重写。

---

## 5. 相关 API（便于进一步阅读）

| API | 作用（简述） |
|-----|----------------|
| `cute.make_layout(shape, stride=…)` | 构造基础 layout |
| `cute.zipped_divide(tensor, tiler)` | `(Tiler, Rest)` 拉链式分块视图 |
| `cute.local_tile(global, tiler, coord, proj=…)` | 按 block/tile 坐标取子张量 |
| `cute.logical_divide` / `tiled_divide` | 其他分块形式；`zipped_divide` 是其「Tiler 与 Rest 分组」变体 |
| `cute.group_modes` | 合并若干 mode，常与 TMA/cp.async 划分配合 |

官方 Python 定义可参考 CUTLASS 树内 `cutlass/cute/core.py` 中 `zipped_divide` 的文档字符串。

---

## 6. 一句话

**Layout = 用 shape 与 stride（可嵌套）精确定义「逻辑多维坐标如何映射到线性存储」。**  
CUTLASS 把它做成核心抽象，是为了在 **不牺牲性能** 的前提下，让索引、切块、向量化与多存储层级 **可组合、可验证、可自动生成高效访存与 MMA**。

---

*文档内容源于对 CuTeDSL `elementwise_add` 教程与 `cutlass/cute/core.py` 中 `zipped_divide` 说明的整理。*
