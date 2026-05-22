# FMHA D>128 (D-chunking) 重设计方案

> 状态: **outer-loop 设计已实施** (2026-05-22), 待 Blackwell 真机验证。
>   - host config 不再 reject D>128
>   - 5 个 warp (LOAD / MMA / softmax / correction / epilogue) 全部加外层
>     `for d_chunk_outer in range(num_d_chunks)` loop
>   - PV 段退化为 D=128 单 V 模式 (不再 inner d_chunk loop)
>   - QK 段保留 inner d_chunk loop (S 必须跨完整 d 累加)
>   - num_d_chunks=1 时 const_expr loop 单次, 完全等价 v3 D=128 path
>
> 历史 (有问题的) 设计:
>   - PV inner d_chunk loop + `v_carry_list`: PV 跨 chunk 累加到同一 tOtO0 →
>     d 维度被 reduce; epilogue 复制同一份 sO 到全部 gO slice → 输出错误
>   - per-d_chunk PV+correction+epilogue (内层 pipeline) 方案: TMEM 不够装
>     num_d_chunks 个 partial O (1-CTA SM100 TMEM 512 cols, 需 768 cols)
>
> 本文档下文保留原诊断和被否决方案的细节, 帮助理解为什么选 outer-loop。

---

## 1. 现状

`study_cute/fmha` 在 `head_dim > 128` 时启用 D-chunking,把 `head_dim` 切成 `num_d_chunks = ceil(head_dim / 128)` 个 `d_chunk_k = 128` 宽的块。

设计意图: MMA / SMEM / TMEM 都按 `d_chunk_k = 128` 宽分配 (与 D=128 完全等价),通过在 d_chunk 上循环来覆盖完整的 head_dim。

实际情况: PV gemm 和 epilogue 的 d_chunk 处理**语义错了**, 算出来的结果不是正确的 D>128 attention。

---

## 2. 诊断证据 (Blackwell 真机)

测试: `D=256, S=128, h_q=h_k=8, b=1`, 输入随机整数 [-2, 1] f16, 跑 prefill Round 1。

```
batch 0: FAIL  max_diff=1.684400  mean_diff=0.501495
[DCHUNK token=0 head=0] (first 8 values per chunk)
  chunk=0 actual=[-1.138, -1.576, -1.294, -0.777, -0.628, -1.036, -1.268, -1.030]
  chunk=0    ref=[-0.209, -0.521, -0.460, -0.688, -0.492, -0.217, -0.540, -0.236]
  chunk=1 actual=[-1.138, -1.576, -1.294, -0.777, -0.628, -1.036, -1.268, -1.030]
  chunk=1    ref=[-0.929, -1.054, -0.834, -0.088, -0.136, -0.818, -0.727, -0.793]
```

**关键观察**:

1. `chunk_0_actual == chunk_1_actual` — 两个 d_chunk slice 的 GPU 输出**完全相同**
2. `actual[i] == chunk_0_ref[i] + chunk_1_ref[i]`,逐元素精确匹配 (8/8):
   - `-0.209 + (-0.929) = -1.138` ✓
   - `-0.521 + (-1.054) = -1.575` ✓
   - ... (其余 6 个全对)

这两条事实**同时成立**,直接证明了下面 §3 的两个设计 bug。

---

## 3. 设计 bug 根因

### Bug 1: PV gemm 在 d_chunk 维度错误累加

PV gemm shape: `O[M, d_chunk_k=128] = P[M, N_kv] @ V[N_kv, d_chunk_k=128]`

当前代码 (`fmha/device/warp_mma.py`):

```python
for d_chunk_idx in cutlass.range_constexpr(self.num_d_chunks):
    v_handle = wait V[d_chunk_idx]
    tOrVi = tOrV[None, None, None, v_handle.index]
    for kphase_idx in ...:
        pv_tiled_mma.set(ACCUMULATE, d_chunk_idx != 0 or kphase_idx != 0)
        cute.gemm(pv_tiled_mma, tOtO0, tOrP0[kphase_coord],
                  tOrVi[kphase_coord], tOtO0)
```

`tOtO0` 是**固定的 TMEM region**(尺寸 `M × d_chunk_k`)。所有 d_chunk 都写到这同一区域,而且第二个及后续 d_chunk `ACCUMULATE = True` →

```
tOtO0_final[m, d_in_chunk]
  = sum_over_d_chunks( P[m, :] @ V[:, d_in_chunk, d_chunk_idx] )
  = P[m, :] @ ( sum_over_d_chunks V[:, d_in_chunk, d_chunk_idx] )
```

**等价于把 V 在 d 维度上 fold (reduce) 掉,而非按 d_chunk 各算各的**。

正确的语义应该是:

```
O_correct[m, d_full] = O_correct[m, d_chunk_idx * 128 + d_in_chunk]
                     = P[m, :] @ V[:, d_in_chunk, d_chunk_idx]
```

不同 `d_chunk_idx` 对应**不同的 d_full slice**,**不能累加**。

### Bug 2: Epilogue 把同一份 sO 复制到 num_d_chunks 个 GMEM slice

`fmha/device/warp_epilogue.py`:

```python
o0_handle = corr_epi_consumer.wait_and_advance()
cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])  # store sO to gO[d=0]
o1_handle = corr_epi_consumer.wait_and_advance()
cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])

if num_d_chunks > 1:
    for d_chunk_idx in range(num_d_chunks - 1):
        d_idx = d_chunk_idx + 1
        gO = gO_qdl[None, None, None, d_idx, ...]
        _, tOgO = tma_partition(..., sO ..., gO ...)
        cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])  # 同一份 sO!
        cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
```

correction warp 也只 fill 了一次 sO (对单个 tOtO0 / tOtO1),因此 epilogue 把这同一份 sO 复制到 num_d_chunks 个 GMEM slice → 每个 d_chunk slice 都得到同一份 (而且是 Bug 1 中错误累加的) 结果。

### Bug 3 (隐性): TMEM 容量装不下 num_d_chunks 个 O

当前 TMEM 布局 (`fmha/host/config.py`):

| Region | 起始列 | 列数 |
|---|---|---|
| S0 | 0 | 128 |
| S1 | 128 | 128 |
| O0 | 256 | 128 |
| O1 | 384 | 128 |
| **总计** | 0–511 | **512 列 (Blackwell TMEM 上限)** |

要让 D=256 同时 hold 两个 d_chunk 的 O,O0/O1 各需要 256 列, 加 S 256 列 = **768 列, 超 TMEM 上限**。

因此**不能**通过"扩 TMEM 区域"修 Bug 1,只能让 PV/correction/epilogue 在 d_chunk 上**串行执行**(分时复用 tOtO0 / tOtO1)。

---

## 4. (被否决) 备选方案: per-d_chunk PV→correction→epilogue (inner-loop)

**先评估的方案**: 把 PV / correction / epilogue 改成 d_chunk inner loop,
每个 d_chunk 串行跑 PV→correction→epilogue, partial O 存 TMEM。

**否决原因**: 这个方案需要 tOtO0 / tOtO1 各自能 hold 一个 d_chunk 的 partial O,
跨 KV iter 累加 partial O。对 num_d_chunks=2, 需要 O0/O1 各 256 cols, 加 S0/S1
各 128 cols, 总 768 cols, **超 1-CTA SM100 TMEM 512 cols 上限**。

cutlass MLA (D_latent=512) 在 2-CTA mode 下能解决这个问题 (ThreadShape=(2,1,1),
TMEM 容量翻倍), 但本项目是 1-CTA 设计, 不能直接套用。

下文 §4.1 / §4.2 是被否决方案的细节,**仅作记录**。实际实施的方案见 §5。

## 4-old (被否决) 重设计方案: per-d_chunk PV→correction→epilogue pipeline

### 4.1 流程对比

**当前 (broken)**:

```
QK (d_chunk 累加 S, S 正确) → softmax → PV (d_chunk 累加 O, 错!)
                                       → correction (1 次) → epilogue (重复 store sO)
```

**目标**:

```
QK (d_chunk 累加 S, 不变) → softmax (不变)
                          → for d_chunk:
                                PV[d_chunk]: gemm O 覆盖 tOtO0
                                correction[d_chunk]: rescale → sO (per-d_chunk)
                                epilogue[d_chunk]: TMA store sO → gO[d_chunk]
```

### 4.2 各 warp 需要的改动

#### MMA warp (`fmha/device/warp_mma.py`)

每个 PV 段 (PV00 / PV1(i-1) / PV0i / tail PV1) 把 `acquire o_handle` 移到 d_chunk loop 内层,每个 d_chunk 单独 `commit`:

```python
# PV00:
for d_chunk_idx in range(num_d_chunks):
    v_handle = wait V
    o0_handle = mma_corr_producer.acquire_and_advance()  # ← per d_chunk
    s0_handle = mma_s0_producer.acquire_and_advance() if d_chunk_idx == 0 else s0_handle
    for kphase: gemm O0  # ACCUMULATE = kphase != 0 (覆盖 tOtO0, 不跨 d_chunk 累加)
    o0_handle.commit()  # ← per d_chunk
    v_carry_list.append((v_handle, ...))
```

注意: `s0_handle` (mma↔softmax 的 P0 ownership) 跨整个 d_chunk loop 共享,不分 d_chunk (P0 与 d_chunk 无关)。但 `o0_handle` (mma→correction 的 O0 ownership) 必须 per-d_chunk,因为 correction 也要 per-d_chunk 处理。

#### Correction warp (`fmha/device/warp_correction.py`)

final correction + epilog 段改为 d_chunk 循环:

```python
for d_chunk_idx in range(num_d_chunks):
    o0_handle = mma_corr_consumer.wait_and_advance()  # ← per d_chunk
    o0_final_handle = corr_epi_producer.acquire_and_advance()
    correction_epilog(pv_thr_mma, tOtO0, ..., sO[None, None, 0])
    o0_handle.release()
    o0_final_handle.commit()
# 类似 O1
```

`correction_rescale` 部分 (在 kv loop 里) 不变,只有 final epilogue 部分要 per-d_chunk。

但 **rescale 也有问题**: 当前 rescale 假设 tOtO0 内是单个 d_chunk 的 partial O,跨 d_chunk 时 tOtO0 会被多次 rescale。要么 (a) 在 d_chunk 外只 rescale 一次,然后 d_chunk 内 PV gemm 不要再覆盖 (那就回到 Bug 1),要么 (b) 把 rescale 移到 d_chunk 内层。需要重新设计:

- 方案 b: rescale + epilog 都在 d_chunk 内层。但 rescale 需要 row-max/row-sum,这些与 d_chunk 无关 (只看 S),所以 rescale 跨 d_chunk 用同一个 scale 值即可。把 scale 计算移到 d_chunk loop 外,每个 d_chunk 内对 tOtO0 应用同一个 scale 即可。

#### Epilogue warp (`fmha/device/warp_epilogue.py`)

把 "store sO + 后面重复 store" 改为 per-d_chunk wait + store:

```python
for d_chunk_idx in range(num_d_chunks):
    o0_handle = corr_epi_consumer.wait_and_advance()  # ← per d_chunk
    gO = gO_qdl[None, None, None, d_chunk_idx, ...]
    _, tOgO = tma_partition(..., gO, ...)
    cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)
    o0_handle.release()
# 类似 o1
```

注意 `tOsO` (sO 的 TMA partition) 不能跨 d_chunk 共用 (gO partition 不同),要在 d_chunk 内部重新 partition。

#### Pipeline depth 调整

`mma_corr` 和 `corr_epi` pipeline 的 stage 数原本按 1 个 KV iter = 1 个 PV (O0+O1) 计。改为 per-d_chunk 后变成 1 个 KV iter = `num_d_chunks` × (O0+O1)。

`acc_stage` 和 `epi_stage` 可能要从 `2` 提到 `2 * num_d_chunks`,或者保持 2 但让 PV warp 阻塞等 correction/epilogue 跟上。需要在 `host/config.py` 的 `_pick_pipeline_stages` 里加 D-chunking 分支。

### 4.3 LOAD warp

LOAD 端 `cute.copy(tma_atom_v, tVgV[..., d_chunk], tVsV[v_handle.index])` 已经 per-d_chunk produce V (`num_d_chunks` 个 V handle per kv tile),**这部分已经对了**,不需要改。

---

## 5. (已实施) outer-loop 方案: d_chunk 外层循环

### 5.1 核心思路

不再尝试在 inner pipeline 解决 D-chunking, 而是**把整个 attention pipeline
跑 num_d_chunks 次**, 每次只用一个 V[d_chunk_outer] 切片, 写到对应的
gO[d_chunk_outer] slice:

```
for d_chunk_outer in range(num_d_chunks):
    # 完整 attention pipeline (Q full, K full, V[d_chunk_outer], O[d_chunk_outer])
    for kv_iter:
        QK:  for d_chunk_inner: gemm S 累加 (S 取决于完整 d, 必须跨 inner 累加)
        softmax(S) → P
        PV:  gemm O = P @ V[d_chunk_outer]  # 单 V, 退化为 D=128 模式
        correction (rescale)
    final correction → sO
    epilogue: TMA store sO → gO[d_chunk_outer]
```

**性能代价**: QK 和 softmax 在 D=256 时被算 2 次 (每个 d_chunk_outer 一次),
PV 不重复。总 latency ~1.5x (vs in-place D=128 设计)。D=128 时
`num_d_chunks=1`, outer loop 单次, 性能完全不变。

**TMEM 用量**: 每个 d_chunk_outer iter 用同一份 tOtO0/tOtO1 (128 cols each),
512 cols TMEM 容量足够, 不用 2-CTA。

### 5.2 各 warp 改动

| Warp | 改动 |
|---|---|
| LOAD | 外层 d_chunk_outer loop; 每个 outer iter 每 kv tile 产 num_d_chunks Q0+K+Q1 + **1 V[d_chunk_outer]** (不再 produce num_d_chunks V) |
| MMA | 外层 d_chunk_outer loop 包整个 prologue+main+tail; **PV 段去掉 inner d_chunk loop** (单 V), `v_carry_list` 退化为单变量 `v_handle`/`tOrVi` |
| softmax | 外层 d_chunk_outer loop 包整个 leading+unmask+trailing+final; 每个 outer iter 重置 (row_max, row_sum) |
| correction | 外层 d_chunk_outer loop 包 KV correction loop + final epilog; final epilog 写 sO 给 epilogue |
| epilogue | 外层 d_chunk_outer loop; 每个 iter wait 一对 corr_epi handles, TMA store `sO → gO[..., d_chunk_outer, ...]` |

`fmha/host/config.py` 移除 D>128 reject, pipeline stage 数不变
(q_stage=2, kv_stage=3, epi_stage=2, acc_stage=1)。

### 5.3 工作量

实际改动: ~250 行, 一次性完成 (2026-05-22)。

---

## 6. 当前状态与下一步

- ✅ D=128 路径完全工作 (v3 修复了 deadlock + 精度,prefill 3 轮 PASS, max_diff=0.0006)
- ✅ D>128 路径在 host config 里**显式 reject**,避免静默错误
- ⏸️ D-chunking 重设计列为 future work, 按本文 §4 方案执行

如果以后要恢复 D>128 支持: 拿掉 `host/config.py` 的 raise, 按 §4.2 改 3 个 warp + pipeline stage, 用 `FMHA_DEBUG_DCHUNK=1` 反复验证每个 d_chunk 的 GPU 输出与 numpy ref 逐元素吻合。
