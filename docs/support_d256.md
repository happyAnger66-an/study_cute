# FMHA 支持 D>128 修改方案

实现仓库：`study_cute/fmha/fmha.py`（实验主线）。问题与复现见 [d_256.md](d_256.md)。

**现象**：D≤128 同参数可跑；D=144 起 `CUDA_ERROR_INVALID_VALUE`（launch 阶段，多为动态 SMEM 超限）；D>128 另受 TMEM 列布局限制。

---

## 目标

| 阶段 | 目标 D | 状态 |
|------|--------|------|
| **P0** | 仅 **SMEM 校验** + `D>128` 提前报错 | ✅ 已实施（**不能**单独支持 D=144） |
| **P1** | 129–256+（D 分块） | 待做 |
| **P2** | TRT AOT + runner | 待做 |

---

## 阶段 P0：Host 校验（已实施，**不**支持 D>128 跑通）

### 实验结论（D=144）

| 现象 | 原因 |
|------|------|
| `q=1` 时 compile 后**卡死** | `q_stage=1` 与双 Q tile 流水线死锁 |
| `q=2,kv=3,epi=1` 后 **ILLEGAL_ADDRESS** | **TMEM**：`tmem_o0=256`、`tmem_o1=384` 间距仅 **128 列**，PV 的 O 宽 = **D=144** → O0 占 256..399，与 O1@384 **重叠**；且 256+2×144>512 |
| 减 `epi_stage` **不能**修 TMEM | 根因是 **D 维列数**，不是 epilogue stage |

**结论：P0 不能替代 P1。`D>128` 必须在 compile 前拒绝，或实现 D 分块。**

### 代码（`validate_config_host`）

- `D_mma > 128` → 明确 `ValueError`（避免 launch 后 ILLEGAL_ADDRESS）
- `D_mma ≤ 128` 且 staged SMEM 估算超预算 → `ValueError`
- Pipeline stage **保持默认** `q=2, kv=3, epi=2`（不再对 D>128 减 stage）

### 验证

```bash
# 应通过
python3 fmha/fmha.py --q_shape 1,968,8,128 --k_shape 1,968,1,128

# 应 compile 前报错（需 P1）
python3 fmha/fmha.py --q_shape 1,968,8,144 --k_shape 1,968,1,144
```

---

## 阶段 P1：D 方向分块（`D_chunk=128`）

### 核心

- **MMA / SMEM / TMEM 的 K 维恒为 128**（与现网 D=128 布局一致）。
- **`num_d_chunks = ceil(head_dim / 128)`**；host `mma_tiler = (128, 128, 128)`。
- **QK**：对每个 KV tile，遍历 `d_chunk`，UMMA 累加到同一 **S**（`ACCUMULATE`）。
- **Softmax**：所有 chunk 的 QK 完成后再做（算法不变）。
- **PV**：每 chunk 计算 `P @ V[:, chunk]`，epilogue 写 O 时 D 坐标 **+ chunk×128**。

### 需改区域

| 模块 | 改动 |
|------|------|
| `__init__` / host `run()` | `d_chunk`, `num_d_chunks`, `mma_tiler` 固定 K=128 |
| Load warp | TMA 源 D 维 `+ d_chunk * 128` |
| MMA warp | 外包 `d_chunk` 循环（QK 累加 S；PV 写 O 分块） |
| Epilogue | TMA store O 的 D offset |
| `__call_vit__` | 同逻辑（若 ViT 也要 D>128） |

### 资源

| D | chunks | SMEM | TMEM |
|---|--------|------|------|
| 256 | 2 | 同 D=128 | 现有 offset 可用 |
| 384 | 3 | 同 D=128 | 可用；主循环 ×3 |

**性能**：QK+PV 约 ×`num_d_chunks`，属预期 trade-off。

---

## 阶段 P2：产品集成（可选）

1. 同步 P1 到 `TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py`。
2. `build_cutedsl.py` 增加 `fmha_d256` variant。
3. `CuteDslFMHARunner::canImplement` 增加 `headSize == 256`。
4. README / [fmha.md](fmha.md) 更新支持列表。

**短期替代**：Edge-LLM plugin 对 D=256 走 **FMHA_v2** cubin（`canImplement` 已含 256）。

---

## 不推荐方案

| 方案 | 原因 |
|------|------|
| 只改 `q_shape` 最后一维 | SMEM/TMEM 仍按 D 分配 |
| 只改 TMEM offset | D=144 仍 SMEM 超 |
| `mma_tiler_mn` 128→64 | softmax/correction/TMEM 均按 128 宽设计，改动面≈重写 |

---

## 实施顺序

```mermaid
flowchart TD
  P0[P0 自适应 stage] --> T144[D=144 测试]
  T144 --> P1[P1 D 分块]
  P1 --> T256[D=256 数值]
  T256 --> P2[P2 TRT AOT]
```

---

## 相关文档

- [d_256.md](d_256.md) — 现象、根因、实验日志
- [fmha.md](fmha.md) — FMHA 架构与 warp 分工
