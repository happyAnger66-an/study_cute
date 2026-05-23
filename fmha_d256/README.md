# fmha_d256 — CUTLASS 官方 d=256 FMHA prefill 迁移

把 CUTLASS 官方 Blackwell d=256 mixed-input FMHA prefill kernel 原样迁移到
study_cute 仓库下,方便在 Jetson Thor (sm_100a) 上跑 head_dim=256 的 FMHA。

## 文件来源 (CUTLASS 官方源)

| 本地路径 | CUTLASS 官方对应 |
| --- | --- |
| `fmha_helpers.py` | `examples/python/CuTeDSL/helpers/fmha_helpers.py` |
| `prefill_helpers.py` | `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/prefill_helpers.py` |
| `mixed_input_fmha_prefill_d256.py` | `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_prefill_d256.py` |
| `fmha_d256.py` | study_cute 自写的轻量 CLI shim |
| `__init__.py` | study_cute 自写,空 (使 `fmha_d256` 变成 Python package) |

`mixed_input_fmha_prefill_d256.py` 只改了两行 import (line 48-58),其余 100%
与官方一致,便于后续 follow upstream。

## 接口契约

| 张量 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| Q | BFloat16 | (B, H_q, S_q, D) | D **必须** 256 |
| K | **Int8** | (B, H_k, S_k, D) | int8 量化, H_q 必须能整除 H_k |
| V | **Int8** | (B, H_k, S_k, D) | int8 量化 |
| O | BFloat16 | (B, H_q, S_q, D) | 输出 |
| scale_k | BFloat16 | (B, H_k, S_k, D//scale_granularity) | per-channel scale |
| scale_v | BFloat16 | (B, H_k, S_k, D//scale_granularity) | per-channel scale |

`scale_granularity` 默认 256 (即每 256 维一个 scale)。也支持 128。

**重要**: 这个 kernel 是 mixed-input 设计 (Q 是 BF16, K/V 是 INT8),不是
study_cute 现有 `fmha/` 模块那种纯 FP16/BF16 接口。如果要接入 LLM prefill
loop (cumulative seqlen + packed cache),需要后续做接口适配 — 见底部"后续
工作"章节。

## 关键设计 (核心差异 vs study_cute `fmha/`)

1. **2-CTA cluster** (`cluster_shape_mn = (2, 1)`, `cta_group = TWO`):
   - `qk_mma_tiler[0] = 256` = 2 × cta_tiler[0]=128, 由 2 个 CTA 协作完成
   - 充分利用 Blackwell 2cta TMA multicast
2. **D=256 在 TMEM 内多 slot 累加**:
   - `iterations_pv = 2`, `tOtO_staged[..., iter]` 不同 iter 用 *不同* TMEM
     ACC slice (O 占 256 cols, S 占 128 cols, 总 384 cols 在 512 限制内)
   - 整个 PV 是 *1 个* acquire+commit, 不需要外层 d_chunk loop 重复整个
     attention
3. **Mixed-input 量化**: 8 个 `transform_warp` 专门做 i8→bf16 dequant; 通过
   独立的 `dequant_kv` pipeline 把转换后的 K/V 喂给 MMA warp

## 在 Jetson Thor (Blackwell sm_100a) 上的运行命令

```bash
cd <study_cute root>
export CUTE_DSL_ARCH=sm_100a    # 关键: 必须显式指定 Blackwell arch

# (1) 基础 smoke test (无 ref check, 最快验证 kernel 能跑通)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,8,256,256 \
    --is_persistent --skip_ref_check

# (2) 带 ref check 的精度验证
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,8,256,256 \
    --is_persistent

# (3) Causal 模式
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,512,256 --k_shape 1,8,512,256 \
    --is_persistent --is_causal

# (4) GQA (H_q != H_k)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,1,256,256 \
    --is_persistent

# (5) 完整 CLI 选项
python3 fmha_d256/fmha_d256.py --help
```

## 本地非 Blackwell GPU 上的限制

本地 sm_89 (RTX 4070 等) GPU 只能跑 import / argparse 验证 (上面 shim 已经
测过)。`cute.compile` 阶段就会因为 `arch=sm_89 != sm_100a` 报错;
`launch` 阶段更不可能。

也就是说,本目录只能在 Jetson Thor / 其他 Blackwell sm_100 GPU 上真正跑通。

## 与 study_cute 现有 `fmha/` 模块的关系

| 项目 | `fmha/` (现有) | `fmha_d256/` (本目录) |
| --- | --- | --- |
| 来源 | 自行重构 + outer-loop D-chunk 设计 | CUTLASS 官方原样迁移 |
| KV dtype | FP16/BF16 (un-quantized) | **INT8 + scale** |
| D 支持 | D≤128 完美; D=256 Round 2 死锁 (见 fmha/STATUS.md) | D=256 原生支持 |
| Cluster | 1-CTA | **2-CTA** |
| LLM prefill 接口 | ✓ 已接入 (cumulative seqlen + packed cache) | ✗ 需后续适配 |
| 代码量 | ~3500 行 (10+ 模块) | ~2400 行 (3 个官方文件 + 1 个 shim) |
| 推荐用途 | 学习 D≤128 的 outer-loop / pipeline 设计 | 跑 D=256 production 路径 |

## 后续工作 (可选)

1. **剥离 mixed-input 量化** — 把 i8 dequant 路径改成 noop, 直接接受
   FP16/BF16 K/V。这样可以跟现有 `fmha/` 模块统一接口。涉及:
   - 删 `transform_warp_ids` 那 8 个 warp + 相关 pipeline
   - 把 `dequant_kv_consumer` 改成直接接 `load_kv_consumer`
   - 把 `kv_dtype` 改成跟 `q_dtype` 一致
   - 删 `scale_k` / `scale_v` 参数
   - 大概 30-40% 代码改动
2. **接 study_cute LLM prefill runner** — 让 `run_llm_multi_round_prefill_test`
   能用本 kernel 跑多轮 prefill。涉及:
   - 适配 cumulative seqlen 输入 (本 kernel 当前用 fixed-size shape)
   - cache K/V 用 INT8 存储 + scale (或者先做 1. 再做这步)
3. **性能对比** — 跟现有 `fmha/` D=128 path 在 Thor 上的 latency 对比, 验证
   2-CTA cluster + iterations 设计的实际收益
