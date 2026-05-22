# FMHA 重构 + D=256 死锁修复 — 进度快照

> 最后更新: 2026-05-22 (Friday) ~22:50 UTC+8
> 状态: **Fix A v3 (4 段精度修复)** 完成, D=128/D=256 都编译通过, 待 Blackwell 真机验证
>
> 修复历史:
>   - v1: 错把 PV1 也 wait V → D=128 死锁 (废弃)
>   - v2: 用 `v_carry_list` 对 V handle "acquire-defer-release" → 死锁消除, 但
>         **合并了 QK0i/QK1i 段**, 让 `s1_handle` 在 PV1 段提前 commit, QK1i 写 S1
>         时 ownership 不在 mma 手里, softmax 读到没写完的 S1 算出错的 P1 →
>         D=128 精度爆掉 `max_diff=99.7` (废弃)
>   - v3 (当前): 严格按原版 4 段顺序 `QK0i → PV1(i-1) → QK1i → PV0i`, s0/s1 跨段
>         共享句柄, 修复 v2 的精度问题, 同时保留 v2 的 V-handle carry list 设计

---

## ✅ 已完成

### 重构 (Day 1)
1. **包骨架**: `fmha/{host,device}/__init__.py` + `fmha/__init__.py` (类绑定中心)
2. **host 层** (4 个文件):
   - `host/tensor_layout.py` — 张量布局/dtype 辅助
   - `host/numpy_ref.py` — numpy 参考实现
   - `host/config.py` — `BlackwellFusedMultiHeadAttentionForward` 类 (仅 host 部分);
     新增 `debug_pipeline = False` 类变量供 `cute.printf` trace 开关
   - `host/launcher.py` — `_call_llm` / `_call_vit` + 共享
     `_build_kernel_inputs_and_launch` + `_make_shared_storage`
3. **device 层** (6 个文件):
   - `device/kernel.py` — 顶层 `@cute.kernel` 外壳: SMEM/TMEM 资源、所有 pipeline 创建、
     warp_idx 分发, 通过 `self.<warp>_body(...)` 调用
   - `device/warp_load.py` — TMA Q/K/V loads, `debug_pipeline` gated trace
   - `device/warp_mma.py` — QK / PV (**已应用 D=256 Fix A**, 见下文)
   - `device/warp_softmax.py` — `softmax` + `softmax_step`
   - `device/warp_correction.py` — `correction_warp_body` + `correction_rescale` +
     `correction_epilog`
   - `device/warp_epilogue.py` — TMA S→G O store
4. **`fmha/runner.py`** — `run()` / `run_llm_multi_round_prefill_test()` + `main()` CLI
5. **`fmha/fmha.py`** — thin shim (~40 行), 转发到 `fmha.runner.main`
6. **`fmha/__init__.py`** — 把所有 @cute.jit 方法 monkey-patch 到
   `BlackwellFusedMultiHeadAttentionForward` 上, 让 DSL 把 `cfg` 当 `self`
   (静态 Python 对象), 避开 `DSLTreeFlattenError`
7. **D=128 / D=256 JIT 编译均通过** (本机 sm_89 仅能验编译, launch 时报
   `cudaErrorNoKernelImageForDevice` 是预期硬件不匹配)

### Fix A v1 (废弃) — 错误分析的教训

第一版以为每个 PV 段 (PV00 / PV1(i-1) / PV0i / PV1 final) 都要自己
`wait_and_advance()` 来获取 V. 这是错的:

- LOAD 端每 kv tile 只产 `num_d_chunks` 个 V
- 但 v1 让 MMA 每 iter 想 wait `2 * num_d_chunks` 个 V (PV1 一遍, PV0 一遍)
- 结果 D=128 / D=256 都立刻在 PV1 wait_and_advance 上死锁

用户在 Blackwell 真机上跑 v1 复现了这个错误: `Compilation time: 3.66s →
LLM multi-round prefill test Round 1 (kernel launch) → GPU 100% / CPU 低`,
device 端 mbarrier 卡住.

### Fix A v2 (废弃) — V handle 对了, QK 段合并错了

v2 把 V-handle 的"acquire-defer-release"对偶模式扩展到 D-chunking 通过 `v_carry_list`,
死锁消除. 但 **v2 把 QK0i + QK1i 合并成一个 d_chunk 循环**, 然后:

- PV1(i-1) 段 acquire s1, gemm O1 (读 P1), `s1_handle.commit()` 立即把 stage 还给 softmax
- 下一 iter 开始的合并 QK 段写 S1: outer scope `s1_handle` 已 commit, ownership 已转给
  softmax → mma 越权写 → softmax 读到没写完的 S1 → 算错 P1 → PV1 用错 P1 算 O1
- 结果: `Round 1/3 batch 0: FAIL max_diff=99.703667 mean_diff=2.616742`, 后续 round 全 NaN
  (NaN 来自 numpy ref 也用错的 cache 进行计算)

### Fix A v3 (当前) — 严格 4 段 + s0/s1 跨段共享

按原版的精确 4 段顺序: `QK0i → PV1(i-1) → QK1i → PV0i`. 关键约束:

- **`s0_handle` 跨段共享**:
  - PV00 (或 prev PV0i) acquire 一个 s0 stage
  - iter i 的 QK0i 写 S0 进这个 stage, 末尾 `s0_handle.commit()` → softmax 算 P0
  - PV0i acquire 新的 s0 stage (给下个 iter QK0i 用)
  - Tail 末尾把最后 PV0i 的 s0 也 commit (清理)
- **`s1_handle` 跨段共享**:
  - PV1(i-1) acquire 一个 s1 stage — 既用来读 P1 (PV1 段), 也用来写新 S1 (QK1i 段)
  - QK1i 末尾 `s1_handle.commit()` → softmax 算下一轮 P1
  - **关键**: v2 错在 PV1 立即 commit s1, v3 把 commit 推迟到 QK1i 末尾
  - Tail 段 acquire 一个 s1 stage 用于 gemm O1 final, 末尾 commit (清理)
- **Q0 / Q1 / K handles 跨段持有**:
  - QK0i wait Q0 + K (push 到 list 跨 d_chunk)
  - QK1i wait Q1, 用 list 里的 K
  - QK1i 末尾统一 release Q0 / Q1 / K
- **V-handle carry list (沿用 v2)**: `v_carry_list = [(v_handle, tOrVi), ...]` 在 PV00
  / PV0i 填充, 在紧跟的 PV1 / tail-PV1 消费 + release

为什么 v2 一开始要合并 QK?
LOAD 端 study_cute 设计每 KV iter 产 (Q0, K, Q1) per d_chunk (跟原版每 iter 只产 K/V
不同), 看起来"全部一起 wait/release"更对称, 但**破坏了 s1 的跨段语义**. v3 接受
"Q0/Q1/K handle 跨段持有"的复杂性来换正确性.

学习了 `TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py` 的原版本,
明白了原版的 V handle 模式:

```
PV00 prologue: v_handle = wait()           # acquire V_0, NO release here
               tOrVi = view(V_0)
               gemm(P0, tOrVi)
               o0_commit                   # defer release

iter i:
  QK0i:        K wait, gemm S0, s0_commit
  PV1(i-1):    # NO new wait! Use tOrVi from outer scope (= V_{i-1})
               gemm(P1, tOrVi)
               o1_commit
               v_handle.release()          # ← release V_{i-1} HERE
  QK1i:        gemm S1, s1_commit, K release
  PV0i:        v_handle = wait()           # acquire V_i, NO release
               tOrVi = view(V_i)
               gemm(P0, tOrVi)
               o0_commit                   # defer release

Tail PV1:      # NO wait! Use tOrVi from outer scope (= V_{N-1})
               gemm(P1, tOrVi)
               o1_commit
               v_handle.release()          # ← release V_{N-1}
```

每个 V handle 精确 1 acquire (在 PV0/PV00) + 1 release (在紧跟着的下一个
PV1/tail PV1). 零泄漏, 完美 1-1 配对.

**D-chunking (D=256) 扩展**: 每个 PV 段需要遍历 `num_d_chunks` 个 V handle
(不同 D 维数据). 用 Python list `v_carry_list = [(v_handle, tOrVi), ...]`
把整组传给下一个 PV 段. PV1 段用 `v_carry_list[d_chunk_idx]` 取对应 d_chunk
的 tOrVi 做 gemm (这同时修复了 D=256 计算错误 — 否则只用 outer-scope 单个
tOrVi 等于丢失了非最后 d_chunk 的 V 数据), 然后释放所有 handle.

**代码变更**: `fmha/device/warp_mma.py` 主循环重写, prologue / tail 微调:

Prologue (不变):
- 合并 QK00 + QK10 跨 d_chunk, acquire s0/s1 → gemm S0/S1 → commit s0/s1
- PV00: acquire o0 + new s0, `v_carry_list` 暂存 `num_d_chunks` 个
  `(v_handle, tOrVi)`, defer 全部 V release, **不** commit s0 (留给 iter 0 QK0i)

Main loop 4 段:
- **QK0i**: per d_chunk wait Q0 + K (push 到 `q0_handles` / `k_handles` list),
  gemm S0 跨 d_chunk 累加, 末尾 `s0_handle.commit()`
- **PV1(i-1)**: acquire o1 + new s1 (这个 s1 也会给 QK1i 用!), 遍历
  `v_carry_list` 用对应 tOrVi gemm O1, commit o1, release `v_carry_list` 全部
  V handle, **不** commit s1
- **QK1i**: per d_chunk wait Q1, 用 `k_handles[d_chunk]` 里的 K, gemm S1 跨
  d_chunk 累加, 末尾 `s1_handle.commit()` + 统一释放 Q0/Q1/K
- **PV0i**: acquire o0 + new s0, 重新初始化 `v_carry_list = []`, 遍历 acquire
  新一轮 V handle gemm O0, commit o0, **不** commit s0 (留给下个 iter QK0i)

Tail:
- acquire o1 + new s1, 遍历 `v_carry_list` gemm O1, commit o1, release V 全部
- **commit s0 + commit s1** (清理最后的 stage)

### 关键时序约束

1. **`cute.gemm` 是 async**: 必须先 `o_handle.commit()` 才能
   `v_handle.release()`. 否则 tensor cores 还在读 V SMEM 时 LOAD 可能 refill
   该 slot.

2. **s1 跨 PV1 + QK1i 共享**: s1 ownership 期间 mma 既读 P1 (PV1 段) 又写新
   S1 (QK1i 段), 同一个 stage 复用 S1 buffer. PV1 提前 commit s1 (v2 bug)
   会让 QK1i 越权写, softmax 读到不一致的 S1.

3. **D-chunked S0/S1/O0/O1 累加**: 跨 d_chunk 在同一个 stage 内累加 (因为
   S0/S1 是 TMEM, O0/O1 是 TMEM 的不同 region), ACCUMULATE 标志按
   `d_chunk!=0 or kphase!=0` 决定. PV1 段的 ACC 多 OR 一个 `pv_whether_acc`
   覆盖 iter 0 第一次 PV1 的 "覆盖 O1" 行为.

4. **Python list 跨段流动 (v_carry_list / q0_handles / k_handles)**: 它们是
   Python compile-time 对象 (`num_d_chunks` 是 const_expr, d_chunk 循环在
   trace 时全部展开). list 中存的 `v_handle` / `tOrVi` / `k_handle` 是 DSL
   值引用, 跨段流动符合 DSL 的 SSA 风格.

---

## 🔜 待办 — 在 Blackwell 真机上跑

本机 sm_89/Ada 跑不了 Blackwell tcgen05 kernel, 以下命令需在 Blackwell GPU 上跑:

```bash
cd /path/to/study_cute
rm -rf fmha/__pycache__ fmha/host/__pycache__ fmha/device/__pycache__

# ① D=128 回归 (v3 必须修复 v2 的精度错: max_diff=99.7)
python3 fmha/fmha.py --q_shape 1,256,8,128 --k_shape 1,256,8,128 --is_persistent
# 预期: 内置 3 轮 prefill ref check 全部 PASS, max_diff < 0.1 (而非 v2 的 99.7)

# ② D=256 单 KV tile (loop_steps=0, 走 PV00 + 尾声 PV1 final)
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 \
  --is_persistent --skip_ref_check
# 预期: kernel 不再死锁, 正常返回 latency

# ③ D=256 多 KV tile (loop_steps>0, 走完整 PV00 → 主循环 → 尾声)
python3 fmha/fmha.py --q_shape 1,256,8,256 --k_shape 1,256,8,256 \
  --is_persistent --skip_ref_check

# ④ D=256 + reference check + 3 轮 prefill (默认就跑)
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent
python3 fmha/fmha.py --q_shape 1,256,8,256 --k_shape 1,256,8,256 --is_persistent
# 预期: 3 轮 prefill 全 PASS, max_diff < 0.1
```

### 如果还死锁

打开 trace 看 LOAD/MMA 在哪个 V slot 卡住:
```python
# fmha/host/config.py:
debug_pipeline = True
```
然后重新编译运行, 看 stderr 上 `LOAD pro V chunk=X slot=Y` 和
`MMA PV00 wait V chunk=X slot=Y` 序列在哪一步停下来. 大概率会暴露:
- LOAD 在某个 V acquire 阻塞 → 消费侧 (MMA) 漏了 release
- MMA 在某个 V wait_and_advance 阻塞 → 生产侧 (LOAD) 没追上

### 如果 ref check FAIL

`v_carry` 持有的 SMEM 槽位被 LOAD 提前覆盖了. 排查方向:
1. 检查 `o_handle.commit()` 时序是否真的覆盖了所有 cute.gemm
2. 检查 `kv_stage` 是否足够 (主循环 PV1 释放 v_carry 之前, LOAD 是否会
   acquire 到同一个 slot)

---

## 📋 TODO 状态镜像

| ID                | 状态        |
| ----------------- | ----------- |
| scaffold-pkg      | ✅           |
| host-tensor       | ✅           |
| host-ref          | ✅           |
| host-config       | ✅           |
| host-launcher     | ✅           |
| device-kernel-shell | ✅         |
| device-load       | ✅           |
| device-mma        | ✅           |
| device-softmax    | ✅           |
| device-correction | ✅           |
| device-epilogue   | ✅           |
| runner            | ✅           |
| regression-d128   | ✅           |
| d256-trace        | ✅           |
| **fix-v-release** | ✅ (今天完成) |
| **validate-d256** | 🟡 等 Blackwell 真机 |
