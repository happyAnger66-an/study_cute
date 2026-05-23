# FMHA 重构 + D=256 死锁修复 — 进度快照

> 最后更新: 2026-05-23 (Saturday) ~22:30 UTC+8
> 状态:
>   - **D=128**: 完全通过 (v3 修复 deadlock + 精度, prefill 3 轮 PASS, max_diff=0.0006)
>   - **D=256 outer-loop**: Round 1 **PASS** (max_diff=0.000612, 单 KV tile),
>     **Round 2 死锁** (multi KV tile, loop_steps≥1)。
>   - 诊断结论:
>     1. `dmesg` 没有 nvidia/xid/gpu fault, 排除硬件错误 (Thor 用 nvgpu 驱动框架)
>     2. `cuda-gdb` host 栈 stuck 在 `ioctl/cuMemcpyDtoHAsync_v2/cupy.ndarray.get()`,
>        `info cuda kernels` "No CUDA kernels" → kernel 没完成
>     3. single launch `--q_shape 1,128,8,256 --k_shape 1,256,8,256 --is_persistent
>        --skip_ref_check` 也死锁, 排除 multi-round/KV cache state 问题
>     4. 死锁仅发生在 D=256 (num_d_chunks=2) + multi KV tile (loop_steps≥1)
>        组合; D=128 multi tile 通过, D=256 single tile 通过
>   - 当前 fix experiment: 在 5 个 active warp body 的 d_outer iter 末尾 加
>     **NamedBarrier(barrier_id=3, num_threads=480)** 强制同步, 排除/确认
>     "跨 d_outer 缺 sync" 是否为 root cause
>   - debug 开关已改为 env 变量: `FMHA_DEBUG_PIPELINE=1` 启用 cute.printf
>     trace; 当前 LOAD/MMA 在每个 d_outer 边界 / main loop phase 都有 trace。
>
> D-chunking 设计演进:
>   - 原设计 (broken): PV inner d_chunk loop, 把多 chunk V 累加到同一个 tOtO0 →
>     d 维度被错误 reduce; epilogue 把同一份 sO copy 到所有 gO slice
>   - 备选: per-d_chunk PV+correction+epilogue (内层 d_chunk pipeline) →
>     **TMEM 不够** (1-CTA SM100 只 512 cols, num_d_chunks=2 partial O 需 256 cols
>     each × 2 = 512 cols, 加 S0/S1 = 768 cols, 超!)
>   - 新设计 (outer-loop, 已实施): d_chunk_outer 外层 loop 包整个 attention pipeline,
>     **QK 段保持 d_chunk_inner 累加 S** (S 必须跨完整 d), **PV 段退化为 D=128 单 V
>     模式** (只用 V[d_chunk_outer], 不累加跨 chunk 的 partial O), correction/epilogue
>     按 d_chunk_outer 写 sO + gO[d_chunk_outer]。TMEM 用量保持 512 cols 内。
>   - 性能代价: QK 和 softmax 重复算 num_d_chunks 次 (D=256 ~1.5x latency,
>     D=128 完全不变)。
>
> 修复历史 (D=128 主循环 4 段):
>   - v1: 错把 PV1 也 wait V → D=128 死锁 (废弃)
>   - v2: 用 `v_carry_list` 对 V handle "acquire-defer-release" → 死锁消除, 但
>         **合并了 QK0i/QK1i 段**, 让 `s1_handle` 在 PV1 段提前 commit, QK1i 写 S1
>         时 ownership 不在 mma 手里, softmax 读到没写完的 S1 算出错的 P1 →
>         D=128 精度爆掉 `max_diff=99.7` (废弃)
>   - v3 (D=128 path 仍保留): 严格按原版 4 段顺序 `QK0i → PV1(i-1) → QK1i → PV0i`,
>         s0/s1 跨段共享句柄, 修复 v2 的精度问题
>   - outer-loop (当前 D>128 path): 在 v3 基础上把整个 prologue+main+tail 包到
>     d_chunk_outer 外层, **PV 段去掉 v_carry_list** (单 V 模式), 5 个 warp 全部
>     按 d_chunk_outer 重复执行

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

**D-chunking (D=256) 扩展**: 历史上 v2/v3 给 PV 段加了 inner d_chunk loop +
`v_carry_list`, 试图让 PV 跨 d_chunk 累加 partial O。**但 TMEM 装不下** (见上),
而且 epilogue 把同一份 sO 复制到全部 d_chunk slice → 输出错误 (`actual ==
sum_of_ref_chunks`, 每个 chunk 输出一样, 见 docs/d_chunk_redesign.md)。

**当前 D>128 设计 (outer-loop)**: PV 段已**退化为 D=128 单 V 模式** (`tOrVi` 来自
outer scope, 无 inner d_chunk loop, 无 `v_carry_list`)。整个 attention pipeline
被外层 `for d_chunk_outer in range(num_d_chunks)` 包起来, 每次跑完整 attention
但**只用 V[d_chunk_outer]**, 写 `gO[..., d_chunk_outer, ...]`。QK 段保留 inner
d_chunk loop (因为 S 必须跨完整 d 维度累加)。

**代码变更 (D=128 v3 / D>128 outer-loop, 共用同一份代码 — num_d_chunks=1 时
退化到 v3 path)**: `fmha/device/warp_mma.py` 主循环结构:

```
for d_chunk_outer in range(num_d_chunks):    # const_expr, num_d_chunks=1 时单次
    Prologue:
        acquire s0/s1; for d_chunk_inner: gemm S0/S1 (累加跨 inner); commit s0/s1
        PV00: acquire o0 + new s0, wait 1 V (V[d_chunk_outer]), gemm O0
              (ACC = kphase!=0), commit o0, defer V release
    Main loop iter i:
        QK0i: per d_chunk_inner wait Q0+K (stash list), gemm S0 累加, commit s0
        PV1(i-1): acquire o1 + new s1, gemm O1 with tOrVi (V_{i-1}),
                  commit o1, release V_{i-1}
        QK1i: per d_chunk_inner wait Q1, 用 stash 的 K, gemm S1 累加, commit s1,
              统一释放 Q0/Q1/K
        PV0i: acquire o0 + new s0, wait 1 V (V_i), gemm O0 (ACC=True),
              commit o0, defer V release
    Tail PV1:
        acquire o1 + new s1, gemm O1 with V_{N-1}, commit o1, release V_{N-1},
        commit s0 + commit s1 (清理)
```

LOAD 端镜像: 外层 d_chunk_outer loop, 每个 outer iter 每 kv tile 产 num_d_chunks
个 Q0+K+Q1 + **1 个 V[d_chunk_outer]** (单 V)。

correction / softmax / epilogue: 各自加外层 d_chunk_outer loop, 内层逻辑不变。
correction final epilog 写 sO; epilogue 取 sO TMA store 到 gO[d_chunk_outer]。

### 关键时序约束

1. **`cute.gemm` 是 async**: 必须先 `o_handle.commit()` 才能
   `v_handle.release()`. 否则 tensor cores 还在读 V SMEM 时 LOAD 可能 refill
   该 slot.

2. **s1 跨 PV1 + QK1i 共享**: s1 ownership 期间 mma 既读 P1 (PV1 段) 又写新
   S1 (QK1i 段), 同一个 stage 复用 S1 buffer. PV1 提前 commit s1 (v2 bug)
   会让 QK1i 越权写, softmax 读到不一致的 S1.

3. **D-chunked S0/S1 跨 inner d_chunk 累加**: 跨 d_chunk_inner 在同一个 S0/S1
   stage 内累加 (S 取决于完整 d 维度), ACCUMULATE 标志按
   `d_chunk_inner!=0 or kphase!=0` 决定. **PV 段不累加跨 chunk** — 每个
   d_chunk_outer 跑完整 PV (V 只用一个 chunk), 写 sO + gO 各自不同 slice。
   PV1 段的 ACC 用 `pv_whether_acc` 决定 (覆盖 iter 0 第一次 PV1)。

4. **Python list 跨段流动 (q0_handles / k_handles)**: QK 段跨 d_chunk_inner 用
   Python list 存 `q0_handle` / `k_handle` 跨 QK0i / QK1i 段 (因为 K 在 QK0i
   和 QK1i 都要用)。Python list 是 compile-time 对象, list 中的 DSL 引用按 SSA
   流动。**`v_carry_list` 已移除** (v3 path 也不再需要, num_d_chunks=1 时
   carry list 退化为单元素, 直接用 `v_handle` 单变量等价)。

---

## 🔜 待办 — D=256 Round 2 死锁诊断

**当前现象** (Blackwell 真机上跑, 2026-05-23):

```
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent
... outer-loop mode, num_d_chunks=2 ...
--- Round 1/3 (pos=0, s_k=128, cap=384) ---
  batch 0: PASS  max_diff=0.000612   ← Round 1 完美通过!
--- Round 2/3 (pos=128, s_k=256, cap=384) ---
  挂死 (无输出)
```

Round 1 走 0 个 main loop iter (单 KV tile, prologue+tail), 通过.
Round 2 走 1 个 main loop iter (2 KV tiles), 挂死.

**已加诊断**:
- `host/config.py`: `debug_pipeline` 改为 env-gated, 通过 `FMHA_DEBUG_PIPELINE=1` 启用.
- `device/warp_mma.py`: main loop 每个 phase (QK0i / PV1 / QK1i / PV0i) 有
  d_outer + iter idx + KV stage idx printf, tail 完成也有 printf.
- `device/warp_load.py`: prologue + main iter 起头 / 每个 K/V acquire / d_outer
  结束都有 printf.

**下一步**: 在 Blackwell 上跑以下命令, 把 trace 输出贴回来定位卡点:

```bash
cd /path/to/study_cute
rm -rf fmha/__pycache__ fmha/host/__pycache__ fmha/device/__pycache__

FMHA_DEBUG_PIPELINE=1 FMHA_DEBUG_ROUNDS=2 \
  python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 \
  --is_persistent 2>&1 | tee /tmp/fmha_d256_trace.log
# 死锁后 Ctrl+C, 看 /tmp/fmha_d256_trace.log 的最后输出.
# 重点看 Round 2 期间最后一条 trace: LOAD 卡在哪个 K/V acquire? MMA 卡在 PV1/PV0i?
```

**其他诊断命令** (Blackwell 上仍可跑的回归):

```bash
# ① D=128 回归: 确认 outer-loop 路径在 num_d_chunks=1 退化时仍 PASS
python3 fmha/fmha.py --q_shape 1,256,8,128 --k_shape 1,256,8,128 --is_persistent
# 预期: 3 轮 prefill PASS, max_diff < 0.1

# ③ 如有需要, 单 KV tile / 多 KV tile + skip ref:
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 \
  --is_persistent --skip_ref_check
python3 fmha/fmha.py --q_shape 1,256,8,256 --k_shape 1,256,8,256 \
  --is_persistent --skip_ref_check
```

### 如果死锁

打开 trace 看 LOAD/MMA 在哪个 V slot 卡住:
```python
# fmha/host/config.py:
debug_pipeline = True
```
然后看 stderr 上的 `LOAD prologue d_outer=X kv=...` 和 `MMA prologue
d_outer=X trip=...` 序列在哪一步停下来。

### 如果 ref check FAIL

可以用 `FMHA_DEBUG_DCHUNK=1` 看每个 d_chunk_outer 的 actual vs ref:

```bash
FMHA_DEBUG_DCHUNK=1 FMHA_DEBUG_ROUNDS=1 python3 fmha/fmha.py \
  --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent
```

期望: 每个 d_chunk 的 actual 与 ref 相近 (max_diff < 0.1), 且不同 d_chunk 的
actual 应该**不同** (验证 V[d_chunk_outer] 切片正确生效)。

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
