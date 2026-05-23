# FMHA 重构 + D=256 死锁修复 — 进度快照

> **🆕 D=256 新路径 (推荐, 不在本目录)**: 已迁移 CUTLASS 官方 d=256 实现到
> `study_cute/fmha_d256/`, 用 2-CTA cluster + iterations_pv=2 (同 MMA session
> 内多 TMEM ACC slot) 设计, 性能 / 健壮性都优于本目录的 outer-loop 设计。
> 见 `fmha_d256/README.md`。本目录 (`fmha/`) 现定位为"学习 D≤128 outer-loop /
> pipeline 设计"的参考。
>
> 最后更新: 2026-05-23 (Saturday) ~23:10 UTC+8
> 状态:
>   - **D=128**: 完全通过 (v3 修复 deadlock + 精度, prefill 3 轮 PASS, max_diff=0.0006)
>   - **D=256 outer-loop**: Round 1 **PASS** (max_diff=0.000612, 单 KV tile),
>     **Round 2 死锁** (multi KV tile, loop_steps≥1)。
>   - 今日诊断结论 (已确认):
>     1. `dmesg` 没有 nvidia/xid/gpu fault, 排除硬件错误
>        (Thor 用 nvgpu 驱动框架, 但 hard fault 仍会进 dmesg)
>     2. `cuda-gdb` host 栈 stuck 在 `ioctl/cuMemcpyDtoHAsync_v2/cupy.ndarray.get()`,
>        `info cuda kernels` 显示 "No CUDA kernels" → **kernel 完全没完成**
>     3. single launch `--q_shape 1,128,8,256 --k_shape 1,256,8,256 --is_persistent
>        --skip_ref_check` 也 hang/被 OOM-killer killed, 排除 multi-round / KV cache
>        reuse / cumulative seqlen state 问题
>     4. **死锁仅发生在 D=256 (num_d_chunks=2) + multi KV tile (loop_steps≥1)** 组合:
>        - D=128 + single tile ✓ | D=128 + multi tile ✓
>        - D=256 + single tile ✓ | D=256 + multi tile ✗
>   - **今日已 falsify 的 hypothesis**:
>     - **(H1) 跨 d_outer 缺 CTA sync** — 加 `NamedBarrier(barrier_id=3,
>       num_threads=480)` 在 5 个 active warp body 的 d_outer iter 末尾,
>       D=128 回归仍 PASS, **D=256 Round 2 仍 hang** → falsified, 已回退
>   - debug 开关: `FMHA_DEBUG_PIPELINE=1` 启用 cute.printf trace; 但 kernel hang 时
>     printf 缓冲不 flush, 无输出。
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

## 🔜 待办 — D=256 Round 2 死锁诊断 (暂停, 下次继续)

### 当前现象 (Blackwell Thor 真机, 2026-05-23 22:00)

```
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent
... outer-loop mode, num_d_chunks=2 ...
--- Round 1/3 (pos=0, s_k=128, cap=384) ---
  batch 0: PASS  max_diff=0.000612   ← Round 1 完美通过!
--- Round 2/3 (pos=128, s_k=256, cap=384) ---
  挂死 (无输出, ctrl+c 无响应)
```

Round 1 走 **0** 个 main loop iter (单 KV tile, prologue+tail), 通过。
Round 2 走 **1** 个 main loop iter (2 KV tiles), 挂死。

**对照试验**:
| Config                              | num_d_chunks | loop_steps | 结果 |
| ----------------------------------- | ------------ | ---------- | --- |
| D=128, s_k=128                      | 1            | 0          | ✓   |
| D=128, s_k=256+ (multi-round R2/R3) | 1            | ≥1         | ✓   |
| D=256, s_k=128 (Round 1)            | 2            | 0          | ✓   |
| D=256, s_k=256+ (Round 2)           | 2            | ≥1         | ✗ hang |

→ **死锁唯一条件**: `num_d_chunks > 1` **且** `loop_steps ≥ 1`。

### 已 falsify 的 hypothesis

#### (H1) 跨 d_outer 缺 CTA-wide sync — falsified (2026-05-23)

**假设**: outer-loop 把 5 个 active warp 各自跨 `for d_chunk_outer` 推进, 但
const_expr unroll 时 pipeline state 跨 d_outer iter carry 不一致, 需要在
d_outer 边界强制全 warp 对齐。

**实验**: 在 `host/config.py` 加 `d_outer_sync_barrier = NamedBarrier(
barrier_id=3, num_threads=480)` (15 active warp × 32 thread, 排除 empty warp);
在 5 个 warp body (load/mma/softmax×2/correction/epilogue) 的 `for d_chunk_outer`
循环末尾加 `arrive_and_wait()`, 用 `const_expr(self.num_d_chunks > 1)` gate。

**结果**:
- 测试 1 D=128 回归: 3 轮 PASS, max_diff=0.0006 ✓ (确认 barrier 不破坏 D=128)
- 测试 2 D=256 single launch (s_k=256): Compilation 完成后被 OOM-killer
  `Killed` (host 端某次 alloc OOM, 跟死锁无直接关系)
- 测试 3 D=256 multi-round Round 2: **仍 hang** ✗

**结论**: cross-d_outer sync 不是 root cause。代码已回退 (commit 待打)。

### 候选 hypothesis (下次继续优先排)

#### (H2) PV `pv_whether_acc` 跨 d_outer 没正确 reset / carry — 重点怀疑

`pv_whether_acc` 是 Python bool, 在 cutlass.range body 内被设 True 后 yield 回
outer scope。每个 d_outer iter 起头我们 reset `pv_whether_acc = False`,
但 **tail PV1 在 cutlass.range 出来后用的是 yield 出来的值** (True),
这是对的 (tail PV1 应该 acc)。但 d_outer=1 的 PV00 起头 reset 为 False 时,
Python 局部变量 `pv_whether_acc` 在 const_expr unroll 下是不是被正确处理?

**验证方法**: 在 mma_warp_body 的 `pv_whether_acc = False` 重置点前后, 加
`cute.printf("[d_outer=%d reset]", d_chunk_outer)` 看是否真的被执行。
或直接把 `pv_whether_acc` 改为通过 `cutlass.const_expr` 包装的形式。

#### (H3) `v_handle.release()` 跨 d_outer 漏一个或多一个

每个 d_outer iter LOAD 产 (1 + loop_steps) 个 V; MMA 在 PV00 acquire 1 个 V
(carry), 每个 main iter 内 PV1 release 旧 V + PV0i acquire 新 V, tail PV1
release 最后一个 V。每 d_outer 共 acquire = (1 + loop_steps), release =
(1 + loop_steps), 平衡。**但跨 d_outer 时, kv_pipeline 的 producer/consumer
state.phase 是否在两端都同步 advance 了**?

LOAD per d_outer: 产 num_d_chunks 个 K + 1 个 V (prologue) +
loop_steps×(num_d_chunks K + 1 V) = num_d_chunks×(1+loop_steps) K +
(1+loop_steps) V。
MMA per d_outer wait: 对称数量。
理论上 producer/consumer state.phase 应该一致。

**验证方法**: 检查 `cute.printf` trace 在 LOAD/MMA 上的 K/V index, 看是否在
d_outer=1 起头第一个 acquire 上卡住。
**但 hang 时 printf 看不到 → 需要换写 GMEM counter 的方案** (见下)。

#### (H4) `s1_handle` 跨 d_outer 时, tail 的 commit + d_outer=1 起头的 acquire 冲突

tail PV1 末尾 `s0.commit(); s1.commit();` 把两个 s 还给 softmax。
d_outer=1 起头 prologue acquire 新 s0/s1。这跨 d_outer 边界, 中间没有 mbar
sync (除非 softmax 已 release)。如果 softmax 还没 release 上一轮的 s, 就 hang。

**但 softmax 也有 d_outer loop**, 每 d_outer iter 它的 release 数应该匹配
MMA 的 commit 数。除非 softmax 内的 d_outer loop 逻辑跟 MMA 不一致。

**验证方法**: 跑 softmax warp 不带 d_outer loop (只跑一次) 看是否 hang
减少, 但这会破坏正确性, 仅用于定位。

#### (H5) `tOrVi` (V SMEM view) 跨 d_outer 没重新创建, 用了 stale view

`tVgV_dkl[None, d_chunk_outer, ...]` 在每个 d_outer iter 起头重新切片, 但
TMA copy 用的是 `tma_atom_v` (固定 descriptor)。如果 tma_atom_v 是按
`pv_mma_tiler` (含 d=128) 创建的, 它处理任意 d_chunk_outer 偏移应该 OK
(由 cute.copy 的 coord 参数决定)。但如果某处缓存了 d_outer=0 的 tOrVi 给
d_outer=1 用就会错。

### 下次继续的诊断 checklist

#### 必做: 换"写 GMEM 计数器"代替 printf

当前 `cute.printf` 在 kernel hang 时不 flush, 全无输出。改成:

1. 在 `device/kernel.py` 起头 alloc 一个 device GMEM array (e.g., 64 × Int32),
   每个 warp 一个 slot 记录最后到达的 phase index。
2. 每个 warp body 的 d_outer/main loop 关键点, 用 `cute.copy` 或直接
   `st.global` 写当前 (d_outer, iter, phase) 到自己的 slot。
3. Host 端 launch 时不调 `.get()` 阻塞, 而是用 `cupy.cuda.Stream` polling +
   timeout, 死锁 N 秒后强制 read GMEM array, 看每个 warp 卡在哪。

或者更简单: kernel launch 一个 grid 后, **不等 stream sync**, 直接 background
sleep + read GMEM counter (绕过 cuMemcpyDtoHAsync 的隐式 sync)。

#### 备选: cuda-gdb 看 GPU warps

用户已试过 `cuda-gdb`, 但 `info cuda kernels` 显示 "No CUDA kernels"。这意味着
Ctrl+C 之前 driver 已经撤销了 kernel。换 attach 已 hang 的 Python (在另一终端):

```bash
# 终端 1
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent
# 死锁后保持

# 终端 2
sudo cuda-gdb -p $(pidof python3)
(cuda-gdb) set pagination off
(cuda-gdb) info cuda kernels     # 看是否能看到 launched kernel
(cuda-gdb) info cuda warps
(cuda-gdb) cuda warp 0
(cuda-gdb) x/10i $pc-20          # 看 SASS, 找 BARRIER.* 或 WAIT
```

#### 备选: 对照原版 fmha (TensorRT-Edge-LLM/.../fmha.py)

原版 `TensorRT-Edge-LLM/kernelSrcs/fmha_cutedsl_blackwell/fmha.py` 也是 D 维度
不分 chunk (只支持 d ≤ 128 直接 single chunk)。**study_cute 的 d_outer loop
是新增逻辑, 原版没有参考**。

但原版有 d=256 实现 (mma_tiler_k=256 直接编)? 不, mma_tiler_k 受 tcgen05 限制
只能 128。原版要支持 d=256 也必须 chunking, 但**原版可能没真的跑过 d=256**。

**应优先**: 在原版 fmha 仓库里搜 `num_d_chunks` / `d_chunk_k` / 类似 D-chunking
逻辑, 看是否有 reference impl。如果没有, 我们就是在做原创设计, debug 难度大。

#### 备选: 用更小 shape 复现 / 简化

试 `--q_shape 1,128,2,256 --k_shape 1,256,2,256 --skip_ref_check` (head_q=2
缩小 grid), 看是否更易 attach / debug。

### 当前代码状态 (本次会话结束时)

- D=128 path (v3) ✓ 完整工作
- D=256 outer-loop path: 单 KV tile (Round 1 / single launch s_k=128) ✓ 工作
- D=256 + multi KV tile: ✗ hang (root cause 未定位)
- H1 (cross-d_outer NamedBarrier) 实验代码已**回退**, repo 干净
- printf trace 代码保留 (env gate `FMHA_DEBUG_PIPELINE=1`), 但 hang 时不 flush

### 复现命令 (下次起手)

```bash
cd /codes/codes/study_cute
rm -rf fmha/__pycache__ fmha/host/__pycache__ fmha/device/__pycache__

# D=128 回归 (~1.1ms, 应 PASS):
python3 fmha/fmha.py --q_shape 1,256,8,128 --k_shape 1,256,8,128 --is_persistent

# D=256 multi-round (Round 2 hang):
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,128,8,256 --is_persistent

# D=256 single launch multi KV tile (也 hang / OOM-killed):
python3 fmha/fmha.py --q_shape 1,128,8,256 --k_shape 1,256,8,256 \
  --is_persistent --skip_ref_check
```

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
| fix-v-release     | ✅           |
| d256-outer-loop-impl | ✅        |
| h1-cross-d_outer-barrier | ❌ falsified, 已回退 |
| **validate-d256** | 🟡 暂停 — Round 2 hang root cause 未定位 |
