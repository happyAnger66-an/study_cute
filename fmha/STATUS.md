# FMHA 重构 + D=256 死锁修复 — 进度快照

> 最后更新: 2026-05-22 (Friday) ~22:30 UTC+8
> 状态: **Fix A v2 重写**完成, D=128/D=256 都编译通过, 待 Blackwell 真机验证
>
> 重要更正: 第一版 Fix A 完全搞错了原版本的 V handle 模式, 导致 D=128 也死锁.
> 第二版按原版本"acquire-defer-release"对偶模式扩展, 用 v_carry_list 在 PV 段间
> 传递 (v_handle, tOrVi) 元组, 同时修复 D=256 计算错误.

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

### Fix A — V-handle release 平衡 (Day 2, 今天)

在 `fmha/device/warp_mma.py` 应用了 4 处修改, 把 V handle 的 wait/release
计数完全配平, 消除 D=256 的死锁根因. 每处都附了详细注释:

- **PV00 (prologue)** L185-L212: 用 Python list `v_handles_pv00` 暂存所有
  d_chunk 的 v_handle, `o0_handle.commit()` 后释放前 `num_d_chunks - 1` 个,
  最后一个作为 `v_carry` 跨给主循环.
- **PV1(i-1) (主循环)** L273-L302: 先 `v_carry.release()` (释放上轮 PV0i 的最后
  defer 或 prologue 的 defer), 然后 gather + commit + 全部释放.
- **PV0i (主循环)** L309-L332: gather + commit 后释放前 N-1 个, 最后一个作为新
  `v_carry`.
- **PV1 final (尾声)** L337-L361: 把原 `v_handle.release()` 改成
  `v_carry.release()`, 释放最后一次 PV0i 的尾巴 (或 loop_steps=0 时的 prologue 尾巴).

### 死锁机理(回顾)

旧代码每 KV iter V 计数失衡:
| 段落 | waits | releases (legacy) | leak |
| --- | --- | --- | --- |
| PV00 prologue | `num_d_chunks` | `0` | `num_d_chunks` |
| PV1(i-1) | `num_d_chunks` | `1` (循环外) | `num_d_chunks - 1` |
| PV0i | `num_d_chunks` | `0` | `num_d_chunks` |
| PV1 final | `0` | `1` | `-1` |

- **D=128** (`num_d_chunks=1`): 总 leak ≈ 1, `kv_stage=3` 能撑住 → **侥幸跑通**
- **D=256** (`num_d_chunks=2`): 单 iter leak `3`, 超 `kv_stage=3` → **LOAD acquire 死锁**

Fix A 后, per iter V waits = `2 * num_d_chunks` = V releases (1 carry + N PV1 + (N-1) PV0).
全局零泄漏: 整个 kernel `waits = D + 2D*N == releases`. ✓

### 关键时序约束

`cute.gemm` 是 async — Python 调用返回时 tensor cores 还在读 V SMEM. 必须
先 `o_handle.commit()` (隐含等待 tensor cores 完成), 然后才 `v_handle.release()`.
所以修复用 Python list 暂存所有 d_chunk 的 v_handle, commit 后才统一 release.
list 本身是 Python compile-time 对象 (`num_d_chunks` 是 const_expr), 不会
进入 DSL 动态域.

---

## 🔜 待办 — 在 Blackwell 真机上跑

本机 sm_89/Ada 跑不了 Blackwell tcgen05 kernel, 以下命令需在 Blackwell GPU 上跑:

```bash
cd /path/to/study_cute
rm -rf fmha/__pycache__ fmha/host/__pycache__ fmha/device/__pycache__

# ① D=128 回归 (确保 fix 不破坏原本就跑通的路径)
python3 fmha/fmha.py --q_shape 1,256,8,128 --k_shape 1,256,8,128 --is_persistent
# 预期: 内置 3 轮 prefill ref check 全部 PASS

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
