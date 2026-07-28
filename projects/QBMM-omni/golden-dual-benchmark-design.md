# QBMM-omni 双标杆 golden 设计方案

> 范围：把 `projects/QBMM-omni/tests/golden.py` 从"合成地板 ratio 制 (new=True)"升级到华为精度标准的**双标杆 ratio 制 (L1)**。
> 本文件只出方案，不改 `golden.py`（用户批准后再分期实现，遵守 CLAUDE.md §2.5 小步迭代）。
>
> 已锁决策（2026-07-28）：
> 1. 精度等级 = **L1**（≥10000 例，ratio 5 / 1.5 / 1.5）
> 2. **双标杆**（精度主判据，2026-07-28 定）：
>    - Golden = 更高精度 CPU（int32 matmul + **fp64 scale**，升级现有 `golden_ref`）
>    - Benchmark = **CPU 小算子拼接**（`np.matmul` + scale + bias + cast，新建 `benchmark_ref`）—— NPU 侧 QBMM 是融合单算子（npu_quant_matmul）无法拆成独立 NPU op 序列，benchmark 只能走 CPU
>    - int8 matmul 三者一致（精确整数），精度差异在 scale/cast 域；golden 须 fp64 才使 benchmark_err>0（Ratio 不退化）
>    - `Ratio = NPU_err / max(benchmark_err, err)` —— 主精度判据
>    - 现有合成地板（`BM_CMP_STD` dbr_max/dbr_avg）+ flat isclose 降为**二次判断/参考列**（§3.5，不 gating）
>    - **确定性（MD5）与 batch 一致性是独立测试**（§8/§9），不混进精度 ratio
> 3. 范围 = 按标准要求（含复检 bootstrap CI）
> 4. **双轨判据**（2026-07-28 追加）：双标杆标准为**主判据**（决定 PASS/FAIL）；现有合成地板 ratio（`BM_CMP_STD` 的 `dbr_max/dbr_avg`）+ flat isclose `err_ratio` 作为**参考列**打印在 reason 里，**不 gating**。主/参考分歧即"特例信号"（对应标准 5.4 特例 CCB 候选）。详见 §3.5。
>
> 权威源：`D:\Desktop\TMP\log.txt`（标准全文，已被 `.claude/skills/ascendc-golden-testing/criteria.md` 提炼）、`D:\Desktop\TMP\tmp.txt`（复检详解）。

---

## 0. 现状盘点（golden.py 现覆盖的量化路径）

`golden_ref`（CPU 高精度 oracle，`np.matmul(int32)` 累加，scale/offset 后处理）现覆盖 7 条路径：

| mode | 路径语义 | out_dtype 现覆盖 | scale 形式 | 转置 oracle | bias |
|---|---|---|---|---|---|
| `pertensor` | T-C，全量单 scale | fp16/bf16/int8 | fp32 / int64(static) / bf16 | swap x1/x2 | float32 / int32 |
| `perchannel` | T-T，per-N scale | fp16/bf16/int8/int32 | fp32 / int64 / bf16 | swap x1/x2 | float32 / int32 |
| `pertoken` | K-C，per-M scale | fp16/bf16 | fp32 | swap | int32 |
| `requant` | int8 输出，s9 截断 + offset | int8 | fp32 | — | int32 |
| `int32` | raw matmul 不乘 scale | int32 | — | — | int32/float32 |
| `int4sym` | int4×int4 对称，per-N scale | fp16 | fp32 | — | — |
| `a8w4` | MSD：int8×int4，x2Scale[N]×x1Scale[M]+yOffset | fp16 | uint64(static) | — | — |
| `perblock` | G-B，per-tile [gM,gN,gK] block scale | bf16 | fp32 2D | swap x1/x2 + scale | float32 |

字段含义见 `golden.py:207-225` 的 `Case` 与各 `_compute_*`（line 134-185、469-492）。

输出 dtype 维度（决定用浮点 ratio 制还是整数 AE 制）：
- **浮点输出**（走 ratio 制 + 双标杆）：`float16` / `bfloat16`
- **整数输出**（走标准 4.4，**不涉 ratio**）：`int32`（BinaryMatch 精确）、`int8`（AE≤1）

> 关键约束：**int32/int8 输出 case 不需要 benchmark**。标准 4.4 是绝对误差，benchmark 无意义。仅 fp16/bf16 输出需要双标杆。

---

## 1. Benchmark 拼接设计（核心）

### 1.1 设计原则（CPU 拼接，2026-07-28 修正回 CPU）

**benchmark = CPU 用 np.matmul + scale + bias + cast 拼接 QBMM 等效计算。**

NPU 侧 QBMM 是融合单算子（`npu_quant_matmul` / `aclnnQuantMatmulV5` 把 matmul+scale+bias+cast 全融），**无法拆成独立 NPU op 序列**做标杆（唯一做这事的 NPU op 就是 DUT 自己，不独立）→ benchmark 只能走 CPU。

- **int8 matmul 在 CPU = 精确整数**：A1 验证（`tests/verify_a1_matmul_accum.py`，CPU torch.matmul fp32 累加）证实，对 int8 输入，fp32 matmul = 精确整数算术（int8 在 fp32 完全精确表示），等同 NPU 的 int32 L0C 累加。**matmul 部分三者（NPU/benchmark/golden）bitwise 一致**。
- **精度差异在 scale/cast**：matmul 之后 scale apply、bias add、fp16 cast 的**精度域差异**（NPU fixpipe 可能用 bf16 截断/特定舍入 vs CPU 干净 fp32）是 Ratio 的实际衡量对象。**确定性 ≠ 精度**——NPU 确定不代表精度好，scale/cast 的系统性误差仍可量。
- **golden 须 fp64 scale**：为使 benchmark_err>0（Ratio 不退化成 0/0），golden 用 fp64 scale（升级现有 `golden_ref`），benchmark 用 fp32 scale。三者关系：golden(fp64) 是真值，benchmark(fp32) 是"干净实现"代表，NPU 是 DUT。
- **不刻意做精 benchmark**：照"常规 CPU 实现"自然写（`np.matmul` fp32 + fp32 scale + fp16 cast），不做 fp64 中间域、不做 K-blocking。
- **round-to-dtype 原则保留**：benchmark 输入照 DUT 输入 dtype（int8 精确）。
- **不复用 `golden_ref` 的 `_compute_*`**：那是 oracle（fp64），benchmark 是独立 fp32 拼接。

### 1.2 通用拼接范式（CPU）

QBMM 本质 = `(int_x1 × int_x2) 累加 → × scale → + bias/offset → cast(out_dtype)`。
benchmark 在 CPU 上用 numpy/torch 拼接（差异在 scale 形式，见 §1.3）：

```python
def benchmark_ref(d, c):
    # 1) dequant int 输入到 out_dtype（这一步是 benchmark 的"量化反量化"语义对齐）
    #    注意：不是 dequant 到 fp32！要 dequant 到 out_dtype（fp16/bf16）。
    x1 = torch.from_numpy(d["x1"]).to(OUT_DT[c.out_dtype])      # int8 → fp16/bf16
    x2 = torch.from_numpy(d["x2"]).to(OUT_DT[c.out_dtype])

    # 2) 转置 case：和 oracle 同样先 swap（benchmark 也算的是转置 matmul）
    if c.trans_x1: x1 = x1.transpose(-1, -2)
    if c.trans_x2: x2 = x2.transpose(-1, -2)

    # 3) matmul 累加 in out_dtype（这一步的累加误差就是 benchmark_err 的来源）
    mm = torch.matmul(x1, x2)     # 默认在 out_dtype 累加（fp16→fp16, bf16→fp32 on some HW）
    # ↑ 关键：torch.matmul 在 fp16 输入下会用 fp16/fp32 混合累加（取决于 oneDNN/MPS 后端），
    #   为强制同精度，需要用 torch.matmul(...).to(out_dtype) 或显式分块累加。

    # 4) apply scale (路径特化，见 1.3)
    # 5) apply bias / offset
    # 6) cast to out_dtype
    return out
```

> **A1 已验证（非阻塞）**：CPU `torch.matmul` 走 fp32 累加（`tests/verify_a1_matmul_accum.py` 三 shape 实测）。**对 int8 输入，fp32 累加 = 精确整数算术**（int8 在 fp32 完全精确表示），等同 NPU 的 int32 L0C 累加 → matmul 部分不引入精度差异，benchmark 直接 `np.matmul(int8.to(fp32))` 即可，**无需 K-chunk**（C7 删）。
> 精度差异只在 scale/cast 阶段（§1.1），靠 golden 升 fp64 scale 提供 benchmark_err>0 的分母。

### 1.3 各路径 benchmark 实现（逐条对齐 NPU 行为）

| mode | benchmark 拼接（除通用骨架外的差异） | 备注 |
|---|---|---|
| `pertensor` | `out = mm * scale[0]`（标量）；scale_generate(int8+fp32 scale+no bias) 后处理也复刻：`scale = scale & 0xFFFFE000` | scale 形式见 `golden.py:108-118` `_needs_scale_generate` |
| `perchannel` | `out = mm * scale[N]`（broadcast 到 N 维）；scale_generate 同上条件触发 | int64/bf16 scale 各自路径 |
| `pertoken` | `out = mm * scale[N] * pertoken_scale[M]`（K-C 双 scale 广播） | bias 在 scale 后（is_bias_vec） |
| `int4sym` | x1/x2 是 int4 值（int8 持），benchmark 直接 `.to(out_dtype)` 后 matmul；scale=[N] perchannel | 与 perchannel 同骨架 |
| `a8w4` | `out = mm * x2Scale[N] * x1Scale[M] + yOffset[N]`（MSD 三段） | x2Scale=uint64 static → `_u64_to_deq_scale` 解码 |
| `perblock` (G-B) | **per-tile 累加**：按 groupSize=[gM,gN,gK] 切块，每块 `mm_tile × (x1Scale_tile × x2Scale_tile)` 累加 | 这是唯一必须分块 matmul 的路径（参考 `golden.py:469-492`） |
| `requant` (int8 out) | **不涉 benchmark**（int 输出走 AE≤1） | 见 §0 |
| `int32` out | **不涉 benchmark**（int 精确） | 见 §0 |

### 1.4 scale/bias/offset 的 apply 顺序（对齐 fixpipe）

照 oracle `_compute_*`（line 134-185）的顺序，但累加域换成 out_dtype：

- **pertensor/perchannel**：`mm(int32) → cast(out_dtype) → ×scale → +bias(if is_bias_vec)`，或 `+int32 bias` 在 cast 前
- **pertoken**：`mm → +int32 bias(前加) → cast(out_dtype) → ×scale ×pertoken_scale → +float bias(if is_bias_vec)`
- **perblock**：每 tile `mm_tile(out_dtype) → ×tile_scale → acc`；最后 `+bias`
- **a8w4**：`mm(out_dtype) → ×x2Scale ×x1Scale → +yOffset`

### 1.5 int 输出场景（标准 4.4，**不涉 ratio**）

- `int32` out：BinaryMatch 逐元素精确（现有 `INT_EXACT={"int32": True}` 不动）
- `int8` out（requant / perchannel_int8）：AE≤1，现有 `INT8_ATOL=1` 不动
- benchmark 可省略实现（`bench_metrics` 中走 `int_exact/int_atol` 分支）

---

## 2. Tolerance 表（L1）

### 2.1 正常值域 Ratio 阈值（标准 4.5，L1）

| 指标 | 定义 | L1 阈值 |
|---|---|---|
| MARE_ratio | `max(NPU_AE / (|golden|+1e-7)) / max(benchmark_AE / (|golden|+1e-7), err)` | ≤ **5** |
| MERE_ratio | `avg(同) / max(avg(同), err)` | ≤ **1.5** |
| RMSE_ratio | `sqrt(mean(NPU_AE²)) / max(sqrt(mean(benchmark_AE²)), err)` | ≤ **1.5** |

> 1e-7 防 golden 除零（标准 4.5 原文）。
> `err` 是分母地板，避免 benchmark_err→0 时 Ratio 爆炸。**不再用 `2^-mantissa`**，改用 §2.2 小值域表里的 `err`（标准 4.5 同源）。

### 2.2 小值域 threshold / err（标准 4.5.2，按 dtype）

| dtype | threshold | err |
|---|---|---|
| FLOAT16 | 2^-11 | 2^-16 |
| BFLOAT16 | 2^-8 | 2^-16 |
| FLOAT32 | 2^-14 | 2^-30 |

> 与现有 `BM_CMP_STD` 的 `small_value/small_value_atol`（fp16=1e-3/1e-5 等"工程数"）**不一致** —— 现有数值来自 ops-nn tolerance 表（非标准），方案要换成标准真值。

### 2.3 小值域通过门（标准 4.5.2）

```
ErrorCount_npu = Σ I( |golden|<threshold ∧ |NPU-golden|>err )
ErrorCount_bmk = Σ I( |golden|<threshold ∧ |benchmark-golden|>err )
通过: ErrorCount_npu / max(ErrorCount_bmk, 1) ≤ 2
```

> 替换现有 `gate_small=0`（合成地板要求 benchmark 零错，过严）。

---

## 3. Ratio 公式实现（替换现有 `bench_metrics` 的 `dbr_*`）

### 3.1 三指标分别算 ratio（关键修正）

现有 `bench_metrics`（line 578-636）算的是 `dbr_max/dbr_avg`（NPU 自身的相对误差，非 ratio）。改造为：

```python
def ratio_metrics(npu, golden, benchmark, out_dtype):
    # 全部拉到 cpu.float 做比对（避免 bf16 无原生 numpy）
    v = npu.reshape(-1).float().cpu()
    g = golden.reshape(-1).float().cpu()
    b = benchmark.reshape(-1).float().cpu()
    n = g.numel()

    # 整数 out: 不走 ratio
    if out_dtype in INT_EXACT: ...  # 保留现有 int_exact/int_atol 分支

    std = STD[out_dtype]            # 新表 (threshold/err from §2.2)
    eps = 1e-7
    denom_abs = (g.abs() + eps)

    # 正常值域 mask
    big = g.abs() >= std["threshold"]

    # 三指标分子分母 (只在 big mask 上统计)
    npu_ae  = (v - g).abs()
    bmk_ae  = (b - g).abs()
    npu_rel = npu_ae / denom_abs        # 逐元素 NPU 相对误差
    bmk_rel = bmk_ae / denom_abs

    # MARE: max(npu_rel[big]) / max(max(bmk_rel[big]), err)
    mare_npu = float(npu_rel[big].max()) if big.any() else 0.0
    mare_bmk = float(bmk_rel[big].max()) if big.any() else 0.0
    mare = mare_npu / max(mare_bmk, std["err"])

    # MERE: avg / max(avg, err)
    mere_npu = float(npu_rel[big].sum() / big.sum()) if big.any() else 0.0
    mere_bmk = float(bmk_rel[big].sum() / big.sum()) if big.any() else 0.0
    mere = mere_npu / max(mere_bmk, std["err"])

    # RMSE: sqrt(mean(ae²[big])) / max(sqrt(mean(bmk_ae²[big])), err)
    rmse_npu = float(torch.sqrt((npu_ae[big]**2).mean())) if big.any() else 0.0
    rmse_bmk = float(torch.sqrt((bmk_ae[big]**2).mean())) if big.any() else 0.0
    rmse = rmse_npu / max(rmse_bmk, std["err"])

    # 小值域 ErrorCount (3.2)
    small = ~big
    ec_npu = int(((npu_ae[small] > std["err"])).sum())
    ec_bmk = int(((bmk_ae[small] > std["err"])).sum())

    return dict(mare=mare, mere=mere, rmse=rmse,
                mare_npu=mare_npu, mare_bmk=mare_bmk,
                mere_npu=mere_npu, mere_bmk=mare_bmk,
                rmse_npu=rmse_npu, rmse_bmk=rmse_bmk,
                ec_npu=ec_npu, ec_bmk=ec_bmk, ec_ratio=ec_npu/max(ec_bmk,1))
```

### 3.2 小值域 ErrorCount ratio

照 §2.3 公式。`ec_ratio ≤ 2` 为通过。

### 3.3 verdict（替换 `bench_verdict` line 639-654）

```python
L1 = dict(mare=5.0, mere=1.5, rmse=1.5, ec=2.0)

def ratio_verdict(m, out_dtype):
    if out_dtype in INT_EXACT: ...        # 保留 int 分支
    rs = []
    if m["mare"] > L1["mare"]: rs.append(f"MARE {m['mare']:.3g}>{L1['mare']}")
    if m["mere"] > L1["mere"]: rs.append(f"MERE {m['mere']:.3g}>{L1['mere']}")
    if m["rmse"] > L1["rmse"]: rs.append(f"RMSE {m['rmse']:.3g}>{L1['rmse']}")
    if m["ec_ratio"] > L1["ec"]: rs.append(f"EC {m['ec_ratio']:.3g}>{L1['ec']}")
    return (len(rs)==0), rs
```

### 3.4 打印（保留诊断列）

- cos / worst 5 元素（line 586-599）保留
- 新增打印：`mare_npu/mare_bmk`、`mere_npu/mere_bmk`、`rmse_npu/rmse_bmk` 双侧绝对值（一眼看是不是 benchmark 太精导致 Ratio 失真）
- 余量百分比：`mare/L1['mare']*100%`（对齐 memory `test-output-verbose-margins`）

### 3.5 双轨判据（主标准 + 参考列，2026-07-28 追加）

**主判据 = 双标杆 `ratio_verdict`**（§3.3），决定 PASS/FAIL。
**参考列 = 现有方法并行算**（**不 gating**，只打印在 reason）：
- 合成地板 ratio：现有 `bench_metrics` 的 `dbr_max/dbr_avg`（`BM_FLOOR×rtol`，保留计算不参与 verdict）
- flat isclose `err_ratio`：`torch.isclose(rtol=atol=tol)` 的不匹配占比（tol fp16/fp32=1e-3、bf16=5e-3，memory `matmul-golden-tol-flat-001` 的 fallback 判据）

**实现**：`run()` 里 `ratio_metrics`（主）+ 现有 `bench_metrics`（参考）并算，verdict 只取 `ratio_verdict`。

**一致性标记**（自动识别特例，对应标准 5.4）：

| 主标准 | 参考列 | 标记 | 处理 |
|---|---|---|---|
| PASS | PASS | 一致 PASS | 稳过 |
| FAIL | FAIL | 一致 FAIL | 真 bug，修算子 |
| FAIL | PASS | **分歧（标准更严）** | 特例候选 → 复检 → 仍 FAIL 则进 CCB 建档（§4.5） |
| PASS | FAIL | 分歧（参考过紧） | 参考方法已知 floor 问题（fp32 累加），可忽略 |

**价值**：切 L1 后若大批 case 新 FAIL（§6.3 回归风险），参考列能秒分"标准收紧致 FAIL"（参考 PASS）vs"真精度回归"（参考也 FAIL），不用逐条人工 trace。

---

## 4. 复检机制（bootstrap CI）

### 4.1 触发条件

单例 `ratio_verdict` 返回 FAIL 时自动启动复检（标准 4.5.3 + `tmp.txt`）。

### 4.2 算法

```python
def recheck_bootstrap(case_factory, c, N=1000, B=2000, seed_offset=0):
    ratios = []     # 收集每次的 (mare, mere, rmse) 三元
    for i in range(N):
        d = case_factory(c, seed=c.seed + seed_offset + i if c.seed else None)
        # ↑ 换种子重新 gen_data, 重新算 golden/benchmark/npu
        g = golden_ref(d, c)
        b = benchmark_ref(d, c)
        out = call_npu(d, c)
        m = ratio_metrics(out, g, b, c.out_dtype)
        ratios.append((m["mare"], m["mere"], m["rmse"]))

    if N < 200:
        return "FUSE", "N<200 熔断, 直接判失败"

    # bootstrap: 对每个指标各做 B 次有放回重采样, 取中位数
    results = {}
    for idx, name in enumerate(["mare", "mere", "rmse"]):
        col = np.array([r[idx] for r in ratios])
        boot_medians = np.median(col[ np.random.randint(0, N, size=(B, N)) ], axis=1)
        ci_lo = np.percentile(boot_medians, 2.5)   # 第 25 小 (B=2000)
        ci_hi = np.percentile(boot_medians, 97.5)
        results[name] = (ci_lo, ci_hi)
        if ci_lo > 1.0:
            return "FAIL", f"{name} CI_Lower={ci_lo:.3g}>1.0 (系统性恶化)"
    return "PASS", results
```

> 关键：CI 基于中位数（非均值），抗离群；判定只看 `CI_Lower > 1.0`。
> 用 `L1` 阈值的 ratio（mare≤5 等）作为单次 ratio 的"通过"，bootstrap CI 检验的是"中位数 ratio 是否系统性 >1.0"。

### 4.3 与现有单次执行的衔接

现有 `run(c, npu, v)`（line 782-845）是"gen → golden → call_npu → bench_metrics → verdict"单次流水线。改造：

1. 单次跑完 `ratio_verdict` 返回 PASS → 直接结案
2. 单次 FAIL → 调 `recheck_bootstrap`（需要传一个 `case_factory` 闭包，能换种子重 gen）
3. 复检 PASS → 最终结 PASS，但打印 `单次 FAIL → 复检 PASS (CI=[lo,hi])`
4. 复检 FAIL / FUSE → 最终结 FAIL

> **NPU 调用代价**：1000 次复检 = 1000 次 NPU 调用。M=N=K=128 case 约 0.1ms/次 → 0.1s，可接受；大 shape（perblock 512×512×512）需评估，必要时降 N=200（仍 >熔断线）或仅对 Mare/Mere/Rmse 中破例最大的一个做复检。

### 4.4 复检不适用于 int 输出

`int32/int8` 走绝对误差，无 ratio 概念，单次不通过直接 FAIL（无系统性恶化的统计空间，因为是二进制/截断确定性）。

### 4.5 失败用例处理（标准 5.4 + tmp.txt 诊断，2026-07-28 追加）

标准对确认 FAIL（复检仍 FAIL）**没有代码层 suppress**（不像 cleancode）—— 确认 FAIL = 要么修算子（真 bug），要么走 **CCB 会议裁决 + 建档存册**（标准 5.4 特例）。golden 脚本职责：如实报告 + 诊断 + 触发复检 + 输出特例建档素材，**不做 suppress / 不标 known-bad**。

**FAIL 时每条 case 输出**：
1. **主标准指标**：MARE/MERE/RMSE ratio + 小值域 ErrorCount ratio（§3）
2. **参考列指标**：合成地板 dbr_max/dbr_avg + flat isclose err_ratio（§3.5）
3. **一致性标记**：一致 FAIL / 分歧类型（§3.5 表）
4. **Top-K Worst Samples**（照 tmp.txt，top 3 by Ratio，已含在 §3.4 worst 诊断）：每条带 位置 / golden 值 / NPU 值 / benchmark 值 / 该点 Ratio（比 tmp.txt 的行号+Ratio 更详细，方便 CCB 取证）
5. **复检结果**（若触发）：N / 中位数 / 95%CI / 判定

**特例建档模板**（确认 FAIL + 主参考分歧时输出，供 CCB 用）：

    特例候选:
      case: <name> (mode=<mode>, out_dtype=<dt>, shape=<M,N,K>)
      主标准: MARE=<x> MERE=<x> RMSE=<x> EC_ratio=<x> → FAIL
      参考列: dbr_max=<x> dbr_avg=<x> err_ratio=<x> → PASS/FAIL
      分歧类型: 标准更严 / 参考过紧
      Top worst: [位置 / Ratio ...]
      可能原因: <待开发者/专家填>

**不做**：不在 golden 里 suppress / 跳过 / 标 known-bad。所有 FAIL 如实上报；特例判定走 CCB 流程（人），golden 只产建档素材。

---

## 5. 改动点清单（分期能力）

### 现状 5 个缺口（用户已识别）

| # | 缺口 | 现状 | 方案目标 |
|---|---|---|---|
| G1 | 无真 GPU/CPU 同精度 benchmark 标杆 | `BM_FLOOR=2^-mantissa` 合成地板 | 新建 `benchmark_ref` 函数（§1） |
| G2 | 未定精度等级 | tolerance 表来自 ops-nn（非标准） | 锁 L1，ratio 阈值 5/1.5/1.5（§2） |
| G3 | RMSE 未算 | `BM_CMP_STD` 有 `rmse_rtol` 字段但 `bench_metrics` 没算 rmse | `ratio_metrics` 三指标全算（§3.1） |
| G4 | 小值域数值不符 | `small_value=1e-3` 等工程数 | 换标准真值 2^-11/2^-8/2^-14（§2.2） |
| G5 | `gate_small=0` + 无复检 | 小值零错 + 单次 FAIL 即 FAIL | ErrorCount ratio + bootstrap CI（§2.3、§4） |

### 一期（核心能力，落地即可对齐标准）

- **C1** 新建 `benchmark_ref(d, c)`（§1）：覆盖 6 条浮点输出路径（pertensor/perchannel/pertoken/int4sym/a8w4/perblock），**CPU 上** `np.matmul(int8→fp32) + fp32 scale + bias + cast` 拼接（int8→fp32 精确整数累加，等价 NPU int32）
- **C2** 新建 `STD_L1` tolerance 表（§2.1、§2.2）+ `ratio_metrics`（§3.1）+ `ratio_verdict`（§3.3，**主判据**）；**保留 `bench_metrics` 作参考列**（§3.5，不 gating），仅 `bench_verdict` 被 `ratio_verdict` 替换
- **C3** `run()` 流水线：`golden → benchmark → npu → ratio_metrics(主) + bench_metrics(参考) → ratio_verdict`
- **C4** 打印保留 cos/worst + 新增双侧绝对值 + 余量（§3.4）+ **参考列 dbr/err_ratio + 一致性标记**（§3.5）+ FAIL 时 Top-K worst + 特例建档模板（§4.5）
- **C5** int 输出（int32/int8）保留现有 `INT_EXACT/INT8_ATOL` 分支不变

> 一期完成即可 claim "对齐标准 4.5 L1（无复检）"。

### 二期（复检 + 工程化）

- **C6** `recheck_bootstrap`（§4）：单次 FAIL 时换种子跑 N=1000、bootstrap B=2000、CI_Lower>1.0 判定
- **C7** （已删：原 CPU fp16 累加 K-chunk，benchmark 改 NPU 后 L0C 同累加域，无需）
- **C8** 用例规模扩到 ≥10000（标准 L1 要求）：现有 CASES 约 50 个，需参数化生成（M/N/K 各 10 档 × 各 mode）或种子扫描
- **C9** per-tile perblock 的 benchmark K-chunk 与 oracle 的 `_compute_per_tile_int8` 对齐验证（转置 case scale swap）
- **C10** 离群点分布输入（标准 L1 要求 20% 用例含 0.1%×1000 离群）—— gen_data 现只有 `integers(-5,5)`/`standard_normal*0.05`，需扩分布

### 不做（明确 scope out）

- 不改 host/tiling/kernel 生产代码
- 不改 binding（benchmark 是 CPU 侧 PyTorch，不经 NPU）
- 不实现 GPU 标杆（用户锁的是"同精度小算子拼接"，非竞品 GPU）

---

## 6. 风险与未决假设

### 6.1 假设状态（2026-07-28 更新，benchmark 回 CPU）

- **A1（已验证·非阻塞）**：CPU `torch.matmul` fp32 累加（`verify_a1_matmul_accum.py` 三 shape 实测确认）。对 int8 输入，fp32 = 精确整数，等同 NPU int32 → matmul 部分无精度差异，benchmark 直接用，无需 K-chunk（C7 删）。
- **A2 scale apply 域**：benchmark scale 用 fp32，golden 升 fp64。验证：NPU_err 与 benchmark_err 应同数量级（Ratio≈1 波动）；若 NPU fixpipe 用 bf16 截断 scale → Ratio>1（双标杆可检出）。
- **A3 perblock G-B 分块**：oracle fp32 累加 per-tile，benchmark 同样 per-tile（fp32），tile 内 matmul + tile scale 顺序对齐 fixpipe。
- **A4（已废弃）**：原 NPU int8 matmul 路径假设。benchmark 回 CPU 后，NPU 侧不参与 benchmark，此假设无关。

### 6.2 已知语义陷阱（保留现有处理）

- **转置 oracle swap**（line 429-433、477-482）：benchmark 也必须照做，否则比的是不同计算
- **scale_generate 截断**（int8 + fp32 pertensor scale + no bias，line 108-118）：benchmark 也要复刻 `& 0xFFFFE000`，否则 scale 精度不一致
- **bf16 scale 截断**（`_fp32_to_bf16_sim`，line 85-90）：benchmark 的 scale 也要截断
- **u64 static scale 解码**（`_u64_to_deq_scale`）：a8w4/perchannel_static benchmark 也要解码
- **int4 pack**：benchmark 用 logical int4 值（int8 持），不走 pack（pack 是 NPU 输入格式）

### 6.3 tolerance 风险

- 现有 `BM_CMP_STD` 的 `max_re_rtol=10/5` 等"ops-nn 工程值"与标准 L1 的 `MARE≤5` 在 pertoken/perchannel 上**可能更严**。切换后部分原本 PASS 的 case 可能变 FAIL —— 这是预期（标准本来就是真值），但要预先在已有 NPU 结果上回归跑一遍，确认不是 oracle/benchmark bug 而是真的精度回归。

---

## 7. 文件落点

- 本方案：`projects/QBMM-omni/golden-dual-benchmark-design.md`（本文件）
- 实现（一期）：改 `projects/QBMM-omni/tests/golden.py`，新增 `benchmark_ref` / `ratio_metrics` / `ratio_verdict` / `recheck_bootstrap`，保留 `golden_ref` / `INT_EXACT` / `cos/worst` 诊断
- 不另起脚本（用户要求改 golden.py，不另写）

---

## 附 A：确定性测试（MD5，独立于精度 ratio）

**维度分离（2026-07-28）**：确定性 ≠ 精度 ≠ batch 一致性，三个独立测试，不混。
- **确定性**：同输入多次跑，输出是否完全一致（MD5/bitwise）—— 跟精度无关
- **精度**：输出 vs golden 的误差大小（MARE/MERE/RMSE ratio，§3）—— 双标杆
- **batch 一致性**：跨 batch 元素精度是否均匀（本节附 B）

标准第 3 节："支持确定性的算子，单用例多次执行结果须一致"。

- 同输入跑 N≥2 次（推荐 5），NPU 输出 bitwise / MD5 比对
- **通过**：所有次输出完全一致（MD5 相同）
- **不通过**：run-to-run 差异 → 算子有非确定行为，独立报告（不混进精度 ratio）
- 与精度 ratio **完全分开**：确定性是 md5 对比，精度是误差量

## 附 B：batch 一致性测试（独立于精度 ratio 与确定性）

跨 batch 元素的精度误差是否均匀（batch_invariant，memory `mm-determinism-batch-invariant-pr`）。

- 构造多 batch 用例（batch > 1），算每个 batch 元素的 MARE/MERE
- **通过**：各 batch 元素误差同量级（无明显离群 batch）
- **不通过**：某 batch 元素误差显著大于其他 → batch 维度精度非一致
- 与精度 ratio / 确定性 **分开**：这是 batch 维度均匀性，不是整体精度也不是 run-to-run
