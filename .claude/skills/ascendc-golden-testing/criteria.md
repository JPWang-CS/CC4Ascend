# Golden 判据

## 权威源

**华为算子精度标准**（`D:\Desktop\TMP\log.txt`，767 行 HTML，wiki 抓取）。本文件提炼其判据体系。
旧辅助门（cos/rel_l2/2^-mantissa）已弃；flat isclose 降为"无 GPU 标杆时的简化 fallback"（见末节）。

## 第一步：算子分类（决定用哪套判据）

标准按计算特性分 4 类，先定位算子属哪类：

| 类 | 例子 | 判据 |
|---|---|---|
| 非计算类（搬移/Cast） | transpose/gather/cast | 二进制一致（Bitwise Match） |
| 整数计算（INT8/16/32） | int matmul | 二进制一致；AE=0 也通过 |
| 量化计算（FP4/FP8/INT8） | **QBMM / quant_matmul** | 整型输出+浮点输入→AE≤1；浮点输出→参考浮点标准 |
| 浮点计算（FP16/FP32 等） | matmul/softmax | MARE/MERE/RMSE ratio（见下） |

## 浮点算子判据（标准 4.5）

### 误差度量

```
AE   = abs(actual - golden)
MARE = max( abs(actual-golden) / (abs(golden)+1e-7) )   # 最大相对误差
MERE = avg( 同 )                                         # 平均相对误差
RMSE = sqrt( mean( (actual-golden)² ) )                 # 均方根误差
```
1e-7 防 golden 除零。

### 比对方法

- **双标杆**（默认）：CPU golden + GPU 标杆。`Ratio = NPU_err / max(GPU_err, err)`
- **单标杆**：单一标杆（CPU/GPU/小算子拼接）直接比
- 精度标杆构造顺序（标准 5.3.1）：竞品对标（业界 CPU/第三方同等算子）→ 小算子拼接（融合算子/量化反量化串联）→ 自行构造 CPU（非标准 dtype 如 HiFLOAT8）
- 自定义/非标准算子无竞品时，用小算子拼接或自构 CPU

### 正常值域通过标准（Ratio 阈值 ×1e-6）

| 等级 | MARE_ratio | MERE_ratio | RMSE_ratio | 用例规模 |
|---|---|---|---|---|
| L0 常规 | ≤10 | ≤2.0 | ≤2.0 | ≥5000 |
| L1 重要 | ≤5 | ≤1.5 | ≤1.5 | ≥10000 |
| L2 关键 | ≤2 | ≤1.2 | ≤1.2 | ≥30000 |

### 小值域通过标准（|golden| < threshold 时）

```
ErrorCount = Σ I( |golden|<threshold ∧ |actual-golden|>err )
通过: ErrorCount_npu / max(ErrorCount_gpu, 1) ≤ 2
```

小值域 threshold / err（按 dtype）：

| dtype | threshold | err |
|---|---|---|
| FLOAT16 | 2^-11 | 2^-16 |
| BFLOAT16 | 2^-8 | 2^-16 |
| FLOAT32 | 2^-14 | 2^-30 |
| HiFLOAT32 | 2^-12 | 2^-28 |
| FLOAT8_E4M3 | 2^-4 | 2^-6 |
| FLOAT8_E5M2 | 2^-3 | 2^-5 |

### 复检（单用例不通过时启动，详见 `D:\Desktop\TMP\tmp.txt`）

**旧计数法（已弃）**：`Count(Ratio>1.2) < Count(Ratio<0.83)` 且 `Count(Ratio>2.0) < Count(Ratio<0.5)`。
4 大缺陷：①对整体偏移不敏感（系统性恶化盲区）②忽略幅度（Ratio=100 等同于 Ratio=1.21）③小样本不稳定（1 个波动就翻转）④阈值 1.2/0.83 是经验值无统计依据。

**新 Bootstrap CI 法**：
- 换随机种子重跑 N 次（推荐 1000），算每次 Ratio
- Bootstrap 2000 次有放回重采样 → 中位数集合
- 95%CI = [2.5% 分位, 97.5% 分位]（N=1000 时取第 25 小、第 976 小）
- **判定：`CI_Lower > 1.0` → FAIL（系统性恶化）；否则 PASS**
- **N < 200 熔断**：直接判失败（小样本不可信）
- 基于**中位数**（非均值），天然抗离群值，不要求数据正态分布

**3 类典型场景**（新法相对旧法的价值，含 Ratio 示例）：
- **系统性微小恶化（Slow Poison）**：Ratio 全 ≈1.1~1.15，旧法 `Count(>1.2)=0` 漏判 PASS；新法中位数 ≈1.14，CI≈[1.12,1.16]，下限 >1.0 → **FAIL** ✓（精准拦截渐进式恶化）
- **高噪声无恶化（Noisy Data）**：Ratio 在 [0.5,1.5] 随机跳，旧法见几个 >1.2 误判 FAIL；新法中位数 ≈1.0，CI 宽（如 [0.7,1.3]）含 1.0 → **PASS** ✓（避免误报）
- **个别离群值（Outlier Spike）**：除一个 Ratio=100 外其余 ≈1.0，旧法被带偏判 FAIL；新法中位数 ≈1.0，重采样中 100.0 难撼动中位数位置，CI≈[0.99,1.01] → **PASS** ✓（抗离群）

**新增特性（适配 ATK）**：动态多输出自动排序、复数（Complex）解耦独立检定、全量透明报告、小样本熔断机制（Validity Gate）。

### INF/NAN（标准 5.3.3）

- `INF_NAN_MODE_ENABLE=0`（910A 或 910B+ 未开启）：inf 不参与比较
- `INF_NAN_MODE_ENABLE=1`（910B+ 开启）：inf 输出须一致，或 NPU 比 GPU 更接近 Golden
- 正确：NPU==标杆；或 Golden 是 inf/nan 时 NPU==Golden
- 错误：Golden 是 inf/nan，NPU≠Golden 但标杆==Golden

## 量化算子判据（标准 4.4，QBMM 属此类）

| 输入\输出 | 整型输出（INT4/8/16） | 浮点输出（FP4/8/16 等） |
|---|---|---|
| 整型输入 | N/A | 参考浮点标准 |
| 浮点输入 | **绝对误差 ≤1** | 参考浮点标准 |

## 执行策略（标准第 3 节）

- L0/L1：单用例执行 50 次；L2：1000 次（压测捕获偶现精度问题）
- 平台覆盖：训练 910A/B2/B3/C；推理 910B2/B3/310P/310B 及后续
- 支持确定性的算子，多次执行结果须一致
- 开始测试前 Device 地址初始化为 NaN，以捕获越界计算

## 简化判据（无 GPU 标杆时的 fallback）

真实源：`projects/MM-确定性/test_matmul_golden.py:67-271`，对齐 omni-ops 官方 `test_ai_infra_matmul.py::verify_result`。

```python
close = torch.isclose(vf, gf, rtol=tol, atol=tol, equal_nan=True)
err_ratio = float((~close).sum()) / float(gf.numel())
passed = (err_ratio <= ERR_RATIO_TOL) and (m["err_nan"] == 0)
```

- flat isclose：`rtol=tol, atol=tol`（同一 tol）
- per-dtype tol：fp16/fp32=1e-3，bf16=5e-3
- PASS：`err_ratio ≤ 1e-4` 且无 NaN
- cos/rel_l2 仅诊断打印，不 gating

**何时用简化**：无 GPU 标杆、自定义算子暂未构造小算子拼接标杆时。能做双标杆的应优先用上面的权威 Ratio 制。

注：`projects/QBMM-omni/tests/golden.py` 现已升级为 ratio 制（`BM_CMP_STD` avg/max/rmse_rtol + `BM_FLOOR`=2^-mantissa 合成地板，`new=True` 模式），介于简化与权威之间——无真实 GPU，用合成地板替代标杆舍入误差。补齐方向见 memory `qbmm-golden-align-precision-standard`。

## 不要用（已弃）

- **2^-mantissa 逐 dtype 推阈值**：fp32 跨实现累加 floor≈`sqrt(K)·eps` 过紧，K≫1 的 fp32 case 必假 FAIL
- **cos 单独 gating**：跨实现累加的乘性偏置让 cos 在大 K 下偏低，误判
