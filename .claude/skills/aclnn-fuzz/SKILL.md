---
name: aclnn-fuzz
description: 华为官方 aclnn fuzz 测试框架（xrunfk）的使用与精度比对。支持两种精度路径：单标杆(checkResultNew 合成地板，NPU vs golden，阈值 10/2/2) + 双标杆(auto_new_precision NPU-VS-GPU ratio，算子精度标准v1.0 完整实现)。覆盖框架结构、用例生成、CPU/GPU 标杆生成、NPU 执行、三精度方法(0/1/2)、双标杆 ratio(最大/平均相对误差比·均方根误差比·ULP比·小值域比)+RMSE直方图。当查官方精度比对怎么实现、双标杆怎么跑、阈值怎么配、怎么加 aclnn 用例时调用。源：D:\Desktop\Code\fuzz\CustomOP\aclnn_fuzz（官方下载）。
---

# aclnn-fuzz 测试框架

华为官方 aclnn 算子 fuzz/精度测试框架。源：`D:\Desktop\Code\fuzz\CustomOP\aclnn_fuzz`（34000+ 文件）。主入口 `xrunfk.py`。

## 框架定位

测 aclnn 算子：生成用例 → 出标杆 → NPU 跑 → 比对精度。是**算子精度标准 v1.0 的官方落地实现**（`D:\Desktop\TMP\log.txt` 是理论文档，本框架是代码实现）。

## 目录结构

| 路径 | 用途 |
|---|---|
| `xrunfk.py` | 主入口（create/bm/run） |
| `Aclnn/` | aclnn C++ 测试框架（gen_code.py 生成、build_*.sh 编译） |
| `configs/` | 系统配置（精度阈值/dtype/op 映射/环境 ini） |
| `libs/tools.py` | **单标杆比对核心**（checkResult / checkResultNew） |
| `auto_new_precision.py` / `_mm.py` | **双标杆自动化**（NPU+GPU+golden 全流程 + ratio 统计） |
| `tools/op_precision/` | 新精度比对工具（new_precision_transfer.py 等） |
| `design/` `excel/` | DesignFile/Excel（用例源头） |
| `case_generator/` | yaml→dtype/shape 泛化用例 |
| `script/` | 脚本（update_precision_threshold.py） |

## 两条精度路径

### 路径 A：单标杆（合成地板，`libs/tools.py::checkResultNew`）

xrunfk.py 正常跑用这个。NPU 输出 vs CPU golden，**无 GPU 标杆**。

- `diff_big_ratio = |NPU - golden| / |golden|`（大值逐元素相对误差）
- 指标：max/avg/rmse 相对误差 + 小值域 err_small + red(相对误差分布)
- 阈值 `configs/aclnn_op_bm_cmp_std.json`：max_re_rtol=10 / avg_re_rtol=2 / rmse_rtol=2（fp32/fp16/bf16 同；配合 dtype 分辨率地板用）
- small_value: fp32=1e-6/fp16=1e-3/bf16=1e-7；small_value_atol: 0/1e-3/4e-3
- red_range: fp32=1e-6/1e-5/1e-4/5e-4，fp16/bf16=1e-3/2e-3/5e-3/1e-2
- inf/nan 处理：golden_inf_switch / inf_clip_switch / nan_toZero_switch（xrunfk.ini）

### 路径 B：双标杆（NPU-VS-GPU ratio，`auto_new_precision.py` / `_mm.py`）

**算子精度标准 v1.0 的完整实现**。NPU + GPU 都跑，CPU（或 GPU）出 golden，算 NPU/GPU 双方对 golden 的误差比。

**全流程**（`auto_new_precision.py main()`）：
```
Step1 清远程 GPU 服务器 → Step2 生成用例 → Step3 在 GPU 上执行 bm(标杆)
→ Step4 拷 GPU 结果回 NPU → Step5 bm_output_gold(CPU golden) → Step6 NPU 执行
→ Step7 转新精度 csv(双标杆 ratio 统计)
```

**ratio 列**（`auto_new_precision_mm.py` result_analysis，全标 "NPU VS GPU"）：
- 最大 ULP 误差比 / 平均 ULP 误差比
- **最大相对误差比 / 平均相对误差比**
- **均方根误差比**（RMSE ratio）
- 小值域误差比
- 逐点绝对误差通过率

**ratio 核心公式**（`tools/op_precision/op_precision_stat.py::BenchmarkCompareStandard._calc_ratio`）：
```python
def _calc_ratio(x, y):  # x=NPU_err, y=GPU_err
    if y == 0 and x == 0: return 1.0
    return x / max(y, small_value)   # ← small_value 当分母地板，防 benchmark_err≈0 退化
```
`small_value` = bm_cmp_std 的 small_value（**fp16=1e-3 / fp32=1e-6 / bf16=1e-7**）。这是**双标杆对 int8 不退化的关键**——int8 时 GPU cuBLAS≈CPU MKL（benchmark_err≈0），但分母地板是 small_value(1e-3)，ratio=NPU_err/1e-3≈1（fp16 cast 也≈1e-3）→ PASS。

**阈值**（`BenchmarkCompareStandard`，来自 `aclnn_op_bm_cmp_std.json`）：`max_re_rtol=10 / avg_re_rtol=2 / rmse_rtol=2`（跟单标杆同套数，**不是标准文档 L1 的 5/1.5/1.5**）。

**三级判定**（`PrecisionCheckResult`，比 PASS/FAIL 更细）：
| ratio | 判定 | 含义 |
|---|---|---|
| ≤ 1.0 | SUCCESS | NPU 不劣于 GPU |
| 1.0 < ratio ≤ rtol | WARNING | NPU 比 GPU 差但在容限 |
| > rtol(10/2/2) | ERROR | NPU 显著差于 GPU |

**GPU 标杆可用性自检**（`gpu_precision_check` flag）：若 GPU 自身误差 >90% 落在最大直方图桶（GPU 太精，benchmark_err≈0），关闭精度检查（防退化误判）。

**RMSE 直方图**（`histogram_cmd`）：ratio 分布 0~0.2 / 0.2~0.4 / ... / 1.0~1.2 / 1.2~1.4 / ... / >2.0——对应标准 L0/L1/L2 阈值区间（L1≈1.5、L2≈1.2）。

**参数**：`--precision_mode benchmark`（双标杆，默认）/ `binary`；`--golden_mode gpu/cpu`（golden 取 GPU 还是 CPU）。

**依赖**：远程 GPU 服务器（`configs/auto_new_precision.ini` 的 n2_host/user/port/ssh_key + `configs/gpu_info.json` passwd）。

## 三精度方法（`precision_method`，路径 A 内细分）

`configs/xrunfk.ini` 或 `configs/aclnn_op_pre_method.json`（算子级优先）切换（仅路径 A）：

| method | 算法 | 默认阈值 |
|---|---|---|
| 0（默认） | 相对误差+准确率 | diff_thd=1e-4 / pct_thd=0.999 / max_diff_thd=0.1 |
| 1 | np.isclose | rtol=0.005 / atol=2.5e-5 |
| 2（新·算子精度标准v1.0） | 标杆对比法(单标杆合成地板) | max_re_rtol=10/avg=2/rmse=2 + small_value/red |

> method 2 是路径 A 的"新精度"模式（单标杆）。**双标杆是路径 B（auto_new_precision）**，独立于 precision_method，需 GPU 服务器。

## 配置阈值（README 七节）

- **路径 A method 2**：`configs/aclnn_op_bm_cmp_std.json`（按算子+dtype）+ `aclnn_op_red_range.json`
- **路径 A method 0/1**：`op_report.json`（按算子名）或 `aclnn_op_dtype.json`（按 dtype，优先级高）
- 配完跑 `script/update_precision_threshold.py -d <op>` 更新 cs（cs 优先级 > ini）

## 工作流（正确命令，照官方攻略）

### 分离式（推荐调试用）

```bash
# 1. 生成用例 (.cs/.json) — 用 script/aclnn_create_json_new.py, 不是 xrunfk.py create!
python script/aclnn_create_json_new.py --op_name=<op> --case_file=excel/<op>.xlsx --sheet_name=<sheet> --use_bin=false --run_type=aclnn

# 2. 生成 golden (CPU) — xrunfk.py bm <op> <case_name> (case_name 是 excel 里的具体用例名, 不是 all)
python3 xrunfk.py bm <op> <case_name>

# 3. NPU 执行 — xrunfk.py npu/npu_off <op> <case_name>
python3 xrunfk.py npu <op> <case_name>

# 4. 精度比对 — xrunfk.py compare <op> <case_name>
python3 xrunfk.py compare <op> <case_name>
```

### 归一化（一步到位）

```bash
bash run.sh --op_name=<op> --case_file=excel/<op>.xlsx --sheet_name=<sheet> \
  --exec_mode=npu --framework=aclnn --graphy_path=0 \
  --use_bin=false --bm=bm --ti=<start>-<end>
```

`--use_bin=true` + 删 `--bm=bm` = 复用已生成 golden 不重跑。

### 代码路径模拟

```
1. aclnn_create_json_new.py
   读 excel/<op>.xlsx (sheet_name 页签) → 生成 testcase/aclnn_case/<op>/*.cs + *.json
   .cs = case spec (shape/dtype/range/attr/threshold)
   .json = case config (input details + bin_path)

2. xrunfk.py bm <op> <case_name>
   读 .cs/.json → 生成 input .bin 文件 → 调 opp/<op>.py::aclnn_op_func(action_type='bm')
   → aclnn_op_func CPU 分支 (torch.matmul + golden) → 存 golden output bin

3. xrunfk.py npu <op> <case_name>
   读 .cs/.json + input bin → 调 opp/<op>.py::aclnn_op_func(action_type='npu')
   → aclnn_op_func NPU 分支 (torch.ops.custom.* 或 l0op::*) → 存 NPU output bin

4. xrunfk.py compare <op> <case_name>
   读 golden bin + NPU bin → libs/tools.py checkResult/checkResultNew → 精度判定
```

### xlsx 要求

- 放在 `excel/` 文件夹内
- sheet 名 = `--sheet_name` 参数值 (如 `level0`)
- 列格式照 Design File 规范 (README section 二)

## 新增算子 fuzz 适配步骤（照 AiInfraMatmul 模板）

1. **`opp/aclnn_op/<op_name>.py`**：golden + NPU 调用。顶层 `import omni_custom_ops`（裸 import，wheel 必须装好，跟 matmul 一致，不加 try/except）。`aclnn_op_func` 分三路：npu=`torch.ops.custom.*`、gpu=golden 搬 GPU、cpu=golden
2. **`opp/aclnn_op/__init__.py`**：追加 `try: import opp.aclnn_op.<op_name> except: pass`
3. **`excel/<op_name>.xlsx`**：用例配置（照 QuantMatmulV5 / AiInfraMatmul 抄）。**sheet 名必须是 `level0`**（不是默认 `Sheet`）。列要全（68 列含 attr_name/attr_type/attr_dtype/attr_value）。放 `excel/` 文件夹
4. **`bm_cmp_std.json` 不改**：框架对未配置算子自动用默认阈值（10/2/2）。加条目是冗余
5. **`pip install tabulate`**：框架依赖，不装会 warning

## 踩坑总结

| 坑 | 现象 | 正解 |
|---|---|---|
| **生成用例用错命令** | `xrunfk.py create` 只出 design2_ xlsx，不出 .cs/.json → `bm_output_gold` 报 "no case" | 用 `script/aclnn_create_json_new.py --op_name=... --case_file=excel/xxx.xlsx --sheet_name=level0 --run_type=aclnn` |
| **xlsx sheet 名错** | sheet 叫 `Sheet`（默认）→ 框架认不到用例 | sheet 名改成 `level0`（跟 AiInfraMatmul 一致） |
| **手写 .cs/.json** | 格式不对，框架读不了 | 不要手写，让 `aclnn_create_json_new.py` 从 xlsx 自动生成 |
| **改了 bm_cmp_std.json** | 加了冗余条目（默认就是 10/2/2） | 不改，框架对未配置算子自动用默认 |
| **import omni_custom_ops 报错** | 模块加载失败，`__init__.py` try/except 静默吞 | 确认 wheel 装好（`pip install` torch_ops_extension wheel） |
| **`bm_output_gold` 直接跑** | "no case"（还没生成 .cs/.json） | 先跑 `aclnn_create_json_new.py` 生成 case，再跑 `xrunfk.py bm` |

## 关键命令速查

```bash
# 0. pip 依赖
pip install tabulate

# 1. 生成用例 (从 xlsx → .cs/.json)
python script/aclnn_create_json_new.py --op_name=<op> --case_file=excel/<op>.xlsx --sheet_name=level0 --use_bin=false --run_type=aclnn

# 2. golden + NPU + 比对 (归一化一步到位)
bash run.sh --op_name=<op> --case_file=excel/<op>.xlsx --sheet_name=level0 --exec_mode=npu --framework=aclnn --graphy_path=0 --use_bin=false --bm=bm --ti=0-0

# 或分离式
python3 xrunfk.py bm <op> <case_name>       # golden
python3 xrunfk.py npu <op> <case_name>      # NPU
python3 xrunfk.py compare <op> <case_name>  # 比对
```

## 边界

- 编译/报错 → ascendc-build-errors
- 差异归因 → ascendc-kernel-semantics-researcher
- golden 判据理论 + project golden 脚本 → ascendc-golden-testing
- 本 skill 讲**官方 aclnn_fuzz 框架**（结构 + 精度比对 + 新增算子适配 + 踩坑）
