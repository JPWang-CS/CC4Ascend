# 框架/注册问题

算子注册链路：op_def（OP_ADD）→ op_graph/proto（REG_OP）→ op_api（aclnn）→ torch binding。任一环缺失或符号不一致都报错。

## OP_LOGD 签名冲突（已验证）

### 现象
opapi 源码用旧风格 `OP_LOGD(fmt, args)`，与 op_common/log/log.h 新签名冲突，编译失败。

### 根因（已验证）
opapi 源码旧风格宏 vs `log.h` 新签名不兼容。不能删 log.h（FOR_INVALID 等宏依赖它）。

### 修法
算子 CMakeLists 内加 `-DOP_LOG_LIBOPAPI_ONLY`，限定日志宏只用 opapi 子集，避开签名冲突。

## 符号找不到 / undefined reference

### 根因
- op_def 没注册（缺 `OP_ADD`）
- op_api 没生成（缺 `aclnn_*.cpp` 或自动生成失败）
- namespace 不一致（迁移代码须 `AiInfraOps::` 非 `Ops::`，避免与 nn 仓符号冲突）
- 依赖算子没一起编译（`--ops` 只编了主算子，漏依赖）

### 修法
1. 确认 `OP_ADD` / `REG_OP` 都在
2. 确认 `aclnn_*.cpp` 生成
3. 迁移代码统一 `AiInfraOps::`
4. `--ops` 列全依赖算子（逗号分隔）

## torch binding 注册问题

### 通路
torch binding 链：C++ wrapper（`ops/csrc/*.cpp`）+ Python JIT builder（`ops/*.py`）+ schema 注册。

### 常见
- schema 不匹配（torch_npu 版本与 schema 定义不一致）
- PYBIND11_MODULE / TORCH_LIBRARY_IMPL 缺失
- DeviceGuard 缺失（torch_extension_guidelines 要求）

### 修法
对照 `ops-transformer_AI/torch_extension/cann_ops_transformer/docs/torch_extension_guidelines.md` 检查目录/命名/每层实现。

## namespace 铁律（已验证）
迁移代码必须 `AiInfraOps::` 非 `Ops::`，与 nn 仓区别，避免符号冲突。matmul 改名规则同此。