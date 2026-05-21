# GMM A2A3 → A5 迁移实例

## 整体架构变化

### A2A3 架构
- Kernel 直接在 `op_kernel/` 下（无子目录）
- MixCore 模式：AIC-AIV 配对协作
- 手动 MNBlockIdxCompute 分核
- 手动 workspace 管理

### A5 架构
- Kernel 在 `op_kernel/arch35/` 下
- cgmct 框架：Scheduler + BlockMmad + BlockEpilogue 模块化
- ASWT 对角线分组调度
- 三种 cube 变体：BasicAPI / OnTheFly / MixOnlineDynamic

## 分核策略变化

### A2A3: 对角线分核
```cpp
MNBlockIdxCompute(mnConfig, block, count, thresholdM_dimN);
// 手动对角线窗口分配
// thresholdBlockNum=8, thresholdDimM=5
```

### A5: cgmct ASWT 调度
```cpp
GroupedMatmulAswtWithTailSplitScheduler scheduler;
// 自动对角线分组 + 尾块处理
// 不再需要手动计算 MNBlockIdx
```

## GGMM (GMM finalize routing) 模板参数扩展

### A2A3: 较少模板参数
```cpp
template <typename T, typename CT, ...>
class GMMFR_A8W8_IMPL { ... };
```

### A5: 新增 5 模板参数
```cpp
template <typename XType, typename YType, Mc2CoreType CoreType, 
          bool basedA2aRsAg, int commMode>
class MatmulAllReduceBase {};
```

新增参数含义：
- `basedA2aRsAg`：是否使用 AlltoAll+ReduceScatter+AllGather 模式
- `commMode`：通信模式选择

## 通信变化（MC2 类）

### A2A3: AICPU 通信
```cpp
Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
// 单 Mc2CcTiling
```

### A5: CCU 通信 + V2 API
```cpp
hccl_.InitV2(GetHcclContext<0>(), tilingData_);
hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTilingComm));
// 双 Mc2CcTiling: mc2CcTiling + mc2CcTilingComm
```

## GMM SwiGLU Quant V1 → V2 升级

### V1 (A2A3):
- TILING_KEY 0: COMMON_TILING_KEY_MODE — 简单 Block 分配
- TILING_KEY 1: SPLITWORKSPACE_TILING_KEY_MODE
- TILING_KEY 2: A8W4_MSD_TILING_KEY_MODE

### V2 (A5):
- 新增 A4W4 支持（TILING_KEY 4）
- 新增 A4W4 transpose weight（TILING_KEY 5）
- 新增 Fusion 路径（TILING_KEY 3）
- groupListType 支持 cumsum 和 direct count 两种
- CUSTOM_CFG_MDL 配置（enableGetTensorC=true）

## Tiling 参数变化

### A2A3: 固定模板枚举
```
baseM = 选择自 UB_SIZE / (baseN * sizeof(half))
baseN = 选择自枚举值
```

### A5: 容量反推
```
从 L0C 容量计算 maxSingleM / maxSingleN / maxSingleK
从 UB 容量计算 tileM（扣除 scale_buf + workspace）
dbL0C: L0C double buffer count
kAL1 / kBL1: L1 K-tile 大小（A5 cgmct 显式管理 L1）
```

## A5 新增特性

1. **MX 量化**：FLOAT8_E8M0 scale, group_size=32，需在 tiling 中区分普通量化 vs MX 量化
2. **Weight quant (a16w8)**：AIC-only 模式，无需 AIV 协作
3. **CUSTOM_CFG_MDL**：enableGetTensorC=true，支持 MatMul 中间结果读取
4. **BlockEpilogueDequant**：PERTENSOR + PERCHANNEL 两种 dequant 模式
