# 算子跨平台迁移指南（通用方法论）

## 硬件架构对比

### A2A3 (Atlas A2)
- **24 核** AI Core，频率 1.8 GHz
- Cube: 353T/376T @BF16,FP16
- Vector: 23.5T @FP16
- **L0C: 128KB**, UB: 192KB(?)  
- Memory: 64GB, 1.6TB/s

### A5 (Ascend 950)
- **32 核** AI Core，频率 1.65 GHz
- Cube: 426T@BF16/FP16, 757T@FP8, 1514T@MXFP4
- Vector: 54T @FP16
- **L0C: 256KB, UB: 256KB**
- Memory: 128GB, 1.6TB/s
- 新增: SIMT, CCU1.0, Regbase, ND DMA

## 硬件能力变更与适配

### 搬运单元变更

| 变更 | 影响 | 适配方案 |
|------|------|----------|
| **删除 L1→GM 通路** | 不能直接 L1 回写 GM | 改为 L1→UB→GM 或 L0C/FIXPIPE→GM |
| **删除 GM→L0A/L0B 通路** | 直连不再可用 | GM→L1→L0A/L0B，重构 MTE1/2 流水 |
| **新增 ND DMA + ND→NZ 随路转换** | 减少中间 buffer | 利用 ND2NZ/DN2NZ 在 MTE2 完成格式转换 |
| **新增 Cube↔Vector 通路** (UB2L1, L0C2UB, FIXP→UB) | 减少 GM 往返 | UB 侧完成中间累加/激活/融合 |
| **引入 CCU1.0** | 通信加速 | Eager: 设置 `NNOPBASE_HCCL_SERVER_TYPE_CCU`; Graph: 使用 CCU 系列 GE 接口 |

### 计算单元变更

| 变更 | 适配方案 |
|------|----------|
| Vector 新增 **Regbase** 范式 | 审查 Membase 访存 pattern，评估迁移收益 |
| Cube **不再支持 int4_t** | 切换到 int8，更新量化解算逻辑 |
| **不支持 4:2 稀疏** | 改用稠密或其他稀疏策略 |

### 存储单元变更

| 变更 | 影响 |
|------|------|
| L0C 128KB→256KB | 可增大基本块与双缓冲容量，减少切 K 轮次 |
| UB 192KB→256KB | 重新评估 L1/L0/UB 配比与 tile 尺寸 |

## 迁移步骤

1. 确认算子涉及的计算单元（Cube/Vector）和数据类型差异
2. 确认数据搬运路径差异（ND→NZ, GM↔Lx, 集合通信）
3. 逐项对照修改（Vector 架构, Cube 类型, L1/L0/UB 大小, CCU 通信）
4. 参考算子迁移样例调整分支逻辑

## A5 性能调优 FaQ

性能不升反降时优先排查：

1. **仍用 A2 错位分核模板** → A5 支持同地址并行，可简化为规则滑动窗口
2. **未开启 CCU 通信** → 仍走 AICPU 通信路径
3. **Tiling 沿用 A2 策略** → 未利用 A5 更大 L0C/UB
4. **异常分支 Sync 不匹配** → A5 要求 CrossCoreSetFlag/WaitFlag 数量严格匹配

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md`
