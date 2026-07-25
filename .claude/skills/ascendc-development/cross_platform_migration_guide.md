# 算子跨平台迁移指南（A2A3 → A5）

真实源：`ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md`（453 行）。**A5 四大范式（SWAT / SIMT / Regbase / CV 融合 / CCU）的真身都在这份文档里。**

> 规格数字（核数/L0C/UB）以 `AscendC_platform/*.ini` 为准；文档表格是"代表性配置"供对比，非某具体 variant 真值。

## 一、硬件能力变更总表（真实，迁移必查）

| 硬件单元 | 变更 | 影响 / 适配 |
|---|---|---|
| 搬运 | **删除 L1→GM 通路** | L1 直接回写 GM 的 kernel 改为 L1→UB→GM 或 L0C/FIXPIPE→GM |
| 搬运 | **删除 GM→L0A/L0B 通路** | GM→L1→L0A/L0B，重构 MTE1/2 流水 |
| 搬运 | **ND DMA + 随路 ND→NZ** | MTE2 阶段完成格式转换，减少中间 buffer；注意步长/对齐/NZ 形状映射 |
| 搬运 | **Cube→Vector 直连**（UB2L1 / L0C2UB / FIXP→UB） | UB 侧做中间累加/激活/融合，减少 GM 往返 |
| 搬运 | **CCU1.0** | 通算融合算子 Eager 改 HcclServerType，Graph 改用 CCU 系列 GE 接口 |
| 计算 | **Vector 新增 Regbase** | Membase 访存 pattern / 对齐 / 寄存器假设需重新审查 |
| 计算 | **Cube 不再支持 int4_t** | 改 int8，更新量化解算 |
| 计算 | **不支持 4:2 稀疏** | 改稠密或其他稀疏策略 |
| 存储 | **L0C 128KB→256KB, UB→256KB** | 可增大基本块与双缓冲，减少切 K 轮次；重评 tile 配比 |
| 其他 | **同地址并行优化** | 矩阵乘相关模板可简化分核（不再需错位规避） |
| 其他 | **SIMT** | 线程级并行处理分支/不规则计算 |

## 二、推荐迁移步骤

1. 确认计算单元（Cube/Vector）和支持数据类型的平台差异
2. 确认数据搬运路径差异（ND→NZ、GM↔Lx、集合通信）
3. 逐项对照修改（Vector 架构 / Cube 类型 / L1/L0/UB / CCU）
4. 参考算子迁移样例调整 A2/A5 分支逻辑

## 三、算子迁移样例（真实文档有完整代码）

### Cube 矩阵计算类
- **同地址冲突优化**：A2 上为"错位规避"设计的分核策略，A5 可简化为规则滑动窗口模板（SWAT）。先功能等价保留 tile 尺寸，再放开分核约束，结合 profiling 看 MAC/MTE2/L2 命中率。
- **Tile 尺寸调整**：A5 L0C 翻倍，可增大 tile 粒度或 K 方向单轮深度，减少切 K 轮次；注意别挤压 A/B/scale 缓冲。

### Vector 向量计算类
- **SIMT**：尾轴 ≤2048 走 SIMT 模板，>2048 走 SIMD 模板。适合离散访存、索引重排、稀疏更新。详见 [SIMT 速查](simt编程范式.md)。
- **Regbase**：MicroAPI / RegTensor / MaskReg，精细寄存器控制。详见 [Regbase 速查](regbase编程范式.md)。

### Cube-Vector 融合类
- **MTE 搬运路径变化**：UB2L1 / L0C2UB 直连，切 K 累加可 L0C 直达 UB。详见 [CV 融合速查](cv融合通路.md)。
- **核间同步严格匹配**：A5 上 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 必须一一匹配，不再有 HWTS 兜底清零，不匹配必死锁。

### 集合通信类
- **CCU1.0 替代 AICPU**：Eager 设 `NnopbaseSetHcclServerType`；Graph 区分 `aicpu kfc server` vs `ccu server`。详见 [CCU 速查](ccu通信适配.md)。

## 四、A5 性能 FAQ（性能不升反降时排查）

1. 是否仍用 A2 的**错位分核模板**（A5 支持同地址并行，可简化）
2. 是否**未开启 CCU** 仍走 AICPU
3. **Tiling 是否沿用 A2 策略**（A5 L0C/UB 更大，应重评）
4. 是否存在**异常分支 Sync 不匹配**（A5 要求 CrossCoreSetFlag/WaitFlag 严格匹配）

---

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md`（453 行，含全部 A5 范式完整代码）