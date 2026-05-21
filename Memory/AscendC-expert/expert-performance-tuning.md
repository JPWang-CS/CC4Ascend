# 性能调优专家技巧

## A5 性能不升反降排查清单

### 1. 错位分核模板（最常见）

A2A3 上为规避同地址访问冲突设计的"错位分核"策略，在 A5 上变成负优化。

**原因**：A5 新增同地址请求并行处理特性，不再需要错位规避。

**修复**：将分核策略简化为规则滑动窗口模板（行组窗口+列向往返扫描）。具体：
```
A2A3: 每个核偏移 offset_i 避免同地址
A5:   所有核用相同起始地址，规则滑动
```

验证指标：MAC 利用率、MTE2 利用率、L2 命中率

### 2. 未开启 CCU 通信

仍走 AICPU 通信路径导致性能浪费。
- 检查 `NnopbaseSetHcclServerType` 是否指向 `CCU`
- Graph 模式检查是否使用 `CreateCcuTask`

### 3. Tiling 沿用 A2 策略

A5 L0C 翻倍(128→256KB)、UB 扩大，旧 tiling 参数浪费了片上缓存。
- 优先增大 tile 块切分粒度
- 提高 K 方向单轮处理深度
- 重新平衡 L1/L0/UB 容量预算
- **不要**盲目放大 L0C 导致 A/B/scale 缓冲被挤压

### 4. 未利用 ND DMA

A5 支持 ND→NZ 随路转换：
```
A2A3: L1(ND) → 手动转 NZ → L0A/L0B
A5:   L1(ND) → MTE2 ND2NZ → L0A/L0B  (一步完成)
```
在搬入阶段直接完成格式转换，减少中间 buffer。

## 通用性能分析流程

1. `msprof op` 看整体指标（Task Duration, Block Dim, 流水占比）
2. 仿真流水图（A5: cannsim, A2A3: msprof simulator）
3. 重点看：Vector 占比、Cube 占比、MTE2/MTE3 占比 → 定位瓶颈
4. 瓶颈在 MTE → 增大 tile 或减少搬运次数
5. 瓶颈在 Compute → 检查流水并行度、双缓冲利用
