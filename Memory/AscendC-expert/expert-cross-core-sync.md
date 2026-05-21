# 核间同步专家技巧

## A5 死锁排查黄金法则

A5 上 `CrossCoreWaitFlag` 和 `CrossCoreSetFlag` 数量必须严格一一匹配。A2A3 的 HWT S 兜底清零机制在 A5 上不存在。

### 排查三步法

**Step 1: 检查异常分支**

```cpp
// 危险模式：异常分支提前返回
if (error) {
    CrossCoreSetFlag(flagId);  // Set 了
    return;                     // 但没走到对应的 Wait
}
// 修复：确保所有分支都成对
```

**Step 2: 检查 flagId 复用**

多 stage 流水复用同一 `flagId` 但生命周期重叠 → 跨阶段串扰。
- 解决：不同 stage 用不同 flagId，或用状态机区分阶段

**Step 3: 检查循环边界对齐**

循环内条件触发同步，但循环次数不一致：
- 生产者循环 N 次 Set，消费者循环 M 次 Wait → N≠M 必死锁
- 修复：统一循环边界变量，或使用共享的结束标志

### A5 新增同步模式 mode=3

```cpp
CrossCoreSetFlag<3, PIPE_S>(flagId);   // 新模式
CrossCoreWaitFlag<3, PIPE_S>(flagId);
```

注意 mode 不匹配也会死锁（Set 用 mode=0，Wait 用 mode=3）。

### 调试方法

复杂同步问题 → 最小数据集单 stage 验证 → 逐步叠加双缓冲和多 stage
