# CC4Ascend

AscendC 算子开发工作区。内部规则见 `CLAUDE.md`。

## 用法

三种触发方式：

- **自然语言（默认）**：直接说目标，Claude 读 `CLAUDE.md` 路由规则自动派 agent。例："把 X 从 910B 迁到 950，先出方案" → architect；"QBMM 编译报 EZ0012" → host-engineer。
- **`@ascendc-<agent>`（强制）**：绕过自动路由，指定某 agent 主持。要硬保证进方案流程用 `@ascendc-architect`。
- **`/ascendc-<skill>`（查知识）**：直接调 skill 查知识，不进工作流。如 `/ascendc-hardware`、`/ascendc-api`。

路由判断顺序：用户意图 → 工作阶段 → 代码位置 → 关键词。自动路由是软约束（靠模型遵守 `CLAUDE.md`）。

## 工作流

非 trivial 工作（新算子、跨芯片迁移、跨仓、语义敏感、需 golden）默认走方案：architect 主持确认需求，按需拉 specialist，方案定稿后给改动点清单，按层实现，verifier 证伪收尾。

- host-engineer — 调用通路 / 注册 / 编译安装 / checker
- tiling-expert — TilingData / TilingKey / split / 预算
- kernel-expert — 执行层 / 流水 / profiling / 优化
- kernel-semantics-researcher — 语义建模 / golden / 差异归因
- verifier — 证伪

小改动（bugfix、编译诊断、单层执行）直接进入实现。

工作偏好：改码前先讨论；上板前先确认；小步迭代；ops-nn 先改先编再 ops-tensor；结论须真实路径 trace + 多例验证。

## 调用通路

每次 host 侧工作须显式判断覆盖哪些：PyTorch binding / aclnn eager / aclnn graph / GE graph。不支持的写 scope out。

## 知识库

- skill（14 个）：`.claude/skills/`，按功能组织，全局共享
- 官网 API 索引：`refs/hiascend-api/`
- 硬件真值：`AscendC_platform/*.ini`
- 项目记忆：`.claude/agent-memory/ascendc-architect/`

## 验证纪律

compile success ≠ runtime correctness；local golden pass ≠ semantics proven；弱证据直说。详见 `CLAUDE.md` §8。

## 兄弟仓

`ops-transformer_AI` / `ops-nn` / `ops-tensor` / `op-plugin` 在 `D:/Desktop/Code/` 下，不在本仓内。
