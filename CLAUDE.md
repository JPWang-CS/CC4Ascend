# CC4Ascend

AscendC 算子开发工作区。

本文件是 **强规则 + 内部路由入口**，不是大而全手册。详细领域知识在 `.claude/skills/`、当前仓代码与项目文档中。

---

## 1. 对外交互原则

- 用户只描述**问题、目标、约束**，**不需要也不应手工选择 agent**。
- Claude 需要**先根据用户需求/意图判断**当前需要哪种能力，再结合实际代码位置、当前工作阶段与已有上下文做内部调度。
- 路由优先级：
  1. **用户需求 / 意图**
  2. **当前工作阶段**
  3. **实际代码位置 / 涉及模块**
  4. **关键词 / 描述信号**

---

## 2. 工作流规则

### 2.1 非 trivial 工作默认流程

对以下类型任务，默认先走方案：
- 新算子 / 大改动
- 跨芯片迁移（如 910B → 950）
- 跨仓改动
- 数学语义敏感改动
- 需要 golden / reference 一起落定的改动

默认流程：
1. **确认用户需求与约束**
2. **方案讨论**
3. **方案定稿**（必要时带 golden 设计）
4. **一级改动点清单**
5. **按层实现**
6. **验证**

### 2.2 小改动例外

以下情况可直接进入实现/诊断，不强行上方案流程：
- 明确的局部 bugfix
- 明确的编译/安装/调用链诊断
- 已有方案且当前只是在执行其中一层

但即使是小改动，也必须遵守验证纪律，不得靠“看起来对”收尾。

### 2.3 方案阶段的协作规则

- 方案阶段由 **`ascendc-architect` 主持**。
- architect 先确认用户需求，再**按需**分发分析，不默认把所有 agent 并行拉起来。
- 如果方案需要带 golden / reference，则由 **`ascendc-kernel-semantics-researcher` 协同**。
- 方案定稿时，如绑定 golden，则：
  - 方案文档里必须有 **golden 设计思路**
  - `project/` 目录下必须有**初版可验证脚本**

### 2.4 实现阶段规则

- **host-engineer** 负责 host / 调用通路 / 工程链
- **tiling-expert** 负责 TilingData / TilingKey / split / 预算契约
- **kernel-expert** 负责执行层实现 / pipeline / profiling / 优化落地
- **kernel-semantics-researcher** 负责语义建模 / oracle / golden / 差异归因 / 优化启发
- **verifier** 负责证伪，不负责设计或实现

### 2.5 工作偏好（必须遵守）

- 改码前先讨论达成一致
- 跑服务器 / 上板前先取得同意
- 小步迭代，不做大包大揽一把改完
- **ops-nn 先改先编，再 ops-tensor**
- 方案文档保持纯净，施工进度另起文档
- 所有结论默认要求**真实路径 trace + 多例验证**；若未泛化验证，必须显式说明

---

## 3. 编码规范

### 3.1 风格硬规则

- **值写字面量**，不要写 `A + 1` 这种绕法
- 表达式禁止裸魔鬼数字，使用具名变量
- 块内值用普通局部变量，不滥用 `constexpr` / 文件级常量
- 不加解释性注释尾巴，保持简洁

### 3.2 文档/措辞

- 保留“矩阵”表述
- 斜杠等同，不滥用“及”
- 文档尽量极简

### 3.3 命名/迁移规则

- 迁移代码优先使用 **`AiInfraOps::`**，不是 `Ops::`
- `arch35` = A5
- `arch22` = A2A3
- `arch31` = 310P
- `_apt` = A5 路径，不是普通缩写
- skill 中 `ops-transformer_AI/` / `ops-nn/` / `ops-tensor/` / `op-plugin/` 路径均指 `D:/Desktop/Code/` 下兄弟仓，非项目内相对路径

---

## 4. 调用通路矩阵（host 侧必须显式考虑）

方案和 host 实现都必须判断本次需求覆盖哪些通路：
- **PyTorch binding**
- **aclnn eager**
- **aclnn graph**
- **GE graph**
- 相关验证入口（如 build/run example 的 eager/graph）

如果本次不支持某条通路，必须在方案里显式写成 scope out，而不是默认遗漏。

---

## 5. 内部 agent 路由规则

> 下列规则是给 Claude 自己看的内部调度规则，不是给用户背的菜单。

### 5.1 `ascendc-architect`
适用：
- 先定方案
- 要多方案比较
- 要定改动点清单
- 要判断是否需要 host / tiling / kernel / semantics 协作
- 要决定是否方案必须带 golden

### 5.2 `ascendc-host-engineer`
适用：
- PyTorch binding / aclnn eager / aclnn graph / GE graph
- op_def / proto / schema / registration
- build / install / checker / stale package
- host 侧实现与验证

### 5.3 `ascendc-tiling-expert`
适用：
- TilingFunc
- TilingData
- TilingKey
- split / core partition
- tile / buffer 预算
- host↔kernel 契约设计

### 5.4 `ascendc-kernel-expert`
适用：
- kernel correctness in execution
- pipeline / sync / buffering reality
- profiling / msprof / cannsim
- Regbase / SIMT / CV
- 执行层优化是否真的值得做

### 5.5 `ascendc-kernel-semantics-researcher`
适用：
- 数学/语义建模
- golden/reference 设计
- 方案带 golden
- golden / harness / tolerance / oracle 差异归因
- 从语义建模中抽取 kernel 优化启发

### 5.6 `ascendc-verifier`
适用：
- 任何“fixed / verified / passing / done / safe to proceed”类结论
- 需要拆证据等级
- 需要排查 false-pass
- 需要判断当前证据是否真的够强

---

## 6. skill 路由规则（按功能，不按角色）

> skill 是全局共享能力库，不按 agent 归属。agent 只会有默认优先查阅顺序，没有 skill 所有权。

当前稳定可用的 skill：

- **`ascendc-api`**
  - AscendC API / DataCopy / SetFlag / high-level API

- **`ascendc-kernel-programming`**
  - kernel 编程范式 / 怎么写 kernel（Membase/Regbase/SIMT/Blaze 骨架）

- **`ascendc-data-context`**
  - dtype / format / quant / broadcast / stride / transpose / 语义规则

- **`ascendc-debug`**
  - msprof / cannsim / PRINTF / 调试与性能定位

- **`ascendc-development`**
  - 开发流程 / 跨代迁移 / A5 范式

- **`ascendc-doc-sync`**
  - 跨仓文档同步排查

- **`ascendc-hardware`**
  - 芯片规格索引与设计影响归纳
  - **原始数字真值只认 `AscendC_platform/*.ini`**

- **`ascendc-install`**
  - build.sh / 编译 / 部署 / 目录结构

- **`ascendc-operator-invocation`**
  - PyTorch / aclnn / GE 图模式调用链
- **`ascendc-aclgraph`**
  - 推理算子入 ACLGraph（Capture/Replay）：meta / tiling 更新接口 / 静态 kernel / SuperKernel

- **`ascendc-operators`**
  - 算子类范式 / 分核 / TilingKey / 流水模式

- **`ascendc-kernel-optimization`**
  - kernel 优化知识库 / base 模板与偏离 / 通用优化技巧（按算子类 + 通用范式组织）

- **`ascendc-build-errors`**
  - 编译 / checker / 安装失效排查（EZ 报错码 / stale / 注册 / 符号）

- **`ascendc-golden-testing`**
  - golden 判据 / 输入构造 / 输出规范 / 可证伪设计

> 注：build-errors / golden-testing / kernel-optimization 为 procedural skill，均落地并回链到 agent。

---

## 7. Memory 与 Skill 的边界

- **Skill**：稳定知识库，按功能组织
- **Memory**：项目 delta / 易变经验 / 验证结论
- 规格原始数字只认 `AscendC_platform/*.ini`
- 不要把稳定 repo 事实重复存进 memory

共享主记忆库当前以：
- `.claude/agent-memory/ascendc-architect/`
作为主库

verifier 可保留少量专属记忆，但共享知识优先写主库。

---

## 8. 验证纪律

- compile success ≠ runtime correctness
- local golden pass ≠ semantics proven
- copied oracle ≠ trustworthy oracle
- absence of print ≠ path not executed
- 若当前证据只是 local / weak，必须直说
- 对 board-only / checker-only / package-selection-only 问题，不得用本地弱证据过度下结论

当任何层（architect / host / tiling / kernel / semantics research）出现“已修好/可继续”的结论时，应考虑触发 **`ascendc-verifier`**。
