---
name: ascendc-kernel-optimization
description: AscendC kernel 优化知识库，按算子种类组织，每个算子类围绕 base 模板展开优化维度（分核/流水掩盖/量化/数据流/A2A3-vs-A5/负优化）。每条优化都经过真实代码路径 trace（op_def→tiling→kernel 入口→具体函数）。当要优化某类算子的 kernel、判断某优化是否值得、查某算子的 base 模板与优化偏离点、或查通用优化技巧时调用。内容真实详细，通用技巧在 common/ 统一归纳，各算子类独有优化在各子目录不漏。
---

# Kernel 优化知识库

本 skill 按算子种类组织 kernel 优化知识。**每条优化都经过真实代码路径 trace 验证**（从 op_def → TilingFunc → kernel 入口 → 具体计算函数），不凭文件名或记忆猜测。

真实源：`ops-transformer_AI/${op_class}/`。

## 组织结构

```
ascendc-kernel-optimization/
├── SKILL.md                      ← 本文件（入口 + 目录索引）
├── common/                       ← 通用方法论与跨算子通用技巧
│   ├── how-to-read.md            ← 怎么读 base 模板与优化偏离
│   ├── general-techniques.md     ← 通用优化技巧（transformer 仓 10 条）
│   ├── elementwise.md            ← elementwise 计算范式（ops-nn activation）
│   ├── reduce-norm.md            ← reduce/norm 范式（ops-nn norm）
│   ├── simt-index.md             ← SIMT 离散访存范式（ops-nn index）
│   ├── foreach-batch.md          ← foreach 批量融合范式（ops-nn foreach）
│   └── quant-tools.md            ← 量化工具 cast/scale 范式（ops-nn quant）
├── attention/                    ← Attention 类优化
├── gmm/                          ← GMM 类优化
├── moe/
├── mc2/
├── ffn/
├── mhc/
└── posembedding/
```

每个算子类子目录至少含：
- `base-template.md` — 该类的 base kernel 模板（A2A3 + A5）与设计理由
- 按优化维度分文件（分核 / 流水 / 量化 / 数据流 / 负优化 等）
- 每条优化写明：优化点、适用条件、收益、风险、真实代码路径

## 怎么用

- **查某算子类怎么优化** → 进对应子目录，先读 base-template，再读相关优化维度文件
- **查通用技巧** → `common/general-techniques.md`
- **判断某优化值不值得** → 看该条的风险/负优化条件 + 对应 base 模板

## 内容准则

1. 每条优化经真实路径 trace（op_def→tiling→kernel→函数）后才写入
2. 写当前真值，不写"与旧版区别"
3. 通用技巧归 common，各算子独有优化不漏
4. 真实详细，具体到函数/变量/数值

## 覆盖范围

- transformer 仓 7 类 base 模板（attention/gmm/moe/mc2/ffn/mhc/posembedding，每条含实现 + WHY + 适用范围 + 代价，带行号）；attention/gmm 含分核/量化维度文件；common 通用技巧 10 条
- ops-nn 仓通用范式 5 类归纳进 common（elementwise / reduce-norm / simt-index / foreach-batch / quant-tools）
- ops-nn 特化算子（matmul/conv/optim 等）按需补充
