---
name: spec-sources
description: Pointers to AscendC design-spec sources — .claude/skills callable skills, ops-transformer docs/zh, and official online AscendC API references
metadata:
  type: reference
---

# AscendC design-spec sources

Where to look for AscendC API constraints, data-type/format/quantization rules, and dev guides. Workspace dir locations are in [[workspace-layout]].

## Local (primary)
- **skills** at `.claude/skills/` — 14 callable skills: `ascendc-api` / `ascendc-kernel-programming` / `ascendc-data-context` / `ascendc-debug` / `ascendc-development` / `ascendc-doc-sync` / `ascendc-hardware` / `ascendc-install` / `ascendc-operator-invocation` / `ascendc-aclgraph` / `ascendc-operators` / `ascendc-kernel-optimization` + procedural `ascendc-build-errors` / `ascendc-golden-testing`. Each = a dir with SKILL.md (entry) + 中文 detail docs (progressive disclosure). Auto-discovered by the harness; trigger by description match or `/ascendc-*`.
- **ops-transformer docs** — the `ops-transformer*/docs/zh/` dir at whatever path the session established (see [[workspace-layout]]; path is not hardcoded): context/ (基本概念, 数据结构/类型/格式, 量化, broadcast), develop/ (aicore/graph/aicpu develop guides, cross_platform_migration_guide), install/, debug/ (cann_sim, op_debug_prof), figures/ (hardware arch diagrams: Atlas A2, Ascend 950; 指令流水图; per-operator flow diagrams), invocation/.

## Online
- ops-transformer docs: https://gitcode.com/cann/ops-transformer/tree/master/docs/zh
- AscendC API: https://hiascend.com/document/redirect/CannCommunityAscendCApi
- Op Dev Guide: https://hiascend.com/document/redirect/CannCommunityOpdevAscendC
- Basic Data Structures: https://hiascend.com/document/redirect/CannCommunitybasicopapi
- Release Management: https://gitcode.com/cann/release-management

See [[workspace-layout]], [[hardware-specs]].
