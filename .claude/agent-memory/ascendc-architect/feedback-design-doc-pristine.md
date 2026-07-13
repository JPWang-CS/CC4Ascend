---
name: feedback-design-doc-pristine
description: Keep 实施方案.md as a pure design doc; track real implementation/progress status in a separate 施工进度 doc
metadata:
  type: feedback
---

For the QBMM-Batch project (and likely other projects in this workspace), keep the design document (`实施方案.md`) as a pristine design spec — do NOT inject implementation/progress status, "already done / pending" markers, or working-tree-vs-design reconciliation into it.

**Why:** When asked to "align the design doc with the partially-started code", the user rejected edits that mutated 实施方案.md in place and instructed: "回退，原有的实施方案当做设计文档，另起一个施工进度" (revert; treat the original 实施方案 as the design doc, start a separate construction-progress doc).

**How to apply:** Implementation status, commit references, change-point completion state, dependency/ordering warnings, and stale-snapshot reconciliations belong in a separate progress doc (e.g. `施工进度.md` in the same project dir), not in the design doc. The design doc may still describe "current code" vs "design" as part of explaining a change, but should not be annotated with project execution state. See [[qbmm-batch-status]], [[ascendc-workflow-prefs]].
