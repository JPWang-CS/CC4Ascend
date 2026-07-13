---
name: ascendc-workflow-prefs
description: How the user wants AscendC operator work done — discuss & agree before code changes, get OK before server experiments, small iterative steps, change+compile ops-nn before ops-tensor
metadata:
  type: feedback
---

# AscendC operator work — workflow preferences

Rules the user holds firmly for operator design/implementation work in this workspace.

- **Discuss and reach agreement before any code change.** Present the plan/approach first; do not start editing until aligned.
- **Get explicit OK before running server / on-board experiments.** Board time is gated by the user.
- **Small iterative steps, no large one-shot edits.** Keep code simple, clear, efficient.
- **Cross-repo build order: change ops-nn first AND compile ops-nn first; only afterwards change ops-tensor.** Drove the QBMM-Batch two-phase plan (Phase 1 = ops-nn host side + no-batch regression check; Phase 2 = ops-tensor kernel + batch validation). Reason: host (ops-nn) gates/checkers must accept the new shapes before the kernel (ops-tensor) change can be exercised, and a host regression must be ruled out first.

**Why:** these came out of the QBMM-Batch work (see [[qbmm-batch-status]]) where premature edits and out-of-order builds would have masked regressions or wasted board runs.

**How to apply:** default to a plan-then-confirm loop for any non-trivial AscendC change; treat build/test ordering as ops-nn → ops-tensor unless the user says otherwise. Design-doc discipline is separate — see [[feedback-design-doc-pristine]].
