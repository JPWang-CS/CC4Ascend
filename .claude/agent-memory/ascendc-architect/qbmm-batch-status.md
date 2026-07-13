---
name: qbmm-batch-status
description: QBMM-Batch project — MX scaleX1/scaleX2 batch dim in quantBatchMatmulV3 on A5/950. Direction REVERTED to backward-compatible (constant=3); code committed on branch QBMM0. Golden coverage expanded to 77 cases; a recent run shows widespread FAILs of unconfirmed provenance.
metadata:
  type: project
---

# QBMM-Batch: MX scale batch support — status (last verified against code 2026-07-06)

**Goal:** extend `quantBatchMatmulV3` MX full-quant path so scaleX1/scaleX2 carry a batch prefix matching x1/x2 (scaleX1 `(M,ceil(K/64),2)`→`(batchA,...)`, scaleX2 `(ceil(K/64),N,2)`→`(batchB,...)`). Motivation: real networks share batch between scale and input; needed for mxfp8/mxfp4 precision acceptance.

## Requirement DIRECTION — backward-compatible (constant = 3), NOT batch-mandatory
The spec flip-flopped (see the parallel auto-memory `qbmm-batch-requirement-direction`). Current code is settled on **backward-compatible**: no-batch (2D x1/x2 → 3D scale) must keep working; batch just adds a prefix. **Verified in current code 2026-07-06:** `MX_SCALE_DIM_NUM = 3` and `MX_SCALE_MIN_DIM_WITH_BATCH = 3`, checks are `< 3` reject / `>= 3` pass. batchDimNum = `scaleDim - 3` degrades to 0 at no-batch, so one code path serves both modes.
- **Doc trap:** `实施方案.md` / `需求.md` / `需求分析.md` / `施工进度.md` still describe the WRONG "≥4 / 强制带 batch / dim=3 不再支持" direction and many stale "待改/工作区未提交" statuses — those are 6/17 leftovers, do NOT trust them over the code. `代码修改说明.md` / `代码评审意见.md` describe the correct `>=3` direction.

## Where the constants actually live (memory's old file:line was slightly off)
- aclnn `MX_SCALE_DIM_NUM`: `matmul/quant_batch_matmul_v3/op_api/quant_matmul_v4_common.h:78` (a SHARED header — not `aclnn_quant_matmul_v4.cpp`; this resolves review item 1.2 "constant defined twice").
- tiling `MX_SCALE_MIN_DIM_WITH_BATCH`: `op_host/op_tiling/arch35/quant_batch_matmul_v3_checker.cpp:27`.

## Code state (verified 2026-07-06, branch QBMM0, ops-nn working tree CLEAN)
The whole change is COMMITTED (HEAD `bc8021c13 【QBMM】scale支持batch维度及tiling修复`), not sitting uncommitted as the docs say.
- **Host checkers (#1-#4):** landed, backward-compatible, batch-aware indexing + `CheckBatchValidInMxPerGroupMode` present (checker.cpp:664, called :853). Review 1.1 diagnostic-log fix applied.
- **Kernel BOTH paths landed** (design's key load-bearing fix): CMCT #5b `IDX_X1SCALE_OFFSET`/`IDX_X2SCALE_OFFSET` at `ops-nn matmul/common/cmct/kernel/kernel_qbmm_mx.h:303-306`; Blaze #5 `scaleAGmAddr_/scaleBGmAddr_ += batch*×(m|n)×scaleKLen` at `ops-tensor include/blaze/gemm/kernel/kernel_qbmm_mx.h:330-331`. Neither uses sizeShift (e8m0 = 1B, unpacked). 施工进度.md still marks #5b ⬜ — stale.
- **op-plugin (#6):** `dim_scale` 0→`dim()-3` on MX branch (commit 6c2885f9).

## OPEN item — FP4 transposeX2=false (实施方案.md Ch.4)
Pure golden-side design (no operator change): int8-passthrough golden can't reach trans_x2=false because fp4's ×2 doubling is on the innermost STORED dim; needs real 2:1 packing (`pack_fp4_lastdim`, x2 stored `(K, ceil(N/2))`). golden_mx_matmul_batch.py has `weight_nz`/`expect_fail` fields + `1b_fp4_tx2*` cases but NOT the packing helper yet. See [[fp4-e2m1-packing]].

## UNVERIFIED red flag — do not trust the on-board "all PASS" framing
`projects/QBMM-Batch/log.txt` is a 77-case run where early basic fp8-batch cases PASS (cos=1, diff=0) but MANY later/extended cases FAIL widely: nearly all `*_nz`, transpose×NZ, big shapes, fp4, and a systematic `4layer_* FAIL / 5layer_* PASS` pattern; many share a repeated ~6,645,xxx garbage max_diff (resembles the deploy garbage-value signature in 施工进度 §6.5 — custom-package-not-in-standard-vendors). **Provenance unknown** (which build/env/date). This is NOT proof of a code regression AND NOT proof of pass; treat as "needs the user to say what run this was" before drawing any correctness conclusion. See [[cann-custom-build-gotchas]] and [[aclnn-error-msg-build-fingerprint]].

**General knowledge this project produced** (reusable):
[[mx-quant-scale-semantics]] · [[fp4-e2m1-packing]] · [[scale-transpose-stride-semantics]] · [[cann-blaze-cmct-gating]] · [[golden-falsifiable-testing]] · [[golden-cancellation-criterion]] · [[device-print-verification]] · [[repo-mxfp4-tests-broken]] · [[cann-custom-build-gotchas]].

## Criterion fix landed (2026-07-06) — distinct from the red flag above
One FAIL CLASS is now diagnosed+fixed and is NOT an operator bug: **large-K + strong symmetric cancellation** (online-regression repro `mxfp8_mixed_e5m2e4m3_B13M1011K19085N3873`, cos=0.999999/diff=256 yet FAIL). The official fixed small-value gate (`small=1e-3`) misclassifies near-zero cancellation residue as "big" and a zero-error kernel FAILs.

## Criterion REPLACED with cos-primary + relative-L2 (2026-07-07, supersedes the pathological-scaling fix above)
User pivoted the whole verdict paradigm: `golden_mx_matmul_batch.py` no longer uses the official `BenchmarkCompareStandard` (max×10/avg×2/RMSE/fixed-small-gate) NOR the 2026-07-06 pathological peak-scaling patch. Both were DELETED (`bench_metrics`/`bench_verdict`/`MX_CANCEL_REL_FLOOR`/`EPS_FP32` gone; `BENCH_TOL` kept print-only). New verdict = **cos-primary + relative-L2 auxiliary double-gate** — see [[golden-cos-rell2-criterion]] for the design, thresholds, and falsification battery. Reason: cos is the industry-standard matmul/quant judge, immune by construction to near-zero/large-K-cancellation/fp4-bias graze misjudgments; the relative-L2 aux gate plugs cos's one blind spot (systematic multiplicative bias). Untouched: `gen_data`/`golden`/`call_npu`, `expect_fail` branch, nan/inf consistency. Verified numpy-mirror only (no torch/NPU on host); py_compile passes; real-tensor behavior INFERRED.

Workflow: [[ascendc-workflow-prefs]]. Design-doc-pristine convention: [[feedback-design-doc-pristine]]. Repos are siblings under D:/Desktop/Code/ — see [[workspace-layout]].
