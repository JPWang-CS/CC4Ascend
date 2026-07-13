---
name: workspace-layout
description: CC4Ascend workspace layout — sibling code repos (ops-nn/ops-tensor/op-plugin/ops-transformer) and their roles, skills/ and agent-memory locations, operator categories, chip support
metadata:
  type: reference
---

# CC4Ascend workspace layout

## Code repos (siblings under D:/Desktop/Code/)
- **ops-nn** — operator impl: host checker / tiling layers, aclnn entry, CMCT (basic-api) kernels. (QBMM V3 checker, MX arch35 host tiling live here.)
- **ops-tensor** — Blaze TensorAPI kernel library (e.g. `kernel_qbmm_mx.h`, `block_mmad_qbmm_mx.h`).
- **op-plugin** — torch adapter layer (`QuantMatmulKernelNpuOpApi.cpp`, op-plugin tests).
- **ops-transformer** (a.k.a. ops-transformer_AI) — CANN ops-transformer; transformer operator library + `docs/zh/` design docs. Shows as deleted/submodule in CC4Ascend git status.
- Build order across repos: ops-nn first — see [[ascendc-workflow-prefs]].

## In-workspace dirs (under D:/Desktop/Code/CC4Ascend/)
- **.claude/skills/** — 8 callable AscendC skills (migrated from the old project-root `skills/` dir): `ascendc-api`, `ascendc-data-context`, `ascendc-debug`, `ascendc-development`, `ascendc-install`, `ascendc-operator-invocation`, `ascendc-operators`, `ascendc-hardware`. Auto-discovered; trigger by `/ascendc-*` or description match. See [[spec-sources]].
- **.claude/agent-memory/ascendc-architect/** — this agent memory (flat 30-file index, general expert knowledge + QBMM project closure).
- **projects/QBMM-Batch/** — QBMM-Batch design docs + golden scripts — see [[qbmm-batch-status]].

## Chip support
- Atlas A2/A3: Ascend 910B, 910C (Da Vinci, dav_c220, arch 2201). A5: Ascend 950 (dav_c310, arch 3510). Hardware specs in [[hardware-specs]]; chip/arch mapping in [[cann-blaze-cmct-gating]].

## ops-transformer operator categories (production)
attention (~50: flash_attention, MLA, NSA, sparse/fused attention), ffn (6), gmm (7: grouped_matmul quant/swiglu/dequant/inplace_add), mc2 (30+: comm+compute fusion, all_gather_matmul, matmul_all_reduce, moe_distribute), mhc (8), moe (25+: routing, permute/unpermute, gating, finalize), posembedding (~10: rope variants, norm_rope_concat). Experimental ops under `experimental/`. Build entry `build.sh` (also generates op scaffolding).

See [[spec-sources]], [[ascendc-workflow-prefs]].
