---
name: project-structure
description: Overview of the ops-transformer_AI project directory structure, operator categories, and chip support matrix
metadata:
  type: reference
---

# Project Structure Overview

## Project Identity
- **Repository**: ops-transformer_AI (CANN ops-transformer)
- **Version**: 9.0.0 (in development, build dependency >= CANN 8.5)
- **Path**: D:/Desktop/代码/Claude-AutoGen/CC4Ascend/ops-transformer_AI/
- **Last CHANGELOG**: 8.5.0-beta.1 (2025-12-30)
- **License**: CANN Open Software License Agreement Version 2.0

## Chip Support
- **Atlas A2/A3 series**: Ascend 910B, 910C (Da Vinci architecture)
- **Ascend 950 series**: 950PR, 950DT, KirinX90

## Operator Categories (production)
1. **attention** (~50 operators): flash_attention, MLA, NSA, sparse attention, fused attention, etc.
2. **ffn** (6 operators): ffn, ffn_worker_batching/scheduler, swin_transformer variants
3. **gmm** (7 operators): grouped_matmul variants (quant, swiglu, dequant, inplace_add)
4. **mc2** (30+ operators): communication+compute fusion (all_gather_matmul, matmul_all_reduce, moe_distribute, etc.)
5. **mhc** (8 operators): mhc_pre/post/res, sinkhorn variants
6. **moe** (25+ operators): moe routing, token permute/unpermute, gating, finalize
7. **posembedding** (~10 operators): rotary_position_embedding, rope variants, norm_rope_concat

## Experimental Operators
Located under `experimental/`: attention (blitz_sparse, compressor, typhoon_mla), mamba (causal_conv1d, chunk_scan), mc2, mhc, moe, posembedding (rope_matrix), select_attention_operators, svd, npu_ops_transformer_ext

## Key Project Files
- `build.sh` -- main build script, also generates operator scaffolding
- `CMakeLists.txt` -- top-level cmake
- `version.cmake` -- version 9.0.0
- `docs/zh/develop/aicore_develop_guide.md` -- AI Core operator development guide
- `docs/zh/context/` -- fundamental concepts (data types, formats, quantization, broadcast, etc.)
- `docs/zh/install/` -- installation, compilation, directory structure docs
- `CONTRIBUTING.md` -- contribution guide with operator directory structure requirements
