#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
"""nn 仓 (ops-nn) 风格 npu_ai_infra_matmul 验证 case — fp32 ND.

逐字对齐 omni-ops/.../ops-nn/matmul/ai_infra_matmul/tests/st/test_ai_infra_matmul.py 约定:
  import / verify_result / @pytest.mark.resources / torch.ops.custom.npu_ai_infra_matmul(a, b) 2-arg
  调用 / torch.matmul(a,b) 作 golden. verify_result 整段照搬 (含其打印格式), 保证可整段贴回该文件.

为何 fp32: 现有 nn 仓 test_ai_infra_matmul.py 只有 fp16/bf16, 缺 fp32. 本 case 补 fp32 ND,
  验证目标 = 910B 上 fp32 matmul |Δ|max~1e-5 ≪ tol 1e-3 → error_ratio≈0 通过 (对齐官方 verify_result).
  对照: projects/MM-确定性/test_matmul_golden.py 旧 2^-24 辅助门对 fp32 跨实现累加 floor 假 FAIL,
  已改 flat 0.001; 本 case 在官方 nn 仓判据下独立复验.

前置: torch_ops_extension wheel (注册 torch.ops.custom.npu_ai_infra_matmul) 已编+装.
运行: pytest test_nn_matmul_fp32.py -s   (或丢进 tests/st/ 由 build.sh --run_example 拉起)
"""
import torch
import torch_npu
import numpy as np
import omni_custom_ops  # noqa: F401
import pytest


def verify_result(output, golden, tol=1e-3):
    output = output.float().cpu().reshape(-1)
    golden = golden.float().cpu().reshape(-1)

    different_element_results = torch.isclose(output, golden, rtol=tol, atol=tol, equal_nan=True)
    different_element_indexes = torch.where(different_element_results == False)[0]

    for index in range(min(len(different_element_indexes), 10)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6d" % (
                real_index.item(), golden_data.item(), output_data.item(),
                abs(output_data - golden_data) / golden_data)
        )
    error_ratio = float(different_element_indexes.size(0)) / float(golden.size(0))
    print("error ratio: %.6f, tolerance: %.6f" % (error_ratio, tol))
    return error_ratio <= 1e-4


@pytest.mark.resources(device="npu:910B", npus_per_node=1)
def test_npu_ai_infra_matmul_fp32():
    M, K, N = 128, 256, 512
    a = torch.randn(M, K, dtype=torch.float32).npu()
    b = torch.randn(K, N, dtype=torch.float32).npu()

    out_custom = torch.ops.custom.npu_ai_infra_matmul(a, b)
    out_golden = torch.matmul(a, b)

    print(f"[matmul fp32] input: [{M},{K}] @ [{K},{N}], output shape: {out_custom.shape}")
    print(f"output[:2, :5]:\n{out_custom[:2, :5].float().cpu()}")
    assert verify_result(out_custom, out_golden)
