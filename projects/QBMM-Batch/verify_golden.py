#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# QuantBatchMatmul Golden 验证脚本
#
# 功能:
#   1. 纯 PyTorch 参考实现 (CPU 可运行, 无需 NPU)
#   2. 内置自测用例, 验证参考实现的正确性
#   3. 预留 torch_npu 对比接口 (后续接入实际 NPU 环境后修改)
#
# 用法:
#   python verify_golden.py                    # 运行所有自测
#   python verify_golden.py --verbose          # 打印详细信息
#   python verify_golden.py --profile          # 打印耗时
#
# 核心公式:
#   dequant_weight = weight_q * antiquant_scale + antiquant_offset
#   output = x @ dequant_weight + bias
#   (optional) output = output * quant_scale + quant_offset
# ----------------------------------------------------------------------------

import math
import struct
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import torch

# ============================================================================
# 工具函数: 模拟低精度数据类型的反量化
# ============================================================================

def dequant_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP8 E4M3 (4 exponent + 3 mantissa bits) → float32
    E4M3 格式: s|eeee|mmm
      - sign: 1 bit
      - exponent: 4 bits, bias=7
      - mantissa: 3 bits
      - 特殊值: NAN=0x7F, -NAN=0xFF
    """
    x_int = x.to(torch.int32)
    sign = (x_int >> 7) & 0x1
    exp = (x_int >> 3) & 0xF
    mant = x_int & 0x7

    # Handle denormals (exp=0): value = (-1)^s * 2^(-6) * mant/8
    is_denorm = (exp == 0)
    # Handle normals (exp=1..14): value = (-1)^s * 2^(exp-7) * (1 + mant/8)
    is_norm = (exp > 0) & (exp < 15)

    sign_val = torch.where(sign == 1, -1.0, 1.0)
    denorm_val = sign_val * (2.0 ** -6) * (mant.float() / 8.0)
    norm_val = sign_val * (2.0 ** (exp.float() - 7.0)) * (1.0 + mant.float() / 8.0)

    result = torch.where(is_norm, norm_val, torch.zeros_like(norm_val))
    result = torch.where(is_denorm, denorm_val, result)
    return result


def dequant_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP8 E5M2 (5 exponent + 2 mantissa bits) → float32
    E5M2 格式: s|eeeee|mm
      - sign: 1 bit
      - exponent: 5 bits, bias=15
      - mantissa: 2 bits
    """
    x_int = x.to(torch.int32)
    sign = (x_int >> 7) & 0x1
    exp = (x_int >> 2) & 0x1F
    mant = x_int & 0x3

    is_denorm = (exp == 0)
    is_norm = (exp > 0) & (exp < 31)

    sign_val = torch.where(sign == 1, -1.0, 1.0)
    denorm_val = sign_val * (2.0 ** -14) * (mant.float() / 4.0)
    norm_val = sign_val * (2.0 ** (exp.float() - 15.0)) * (1.0 + mant.float() / 4.0)

    result = torch.where(is_norm, norm_val, torch.zeros_like(norm_val))
    result = torch.where(is_denorm, denorm_val, result)
    return result


def dequant_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP4 E2M1 (2 exponent + 1 mantissa bits) → float32
    输入: int8 tensor, 每 byte 包含 2 个 fp4 值 (低 4bit 为第一个值)
    E2M1 格式: s|ee|m
      - sign: 1 bit, exponent: 2 bits (bias=1), mantissa: 1 bit
    """
    # 解包 nibbles: low nibble = first value, high nibble = second value
    x_int = x.to(torch.int32)
    low = x_int & 0xF
    high = (x_int >> 4) & 0xF

    def _decode_one(val):
        sign = (val >> 3) & 0x1
        exp = (val >> 1) & 0x3
        mant = val & 0x1
        sign_val = torch.where(sign == 1, -1.0, 1.0)
        # Normal: val = (-1)^s * 2^(exp-bias) * (1 + mant/2), bias=1
        # Denormal (exp=0): val = (-1)^s * 2^(1-bias) * mant/2 = (-1)^s * mant/2
        is_denorm = (exp == 0)
        is_norm = ~is_denorm
        denorm_val = sign_val * mant.float() / 2.0
        norm_val = sign_val * (2.0 ** (exp.float() - 1.0)) * (1.0 + mant.float() / 2.0)
        result = torch.where(is_norm, norm_val, torch.zeros_like(norm_val))
        result = torch.where(is_denorm, denorm_val, result)
        return result

    return _decode_one(low), _decode_one(high)


def dequant_fp8_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """
    模拟 FP8 E8M0 (8 exponent bits, 0 mantissa bits) → float32
    纯指数格式: value = 2^(e8m0_val)
    用于 MX format 的 scale
    """
    scale_int = scale.to(torch.int32)
    return 2.0 ** (scale_int.float())


# ============================================================================
# 核心: QuantBatchMatmul Golden 参考实现
# ============================================================================

def weight_quant_batch_matmul_golden(
    x: torch.Tensor,                          # activation: [..., M, K]
    weight: torch.Tensor,                     # quantized weight: [K, N] or [N, K] (depends on transpose)
    antiquant_scale: torch.Tensor,            # dequant scale: scalar or [N] or [N, K/group]
    antiquant_offset: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    quant_offset: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,            # 0=per-channel, >0=per-group
    dtype: int = -1,                           # -1=inherit x dtype, 1=FP16, 27=BF16
    inner_precise: int = 0,                    # 对 golden 无影响 (始终 float32 内部计算)
) -> torch.Tensor:
    """
    QuantBatchMatmul 的纯 PyTorch 参考实现 (CPU 可运行).

    算法:
      1. Transpose (按需)
      2. Weight Dequant: w_deq = w_q * scale + offset
      3. MatMul: output = x @ w_deq
      4. Bias Add (可选)
      5. Output Requant (可选): output = output * q_scale + q_offset
      6. Cast to target dtype
    """
    # ---- Step 0: 类型提升到 float32 ----
    x_f32 = x.float()
    w_f32 = weight.float()
    scale_f32 = antiquant_scale.float()
    offset_f32 = antiquant_offset.float() if antiquant_offset is not None else None
    q_scale_f32 = quant_scale.float() if quant_scale is not None else None
    q_offset_f32 = quant_offset.float() if quant_offset is not None else None
    bias_f32 = bias.float() if bias is not None else None

    # ---- Step 1: Transpose ----
    if transpose_x:
        x_f32 = x_f32.transpose(-2, -1)
        # [..., M, K] -> [..., K, M]

    if transpose_weight:
        w_f32 = w_f32.transpose(-2, -1)
        # [N, K] -> [K, N]  or [..., N, K] -> [..., K, N]

    # ---- Step 2: Weight Dequant ----
    # 现在 w_f32 shape = [..., K, N]
    K_dim = -2
    N_dim = -1
    output_channels = w_f32.shape[N_dim]
    inner_dim = w_f32.shape[K_dim]  # K

    if antiquant_group_size == 0:
        # Per-channel: scale shape = [N] 或其他可广播到 [N] 的形状
        # 需要 scale 维度对齐到 weight 的 N 维
        w_deq = w_f32 * scale_f32 + (offset_f32 if offset_f32 is not None else 0.0)
    else:
        # Per-group: scale shape = [N, ceil(K/group_size)] 或类似
        # 在 K 维上按 group_size 分组, 每组共享一个 scale
        group_size = antiquant_group_size
        num_groups = (inner_dim + group_size - 1) // group_size

        # 将 w 重塑为 [..., N, num_groups, group_size]
        # 需要先做 padding
        padded_K = num_groups * group_size
        if padded_K > inner_dim:
            pad_amount = padded_K - inner_dim
            w_f32 = torch.nn.functional.pad(w_f32, (0, 0, 0, pad_amount))

        w_reshaped = w_f32.reshape(*w_f32.shape[:-2], output_channels, num_groups, group_size)

        # scale 通常 shape = [N, num_groups]
        # 展开 scale 到 [..., 1, N, num_groups, 1] 来广播
        scale_reshaped = scale_f32.reshape(*([1] * (w_f32.dim() - 2)), output_channels, num_groups, 1)
        offset_reshaped = None
        if offset_f32 is not None:
            offset_reshaped = offset_f32.reshape(*([1] * (w_f32.dim() - 2)), output_channels, num_groups, 1)

        w_deq_group = w_reshaped * scale_reshaped
        if offset_reshaped is not None:
            w_deq_group = w_deq_group + offset_reshaped

        w_deq = w_deq_group.reshape(*w_f32.shape[:-2], output_channels, padded_K).transpose(-2, -1)

        # 去掉 padding
        if padded_K > inner_dim:
            w_deq = w_deq[..., :inner_dim, :]

    # w_deq 现在的 shape 应该是 [..., K, N]

    # ---- Step 3: MatMul ----
    output = torch.matmul(x_f32, w_deq)
    # output shape: [..., M, N]

    # ---- Step 4: Bias Add ----
    if bias_f32 is not None:
        output = output + bias_f32

    # ---- Step 5: Output Requant (可选) ----
    if q_scale_f32 is not None:
        output = output * q_scale_f32
    if q_offset_f32 is not None:
        output = output + q_offset_f32

    # ---- Step 6: Cast to target dtype ----
    if dtype == 1:          # FP16
        output = output.half()
    elif dtype == 27:       # BF16
        output = output.bfloat16()
    elif dtype == -1:
        # 继承 x 的 dtype
        output = output.to(x.dtype)
    # else: 保持 float32

    return output


# ============================================================================
# torch_npu 对比接口 (预留, 后续接入 NPU 后取消注释)
# ============================================================================

def compare_with_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    antiquant_scale: torch.Tensor,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    对比 PyTorch Golden 结果和 torch_npu 算子结果.

    在有 NPU 环境中调用:
        golden_out, npu_out, passed = compare_with_npu(x, weight, scale, ...)

    Args:
        与 weight_quant_batch_matmul_golden 相同的参数.

    Returns:
        (golden_output, npu_output, is_pass)
    """
    # Golden 参考结果
    golden_out = weight_quant_batch_matmul_golden(
        x, weight, antiquant_scale, **kwargs
    )

    # 转为 CPU 用于比较
    golden_out_cpu = golden_out.cpu()

    # TODO: 替换为实际的 torch_npu 调用
    # ------------------------------------------------------------------
    # import torch_npu
    #
    # x_npu = x.npu() if not x.is_npu else x
    # weight_npu = weight.npu() if not weight.is_npu else weight
    # scale_npu = antiquant_scale.npu() if not antiquant_scale.is_npu else antiquant_scale
    # bias_npu = kwargs.get('bias')
    # if bias_npu is not None and not bias_npu.is_npu:
    #     bias_npu = bias_npu.npu()
    #
    # npu_out = torch_npu.npu_weight_quant_batch_matmul_v2(
    #     x_npu, weight_npu, scale_npu,
    #     antiquant_offset=kwargs.get('antiquant_offset'),
    #     quant_scale=kwargs.get('quant_scale'),
    #     quant_offset=kwargs.get('quant_offset'),
    #     bias=bias_npu,
    #     transpose_x=kwargs.get('transpose_x', False),
    #     transpose_weight=kwargs.get('transpose_weight', False),
    #     antiquant_group_size=kwargs.get('antiquant_group_size', 0),
    #     dtype=kwargs.get('dtype', -1),
    #     inner_precise=kwargs.get('inner_precise', 0),
    # )
    # npu_out_cpu = npu_out.cpu()
    # ------------------------------------------------------------------

    # 当前使用 Golden 自身作为占位 (无 NPU 环境时)
    npu_out_cpu = golden_out_cpu.clone()

    # 比较
    is_pass = _compare_tensors(golden_out_cpu, npu_out_cpu, golden_out_cpu.dtype)

    return golden_out, npu_out_cpu, is_pass


# ============================================================================
# 比较工具
# ============================================================================

def _compare_tensors(
    ref: torch.Tensor,
    actual: torch.Tensor,
    dtype: torch.dtype,
    cos_sim_threshold: float = 0.99,
    max_rel_err_threshold: float = 5e-3,
) -> bool:
    """比较两个 tensor 是否在误差范围内一致."""
    ref_f32 = ref.float()
    actual_f32 = actual.float()

    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten().unsqueeze(0),
        actual_f32.flatten().unsqueeze(0)
    ).item()

    # 最大相对误差
    abs_diff = torch.abs(ref_f32 - actual_f32)
    ref_abs = torch.abs(ref_f32)
    # 避免除零
    rel_err = abs_diff / (ref_abs + 1e-8)
    max_rel_err = rel_err.max().item()

    # MSE
    mse = torch.nn.functional.mse_loss(ref_f32, actual_f32).item()

    if dtype == torch.float32:
        pass_cos = cos_sim >= 0.999
        pass_rel = max_rel_err <= 1e-5
    elif dtype == torch.float16:
        pass_cos = cos_sim >= 0.99
        pass_rel = max_rel_err <= 5e-3
    elif dtype == torch.bfloat16:
        pass_cos = cos_sim >= 0.98
        pass_rel = max_rel_err <= 1e-2
    else:
        pass_cos = cos_sim >= 0.99
        pass_rel = max_rel_err <= 5e-3

    passed = pass_cos and pass_rel

    # 打印
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} | cos_sim={cos_sim:.6f} | max_rel_err={max_rel_err:.6e} | MSE={mse:.6e}")
    return passed


# ============================================================================
# 自测用例
# ============================================================================

class TestCase:
    """单个测试用例."""
    def __init__(self, name: str, description: str, params: dict):
        self.name = name
        self.description = description
        self.params = params

    def run(self, verbose: bool = False) -> bool:
        """
        运行测试: 生成随机输入 → 调用 golden → 验证一致性.

        通过两次调用 golden (不同初始化) 验证确定性,
        以及与手动 PyTorch 分步计算对比.
        """
        p = self.params
        M, K, N = p.get('M', 32), p.get('K', 128), p.get('N', 64)
        batch = p.get('batch', 1)
        transpose_x = p.get('transpose_x', False)
        transpose_weight = p.get('transpose_weight', False)
        has_bias = p.get('has_bias', True)
        has_offset = p.get('has_offset', False)
        antiquant_group_size = p.get('antiquant_group_size', 0)
        out_dtype_val = p.get('dtype', -1)
        weight_dtype = p.get('weight_dtype', torch.int8)
        act_dtype = p.get('act_dtype', torch.float16)
        scale_dtype = p.get('scale_dtype', torch.float16)
        has_output_quant = p.get('has_output_quant', False)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Test: {self.name}")
            print(f"  Description: {self.description}")
            print(f"  Shape: batch={batch}, M={M}, K={K}, N={N}")
            print(f"  Flags: transX={transpose_x}, transW={transpose_weight}, "
                  f"group_size={antiquant_group_size}, dtype={out_dtype_val}")
            print(f"  Features: bias={has_bias}, offset={has_offset}, "
                  f"output_quant={has_output_quant}")

        # ---- 生成输入 ----
        if batch > 1:
            x_shape = (batch, M, K)
            bias_shape = (batch, N) if has_bias else None
        else:
            x_shape = (M, K)
            bias_shape = (N,) if has_bias else None

        # Activation
        x = torch.randn(x_shape, dtype=act_dtype)

        # 量化权重 (INT8)
        if weight_dtype == torch.int8:
            weight = torch.randint(-128, 127, (K, N), dtype=torch.int8)
        elif weight_dtype == torch.float8_e4m3fn:
            # PyTorch >= 2.1 supports float8
            weight = torch.randn(K, N).to(torch.float8_e4m3fn)
        else:
            weight = torch.randn(K, N).to(weight_dtype)

        # Scale
        if antiquant_group_size == 0:
            # Per-channel: scale shape = (N,)
            scale_shape = (N,)
        else:
            # Per-group: scale shape = (N, ceil(K/group_size))
            num_groups = (K + antiquant_group_size - 1) // antiquant_group_size
            scale_shape = (N, num_groups)

        scale = torch.randn(scale_shape, dtype=scale_dtype).abs() + 0.01

        # Offset (可选)
        offset = torch.randn(scale_shape, dtype=scale_dtype) * 0.01 if has_offset else None

        # Bias (可选)
        bias = torch.randn(bias_shape, dtype=act_dtype) * 0.01 if has_bias else None

        # 输出重量化 scale/offset (可选)
        q_scale = torch.randn(1, dtype=torch.float32).abs() + 0.1 if has_output_quant else None
        q_offset = torch.randn(1, dtype=torch.float32) * 0.01 if has_output_quant else None

        # ---- 运行 Golden ----
        t0 = time.time()
        out1 = weight_quant_batch_matmul_golden(
            x, weight, scale,
            antiquant_offset=offset,
            quant_scale=q_scale,
            quant_offset=q_offset,
            bias=bias,
            transpose_x=transpose_x,
            transpose_weight=transpose_weight,
            antiquant_group_size=antiquant_group_size,
            dtype=out_dtype_val,
        )
        t1 = time.time()

        # ---- 确定性验证: 第二次运行应该得到完全一样的结果 ----
        out2 = weight_quant_batch_matmul_golden(
            x, weight, scale,
            antiquant_offset=offset,
            quant_scale=q_scale,
            quant_offset=q_offset,
            bias=bias,
            transpose_x=transpose_x,
            transpose_weight=transpose_weight,
            antiquant_group_size=antiquant_group_size,
            dtype=out_dtype_val,
        )
        t2 = time.time()

        # 检查确定性
        if not torch.equal(out1, out2):
            print(f"  ❌ DETERMINISM FAIL: 两次调用结果不一致!")
            return False

        # ---- 分步验证: 手动对比 ----
        # 逐步用 PyTorch 基础操作计算参考结果
        x_ref = x.float()
        w_ref = weight.float()

        if transpose_x:
            x_ref = x_ref.transpose(-2, -1)
        if transpose_weight:
            w_ref = w_ref.transpose(-2, -1)

        # 手动反量化
        if antiquant_group_size == 0:
            w_deq_ref = w_ref * scale.float()
            if offset is not None:
                w_deq_ref = w_deq_ref + offset.float()
        else:
            gs = antiquant_group_size
            ng = (K + gs - 1) // gs
            pad_k = ng * gs
            w_pad = torch.nn.functional.pad(w_ref, (0, 0, 0, pad_k - K))
            w_reshape = w_pad.reshape(N, ng, gs)
            s_reshape = scale.float().reshape(1, N, ng, 1)
            w_deq_g = w_reshape * s_reshape
            if offset is not None:
                o_reshape = offset.float().reshape(1, N, ng, 1)
                w_deq_g = w_deq_g + o_reshape
            w_deq_ref = w_deq_g.reshape(N, pad_k)[:K, :].transpose(-2, -1)

        ref = torch.matmul(x_ref, w_deq_ref)
        if bias is not None:
            ref = ref + bias.float()
        if q_scale is not None:
            ref = ref * q_scale.float()
        if q_offset is not None:
            ref = ref + q_offset.float()

        target_dtype = {
            -1: x.dtype,
            1: torch.float16,
            27: torch.bfloat16,
        }.get(out_dtype_val, torch.float32)
        ref = ref.to(target_dtype)

        # 比较 golden 和分步参考
        print(f"  [Golden vs Stepwise]")
        passed_step = _compare_tensors(ref, out1, target_dtype)

        # 耗时
        if verbose:
            print(f"  Golden time: {(t1 - t0) * 1000:.3f} ms")

        return passed_step


# ============================================================================
# 测试套件
# ============================================================================

def get_test_cases() -> list:
    """返回所有内置测试用例."""
    return [
        # ---- Per-tensor ----
        TestCase(
            "per_tensor_fp16",
            "W8A16, per-tensor (scalar scale), FP16 input/output",
            dict(M=32, K=128, N=64, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        # ---- Per-channel ----
        TestCase(
            "per_channel_fp16_small",
            "W8A16, per-channel scale, M=32, K=128, N=64",
            dict(M=32, K=128, N=64, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "per_channel_fp16_medium",
            "W8A16, per-channel, M=256, K=512, N=256",
            dict(M=256, K=512, N=256, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "per_channel_bf16",
            "W8A16, per-channel, BF16 input/output",
            dict(M=128, K=256, N=128, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=27, act_dtype=torch.bfloat16,
                 scale_dtype=torch.bfloat16)
        ),
        TestCase(
            "per_channel_no_bias",
            "W8A16, per-channel, no bias",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=0,
                 has_bias=False, dtype=1)
        ),
        TestCase(
            "per_channel_with_offset",
            "W8A16, per-channel, with antiquant offset",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=0,
                 has_bias=True, has_offset=True, dtype=1)
        ),

        # ---- Per-group ----
        TestCase(
            "per_group_128",
            "W8A16, per-group (group_size=128), K=256 → 2 groups",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=128,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "per_group_32",
            "W8A16, per-group (group_size=32, MX-style), K=128 → 4 groups",
            dict(M=32, K=128, N=64, batch=1, antiquant_group_size=32,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "per_group_non_divisible",
            "W8A16, group_size=128 but K=200 (non-divisible → padding)",
            dict(M=32, K=200, N=64, batch=1, antiquant_group_size=128,
                 has_bias=True, dtype=1)
        ),

        # ---- Transpose ----
        TestCase(
            "transpose_weight",
            "W8A16, transposeWeight=True (weight shape N×K)",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=0,
                 transpose_weight=True, has_bias=True, dtype=1)
        ),
        TestCase(
            "transpose_x",
            "W8A16, transposeX=True (activation shape K×M)",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=0,
                 transpose_x=True, has_bias=True, dtype=1)
        ),

        # ---- Batch ----
        TestCase(
            "batch_matmul",
            "W8A16, 3D batch (batch=4), per-channel",
            dict(M=32, K=256, N=128, batch=4, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "batch_per_group",
            "W8A16, 3D batch (batch=2), per-group group_size=64",
            dict(M=64, K=256, N=128, batch=2, antiquant_group_size=64,
                 has_bias=True, dtype=1)
        ),

        # ---- LLM 典型场景 ----
        TestCase(
            "llm_decode",
            "LLM Decode: M=1 (single token), K=4096, N=4096, per-channel",
            dict(M=1, K=4096, N=4096, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "llm_prefill",
            "LLM Prefill: M=128 (128 tokens), K=4096, N=4096, per-channel",
            dict(M=128, K=4096, N=4096, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),

        # ---- 输出重量化 ----
        TestCase(
            "output_requant_fp16",
            "W8A16 + output requantization (quant_scale + quant_offset)",
            dict(M=64, K=256, N=128, batch=1, antiquant_group_size=0,
                 has_bias=True, has_output_quant=True, dtype=1)
        ),

        # ---- Edge cases ----
        TestCase(
            "edge_small",
            "边界: 极小形状 M=1, K=32, N=16",
            dict(M=1, K=32, N=16, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
        TestCase(
            "edge_non_align",
            "边界: 非对齐 K=33, N=17",
            dict(M=16, K=33, N=17, batch=1, antiquant_group_size=0,
                 has_bias=True, dtype=1)
        ),
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QuantBatchMatmul Golden 验证脚本"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细测试信息")
    parser.add_argument("--profile", "-p", action="store_true",
                        help="打印耗时统计")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="只运行名称包含指定字符串的测试 (如 'per_channel')")
    parser.add_argument("--list", "-l", action="store_true",
                        help="仅列出所有测试用例")

    args = parser.parse_args()

    test_cases = get_test_cases()

    if args.list:
        print(f"共 {len(test_cases)} 个测试用例:\n")
        for tc in test_cases:
            print(f"  [{tc.name}] {tc.description}")
        return

    # 过滤
    if args.filter:
        test_cases = [tc for tc in test_cases if args.filter.lower() in tc.name.lower()]
        if not test_cases:
            print(f"没有匹配 '{args.filter}' 的测试用例")
            return
        print(f"过滤后: {len(test_cases)} 个测试用例匹配 '{args.filter}'")

    print(f"\n{'='*60}")
    print(f"QuantBatchMatmul Golden 验证")
    print(f"共 {len(test_cases)} 个测试用例")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    times = []

    for tc in test_cases:
        print(f"\n[{tc.name}] {tc.description}")
        t0 = time.time()
        try:
            ok = tc.run(verbose=args.verbose)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            ok = False

        elapsed = time.time() - t0
        times.append((tc.name, elapsed))

        if ok:
            passed += 1
        else:
            failed += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"结果: {passed} passed, {failed} failed, {len(test_cases)} total")
    print(f"{'='*60}")

    if args.profile:
        print(f"\n耗时统计:")
        for name, t in sorted(times, key=lambda x: -x[1]):
            print(f"  {name:40s} {t*1000:8.3f} ms")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
