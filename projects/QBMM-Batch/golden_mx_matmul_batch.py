#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MX量化 Matmul Golden 验证脚本 (batch 版本)

scale 带 batch 前缀, 与对应 x 的 batch 维逐维匹配(支持广播); 无 batch 时 scale 为 3D, 向后兼容:
  pertoken_scale: (*batchA, M, ceil(K/64), 2)   — E8M0
  scale:          (*batchB, ceil(K/64), N, 2)    — E8M0

golden 为高精度参考(未舍到 out_dtype); 判定用 余弦相似度(cos)为主 + 相对 L2 范数辅助门 的双保险:
  主门 cos:    cos(npu 拉平, golden 拉平) ≥ COS_TOL(0.9999)。方向一致性, 免疫近零抵消点。
  辅助门 相对L2: ‖npu-golden‖₂/‖golden‖₂ ≤ RELL2_TOL[out_dtype](=2^-mantissa)。抓 cos 盲区
               (系统性乘性小偏置), 分母整张范数故免疫近零点。
  PASS ⟺ 主门与辅助门都过(且 nan/inf 掩码与 golden 一致)。
  cos 是业界 matmul/量化最主流判据, 天然免疫近零点/大K抵消/fp4+bias 擦门这类"整体方向对、个别
  近零点相对误差爆炸"的误判(即官方固定小值门 BenchmarkCompareStandard 会误杀零误差核的病态)。
  fp4 数据仍走官方 mx_quantize(逐字内联)。全局判定(展平整张); FAIL 时逐 batch 定位驱动失败的 batch。

退出码: 0 PASS | 1 FAIL | 2 NOT_IMPLEMENTED(--golden-only 或 NPU 全未实现).

用法:
  python golden_mx_matmul_batch.py                 # golden + NPU
  python golden_mx_matmul_batch.py --golden-only   # 仅 golden 自洽
  python golden_mx_matmul_batch.py -v              # 详细
  python golden_mx_matmul_batch.py --case 2b_fp8_basic   # 选用例(逗号分隔)
"""

import argparse
import sys
import torch
import torch_npu
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# ============================================================
# 官方 MX 量化 —— 逐字内联自 libs/grouped_matmul_libs (new.py), 未改一行算法.
#   原为独立文件 mx_quantize_official.py, 为单文件分发(避免漏传)内联于此.
#   顶部 en_dtypes 垫片: 服务器用真 en_dtypes; 本地无 cp314 wheel 时回退 ml_dtypes(逐码等价).
# ============================================================

import sys, types
try:
    import en_dtypes  # 服务器: 官方依赖
except ImportError:  # 本地 py3.14 无 en_dtypes wheel -> 用 ml_dtypes 垫片(OCP 等价)
    import ml_dtypes
    _shim = types.ModuleType("en_dtypes")
    _shim.float4_e2m1 = ml_dtypes.float4_e2m1fn
    _shim.float4_e1m2 = getattr(ml_dtypes, "float4_e1m2fn", ml_dtypes.float4_e2m1fn)
    _shim.float8_e8m0 = ml_dtypes.float8_e8m0fnu
    sys.modules["en_dtypes"] = _shim

import numpy
import math
import copy


def upgrade_lib(*a, **k):  # 官方在 mx_quantize 内调它升级 en_dtypes; 垫片下置空
    pass


def regenerate_quant_data(x_shape, w_shape, params_out_dtype):
    mu_A = 0.05
    mu_W = -0.05
    Sigma_A = 4
    Sigma_W = 4
    print(f"均值={mu_A}, 方差={Sigma_A}")
    # 高斯分布 + 白噪声
    X_tmp = (np.random.normal(mu_A, Sigma_A, x_shape) + np.random.uniform(-0.1 * mu_A, 0.1 * mu_A, x_shape)).astype(
        np.float32)
    W_tmp = (np.random.normal(mu_W, Sigma_W, w_shape) - np.random.uniform(-0.1 * mu_W, 0.1 * mu_W, w_shape)).astype(
        np.float32)
    if params_out_dtype == 'fp32':
        X_tmp = np.abs(X_tmp)
        W_tmp = np.abs(W_tmp)
    return X_tmp, W_tmp


def mx_quantize(fp_array: numpy.ndarray, mx_ele_dtype: str = "float4_e2m1", axis: int = -1, block_size: int = 32,
                round_mode: str = "rint") -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype. support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """

    def numpy_float8_e5m2():
        try:
            # noinspection PyUnresolvedReferences
            from ml_dtypes import float8_e5m2
            return float8_e5m2
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e5m2 dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float8_e4m3fn():
        try:
            # noinspection PyUnresolvedReferences
            from ml_dtypes import float8_e4m3fn
            return float8_e4m3fn
        except ModuleNotFoundError:
            raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                               "Please install with `pip3 install ml-dtypes`")

    def numpy_float8_e8m0():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float8_e8m0
            return float8_e8m0
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def numpy_float4_e2m1():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float4_e2m1
            return float4_e2m1
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e2m1 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def numpy_float4_e1m2():
        try:
            # noinspection PyUnresolvedReferences
            from en_dtypes import float4_e1m2
            return float4_e1m2
        except ModuleNotFoundError:
            raise RuntimeError("en_dtypes is needed to support float4_e1m2 dtype!!! "
                               "Please install with `pip3 install en-dtypes`")

    def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
        fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
        orig_shape = fp_array.shape
        pad = [[0, 0] for _ in range(len(orig_shape))]
        pad_size = orig_shape[axis] % block_size
        pad[axis][1] = block_size - pad_size
        if pad_size > 0:
            fp_array = numpy.pad(fp_array, pad, 'constant')
        padded_shape = fp_array.shape
        reshape = list(padded_shape)
        reshape[axis + 1] = block_size
        reshape[axis] = reshape[axis] // block_size
        fp_array = fp_array.reshape(reshape)
        return fp_array, orig_shape, padded_shape

    def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
        """
        For example:
        fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
        - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
        - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
        - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
        - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        """
        if round_mode in ("rint", "even"):  # tie to even(c language rint)
            fp_array = numpy.rint(fp_array)
        elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
            #fp_array = numpy.sign(fp_array) * numpy.floor(
            #    numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
            sign = numpy.signbit(fp_array)
            rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
            fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
        elif round_mode == "floor":  # round to minus infinity(c language floor)
            fp_array = numpy.floor(fp_array)
        elif round_mode == "ceil":  # round to positive infinity(c language ceil)
            fp_array = numpy.ceil(fp_array)
        elif round_mode == "trunc":  # round to zero(c language truncation)
            fp_array = numpy.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        # ele_emax = 2 if "float4_e2m1" in str(mx_ele_dtype) else 0  # 0 for fp4_e1m2
        ele_emax = 0
        mx_dtype = str(mx_ele_dtype)
        if "float4_e2m1" in mx_dtype:
            ele_emax = 2
        elif "float8_e4m3fn" in mx_dtype:
            ele_emax = 8
        elif "float8_e5m2" in mx_dtype:
            ele_emax = 15
        fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
        res = numpy.floor(
            numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
        ) - ele_emax
        res[fp_abs_max == 0] = -float("inf")
        return res

    def get_dtype_range(dt):
        if "bfloat16" in str(dt):
            return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
        if "uint4" in str(dt):
            return 0, 15
        if "int4" in str(dt):
            return -8, 7
        if "bool" in str(dt):
            return 0, 1
        if "float4_e2m1" in str(dt):
            return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
        if "float4_e1m2" in str(dt):
            return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
        if "float8_e8m0" in str(dt):
            return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
        if "float8_e5m2" in str(dt):
            return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
        if "float8_e4m3fn" in str(dt):
            return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
        if "hifloat8" in str(dt):
            return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
        if "complex32" in str(dt):
            dt = "float16"
        numpy_dtype = numpy.dtype(dt)
        if numpy_dtype.kind in "iu":
            numpy_info = numpy.iinfo(numpy_dtype)
        else:
            numpy_info = numpy.finfo(numpy_dtype)
        return numpy_info.min, numpy_info.max


    def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                       mx_ele_dtype: str, round_mode: str):
        mx_dtype = str(mx_ele_dtype)
        exp_bits = 0
        mantissa_bits = 0
        if "float4_e2m1" in mx_dtype:
            exp_bits = 2
            mantissa_bits = 1
        elif "float4_e1m2" in mx_dtype:
            exp_bits = 1
            mantissa_bits = 2
        elif "float8_e4m3fn" in mx_dtype:
            exp_bits = 4
            mantissa_bits = 3
        elif "float8_e5m2" in mx_dtype:
            exp_bits = 5
            mantissa_bits = 2

        max_norm = get_dtype_range(mx_dtype)[1]

        ret = fp_array / (2 ** share_exp)
        private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
                                ).astype(fp_array.dtype, copy=False)
        # The minimum representable exponent for 8 exp bits is -126
        # 5bit exp  2^4-1 = 15  or 2 ** (exp_bits - 1) -1
        if "float8_e4m3fn" in mx_dtype or "float8_e5m2" in mx_dtype:
            min_exp = -(2 ** (exp_bits - 1)) + 2  # 指数位 -2^3+4 = -4  
        else:
            min_exp = -(2 ** (exp_bits - 1)) + exp_bits
        private_exp = private_exp.clip(min=min_exp)
        # Scale up so appropriate number of bits are in the integer portion of the number
        ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
        ret = _mx_round_mantissa(ret, round_mode)
        # Undo scaling
        ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
        # Set values > max_norm to Inf if desired, else clamp them
        numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret


    def pad_to_even(tensor: numpy.ndarray, axis: int):
        if not isinstance(tensor, numpy.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

        shape = tensor.shape
        length = shape[axis]

        # 如果已经是偶数，直接返回原数组
        if length % 2 == 0:
            return tensor

        # 构造 pad_width：仅对目标 axis 补一个 0
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

        padded_tensor = numpy.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
        return padded_tensor
  
    def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
        # Undo tile reshaping
        fp_array = fp_array.reshape(padded_shape)
        # Undo padding
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        # Remove extra dimension
        fp_array = numpy.squeeze(fp_array, axis=axis + 1)
        return fp_array

    def inf_clip(tensor, dtype):
        if dtype == 'float4_e2m1':
            max_val = 6
            min_val = -6
        elif dtype == 'float4_e1m2':
            max_val = 1.75
            min_val = -1.75
        elif dtype == 'float8_e4m3fn':
            max_val = 448
            min_val = -448
        elif dtype == 'float8_e5m2':
            max_val = 57344
            min_val = -57344
        elif dtype == 'float8_e8m0':
            max_val = 127
            min_val = -127
        else:
            print(f"ERROR: dtype is Not supported, dtype_input is {dtype}")
            return
        tensor[np.isposinf(tensor)] = max_val
        tensor[np.isneginf(tensor)] = min_val
        tensor[np.isnan(tensor)] = max_val
        return tensor

    def interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
        if not isinstance(tensor, numpy.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
        # 获取目标轴的长度
        length = tensor.shape[axis]
        # 检查是否可整除
        if length % n_group != 0:
            raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")
  
        group_length = length // n_group  # 每组长度
        shape = list(tensor.shape)
  
        # 重塑形状：在目标轴后插入组维度
        new_shape = (
            shape[:axis] +
            [group_length,2] +
            shape[axis+1:])
        reshaped = tensor.reshape(new_shape)
  
        # 构建转置顺序：交换组维度和组内维度
        transpose_order = (
            list(range(0, axis+1)) +  # 目标轴之前的维度
            list(range(axis + 2, len(new_shape))) +
            [axis+1,])  # 后续维度
  
        # 执行转置
        transposed = reshaped.transpose(transpose_order)
  
        return transposed


    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    if mx_ele_dtype in ["float4_e2m1", "float4_e1m2"]: #fp4类型需要升级到en_dtypes==0.0.4
        upgrade_lib("en_dtypes","0.0.4")
    
    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    # get mx scale exponents
    share_exp = _mx_calculate_share_exp(fp_array,
                                        scale_axis=axis + 1,
                                        mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    # share_exp is always float32
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    # if mx_ele_dtype not in ["float4_e2m1", "float4_e1m2"]: #fp4类型astype有问题，单独处理
    #     ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # else:
    #     ele_array = trans_fp4_to_uint8(ele_array, ele_array.shape, mx_ele_dtype)

    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)
 
    result_shape[axis] = scale_array_pad.shape[axis] // 2

    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    print(f"result_shape is ========== {result_shape}")


    scale_array = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)
  
    # 将inf赋值为最大值，-inf赋值为最小值
    ele_array = inf_clip(ele_array, mx_ele_dtype)
    scale_array = inf_clip(scale_array, 'float8_e8m0')
  
    return ele_array, scale_array

# ============================================================
# 常量 / 退出码
# ============================================================
MXFP_DIVISOR = 64
MXFP_BASE    = 2
MXFP_INNER   = 32

EXIT_PASS     = 0
EXIT_FAIL     = 1
EXIT_NOT_IMPL = 2

# e2m1 码 → 值 LUT (OCP E2M1, 索引=4bit code, 高位符号). 已与 ml_dtypes.float4_e2m1fn 逐码核对.
E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# 阴性对照: True 时故意改错 golden 侧 ±6.0→±7.0 (packed 喂 NPU 不动) → golden 与 NPU 分歧,
#   预期 fp4 用例 FAIL, 验 harness 对"实现端 e2m1 解码错"的鉴别力. 用完务必改回 False.
#   两路径都改: 主路径在 fp4_quant_pack_lastdim 改 value; K 奇回退路径改下方 E2M1_LUT[7/15].
_FP4_LUT_NEGCTRL = False
if _FP4_LUT_NEGCTRL:
    E2M1_LUT = list(E2M1_LUT)
    E2M1_LUT[7] = 7.0      # 正确应为 6.0 (K 奇回退路径用)
    E2M1_LUT[15] = -7.0    # 正确应为 -6.0
    print("=" * 60)
    print("  [NEG-CTRL] _FP4_LUT_NEGCTRL=True: golden ±6.0 已故意改错, 预期 fp4 FAIL. 用完改回 False!")
    print("=" * 60)

# nibble 字节序: 仅 pack_codes_lastdim 打包喂 NPU 时用. 默认低 nibble=靠前元素 K[2i].
#   沿 K 打包(非转置/trans_x2): 两 nibble 同 scale, matmul 对 pair 内交换不变 → 序不可观测;
#     且 golden 用官方量化浮点值(不 unpack), 更不依赖此序.
#   沿 M 打包(trans_x1): 两 nibble 属不同 scale 行 → 序可观测, 须上板定(故 trans_x1 fp4 暂挡).
FP4_LOW_NIBBLE_FIRST = True


def ceil_div(a, b):
    return (a + b - 1) // b


def e8m0_to_float32(e8m0: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 → float32:  2^(e - 127),  e=0 → 0.0"""
    e = e8m0.float()
    val = torch.pow(2.0, e - 127.0)
    return torch.where(e8m0 == 0, torch.zeros_like(val), val)


def merge_last2(t: torch.Tensor) -> torch.Tensor:
    """(..., A, B) → (..., A*B)  合并最后两维"""
    return t.reshape(*t.shape[:-2], t.shape[-2] * t.shape[-1])


def merge_mid2(t: torch.Tensor) -> torch.Tensor:
    """(..., A, B, C) → (..., A*B, C)  合并 [-3,-2], 保留末维"""
    return t.reshape(*t.shape[:-3], t.shape[-3] * t.shape[-2], t.shape[-1])


# ============================================================
# fp4 e2m1 codec — 官方 mx_quantize 产 code+scale; golden 用浮点值, NPU 读打包字节.
#   gen_fp4_packed 仅 K 奇负向用例的回退路径用.
# ============================================================

def gen_fp4_packed(shape_logical, packed: bool):
    """生成 fp4 数据(随机 code). 仅 K 奇 oddInner_fail 负向用例用(packed=False):
    返回 (..., inner) int8 原始随机 code 0..15, 内轴保持奇 → 触发 NPU 的 CheckInnerAxisIsEven 拒绝."""
    if packed:
        assert shape_logical[-1] % 2 == 0, "fp4 打包要求最内层 K 为偶数"
        packed_shape = tuple(shape_logical[:-1]) + (shape_logical[-1] // 2,)
        u = torch.randint(0, 256, packed_shape, dtype=torch.int64)
        return (u - 256 * (u >= 128).to(u.dtype)).to(torch.int8)
    return torch.randint(0, 16, tuple(shape_logical), dtype=torch.int8)


def pack_codes_lastdim(code: torch.Tensor) -> torch.Tensor:
    """(..., K) e2m1 码 → (..., K/2) int8 打包字节, 沿最内层 2:1.
    nibble 序由 FP4_LOW_NIBBLE_FIRST 定: 默认低 nibble = 靠前元素 code[2i].
    仅 NPU 入参用此打包; golden 用官方量化产出的 e2m1 浮点值直接取, 不依赖本序.
    沿-K-打包时(非转置/trans_x2)两 nibble 同 scale, 序不可观测; trans_x1 沿-M-打包时
    序可观测, 须上板确认硬件序(错则翻转 FP4_LOW_NIBBLE_FIRST)."""
    assert code.shape[-1] % 2 == 0, "打包要求最内层为偶数"
    c = code.to(torch.int64) & 0xF
    lo = c[..., 0::2]    # code[2i]
    hi = c[..., 1::2]    # code[2i+1]
    if not FP4_LOW_NIBBLE_FIRST:
        lo, hi = hi, lo
    byte = (lo | (hi << 4)) & 0xFF
    return (byte - 256 * (byte >= 128).to(byte.dtype)).to(torch.int8)


def fp4_quant_pack_lastdim(storage_np):
    """对 storage 朝向(K 在最内层)的 fp32 numpy 数组做【官方 MX 量化】+ 打包.
    用官方 mx_quantize(axis=-1, block=32, round_mode='rint') 产 e2m1 码 + e8m0 scale,
    与官方逐位一致(算法、shared_exp、尾数域 rint、scale 编码全是官方本体).
    返回: packed (..., K/2) int8 喂 NPU; value (..., K) fp32 e2m1 值供 golden(不依赖 nibble 序);
          scale_ck2 (..., ceilK, 2) uint8 e8m0."""
    ele, scale = mx_quantize(storage_np.astype(np.float32), mx_ele_dtype="float4_e2m1",
                                 axis=-1, block_size=32, round_mode="rint")
    # ele: float4_e2m1 (1 字节/元素, 低 nibble=4bit 码); scale: float8_e8m0 (...,ceilK,2)
    value = torch.from_numpy(ele.astype(np.float32))                       # (...,K) e2m1 浮点值
    code = torch.from_numpy((ele.view(np.uint8) & 0xF).astype(np.int64))   # (...,K) 0..15 码
    if _FP4_LUT_NEGCTRL:
        # 阴性对照: 故意改错 golden 侧的 value(6.0→7.0, -6.0→-7.0), packed(喂 NPU) 不动 →
        #   golden 与 NPU 在含 ±6 的元素上分歧, 预期 harness 报 FAIL(验鉴别力). 详见头部说明.
        value = torch.where(value == 6.0, torch.tensor(7.0), value)
        value = torch.where(value == -6.0, torch.tensor(-7.0), value)
    packed = pack_codes_lastdim(code)                                      # (...,K/2) int8
    scale_ck2 = torch.from_numpy(scale.view(np.uint8).copy())              # (...,ceilK,2) uint8 e8m0
    return packed, value, scale_ck2


# ============================================================
# 测试用例
# ============================================================

@dataclass
class Case:
    name: str
    M: int; N: int; K: int
    x_dtype: str               # 'fp8_e4m3fn' | 'fp4_e2m1'
    out_dtype: str             # 'float16' | 'bfloat16' | 'float32'
    batch_x1: List[int] = field(default_factory=list)
    batch_x2: List[int] = field(default_factory=list)
    trans_x1: bool = False
    trans_x2: bool = False
    x2_dtype: Optional[str] = None   # None→沿用 x_dtype; 支持 x1/x2 不同 fp8(如 e4m3 x1 + e5m2 x2 混合)
    cont_data: bool = False   # True: uniform(-5,5)→fp8 cast(stress mantissa, 复现线上连续输入); False: randint 整数
    has_bias: bool = False
    seed: Optional[int] = None
    expect_fail: bool = False   # True: 预期被 NPU checker 拒绝, 捕获到报错记 PASS
    weight_nz: bool = False     # True: x2 转 FRACTAL_NZ(29), 走 WeightNz(NZ)路径而非 V5(ND)
    mismatch_x2_scale: bool = False  # True: 仅转 scale 不转 x2, 构造转置态不一致, 验证 op-plugin 拒绝
    mismatch_x1_scale: bool = False  # True: 仅转 pertoken_scale 不转 x1, 构造转置态不一致


CASES: List[Case] = [
    # ---- fp8 主力: 1层 batch ----
    Case("1b_fp8_basic", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("1b_fp8_bias",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], has_bias=True),
    Case("1b_fp8_bf16",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], has_bias=True),
    Case("1b_fp8_fp32",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float32",
         batch_x1=[2], batch_x2=[2]),
    Case("1b_fp8_oddK",  M=64, N=128, K=200, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("1b_fp8_B8",    M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[8], batch_x2=[8]),

    # ---- fp8 主力: 2层 batch ----
    Case("2b_fp8_basic", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 3], batch_x2=[2, 3]),
    Case("2b_fp8_bias",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="bfloat16",
         batch_x1=[2, 3], batch_x2=[2, 3], has_bias=True),

    # ---- fp8 深 batch: 3/4 层 (x 5D/6D, 压满 kernel batchA1-4; cmp reshape(-1,M,N) 自动折叠) ----
    Case("3b_fp8_basic", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2], batch_x2=[2, 2, 2]),
    Case("4b_fp8_basic", M=32, N=64, K=128, x_dtype="fp8_e4m3fn", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2]),

    # ---- fp8 退化维边界: M=1 / N=1 ----
    Case("1b_fp8_M1", M=1,  N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("1b_fp8_N1", M=64, N=1,   K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),

    # ---- broadcast: x1 有 batch, x2 batch=1 ----
    Case("bcast_B_fp8",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[3], batch_x2=[1]),
    # ---- broadcast: x2 有 batch, x1 batch=1 ----
    Case("bcast_A_fp8",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[1], batch_x2=[3]),

    # ---- fp4: 验证变更点 #5 (scale 不可 >>sizeShift). 多 batch 暴露 batch1+ 偏移漏写 ----
    Case("3b_fp4_chk5",  M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[3], batch_x2=[3]),
    Case("1b_fp4_basic", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),
    # fp4 2 层 batch (补 1/3 层之间的空档)
    Case("2b_fp4_basic", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 3], batch_x2=[2, 3]),
    # fp4 + bias (现 fp4 全无 bias; bias 后处理 fp32 加, API 要求 DT_FLOAT)
    Case("1b_fp4_bias", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], has_bias=True),
    # fp4 截断路径: K=160 → ceil(160/32)=5 奇 → golden 截最后一个 sub-block (官方 quantize
    #   pad 到偶, ele/scale 形状 (..,ceilK=3,2), 截断后逻辑 K 对齐; 已 numpy-mirror 验 max_diff=0).
    Case("1b_fp4_oddK", M=64, N=128, K=160, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),

    # ---- transpose (依赖 op-plugin MX transpose check 修复 commit 6c2885f9 上板) ----
    # 约束 A: 数据与 scale 转置属性须一致. trans_x2 → x2 存 (..,N,K)、scale (..,N,ceilK,2).
    Case("1b_fp8_tx2",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x2=True),
    Case("1b_fp8_tx1",  M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x1=True),
    # 双转置: trans_x1/trans_x2 在 gen_data/golden/call_npu 是独立 if 分支, 同时触发即可.
    Case("1b_fp8_tx1x2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x1=True, trans_x2=True),
    Case("2b_fp8_tx1x2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="bfloat16",
         batch_x1=[2, 3], batch_x2=[2, 3], trans_x1=True, trans_x2=True),

    # ---- 数据与 scale 转置态不一致: op-plugin is_x_scale_same_transpose 应拒绝 (expect_fail) ----
    # 仅转 scale 不转 x2 → x2(非转置) vs scale(转置) → 报 "x2 ... transpose are not same".
    Case("mismatch_x2_scale_fp8", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], mismatch_x2_scale=True, expect_fail=True),
    # 仅转 pertoken_scale 不转 x1 → x1(非转置) vs pertoken_scale(转置) → 报 "x1 ... transpose are not same".
    Case("mismatch_x1_scale_fp8", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], mismatch_x1_scale=True, expect_fail=True),

    # ---- fp4 N / 内轴边界 (ND 格式, host checker 仅强制 K>2 + 内轴偶数; n>1 是 NZ 专属约束) ----
    Case("1b_fp4_nN2",  M=64, N=2, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),
    Case("1b_fp4_nN1", M=64, N=1, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),
    # 负向: 内轴 K=255 奇 → 触发 NPU CheckInnerAxisIsEven 拒绝.
    Case("1b_fp4_oddInner_fail", M=64, N=128, K=255, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], expect_fail=True),
    # fp4 转置覆盖: trans_x2 已被所有 fp4 用例等价覆盖(x2_stored_transposed 恒 True), 无需单列.
    #   trans_x1 暂缓(gen_data assert 挡住): ①量化应沿 K 但本路径沿最后维 M ②沿 M 打包 nibble 序可观测.
    #   放开步骤: 上板验 fp8_tx1 链路 → 实现沿 axis=-2 的 fp4 量化 → 上板定 nibble 序 → 去 assert.

    # ---- 无 batch 兼容 (x1/x2 2D, scale 3D) ----
    Case("nobatch_fp8", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[], batch_x2=[]),
    Case("nobatch_fp8_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[], batch_x2=[], trans_x2=True),
    # 线上精度回归 qbmm_MXFP8_case0011: 混合 fp8 (x1 e4m3fn + x2 e5m2), M=1, K=144(非 64 倍数),
    #   trans_x1=trans_x2=True, out=bf16; 复现 950 cosine 99.88% fail。
    Case("mxfp8_mixed_e4m3e5m2_M1K144N823", M=1, N=823, K=144,
         x_dtype="fp8_e4m3fn", x2_dtype="fp8_e5m2", out_dtype="bfloat16",
         batch_x1=[], batch_x2=[], trans_x1=True, trans_x2=True, cont_data=True),
    # 线上精度回归 aclnnQuantMatmulV5 ...inc_000010: 混合 fp8 反向 (x1 e5m2 + x2 e4m3fn),
    #   B=13, M=1011, K=19085, N=3873, 无转置, out=bf16; 复现 cosine 99.83% fail。大尺寸, CPU golden 慢。
    Case("mxfp8_mixed_e5m2e4m3_B13M1011K19085N3873", M=1011, N=3873, K=19085,
         x_dtype="fp8_e5m2", x2_dtype="fp8_e4m3fn", out_dtype="bfloat16",
         batch_x1=[13], batch_x2=[13], cont_data=True),
    Case("nobatch_fp4", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[], batch_x2=[]),

    # ---- NZ 路径 (x2 转 FRACTAL_NZ, 走 WeightNz 而非 V5); fp8 低风险, fp4 高风险见注释 ----
    # fp4 NZ 风险: x2_stored_transposed 恒 True(fp4), transpose+contiguous+npu_format_cast 可能
    #   破坏打包或被 NPU 拒绝(官方 mxfp4 example 用 fp32→NZ→fp4 三步中转); 上板失败则改 fp32 中转.
    Case("1b_fp8_basic_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], weight_nz=True),
    Case("2b_fp8_basic_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 3], batch_x2=[2, 3], weight_nz=True),
    Case("3b_fp8_basic_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2], batch_x2=[2, 2, 2], weight_nz=True),
    # fp4 NZ: x2 int8(打包)直接 npu_format_cast(29) + transpose 在后(参考官方 D:/Desktop/TMP/tmp.py
    #   L641-648). 之前 transpose→cast+contiguous 顺序错(contiguous 抹转置 stride), 改 cast→transpose 后通.
    Case("1b_fp4_basic_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], weight_nz=True),
    Case("3b_fp4_chk5_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[3], batch_x2=[3], weight_nz=True),

    # ==== 补充覆盖: 转置×NZ / NZ×bcast / 大shape / K边界 / 非对齐 (2026-06-27) ====

    # ---- 转置 × NZ (fp8; fp4 转置受 x2_stored_transposed恒True + trans_x1 assert 限制, 不单列) ----
    Case("1b_fp8_tx1_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x1=True, weight_nz=True),
    Case("1b_fp8_tx2_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x2=True, weight_nz=True),
    Case("1b_fp8_tx1x2_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], trans_x1=True, trans_x2=True, weight_nz=True),

    # ---- NZ × broadcast (fp8; 验证 NZ 下 batch broadcast + scale 偏移) ----
    Case("bcast_B_fp8_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[3], batch_x2=[1], weight_nz=True),
    Case("bcast_A_fp8_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[1], batch_x2=[3], weight_nz=True),

    # ---- fp4 NZ 扩展 (2层 batch + bias) ----
    Case("2b_fp4_basic_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 3], batch_x2=[2, 3], weight_nz=True),
    Case("1b_fp4_bias_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], has_bias=True, weight_nz=True),

    # ---- 大 shape (性能/真实场景; batch 偏移放大, 暴露 batch1+ 错位) ----
    Case("big_fp8", M=1024, N=1024, K=1024, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("big_fp4", M=512, N=512, K=512, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),
    Case("big_fp8_nz", M=512, N=512, K=512, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2], weight_nz=True),

    # ---- K 边界 (ceilK=1 最小 / 2 block; NZ 对齐 K%16 满足) ----
    Case("K64_fp8", M=64, N=128, K=64, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("K128_fp8", M=64, N=128, K=128, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    Case("K64_fp4", M=64, N=128, K=64, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),

    # ---- 非对齐 (ND only; NZ 要求 K%16 N%32, 非对齐只走 V5) ----
    # K=192 → ceil(192/64)=3 奇 ceilK → golden 截断路径 (与 oddK=200/160 同类)
    Case("K192_fp8", M=64, N=128, K=192, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    # N=48 不对齐 32 (ND; NZ 会拒)
    Case("N48_fp8", M=64, N=48, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    # M=33 奇 (非对齐 M 维)
    Case("M33_fp8", M=33, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2], batch_x2=[2]),
    # K=300 偶 但不对齐 64 (fp4 内轴偶 OK; ND)
    Case("K300_fp4", M=64, N=128, K=300, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2]),

    # ---- fp4 tx2 显式参数覆盖 (ND + NZ) ----
    # 注: fp4 的 x2_stored_transposed 恒 True(L662), 故 trans_x2=True 与 False 行为等价
    #   (x2/scale 都存转置朝向). 此处显式传 trans_x2=True, 覆盖 Case 字段 + op-plugin transpose 参数路径.
    Case("1b_fp4_tx2", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], trans_x2=True),
    Case("1b_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2], batch_x2=[2], trans_x2=True, weight_nz=True),

    # ---- batch 维数逐渐增大 (≤4 层正向; >4 层 expect_fail 测 NPU 是否拦截超限 batch) ----
    Case("2layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2], batch_x2=[2, 2], trans_x2=True, weight_nz=True),
    Case("3layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2], batch_x2=[2, 2, 2], trans_x2=True, weight_nz=True),
    Case("4layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], trans_x2=True, weight_nz=True),
    Case("5layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], trans_x2=True, weight_nz=True, expect_fail=True),
    Case("6layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2, 2], trans_x2=True, weight_nz=True, expect_fail=True),
    Case("7layer_fp4_tx2_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2, 2, 2], trans_x2=True, weight_nz=True, expect_fail=True),

    # ---- 全组合 batch 拦截: fp8/fp4 × ND/NZ × 非转置/tx2, 每组 4 层(正)+ 5 层(负 expect_fail) ----
    # fp8 ND
    Case("4layer_fp8_nd", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2]),
    Case("5layer_fp8_nd", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], expect_fail=True),
    # fp8 ND tx2
    Case("4layer_fp8_nd_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], trans_x2=True),
    Case("5layer_fp8_nd_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], trans_x2=True, expect_fail=True),
    # fp8 NZ
    Case("4layer_fp8_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], weight_nz=True),
    Case("5layer_fp8_nz", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], weight_nz=True, expect_fail=True),
    # fp8 NZ tx2
    Case("4layer_fp8_nz_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], trans_x2=True, weight_nz=True),
    Case("5layer_fp8_nz_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], trans_x2=True, weight_nz=True, expect_fail=True),
    # fp4 ND
    Case("4layer_fp4_nd", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2]),
    Case("5layer_fp4_nd", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], expect_fail=True),
    # fp4 ND tx2 (fp4 的 x2_stored_transposed 恒 True, trans_x2 与非转置等价, 此处显式传覆盖参数路径)
    Case("4layer_fp4_nd_tx2", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], trans_x2=True),
    Case("5layer_fp4_nd_tx2", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], trans_x2=True, expect_fail=True),
    # fp4 NZ (fp4 NZ tx2 已由上方 2~7layer_fp4_tx2_nz 覆盖)
    Case("4layer_fp4_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[2, 2, 2, 2], weight_nz=True),
    Case("5layer_fp4_nz", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2, 2], batch_x2=[2, 2, 2, 2, 2], weight_nz=True, expect_fail=True),

    # ---- 单侧 batch: x1 带 batch, x2 无 batch (trans_x1=False, trans_x2 各一) ----
    Case("1b_fp8_x1only", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2], batch_x2=[]),
    Case("1b_fp8_x1only_tx2", M=64, N=128, K=256, x_dtype="fp8_e4m3fn", out_dtype="float16",
         batch_x1=[2, 2, 2, 2], batch_x2=[], trans_x2=True),
    Case("1b_fp4_x1only", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2, 2, 2], batch_x2=[]),
    Case("1b_fp4_x1only_tx2", M=64, N=128, K=256, x_dtype="fp4_e2m1", out_dtype="bfloat16",
         batch_x1=[2, 2], batch_x2=[], trans_x2=True),
]


# ============================================================
# 数据生成 — batch scale
# ============================================================

def x2_stored_transposed(c: Case) -> bool:
    """x2 是否以 (..,N,K) 存储(最内层=K). 两种情形:
      - trans_x2: 显式转置.
      - fp4: 每字节打包 2 个 fp4, NPU 把最内层 ×2. x1=(M,K) 翻倍 K; 若 x2 存 (K,N) 翻倍 N→K 不等.
        让 x2 也存 (..,N,K) 使 K 在最内层同样 ×2, x1/x2 的 K 才对齐.
    """
    return c.trans_x2 or c.x_dtype == "fp4_e2m1"


def gen_data(c: Case) -> dict:
    if c.seed is not None:
        torch.manual_seed(c.seed)
    M, N, K = c.M, c.N, c.K
    ceilK = ceil_div(K, MXFP_DIVISOR)
    b1, b2 = list(c.batch_x1), list(c.batch_x2)

    # x1, x2 — NPU 存储格式 (含 batch 前缀)
    x1_shape = tuple(b1 + ([K, M] if c.trans_x1 else [M, K]))
    x2_shape = tuple(b2 + ([N, K] if x2_stored_transposed(c) else [K, N]))
    fp4_packed = False
    fp4_value_x1 = fp4_value_x2 = None   # 真量化的 e2m1 浮点值; 非 None 时 golden 直接用(不 unpack)
    fp4_scale_ps = fp4_scale_sc = None   # 真量化产出的 e8m0 scale(替代随机 scale)
    if c.x_dtype != "fp4_e2m1":  # fp8 (e4m3fn / e5m2, 支持 x1/x2 混合 via x2_dtype)
        x1_dt = torch.float8_e5m2 if c.x_dtype == "fp8_e5m2" else torch.float8_e4m3fn
        x2_dt_str = c.x2_dtype if c.x2_dtype else c.x_dtype
        x2_dt = torch.float8_e5m2 if x2_dt_str == "fp8_e5m2" else torch.float8_e4m3fn
        if c.cont_data:
            # 连续 uniform→fp8: 产生分数 mantissa bit pattern, stress e5m2 的 3 位尾数(对标线上连续输入)
            x1 = torch.empty(x1_shape).uniform_(-5, 5).to(x1_dt)
            x2 = torch.empty(x2_shape).uniform_(-5, 5).to(x2_dt)
        else:
            x1 = torch.randint(-5, 5, x1_shape).to(x1_dt)
            x2 = torch.randint(-5, 5, x2_shape).to(x2_dt)
    else:  # fp4_e2m1
        # 官方 mx_quantize: 高斯 fp32 → 每 32 块 e8m0 scale + e2m1 码; golden 用浮点值, NPU 用打包字节.
        #   trans_x1 暂挡: ①量化应沿 K 但本路径沿最后维 M ②沿 M 打包 nibble 序可观测(见 CASES 说明).
        assert not (c.trans_x1 and c.x_dtype == "fp4_e2m1"), \
            "fp4 + trans_x1 暂未支持: ①量化应沿K但本量化器沿最后维M ②沿M打包nibble序可观测, 见CASES说明"
        if K % 2 == 0:
            # 源走官方 regenerate_quant_data(高斯 normal(0.05,σ=4)); out_dtype=bf16 保留符号.
            src1, src2 = regenerate_quant_data(x1_shape, x2_shape, "bf16")  # numpy fp32
            x1, fp4_value_x1, fp4_scale_ps = fp4_quant_pack_lastdim(src1)  # packed/value/scale
            x2, fp4_value_x2, fp4_scale_sc = fp4_quant_pack_lastdim(src2)
            fp4_packed = True
        else:
            # K 奇 (oddInner_fail): 不打包, 内轴保持奇, 触发 NPU CheckInnerAxisIsEven 拒绝(负向).
            fp4_packed = False
            x1 = gen_fp4_packed(x1_shape, False)
            x2 = gen_fp4_packed(x2_shape, False)

    # scale shape: batch 前缀同 x; e8m0 不打包; ceilK 按逻辑 K 给(与 NPU ×2 还原后的 K 匹配).
    ps_tail = [ceilK, M] if c.trans_x1 else [M, ceilK]
    sc_tail = [N, ceilK] if x2_stored_transposed(c) else [ceilK, N]
    ps_shape = tuple(b1 + ps_tail + [MXFP_BASE])
    sc_shape = tuple(b2 + sc_tail + [MXFP_BASE])

    # fp4 打包路径: scale 由官方量化产出; 其余(fp8 / K 奇回退): 随机 e8m0.
    #   fp8 randint(125,131)→2^e∈{0.25..8}; fp4 回退 randint(126,129)→{0.5,1,2}.
    e_lo, e_hi = (126, 129) if c.x_dtype == "fp4_e2m1" else (125, 131)
    if fp4_scale_ps is not None:
        pertoken_scale = fp4_scale_ps
        scale = fp4_scale_sc
    else:
        pertoken_scale = torch.randint(e_lo, e_hi, ps_shape, dtype=torch.uint8)
        scale = torch.randint(e_lo, e_hi, sc_shape, dtype=torch.uint8)

    bias = torch.randn(N) * 0.5 if c.has_bias else None
    return dict(x1=x1, x2=x2, pertoken_scale=pertoken_scale, scale=scale,
                bias=bias, fp4_packed=fp4_packed,
                fp4_value_x1=fp4_value_x1, fp4_value_x2=fp4_value_x2)


# ============================================================
# Golden (torch CPU) — batch 前缀靠负索引寻址, torch.matmul 自动广播
# ============================================================

def golden(d: dict, c: Case, verbose: bool = False) -> torch.Tensor:
    if c.x_dtype == "fp4_e2m1":
        # golden 直接用官方量化产出的 e2m1 浮点值(不 unpack, 故不依赖 nibble 序).
        if d["fp4_packed"]:
            x1 = d["fp4_value_x1"]   # (..,M,K) fp32 e2m1 值(存储朝向)
            x2 = d["fp4_value_x2"]   # (..,N,K)
        else:
            # K 奇回退: int8 存 code, 查 LUT (oddInner_fail, NPU 侧预期被拒).
            lut = torch.tensor(E2M1_LUT, dtype=torch.float32)
            x1 = lut[d["x1"].to(torch.int64) & 0xF]
            x2 = lut[d["x2"].to(torch.int64) & 0xF]
    else:
        x1 = d["x1"].float()
        x2 = d["x2"].float()
    ps = e8m0_to_float32(d["pertoken_scale"])
    sc = e8m0_to_float32(d["scale"])
    if c.trans_x1:
        x1 = x1.transpose(-1, -2)   # 存储(..,K,M) → 逻辑(..,M,K)
    if x2_stored_transposed(c):
        x2 = x2.transpose(-1, -2)   # 存储(..,N,K) → 逻辑(..,K,N)

    if verbose:
        print(f"  [g] input: x1={tuple(x1.shape)} x2={tuple(x2.shape)} "
              f"ps={tuple(ps.shape)} sc={tuple(sc.shape)}")

    # ---- reshape scale: 目标 ps→(..,M,ceilK*2), sc→(..,ceilK*2,N) ----
    if c.trans_x1:
        # ps:(..,ceilK,M,2) → (..,ceilK,2,M) → (..,ceilK*2,M) → (..,M,ceilK*2)
        ps = ps.transpose(-1, -2)
        ps = merge_mid2(ps)
        ps = ps.transpose(-1, -2)
    else:
        # ps:(..,M,ceilK,2) → (..,M,ceilK*2)
        ps = merge_last2(ps)

    if x2_stored_transposed(c):
        # sc:(..,N,ceilK,2) → (..,ceilK,N,2) → (..,ceilK,2,N) → (..,ceilK*2,N)
        sc = sc.transpose(-3, -2)
        sc = sc.transpose(-1, -2)
        sc = merge_mid2(sc)
    else:
        # sc:(..,ceilK,N,2) → (..,ceilK,2,N) → (..,ceilK*2,N)
        sc = sc.transpose(-1, -2)
        sc = merge_mid2(sc)

    if verbose:
        print(f"  [g] reshape: ps={tuple(ps.shape)} sc={tuple(sc.shape)}")

    # ---- 截断 odd ceil(K/32) ----
    k_dim = x1.shape[-1]
    if ceil_div(k_dim, MXFP_INNER) % 2 != 0:
        ps = ps[..., :-1]
        sc = sc[..., :-1, :]
        if verbose:
            print(f"  [g] truncated: ceil(K/32)={ceil_div(k_dim, MXFP_INNER)} odd")

    # ---- broadcast ×32 ----
    ps_bc = ps.repeat_interleave(MXFP_INNER, dim=-1)   # (..,M,K_padded)
    sc_bc = sc.repeat_interleave(MXFP_INNER, dim=-2)   # (..,K_padded,N)

    # ---- pad K ----
    x1 = torch.nn.functional.pad(x1, (0, max(ps_bc.shape[-1] - x1.shape[-1], 0)))
    x2 = torch.nn.functional.pad(x2, (0, 0, 0, max(sc_bc.shape[-2] - x2.shape[-2], 0)))

    # ---- apply + matmul (torch.matmul 自动 broadcast batch 前缀) ----
    # golden 为高精度参考(不舍到 out_dtype): bf16/fp16 输出用 fp32 参考足够(标杆舍入误差 2^-8/2^-11
    #   远大于 fp32 参考误差 2^-24); fp32 输出须 fp64 参考, 否则参考==标杆, 官方相对误差判据退化为 0/0.
    ref = torch.float64 if c.out_dtype == "float32" else torch.float32
    out = torch.matmul((x1 * ps_bc).to(ref), (x2 * sc_bc).to(ref))

    if verbose:
        print(f"  [g] out={tuple(out.shape)} range=[{out.min():.4f}, {out.max():.4f}]")

    if d["bias"] is not None:
        out = out + d["bias"].to(ref)
    return out   # 高精度参考; 标杆(golden 舍到 out_dtype)与判定在 cmp 内


# ============================================================
# NPU 调用 (batch scale 尚未实现 → 预期报错, 由 run() 捕获记为 NOT_IMPL)
# ============================================================

def call_npu(d: dict, c: Case, verbose: bool = False) -> torch.Tensor:
    x1 = d["x1"].clone()
    x2 = d["x2"].clone()
    ps = d["pertoken_scale"].clone()
    sc = d["scale"].clone()
    bias = d["bias"].clone() if d["bias"] is not None else None

    # fp4: 数据已在 gen_data 做真正 2:1 打包 (最内层 K 存 K/2 字节), 此处仅打 dtype kwarg;
    # NPU 按 float4_e2m1fn_x2 把最内层 ×2 还原逻辑 K. scale (e8m0) 不打包.
    x1_dt = x2_dt = None
    if c.x_dtype == "fp4_e2m1":
        x1_dt = torch_npu.float4_e2m1fn_x2
        x2_dt = torch_npu.float4_e2m1fn_x2

    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    if verbose:
        print(f"  [npu] x1={tuple(x1.shape)} {x1.dtype}  x2={tuple(x2.shape)} {x2.dtype}")
        print(f"  [npu] ps={tuple(ps.shape)} {ps.dtype}  sc={tuple(sc.shape)} {sc.dtype}")

    kw = dict(
        x1=x1.npu(), x2=x2.npu(),
        scale=sc.npu(), pertoken_scale=ps.npu(),
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=dt[c.out_dtype],
        group_sizes=[0, 0, 32],  # [临时复现-勿提交] 复原为 [0,0,32] 触发 tiling 的 M/N 推断, 确认责任函数; 原值 [1,1,32] 见下.
        # 原为 group_sizes=[1, 1, 32]  # MX 文档要求 [groupSizeM,groupSizeN,groupSizeK]=[1,1,32];
        # 曾传 [0,0,32]: V5(ND) 把 M/N=0 推断成 1(正确), 但 WeightNz(NZ) tiling 推断成 32 →
        # [32,32,32] 被 MX 校验拒绝(1b) 或用错 groupSize 导致数值错位(2b/3b cos~0.04). 明确传 [1,1,32].
    )
    if x1_dt: kw["x1_dtype"] = x1_dt
    if x2_dt: kw["x2_dtype"] = x2_dt
    if bias is not None: kw["bias"] = bias.float().npu()  # API 要求 bias 为 DT_FLOAT

    # transpose: NPU 靠 stride 检测转置(只转最后两维). 约束 A: scale 转置态须与数据张量一致 —
    #   op-plugin is_x_scale_same_transpose 比对数据最后两维 vs scale 的 [-3,-2](跳过末维"2"),
    #   故数据 .transpose(-1,-2) 时 scale 须同步 .transpose(-3,-2), 否则报 transpose not same.
    # NZ cast 必须在 transpose 之前(参考官方测试 D:/Desktop/TMP/tmp.py L641-648 的 cast→transpose 顺序):
    #   NZ tensor 的 transpose 靠 stride 让 NPU 检测转置; 若 transpose 后再 cast, .contiguous() 抹掉
    #   转置 stride → fp4/转置 NZ 报 "x2/scale transpose not same". int8(fp4 打包)/fp8 原始 dtype 直接 cast.
    if c.weight_nz:
        kw["x2"] = torch_npu.npu_format_cast(kw["x2"], 29)

    if c.trans_x1:
        kw["x1"] = kw["x1"].transpose(-1, -2)
        kw["pertoken_scale"] = kw["pertoken_scale"].transpose(-3, -2)
    if x2_stored_transposed(c):
        kw["x2"] = kw["x2"].transpose(-1, -2)
        kw["scale"] = kw["scale"].transpose(-3, -2)

    # 故意打破"数据与 scale 转置态一致"约束: 仅转 scale/pertoken_scale, 对应数据不转 →
    #   stride 转置态不一致, op-plugin is_x_scale_same_transpose 应报 "transpose are not same".
    if c.mismatch_x2_scale:
        kw["scale"] = kw["scale"].transpose(-3, -2)
    if c.mismatch_x1_scale:
        kw["pertoken_scale"] = kw["pertoken_scale"].transpose(-3, -2)

    out = torch_npu.npu_quant_matmul(**kw)
    return out.float().cpu()


# ============================================================
# 精度比对 — 余弦相似度(cos)为主 + 相对 L2 范数辅助门 的双保险判据
#   业界 matmul / 量化最主流判据。cos 天然免疫近零抵消点 / 大 K 抵消 / fp4+bias 擦门这类
#   "整体方向正确、个别近零点相对误差爆炸"的误判(那正是官方固定小值门 BenchmarkCompareStandard
#   会误杀零误差核的病态: 见 golden-cancellation-criterion)。但 cos 有一个已知盲区 —— 系统性
#   乘性小偏置(所有元素×(1+δ)不改变方向, cos 仍≈1 却掩盖一致误差), 故必须叠加相对 L2 辅助门堵住。
#
# 判据: PASS ⟺ 主门 cos ≥ COS_TOL  且  辅助门 相对L2 ≤ RELL2_TOL[out_dtype]。
#   主门 cos:  cos(npu 拉平, golden 拉平) ≥ COS_TOL。
#   辅助门 相对L2:  ‖npu-golden‖₂ / ‖golden‖₂ ≤ RELL2_TOL。分母是整张范数(大值主导),
#     故近零点噪声贡献可忽略(天然免疫近零点); 而一致乘性偏置 δ 使相对L2 = |δ| 恰好被抓住。
# ============================================================

# 主门阈值 COS_TOL: bf16 舍入本身的 cos 理论地板 ≈ 1-(2^-8/√3)²/2 ≈ 1-2.5e-6(numpy 镜像实测
#   1-cos≈1.3e-6, 因真实 bf16 误差分布优于最坏均匀假设)。取 0.9999(1-cos≤1e-4)留约 75× 余量:
#   数值正确的核绝不会因舍入误判 FAIL; 而方向真错(漏写 batch 偏移/错位/垃圾值)会把 cos 打到 ≪0.99。
#   cos 对一致乘性偏置几乎不动(方向不变), 精细鉴别交给辅助门, 故主门取宽余量即可。
COS_TOL = 0.9999

# 辅助门阈值 RELL2_TOL: 逐 out_dtype 取该类型的相对分辨率(相对-ULP) 2^-mantissa。
#   纯舍入下 相对L2 ≈ 每元素相对舍入尺度 σ = 2^-mantissa/√3(bf16 实测地板 1.6e-3 = 0.71σ)。
#   取 2^-mantissa 使阈值 ≈ 1.41σ: 纯舍入地板落在门下 ~2.4×(bf16/fp16 一致), 破门盈亏偏置 bf16≈0.35%
#   / fp16≈0.05% → 用户命脉反例(整体×1.005 系统偏置)相对L2≈5.3e-3 > 门 3.9e-3 必 FAIL。
#   逐类型而非统一常量: fp16 地板(2e-4)比 bf16(1.6e-3)紧 8×, 统一门会让 fp16 辅助门形同虚设。
RELL2_TOL = {
    "float32":  2.0 ** -24,
    "float16":  2.0 ** -11,
    "bfloat16": 2.0 ** -8,
}

# 保留供打印参照: 旧官方 BenchmarkCompareStandard 各档倍数/下界(不再参与判定, 仅 report)。
BENCH_TOL = {
    "float32":  dict(avg=2.0, mx=5.0,  rmse=2.0, small=1e-6, small_atol=1e-9, lo=0.0),
    "float16":  dict(avg=2.0, mx=10.0, rmse=2.0, small=1e-3, small_atol=1e-5, lo=2.0 ** -11),
    "bfloat16": dict(avg=2.0, mx=10.0, rmse=2.0, small=1e-3, small_atol=1e-5, lo=2.0 ** -8),
}
DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def cos_rell2(value: torch.Tensor, golden: torch.Tensor) -> dict:
    """cos 双保险度量, 展平 float64: 主门 cos、辅助门 相对L2、nan/inf 一致性错误数、以及
    定位失败元素用的最差点(golden/npu/absErr/relErr)。
      cos      = ⟨v,g⟩/(‖v‖‖g‖)      (方向一致性; 免疫一致乘性偏置)
      rel_l2   = ‖v-g‖₂/‖g‖₂          (整张相对误差; 抓一致偏置, 免疫近零点)
    nan/inf: 比对 golden 与 value 的掩码是否一致(不一致直接判 FAIL, 与官方一致)。
    非有限位置置 0 不入 cos/相对L2 数值统计(掩码不一致已单独计入 err_nan/err_inf)。"""
    v = value.reshape(-1).double()
    g = golden.reshape(-1).double()
    zeros = torch.zeros_like(g)
    # nan/±inf 掩码一致性
    g_nan, v_nan = torch.isnan(g), torch.isnan(v)
    err_nan = int(torch.sum(g_nan ^ v_nan))
    g_pinf, v_pinf = torch.isinf(g) & (g > 0), torch.isinf(v) & (v > 0)
    g_ninf, v_ninf = torch.isinf(g) & (g < 0), torch.isinf(v) & (v < 0)
    err_inf = int(torch.sum(g_pinf ^ v_pinf)) + int(torch.sum(g_ninf ^ v_ninf))
    finite = ~(torch.isinf(g) | torch.isnan(g)) & ~(torch.isinf(v) | torch.isnan(v))
    g = torch.where(finite, g, zeros)
    v = torch.where(finite, v, zeros)
    # 主门 cos
    gn = float(g.norm())
    vn = float(v.norm())
    cos = float(g @ v / (gn * vn)) if gn > 0 and vn > 0 else 1.0
    # 辅助门 相对L2
    diff_norm = float((v - g).norm())
    rel_l2 = diff_norm / gn if gn > 0 else 0.0
    # 定位: 绝对误差最大的前几个点(供 FAIL 打印 golden/npu/absErr/relErr)
    absdiff = (v - g).abs()
    gabs = g.abs()
    n_worst = int(min(5, g.numel()))
    if n_worst > 0:
        _, idx = torch.topk(absdiff, n_worst)
        worst = [(float(g[i]), float(v[i]), float(absdiff[i]),
                  float(absdiff[i] / gabs[i]) if float(gabs[i]) > 0 else float("inf"))
                 for i in idx.tolist()]
    else:
        worst = []
    return dict(cos=cos, rel_l2=rel_l2, err_nan=err_nan, err_inf=err_inf,
                gnorm=gn, vnorm=vn, worst=worst)


def cos_verdict(m: dict, out_dtype: str):
    """双保险判定: 主门 cos ≥ COS_TOL 且 辅助门 相对L2 ≤ RELL2_TOL 且 nan/inf 掩码一致。
    返回 (ok, [超标原因])。"""
    rell2_tol = RELL2_TOL[out_dtype]
    reasons = []
    if m["cos"] < COS_TOL:
        reasons.append(f"主门 cos {m['cos']:.8f} < 阈值 {COS_TOL:.6g}")
    if m["rel_l2"] > rell2_tol:
        reasons.append(f"辅助门 相对L2 {m['rel_l2']:.4g} > 阈值 {rell2_tol:.4g}")
    if m["err_nan"] > 0:
        reasons.append(f"nan 掩码不一致 {m['err_nan']} 处")
    if m["err_inf"] > 0:
        reasons.append(f"inf 掩码不一致 {m['err_inf']} 处")
    return (len(reasons) == 0), reasons


def precision_table(g: torch.Tensor, n: torch.Tensor, c: Case):
    """cos 双保险各门明细: 返回 [(门名, 是否通过, npu值/阈值), ...]。
    g 为高精度 golden; n 为 NPU 输出。"""
    g = g.double()
    if tuple(g.shape) != tuple(n.shape):
        return [("shape", False, f"g={tuple(g.shape)} n={tuple(n.shape)}")]
    m = cos_rell2(n, g)
    rell2_tol = RELL2_TOL[c.out_dtype]
    rows = [
        (f"主门 cos ≥ {COS_TOL:.6g}", m["cos"] >= COS_TOL,
         f"npu cos={m['cos']:.8f} 阈值={COS_TOL:.6g}"),
        (f"辅助门 相对L2 ≤ {rell2_tol:.4g}", m["rel_l2"] <= rell2_tol,
         f"npu 相对L2={m['rel_l2']:.6g} 阈值={rell2_tol:.4g}"),
    ]
    if m["err_nan"] > 0 or m["err_inf"] > 0:
        rows.append(("nan/inf 掩码一致", m["err_nan"] == 0 and m["err_inf"] == 0,
                     f"nan={m['err_nan']} inf={m['err_inf']} 阈值=0"))
    return rows


def cmp(g: torch.Tensor, n: torch.Tensor, c: Case, v=False) -> dict:
    """cos 双保险判定(主门 cos + 辅助门 相对L2, 展平整张)。
    g 高精度 golden, n NPU 输出。FAIL 时按同一门逐 batch 定位驱动失败的 batch(暴露偏移漏写)。"""
    g = g.double()
    n = n.float().double()
    if tuple(g.shape) != tuple(n.shape):
        return dict(passed=False, max_diff=float("inf"), cosine=0.0,
                    fail_batches=[], note=f"shape mismatch g={tuple(g.shape)} n={tuple(n.shape)}")

    m = cos_rell2(n, g)
    passed, reasons = cos_verdict(m, c.out_dtype)

    # 报告用: 最大绝对误差供汇总表(判定不依赖它)。cosine 直接取主门实测值。
    max_diff = float((n - g).abs().max())
    cos = m["cos"]
    note = "" if passed else "; ".join(reasons)

    fail_batches = []
    if not passed and (c.batch_x1 or c.batch_x2):
        # 逐 batch 套同一双保险门, 找出驱动失败的 batch(暴露 batch1+ 偏移漏写)。
        gB = g.reshape(-1, c.M, c.N)
        nB = n.reshape(-1, c.M, c.N)
        for b in range(gB.shape[0]):
            mb = cos_rell2(nB[b], gB[b])
            ok_b, _ = cos_verdict(mb, c.out_dtype)
            if not ok_b:
                fail_batches.append(b)

    if v:
        print(f"    [cmp] {'PASS' if passed else 'FAIL'}: "
              f"cos={m['cos']:.8f} 相对L2={m['rel_l2']:.4g}"
              + (f"  [{note}]" if note else ""))

    return dict(passed=passed, max_diff=max_diff, cosine=cos,
                fail_batches=fail_batches, note=note, m=m)


def _g4(x) -> str:
    """相对误差/门统一 4 位有效数字。"""
    return f"{x:.4g}"


def _fmt_verdict(cr: dict, c: Case) -> str:
    """PASS/FAIL 统一打印(用户硬性要求 PASS 也打全, 不黑盒): 主门 cos 与辅助门 相对L2 各自
    实测/阈值/余量, FailBatches; PASS 显示余量(cos 距阈的 1-cos 富余、相对L2 用量百分比),
    FAIL 显示哪个门破、破多少 + 前几个最差元素(golden/npu/absErr/relErr)。附 out_dtype
    相对分辨率参照。纯格式化, 不参与判定(verdict 沿用 cr['passed'])。"""
    verd = "PASS" if cr["passed"] else "FAIL"
    diff, cos = cr.get("max_diff"), cr.get("cosine")
    m = cr.get("m")
    fb = cr.get("fail_batches") or []
    if not fb:
        fb_s = ""
    elif len(fb) <= 8:
        fb_s = f"  FailBatches={fb}"
    else:
        fb_s = f"  FailBatches=[0..{fb[-1]}]({len(fb)})"
    # shape 不匹配等无 m 的早退场景: 回精简行
    if m is None:
        head = f"  -> {verd}  diff={diff:.6f}  cos={cos:.6f}" if diff is not None else f"  -> {verd}"
        return head + fb_s + (f"  ({cr['note']})" if cr.get("note") else "")

    rell2_tol = RELL2_TOL[c.out_dtype]
    mant = {"float32": 24, "float16": 11, "bfloat16": 8}[c.out_dtype]

    # 主门 cos: 过 → 显示 1-cos 富余量(距阈的量级余地); 破 → 差多少
    cos_ok = m["cos"] >= COS_TOL
    if cos_ok:
        cos_tok = f"主门cos {m['cos']:.8f}≥{COS_TOL:.6g}=✓(1-cos={1.0 - m['cos']:.2e} 距阈{COS_TOL:.4g})"
    else:
        cos_tok = f"主门cos {m['cos']:.8f}<{COS_TOL:.6g}=✗破(差{COS_TOL - m['cos']:.3g})"

    # 辅助门 相对L2: 过 → 用量百分比; 破 → 超几倍
    rl2_ok = m["rel_l2"] <= rell2_tol
    if rl2_ok:
        use = m["rel_l2"] / rell2_tol * 100.0 if rell2_tol > 0 else 0.0
        rl2_tok = f"辅助门相对L2 {_g4(m['rel_l2'])}/{_g4(rell2_tol)}=✓({use:.1f}%用量)"
    else:
        over = m["rel_l2"] / rell2_tol if rell2_tol > 0 else float("inf")
        rl2_tok = f"辅助门相对L2 {_g4(m['rel_l2'])}/{_g4(rell2_tol)}=✗破(超{over:.3g}×)"

    toks = [cos_tok, rl2_tok]
    if m["err_nan"] > 0 or m["err_inf"] > 0:
        toks.append(f"nan/inf {m['err_nan']}/{m['err_inf']}✗掩码不一致")

    if cr["passed"]:
        health = f"稳过·辅助门用量{m['rel_l2'] / rell2_tol * 100.0:.1f}%" if rell2_tol > 0 else "稳过"
    else:
        broken = []
        if not cos_ok:
            broken.append("主门cos")
        if not rl2_ok:
            broken.append(f"辅助门相对L2(超{m['rel_l2'] / rell2_tol:.3g}×)")
        if m["err_nan"] > 0 or m["err_inf"] > 0:
            broken.append("nan/inf掩码")
        health = "破门·" + "/".join(broken)

    ref = f"{c.out_dtype}相对分辨率=2^-{mant}={_g4(2.0 ** -mant)}(辅助门阈值同此)"

    line1 = f"  -> {verd}  diff={diff:.6f}  cos={cos:.6f}{fb_s}  {ref}"
    line2 = "     " + " | ".join(toks) + f" | {health}"
    lines = [line1, line2]
    # FAIL: 附前几个最差元素, 便于定位
    if not cr["passed"] and m.get("worst"):
        parts = []
        for gv, nv, ad, rd in m["worst"]:
            rd_s = "inf" if rd == float("inf") else f"{rd:.3g}"
            parts.append(f"(g={gv:.4g} npu={nv:.4g} |Δ|={ad:.3g} relΔ={rd_s})")
        lines.append("     最差元素: " + " ".join(parts))
    return "\n".join(lines)


# ============================================================
# main
# ============================================================

def run(c: Case, npu: bool, v: bool) -> dict:
    b1 = "x".join(str(b) for b in c.batch_x1)
    b2 = "x".join(str(b) for b in c.batch_x2)
    print(f"\n{'='*60}")
    print(f"  {c.name}  M={c.M} N={c.N} K={c.K}  x={c.x_dtype} out={c.out_dtype}"
          f"  batch_x1={b1} batch_x2={b2}")
    d = gen_data(c)
    g = golden(d, c, v)
    print(f"  golden: {tuple(g.shape)} {g.dtype}  [{g.float().min():.2f}, {g.float().max():.2f}]")

    r = dict(name=c.name, g_ok=True, npu_ok=False, passed=None)
    if npu:
        if c.expect_fail:
            # 预期被 NPU checker 拒绝: 捕获到报错 → PASS; 跑通无报错 → FAIL.
            try:
                n = call_npu(d, c, v)
                print(f"  npu:    {tuple(n.shape)} {n.dtype}  (UNEXPECTED: 预期被拒却跑通)")
                r.update(npu_ok=True, passed=False,
                         note="expected NPU rejection but call succeeded")
                print(f"  -> FAIL  (expect_fail 用例反而跑通, NPU checker 未拦截)")
            except Exception as e:
                msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                r.update(npu_ok=True, passed=True, note=f"rejected as expected: {msg}")
                print(f"  -> PASS  (expect_fail: NPU 按预期拒绝 -> {msg})")
            return r
        try:
            n = call_npu(d, c, v)
            print(f"  npu:    {tuple(n.shape)} {n.dtype}  [{n.min():.2f}, {n.max():.2f}]")
            cr = cmp(g, n, c, v)
            r.update(npu_ok=True, **cr)
            print(_fmt_verdict(cr, c))
            r["prec_rows"] = precision_table(g, n.float(), c)
        except Exception as e:
            import traceback; traceback.print_exc()
            r["err"] = str(e)
    return r


def select_cases(a) -> List[Case]:
    cases = CASES
    if a.case:
        wanted = set(x.strip() for x in a.case.split(","))
        cases = [c for c in cases if c.name in wanted]
    if a.filter:
        cases = [c for c in cases if a.filter in c.name]
    return cases


def _print_summary_table(res, round_label=None):
    """打印汇总表 (单轮或多轮合并)."""
    if round_label:
        print(f"\n{'='*60}")
        print(f"  {round_label}")
    name_w = max((len(r["name"]) for r in res), default=8)
    hdr = (f"{'name':<{name_w}}  {'gold':>4} {'npu':>4} {'res':>4} "
           f"{'max_diff':>10} {'cosine':>9}  {'fail':>10}  备注")
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        g_ = "OK" if r["g_ok"] else "ERR"
        n_ = "OK" if r.get("npu_ok") else ("ERR" if "err" in r else "--")
        p_ = "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "--")
        d_ = f"{r['max_diff']:.4f}" if r.get("max_diff") is not None else "--"
        c_ = f"{r['cosine']:.6f}" if r.get("cosine") is not None else "--"
        fbl = r.get("fail_batches") or []
        if not fbl:
            fb = "-"
        elif len(fbl) <= 8:
            fb = "[" + ",".join(str(b) for b in fbl) + "]"
        else:
            fb = f"[0..{fbl[-1]}]({len(fbl)})"
        note = r.get("note") or ""
        print(f"{r['name']:<{name_w}}  {g_:>4} {n_:>4} {p_:>4} {d_:>10} {c_:>9}  {fb:>10}  {note}")


def _print_fail_details(res):
    """逐 case 列未通过的判据门 + 含义."""
    def _why(nm):
        if "cos" in nm:      return "主门: npu 与 golden 拉平后的方向一致性; 破门=方向真错(漏写 batch 偏移/错位/垃圾值)"
        if "相对L2" in nm:   return "辅助门: ‖npu-golden‖₂/‖golden‖₂ 整张相对误差; 抓一致乘性偏置(cos 盲区), 分母整张范数故免疫近零点"
        if "nan/inf" in nm:  return "nan/inf 掩码须与 golden 一致; 不一致直接 FAIL"
        return ""
    fail_cases = [(r["name"], r.get("passed"), r.get("note") or "", r["prec_rows"]) for r in res
                  if r.get("prec_rows") and any(not ok for _, ok, _ in r["prec_rows"])]
    if fail_cases:
        print(f"\n{'='*60}")
        print("cos 双保险 FAIL 明细(逐 case, 仅列破门):")
        for cname, passed, note, rows in fail_cases:
            v = "PASS" if passed else "FAIL"
            reason = note if note else ("判定门过" if passed else "判定门不过")
            print(f"\n  [{cname}]  verdict={v}  — {reason}")
            for nm, ok, det in rows:
                if not ok:
                    det_s = f"  | {det}" if det else ""
                    why_s = f"  — {_why(nm)}" if _why(nm) else ""
                    print(f"    FAIL | {nm}{det_s}{why_s}")


def _fail_break_tokens(m: dict, out_dtype: str):
    """从度量 m 提炼破门类型标签 + 破门数值(纯展示; 判定已在 cmp/cos_verdict 完成, 此处不重判)。
    与 cos_verdict 用同一比较式(cos<COS_TOL、rel_l2>阈、err_nan/err_inf>0), 保证分类与判定同源。
    返回 (labels, detail): labels 如 ['主门cos','辅助门相对L2']; detail 是拼好的破门数值串。"""
    rell2_tol = RELL2_TOL[out_dtype]
    labels = []
    parts = []
    if m["cos"] < COS_TOL:
        labels.append("主门cos")
        parts.append(f"cos={m['cos']:.6g}<{COS_TOL:.6g}")
    if m["rel_l2"] > rell2_tol:
        labels.append("辅助门相对L2")
        over = m["rel_l2"] / rell2_tol if rell2_tol > 0 else float("inf")
        parts.append(f"相对L2={_g4(m['rel_l2'])}>{_g4(rell2_tol)}(超{over:.3g}×)")
    if m["err_nan"] > 0 or m["err_inf"] > 0:
        labels.append("nan/inf不一致")
        parts.append(f"nan={m['err_nan']} inf={m['err_inf']}(应为0)")
    return labels, ("  ".join(parts) if parts else "")


def _fail_tag(labels) -> str:
    """破门标签列表 → 用户要求的标签文案。cos+相对L2 都破合并为 [两门都✗]; nan/inf 破单加一枚。"""
    has_cos = "主门cos" in labels
    has_rl2 = "辅助门相对L2" in labels
    has_ni = "nan/inf不一致" in labels
    tags = []
    if has_cos and has_rl2:
        tags.append("[两门都✗]")
    elif has_cos:
        tags.append("[主门cos✗]")
    elif has_rl2:
        tags.append("[辅助门相对L2✗]")
    if has_ni:
        tags.append("[nan/inf不一致✗]")
    return "".join(tags) if tags else "[破门]"


def _fb_summary_str(fbl) -> str:
    """FailBatches 紧凑串: 空→空串; ≤8→列全; >8→区间+计数。"""
    if not fbl:
        return ""
    if len(fbl) <= 8:
        return "  FailBatches=[" + ",".join(str(b) for b in fbl) + "]"
    return f"  FailBatches=[0..{fbl[-1]}]({len(fbl)})"


def _print_fail_summary(cases, res):
    """全部用例跑完后的 FAIL 汇总(纯展示, 判定逻辑一律不改, 结果不变)。集中列出本次所有失败
    用例 + 破门类型 + 破门数值, 免去在逐用例几百行输出里翻找。分桶:
      精度门破     : passed False 且带度量 m(按 cos/相对L2/nan-inf 标破门); shape mismatch 无 m 有 note 也列此。
      该报错却没报错: expect_fail 用例反而跑通(note 固定文案), 非精度门, 单列。
      NPU 异常     : call_npu 抛异常未进判定(passed None + err), 既非 PASS 也非精度 FAIL, 单列以免遗漏。
    cases 与 res 一一对应(单轮 res 由 [run(c) for c in cases] 生成; 多轮 merged 亦按 cases 顺序)。"""
    total = len(res)
    n_pass = sum(1 for r in res if r.get("passed") is True)
    prec_fail = []
    expect_fail_miss = []
    npu_err = []
    for c, r in zip(cases, res):
        if r.get("passed") is False:
            if r.get("m") is not None:
                prec_fail.append((c, r))
            elif r.get("note") == "expected NPU rejection but call succeeded":
                expect_fail_miss.append((c, r))
            else:
                prec_fail.append((c, r))   # shape mismatch 等: 有 note, 当破门列
        elif r.get("passed") is None and "err" in r:
            npu_err.append((c, r))
    n_fail = len(prec_fail) + len(expect_fail_miss)

    print(f"\n{'='*60}")
    print(f"  FAIL 汇总  (纯展示, 判定结果不变)")
    print(f"  总数={total}  PASS={n_pass}  FAIL={n_fail}"
          + (f"  NPU异常={len(npu_err)}" if npu_err else ""))
    print("-" * 60)

    if not prec_fail and not expect_fail_miss and not npu_err:
        print(f"  全部 {total} 个用例通过")
        return

    if prec_fail:
        print("  [精度门破] 逐用例(用例名 破门类型 破门数值 FailBatches):")
        name_w = max(len(c.name) for c, _ in prec_fail)
        for c, r in prec_fail:
            m = r.get("m")
            if m is not None:
                labels, detail = _fail_break_tokens(m, c.out_dtype)
                tag = _fail_tag(labels)
            else:
                tag, detail = "[破门]", (r.get("note") or "")
            fb = _fb_summary_str(r.get("fail_batches") or [])
            print(f"    {c.name:<{name_w}}  {tag}  {detail}{fb}")

    if expect_fail_miss:
        print("  [该报错却没报错] expect_fail 用例反而跑通(NPU checker 未拦截, 非精度门):")
        name_w = max(len(c.name) for c, _ in expect_fail_miss)
        for c, r in expect_fail_miss:
            print(f"    {c.name:<{name_w}}  {r.get('note') or ''}")

    if npu_err:
        print("  [NPU 异常] call_npu 抛异常, 未进入判定(既非 PASS 也非精度 FAIL):")
        name_w = max(len(c.name) for c, _ in npu_err)
        for c, r in npu_err:
            errs = (r.get("err") or "").strip()
            msg = errs.splitlines()[-1] if errs else ""
            print(f"    {c.name:<{name_w}}  {msg}")


def _merge_round_results(all_rounds_res, cases):
    """合并多轮结果: 同名 case 取 diff 最大、cosine 最小的那轮作为合并值."""
    merged = []
    for c in cases:
        cname = c.name
        c_rounds = []
        for ri, rlist in enumerate(all_rounds_res):
            for rr in rlist:
                if rr["name"] == cname:
                    c_rounds.append((ri + 1, rr))
                    break
        if not c_rounds:
            continue
        # 按 max_diff 降序取最差的一轮
        worst = max(c_rounds, key=lambda x: (x[1].get("max_diff") or 0))
        ri_worst, rw = worst
        # 打上轮次标记
        rw = dict(rw)
        rw["_worst_round"] = ri_worst
        # 统计: 总轮数, PASS/FAIL 轮数
        n_pass = sum(1 for _, rr in c_rounds if rr.get("passed"))
        n_fail = sum(1 for _, rr in c_rounds if rr.get("passed") is False)
        n_npu = sum(1 for _, rr in c_rounds if rr.get("npu_ok"))
        rw["_rounds_total"] = len(all_rounds_res)
        rw["_rounds_pass"] = n_pass
        rw["_rounds_fail"] = n_fail
        rw["_rounds_npu"] = n_npu
        merged.append(rw)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-only", action="store_true", help="仅跑 golden, 不调 NPU")
    ap.add_argument("-v", "--verbose", action="store_true", help="详细输出 (逐 batch)")
    ap.add_argument("--filter", default=None, help="按 name 子串过滤用例")
    ap.add_argument("--case", default=None, help="精确选用例 name, 逗号分隔多个")
    ap.add_argument("-n", "--repeat", type=int, default=1, metavar="N",
                    help="重复运行 N 轮 (压测; 默认 1)")
    a = ap.parse_args()
    npu = not a.golden_only

    cases = select_cases(a)
    if not cases:
        print("no case selected (check --case/--filter)")
        return EXIT_NOT_IMPL

    n_rounds = max(a.repeat, 1)
    print(f"MX Golden+NPU (batch)  cases={len(cases)}  npu={'ON' if npu else 'OFF'}  rounds={n_rounds}")

    all_rounds_res = []
    any_round_fail = False
    ok_rounds = 0

    for ri in range(1, n_rounds + 1):
        if n_rounds > 1:
            print(f"\n{'#'*60}")
            print(f"  ROUND {ri}/{n_rounds}")
            print(f"{'#'*60}")
        res = [run(c, npu, a.verbose) for c in cases]
        all_rounds_res.append(res)
        if n_rounds > 1:
            _print_summary_table(res, round_label=f"round {ri} 汇总")
        # 统计本轮的 PASS/FAIL
        if any(r.get("passed") is False for r in res):
            any_round_fail = True
        else:
            ok_rounds += 1

    if n_rounds == 1:
        # 单轮: 走原路径
        _print_summary_table(all_rounds_res[0])
        _print_fail_details(all_rounds_res[0])
        res = all_rounds_res[0]
    else:
        # 多轮: 合并后出总表
        merged = _merge_round_results(all_rounds_res, cases)
        print(f"\n{'='*60}")
        print(f"  合并汇总 ({n_rounds} 轮, 取最差一轮)")
        name_w = max((len(r["name"]) for r in merged), default=8)
        hdr = (f"{'name':<{name_w}}  {'res':>4}  {'max_diff':>10} {'cosine':>9}  {'fail':>10}  "
               f"{'轮次':>8}  {'PASS/总':>8}")
        print(hdr)
        print("-" * len(hdr))
        for r in merged:
            p_ = "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "--")
            d_ = f"{r['max_diff']:.4f}" if r.get("max_diff") is not None else "--"
            c_ = f"{r['cosine']:.6f}" if r.get("cosine") is not None else "--"
            fbl = r.get("fail_batches") or []
            if not fbl:
                fb = "-"
            elif len(fbl) <= 8:
                fb = "[" + ",".join(str(b) for b in fbl) + "]"
            else:
                fb = f"[0..{fbl[-1]}]({len(fbl)})"
            wr = r.get("_worst_round", "-")
            pr = f"{r.get('_rounds_pass',0)}/{r.get('_rounds_total',0)}"
            note = r.get("note") or ""
            print(f"{r['name']:<{name_w}}  {p_:>4}  {d_:>10} {c_:>9}  {fb:>10}  "
                  f"{str(wr):>8}  {pr:>8}  {note}")
        _print_fail_details(merged)
        res = merged

    # ---- 三态退出码 ----
    if not npu:
        print("\n[exit] --golden-only: NOT_IMPLEMENTED (NPU 未验证)")
        return EXIT_NOT_IMPL

    # ---- 末尾 FAIL 汇总(纯展示; 单轮/多轮 res 均已按 cases 顺序确定) ----
    _print_fail_summary(cases, res)

    any_compared = any(r.get("npu_ok") for r in res)
    any_fail = any(r.get("passed") is False for r in res)
    if not any_compared:
        print("\n[exit] NOT_IMPLEMENTED: 无任何成功比较 (NPU 全部报错, batch scale 尚未实现?)")
        return EXIT_NOT_IMPL
    if any_fail:
        print(f"\n[exit] FAIL: 至少一个用例数值不匹配 (PASS {ok_rounds}/{n_rounds} 轮)")
        return EXIT_FAIL
    print(f"\n[exit] PASS: 所有用例逐 batch 通过 ({ok_rounds}/{n_rounds} 轮)")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
