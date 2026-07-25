#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QBMM-omni V3 全量验证 golden.py

全量验证套件 — V3 全分支 (aclnnAiInfraQuantMatmulV5 ND / WeightNz NZ → l0op::AiInfraQuantBatchMatmulV3):
  模式: pertensor / perchannel / pertoken / requant / int32 / int4sym
  scale: fp32(dynamic) | int64(static T-C)
  x2 格式: ND / NZ(weight_nz)
  转置: none / t_x1 / t_x2 / t_x1x2
  out_dtype: float16 / bfloat16 / int8(requant) / int32
  bias: float32 / int32

oracle 忠实移植自 ops-nn matmul/quant_batch_matmul_v3/tests/assets/golden.py (砍 A5: _compute_mx /
fp8-perblock; 砍 G-G/B-B 模式判定)。框架 (Case dataclass, gen/golden/call/cmp 流水线,
--golden-only 离线模式) 参考 QBMM-Batch/golden_mx_matmul_batch.py。int4 int32 打包
(pack_int4_lastdim, 沿末维 8:1) 参考其 pack_codes_lastdim。

判定: 对齐 nn 仓迁移范式 ai_infra_matmul test_ai_infra_matmul.py::verify_result —
  浮点(fp16/fp32 tol=1e-3, bf16 tol=5e-3): isclose(rtol=atol=tol) 逐元素, error_ratio=不匹配
  占比 ≤ 1e-4 即 PASS; int32=BinaryMatch(exact); int8=IsClose rtol=0 atol=1 (ops-nn v3 golden
  tolerance 表)。cos 仅作打印参考列不参与判定。FAIL 逐 case 打破门 + 最差元素。

用法:
  python golden.py                  # 跑 enabled case, 调 NPU
  python golden.py --golden-only    # 仅 oracle (不调 NPU, 验 oracle 自洽: 能跑出有限值)
  python golden.py --self-test      # oracle 数学手算硬编码 tiny case (离线验数学正确)
  python golden.py --list           # 列 case + enabled
  python golden.py --case v3_pertoken_nd,v3_int4sym_nd
  python golden.py --enable-all     # 无视 enabled, 跑全部

前置 (NPU 模式): torch_ops_extension wheel 已编+装 (torch.ops.custom.npu_ai_infra_quant_matmul 注册)。
退出码: 0 PASS | 1 FAIL | 2 NOT_IMPL(--golden-only 或 NPU 全未实现)。
"""

import argparse
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List

# torch_npu 仅 NPU 模式需要; --golden-only / --self-test 无 NPU 也能跑 (本地无 torch_npu 时不阻断)
try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

# omni_custom_ops: 注册 torch.ops.custom.* (TORCH_LIBRARY_FRAGMENT 在 import 时触发)。
# 必须 import, 否子 _op_registered() 恒 False (探针 test_qbmm.py 显式 import 故不漏; golden 漏了 → 修)
try:
    import omni_custom_ops  # noqa: F401
    _HAS_WHEEL = True
except ImportError:
    _HAS_WHEEL = False

# NPU 初始化 (仅 NPU 模式; 失败不阻断 golden-only/self-test)
if _HAS_TORCH_NPU:
    try:
        torch.npu.set_device(1)
        # NZ 探针依赖 npu_format_cast(x2, FRACTAL_NZ); 不开则 cast 被拒、x2 退回 base(ND) 格式,
        # binding is_nz_format 返回 false → 误走 V5 分支 → V5 没编时报 V5 not in libopapi.so
        torch.npu.config.allow_internal_format = True
    except Exception:
        pass

# ============================================================
# V3 oracle — 忠实移植 ops-nn golden.py (砍 A5), 逐函数对齐
# ============================================================
INT4_NUMS_IN_INT32 = 8  # binding 同名常量; int32 打包 8 个 int4


def _f32_2_s9(arr):
    # ops-nn _f32_2_s9: round + clip[-256,255] (requant 9bit 量化截断)
    return np.clip(np.round(arr), -256, 255)


def _scale_generate(fp32_arr):
    # ops-nn _scale_generate: fp32 高 19 位掩码 (fixpipe scale 精度截断, int8/int4 pertensor fp32 scale)
    u32 = np.ascontiguousarray(fp32_arr, dtype=np.float32).view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return u32.view(np.float32)


def _fp32_to_bf16_sim(arr):
    # 模拟 fp32→bf16 截断 (round-to-nearest-even, 低 16 位置 0), 对齐 NPU bf16 scale 精度。
    # ops-nn WeightNz case_01/n_equal_k 用 BF16 scale 直接传 (不走 trans_quant_param)。
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32).copy()
    u32 = (u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u32.view(np.float32)


def _u64_to_deq_scale(u64_scale):
    # ops-nn _u64_to_deq_scale: uint64 → 低32位作 uint32 掩码 → view fp32 (T-C static deq_scale)
    deq_u32 = u64_scale.astype(np.uint64).astype(np.uint32).copy()
    deq_u32 &= np.uint32(0xFFFFE000)
    return deq_u32.view(np.float32).reshape(u64_scale.shape)


def _cast_output(arr, dtype_name):
    # ops-nn _cast_output_dtype (砍 A5 dtype); bfloat16 numpy 无原生 → 留 fp32 高精参考, tol 吸收舍入
    m = {"float16": np.float16, "float32": np.float32, "int32": np.int32, "int8": np.int8}
    if dtype_name == "bfloat16":
        return arr.astype(np.float32)
    return arr.astype(m.get(dtype_name, np.float32))


def _needs_scale_generate(x1_dtype, scale_dtype, x2_scale, bias_dtype):
    # ops-nn _needs_scale_generate: int8/int4 + 非 uint64 scale + pertensor(shape[0]==1) + 非 bf16/f32 bias
    if x1_dtype not in ("int8", "int4"):
        return False
    if scale_dtype in ("uint64", "int64", "bfloat16"):
        return False
    if x2_scale.ndim >= 1 and x2_scale.shape[0] != 1:
        return False
    if bias_dtype in ("bfloat16", "float32"):
        return False
    return True


def _is_bias_vec(x1_dtype, bias_dtype):
    # ops-nn: int8/int4 + float bias → is_bias_vec (bias 在 scale 后加, fixpipe 后处理)
    return x1_dtype in ("int8", "int4") and bias_dtype is not None and \
        bias_dtype in ("bfloat16", "float16", "float32")


def _br(sc, out_ndim, axis):
    # broadcast scale 到 out_ndim (插维对齐): [N]→[1,N], [B,N]→[B,1,N]; [M]→[M,1], [B,M]→[B,M,1]
    while sc.ndim < out_ndim:
        sc = np.expand_dims(sc, axis)
    return sc.astype(np.float32)


def _compute_tc(mm, x2_scale, bias, bias_dtype, do_scale_gen, is_bias_vec, y_dtype):
    # ops-nn _compute_tc: T-C (pertensor/perchannel/int4sym). bias 位置由 is_bias_vec 定。
    out = mm
    if not is_bias_vec and bias is not None:
        if bias_dtype == "int32":
            out = out + bias  # int32 域前加
        elif bias_dtype in ("float32", "bfloat16"):
            out = out.astype(np.float32) + bias.astype(np.float32)
    out = out.astype(np.float32)
    if do_scale_gen:
        x2_scale = _scale_generate(x2_scale)
    out = out * _br(x2_scale, out.ndim, -2)
    if is_bias_vec and bias is not None:
        return _cast_output(out + bias.astype(np.float32), y_dtype)
    return _cast_output(out, y_dtype)


def _compute_pertoken(mm, x2_scale, x1_scale, bias, bias_dtype, is_bias_vec, y_dtype):
    # ops-nn _compute_pertoken: K-C (int8 pertoken, A2A3 非 is_two_scale 双标量分支)。
    out = mm
    if bias is not None and bias_dtype == "int32":
        out = out + bias  # int32 bias 前加
    out = out.astype(np.float32)
    out = out * _br(x2_scale, out.ndim, -2) * _br(x1_scale, out.ndim, -1)  # [..,1,N] × [..,M,1]
    if not is_bias_vec and bias is not None and bias_dtype == "float32":
        out = out + bias.astype(np.float32)
    if is_bias_vec and bias is not None:
        return _cast_output(out + bias.astype(np.float32), y_dtype)
    return _cast_output(out, y_dtype)


def _compute_requant(mm, x2_scale, offset, bias, bias_dtype):
    # ops-nn _compute_requant: y=int8. f32_2_s9(out×scale) [+ f32_2_s9(offset)] → clip[-128,127]。
    out = mm
    if bias is not None and bias_dtype == "int32":
        out = out + bias
    out = out.astype(np.float32)
    out = _f32_2_s9(out * _br(x2_scale, out.ndim, -2))
    if offset is not None:
        out = _f32_2_s9(out) + _f32_2_s9(_br(offset, out.ndim, -2))
    return np.clip(out, -128, 127).astype(np.int8)


def _compute_int32(mm, bias, bias_dtype):
    # ops-nn _compute_int32: y=int32. scale 不参与, [+int32/float bias] → int32。
    out = mm
    if bias is not None:
        if bias_dtype == "int32":
            out = out + bias
        else:
            out = out.astype(np.float32) + bias.astype(np.float32)
    return out.astype(np.int32)


# ============================================================
# int4 → int32 打包 (参考 QBMM-Batch pack_codes_lastdim; 沿末维 8:1)
# ============================================================
def pack_int4_lastdim(code):
    """(..., D) int4 codes (int8 持 -8..7 两补值) → (..., D/8) int32, 沿末维 8 int4 打 1 int32。
    nibble 序: 低 nibble = 靠前元素 (code[..., 0] 占 bit0-3, code[...,7] 占 bit28-31)。
    int4 两补: -8→0x8, -1→0xF, 7→0x7。int32 仅作 32bit 容器, 符号位无意义 (NPU 按 nibble 解包)。"""
    assert code.shape[-1] % INT4_NUMS_IN_INT32 == 0, \
        f"int4 打包要求末维 %{INT4_NUMS_IN_INT32} == 0, got shape {code.shape}"
    c = (code.astype(np.int64) & 0xF).astype(np.int32)  # 取低 4 bit (两补 nibble)
    packed = np.zeros(code.shape[:-1] + (code.shape[-1] // INT4_NUMS_IN_INT32,), dtype=np.int32)
    for j in range(INT4_NUMS_IN_INT32):
        packed |= (c[..., j::INT4_NUMS_IN_INT32] << (4 * j))
    return packed


# ============================================================
# Case 框架 (参考 QBMM-Batch @dataclass Case)
# ============================================================
@dataclass
class Case:
    name: str
    mode: str                  # pertensor|perchannel|pertoken|requant|int32|int4sym|a8w4|perblock
    M: int
    N: int
    K: int
    out_dtype: str = "float16"
    scale_dtype: str = "float32"   # float32 (dynamic T-C) | int64 (static T-C, pertensor/perchannel)
    batch: List[int] = field(default_factory=list)
    trans_x1: bool = False
    trans_x2: bool = False
    weight_nz: bool = False
    has_bias: bool = False
    bias_dtype: str = "float32"    # float32 | int32
    enabled: bool = False
    expect_fail: bool = False
    seed: Optional[int] = None
    group_sizes: Optional[List[int]] = None  # G-B/K-G perblock/pergroup: [gM, gN, gK]


# ---- V3 全分支 case 矩阵 ----
# enabled 首批放开核心 5 模式 ND + int4sym (P4 V3 路径); NZ/transpose/bias/out_dtype 变体挨个放开。
# ND 路径 binding 走 aclnnAiInfraQuantMatmulV5, V5 入口层已迁 (v4/op_api) → 编+装后 ND 可跑。
CASES: List[Case] = [
    # ===== V3 核心 5 模式 + int4sym (ND) — P4 首批; V5 入口层已迁 (增量①), 编+装 .run 后放开 =====
    Case("v3_pertensor_dyn_nd",   "pertensor",  128, 128, 256, out_dtype="float16", scale_dtype="float32", enabled=True),
    Case("v3_pertensor_stat_nd",  "pertensor",  128, 128, 256, out_dtype="float16", scale_dtype="int64",   enabled=True),
    Case("v3_perchannel_dyn_nd",  "perchannel", 128, 128, 256, out_dtype="float16", scale_dtype="float32", enabled=True),
    Case("v3_perchannel_stat_nd", "perchannel", 128, 128, 256, out_dtype="float16", scale_dtype="int64",   enabled=True),
    Case("v3_pertoken_nd",        "pertoken",   128, 128, 256, out_dtype="bfloat16", enabled=True),
    Case("v3_requant_nd",         "requant",    128, 128, 256, out_dtype="int8",     enabled=True),
    Case("v3_int32_nd",           "int32",      128, 128, 256, out_dtype="int32",    enabled=True),

    # ===== out_dtype 变体 (perchannel 可出 int8/int32; pertoken fp16) =====
    Case("v3_perchannel_int8_nd",  "perchannel", 128, 128, 256, out_dtype="int8",   enabled=True),
    Case("v3_perchannel_int32_nd", "perchannel", 128, 128, 256, out_dtype="int32",  enabled=True),
    Case("v3_pertoken_fp16_nd",    "pertoken",   128, 128, 256, out_dtype="float16", enabled=True),
    Case("v3_pertensor_int8_nd",   "pertensor",  128, 128, 256, out_dtype="int8",   enabled=True),

    # ===== bias (float32 after-scale / int32 before-scale) =====
    # ⚠ fbias disabled: perchannel+fp32 scale → trans_quant_param 转 int64(static T-C) → EZ0020 要求 bias=INT32, float32 无效
    Case("v3_perchannel_fbias_nd", "perchannel", 128, 128, 256, out_dtype="float16", has_bias=True, bias_dtype="float32", enabled=False),
    Case("v3_perchannel_ibias_nd", "perchannel", 128, 128, 256, out_dtype="float16", has_bias=True, bias_dtype="int32",   enabled=True),
    Case("v3_int32_ibias_nd",      "int32",      128, 128, 256, out_dtype="int32",   has_bias=True, bias_dtype="int32",   enabled=True),

    # ===== transpose (binding transpose1/2 恒 false, 靠 stride; M=N=K=128 因 GetTransposeAttrValue swap+flip 与 checker 读 K 维交互, M≠K 时 EZ0027 K 不匹配) =====
    Case("v3_perchannel_tx1_nd",   "perchannel", 128, 128, 128, out_dtype="float16", trans_x1=True,  enabled=True),
    Case("v3_perchannel_tx2_nd",   "perchannel", 128, 128, 128, out_dtype="float16", trans_x2=True,  enabled=True),
    Case("v3_perchannel_tx1x2_nd", "perchannel", 128, 128, 128, out_dtype="float16", trans_x1=True, trans_x2=True, enabled=True),
    Case("v3_pertoken_tx2_nd",     "pertoken",   128, 128, 128, out_dtype="bfloat16", trans_x2=True, enabled=True),

    # ===== NZ (x2 → FRACTAL_NZ, 走 aclnnAiInfraQuantMatmulWeightNz) =====
    # V3 ND/NZ 双路径全覆盖: 每个模式 × ND/NZ 各一 (ND 在上方; NZ 在此)
    Case("v3_pertensor_dyn_nz",  "pertensor",  128, 128, 256, out_dtype="float16", weight_nz=True, enabled=True),
    Case("v3_pertensor_stat_nz", "pertensor",  128, 128, 256, out_dtype="float16", scale_dtype="int64", weight_nz=True, enabled=True),
    Case("v3_perchannel_dyn_nz", "perchannel", 128, 128, 256, out_dtype="float16", weight_nz=True, enabled=True),
    Case("v3_perchannel_stat_nz","perchannel", 128, 128, 256, out_dtype="float16", scale_dtype="int64", weight_nz=True, enabled=True),
    Case("v3_pertoken_nz",       "pertoken",   128, 128, 256, out_dtype="bfloat16", weight_nz=True, enabled=True),
    Case("v3_requant_nz",        "requant",    128, 128, 256, out_dtype="int8",    weight_nz=True, enabled=True),
    Case("v3_int32_nz",          "int32",      128, 128, 256, out_dtype="int32",   weight_nz=True, enabled=True),

    # ===== W8A8 BF16-scale NZ (ops-nn WeightNz case_01/n_equal_k 风格; BF16 scale 直接传, 不走 trans_quant_param) =====
    Case("v3_pertensor_bf16sc_nz",   "pertensor",  128, 128, 256, out_dtype="bfloat16", scale_dtype="bfloat16", weight_nz=True, enabled=True),
    Case("v3_perchannel_bf16sc_nz",  "perchannel", 128, 128, 256, out_dtype="bfloat16", scale_dtype="bfloat16", weight_nz=True, enabled=True),

    # ===== 边界 shape (A2A3 约束: 2D only; 无 K 对齐要求 for INT8 T-C/T-T/K-C) =====
    # GEMV(M=1) / 单列(N=1) / 非方 / 小shape / 大K累加 / K对齐边界 / 各mode+ND/NZ
    Case("bd_perchannel_m1_nd",     "perchannel", 1,   128, 256, out_dtype="float16", enabled=True),
    Case("bd_perchannel_n1_nd",     "perchannel", 128, 1,   256, out_dtype="float16", enabled=True),
    Case("bd_perchannel_nonsq_nd",  "perchannel", 64,  256, 512, out_dtype="float16", enabled=True),
    Case("bd_perchannel_nonsq2_nd", "perchannel", 256, 64,  512, out_dtype="float16", enabled=True),
    Case("bd_perchannel_small_nd",  "perchannel", 32,  32,  32,  out_dtype="float16", enabled=True),
    Case("bd_perchannel_k64_nd",    "perchannel", 128, 128, 64,  out_dtype="float16", enabled=True),
    Case("bd_pertensor_m1_nd",      "pertensor",  1,   128, 256, out_dtype="float16", scale_dtype="float32", enabled=True),
    Case("bd_pertoken_m1_nd",       "pertoken",   1,   128, 256, out_dtype="bfloat16", enabled=True),
    Case("bd_pertoken_fp16_m1_nd",  "pertoken",   1,   64,  128, out_dtype="float16", enabled=True),
    Case("bd_int32_bigk_nd",        "int32",      128, 128, 1024, out_dtype="int32",  enabled=True),
    Case("bd_requant_bigk_nd",      "requant",    128, 128, 512,  out_dtype="int8",   enabled=True),
    Case("bd_pertensor_stat_nonsq_nd","pertensor",64,  256, 512, out_dtype="float16", scale_dtype="int64", enabled=True),
    # NZ 边界
    Case("bd_perchannel_m1_nz",     "perchannel", 1,   128, 256, out_dtype="float16", weight_nz=True, enabled=True),
    Case("bd_perchannel_nonsq_nz",  "perchannel", 64,  256, 512, out_dtype="float16", weight_nz=True, enabled=True),
    Case("bd_pertoken_m1_nz",       "pertoken",   1,   128, 256, out_dtype="bfloat16", weight_nz=True, enabled=True),

    # ===== NZ 扩展: transpose + bias + out_dtype 变体 (补齐 WeightNz 路径覆盖) =====
    Case("ext_perchannel_tx2_nz",   "perchannel", 128, 128, 128, out_dtype="float16", weight_nz=True, trans_x2=True, enabled=True),
    Case("ext_perchannel_ibias_nz",  "perchannel", 128, 128, 256, out_dtype="float16", weight_nz=True, has_bias=True, bias_dtype="int32", enabled=True),
    Case("ext_perchannel_int8_nz",   "perchannel", 128, 128, 256, out_dtype="int8",    weight_nz=True, enabled=True),
    Case("ext_perchannel_int32_nz",  "perchannel", 128, 128, 256, out_dtype="int32",   weight_nz=True, enabled=True),
    Case("ext_perchannel_bf16sc_nd", "perchannel", 128, 128, 256, out_dtype="bfloat16", scale_dtype="bfloat16", enabled=True),
    Case("ext_pertensor_bf16sc_nd",  "pertensor",  128, 128, 256, out_dtype="bfloat16", scale_dtype="bfloat16", enabled=True),

    # ===== pertoken 扩展: bias + fp16 out (K-C bias 变体) =====
    Case("ext_pertoken_ibias_nd",   "pertoken",   128, 128, 256, out_dtype="bfloat16", has_bias=True, bias_dtype="int32", enabled=True),
    Case("ext_perchannel_stat_ibias_nd","perchannel",128,128, 256, out_dtype="float16", scale_dtype="int64", has_bias=True, bias_dtype="int32", enabled=True),

    # ===== 占位: ops-nn 有但当前测不了的 WeightNz case =====
    # int4×int4 perchannel NZ (ops-nn a4w4_case_01~12): binding npu_ai_infra_quant_matmul.cpp:228 `!is_a4w4` 条件
    #   把 int4 NZ 路由到 V5 (非 WeightNz); V5 没迁 → 测不了。要测需改 binding 条件或迁 V5。
    # A8W4-int NZ (ops-nn a8w4_case_1~5: int8×int32 + FLOAT x1Scale + UINT64 x2Scale + FLOAT yOffset → FP16):
    #   binding 走 WeightNz (is_a4w4=false ✓), 但 oracle 需扩 int4 打包反解 + yOffset 语义, 待实现。
    # ⚠ A8W4-int disabled: EH0012 "Current version do not support yOffset" — A8W4-int 用 yOffset, 当前版本不支持
    Case("v3_a8w4int_nz",         "a8w4",    128, 128, 256, out_dtype="float16", weight_nz=True, enabled=False),

    # ===== V4 核: G-B perblock (int8×int8 + FP32 block scales, groupSize=[1,128,128]; V5→l0op::V4) =====
    # aclnn V5 G-B 约束(line 662-663): k 须 4*128=512 倍数, n 须 256 倍数, transposeX2=true; out=BFLOAT16
    # ===== V4 核: G-B perblock (int8×int8 + FP32 block scales, groupSize=[1,128,128]; V5→l0op::V4) =====
    # aclnn V5 G-B 约束(line 662-663): k 须 4*128=512 倍数, n 须 256 倍数, transposeX2=true, out=BFLOAT16
    # K=N=512: 满足约束 + mask checker EZ0027 (GetTransposeAttrValue swap+flip, M≠K 时 K 维读错)
    # aclnn V5 G-B 约束(line 651-663): k=4*128倍数, n=256倍数, transposeX2=true, bias=FLOAT32(必选), out=BFLOAT16
    Case("v4_perblock_gb_nd",  "perblock", 128, 512, 512, out_dtype="bfloat16",
         trans_x2=True, group_sizes=[1, 128, 128], has_bias=True, bias_dtype="float32", enabled=True),

    # ===== batch (numpy 式右对齐广播; 单层 2) — ⚠ A2A3 V5 仅支持 1~2D, batch(3D+) 非 A2A3, 禁 =====
    Case("v3_perchannel_b2_nd",  "perchannel", 128, 128, 256, out_dtype="float16", batch=[2], enabled=False),
    Case("v3_pertoken_b2_nd",    "pertoken",   128, 128, 256, out_dtype="bfloat16", batch=[2], enabled=False),
]


# ============================================================
# 数据生成 — 按 mode 生正确 dtype; int4 同时给 logical(oracle) + packed(NPU)
# ============================================================
def gen_data(c: Case) -> dict:
    if c.seed is not None:
        torch.manual_seed(c.seed)
    M, N, K = c.M, c.N, c.K
    B = tuple(c.batch)
    rng = np.random.default_rng((hash(c.name) & 0xffff))
    d = dict(mode=c.mode, out_dtype=c.out_dtype, trans_x1=c.trans_x1, trans_x2=c.trans_x2,
             weight_nz=c.weight_nz, bias_dtype=(c.bias_dtype if c.has_bias else None))

    if c.mode in ("pertensor", "perchannel", "pertoken", "requant", "int32"):
        x1 = rng.integers(-5, 5, B + (M, K), dtype=np.int8)
        x2 = rng.integers(-5, 5, B + (K, N), dtype=np.int8)
        d["x1"], d["x2"] = x1, x2
        d["x1_dtype"], d["x2_dtype"] = "int8", "int8"
        # scale shape: pertensor=[1] (标量 broadcast); 其余=[N]
        sc_shape = (1,) if c.mode == "pertensor" else (N,)
        if c.scale_dtype == "int64":  # static T-C: fp32→uint64 编码
            # 不能直接随机 uint64 (_u64_to_deq_scale 取低32bit view fp32 多半得 denormal/0/inf)。
            # 正确: 生成 fp32 → scale_generate 截断 (低13位置0) → view uint32 → 升 uint64;
            #   这样 _u64_to_deq_scale(uint64) = scale_generate(fp32) round-trip 回合理 fp32。
            fp32_sc = (rng.standard_normal(B + sc_shape) * 0.05).astype(np.float32)
            d["scale"] = _scale_generate(fp32_sc).view(np.uint32).astype(np.uint64)
            d["scale_dtype_str"] = "uint64"
        elif c.scale_dtype == "bfloat16":  # ops-nn WeightNz case_01/n_equal_k: BF16 scale 直接传 (不走 trans_quant_param)
            d["scale"] = (rng.standard_normal(B + sc_shape) * 0.05).astype(np.float32)
            d["scale_dtype_str"] = "bfloat16"
        else:  # dynamic T-C: fp32 (binding 走 trans_quant_param)
            d["scale"] = (rng.standard_normal(B + sc_shape) * 0.05).astype(np.float32)
            d["scale_dtype_str"] = "float32"
        d["pertoken_scale"] = (rng.standard_normal(B + (M,)) * 0.05).astype(np.float32) \
            if c.mode == "pertoken" else None
        d["offset"] = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32) \
            if c.mode == "requant" else None
        if c.has_bias:
            if c.bias_dtype == "int32":
                d["bias"] = rng.integers(-5, 5, B + (N,), dtype=np.int32)
            else:
                d["bias"] = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32)
        else:
            d["bias"] = None

    elif c.mode == "int4sym":
        assert K % INT4_NUMS_IN_INT32 == 0 and N % INT4_NUMS_IN_INT32 == 0, \
            f"int4sym 要求 K,N %{INT4_NUMS_IN_INT32}==0 (int32 打包 8 int4)"
        # int4 范围 -8..7; 取 -7..7 避 -8 边界 (符号打包两补仍正确, 仅规避累加溢出 int32 的极端)
        x1 = rng.integers(-7, 8, B + (M, K), dtype=np.int8)
        x2 = rng.integers(-7, 8, B + (K, N), dtype=np.int8)
        d["x1"], d["x2"] = x1, x2                                 # logical int4 (int8 持值) 供 oracle
        d["x1_packed"] = pack_int4_lastdim(x1)                    # (B,M,K/8) int32 供 NPU
        d["x2_packed"] = pack_int4_lastdim(x2)                    # (B,K,N/8) int32 供 NPU
        d["x1_dtype"], d["x2_dtype"] = "int4", "int4"
        d["scale"] = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32)
        d["scale_dtype_str"] = "float32"
        d["pertoken_scale"] = d["offset"] = d["bias"] = None

    elif c.mode == "a8w4":
        # A8W4-int (MSD): x1 int8[M,K] × x2 int4[K,N](int32 N-packed) → fp16
        #   x2Scale=scale(uint64[N]), x1Scale=pertoken_scale(fp32[M]), yOffset=offset(fp32[N])
        assert N % INT4_NUMS_IN_INT32 == 0, f"A8W4 N%{INT4_NUMS_IN_INT32}==0 (int32 N-packed)"
        x1 = rng.integers(-5, 5, B + (M, K), dtype=np.int8)
        x2 = rng.integers(-7, 8, B + (K, N), dtype=np.int8)
        d["x1"], d["x2"] = x1, x2
        d["x2_packed"] = pack_int4_lastdim(x2)
        d["x1_dtype"], d["x2_dtype"] = "int8", "int4"
        fp32_sc = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32)
        d["scale"] = _scale_generate(fp32_sc).view(np.uint32).astype(np.uint64)
        d["scale_dtype_str"] = "uint64"
        d["pertoken_scale"] = (rng.standard_normal(B + (M,)) * 0.05).astype(np.float32)
        d["offset"] = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32)
        d["bias"] = None

    elif c.mode == "perblock":
        # G-B perblock: int8[M,K]×int8[K,N] + FP32 block scales, groupSize=[1,128,128]
        #   x1Scale(pertoken_scale)=fp32[M,⌈K/gK⌉], x2Scale(scale)=fp32[⌈K/gK⌉,⌈N/gN⌉], bias=fp32[N]
        #   call_nz trans_x2: x2/scale 传转置存储; oracle 用逻辑布局算
        gM, gN, gK = c.group_sizes if c.group_sizes else (1, 128, 128)
        assert K % gK == 0 and N % gN == 0, f"perblock K%{gK}==0 N%{gN}==0 (groupSize)"
        x1 = rng.integers(-5, 5, B + (M, K), dtype=np.int8)
        x2 = rng.integers(-5, 5, B + (K, N), dtype=np.int8)
        d["x1"], d["x2"] = x1, x2
        d["x1_dtype"], d["x2_dtype"] = "int8", "int8"
        d["pertoken_scale"] = (rng.standard_normal(B + (M, K // gK)) * 0.05).astype(np.float32)
        d["scale"] = (rng.standard_normal(B + (K // gK, N // gN)) * 0.05).astype(np.float32)
        d["scale_dtype_str"] = "float32"
        d["offset"] = None
        d["bias"] = (rng.standard_normal(B + (N,)) * 0.05).astype(np.float32)
    else:
        raise ValueError(f"unknown mode {c.mode}")
    return d


# ============================================================
# oracle 入口 (按 mode 分派到移植的 _compute_*; 模式由 Case 定, 不重做 _determine_quant_mode 推断)
# ============================================================
def golden_ref(d: dict, c: Case) -> np.ndarray:
    x1dt, x2dt = d["x1_dtype"], d["x2_dtype"]
    # int8/int4 → int32 精确累加 (对齐 ops-nn: x1.astype(int32) × x2.astype(int32))
    x1 = d["x1"].astype(np.int32) if x1dt in ("int8", "int4") else d["x1"].astype(np.float32)
    x2 = d["x2"].astype(np.int32) if x2dt in ("int8", "int4") else d["x2"].astype(np.float32)
    # oracle 须对转置 case 先 swap x1/x2 再 matmul (照 ops-nn golden:95-98; op 拿到转置 input + transposeX flag 后算的是转置 matmul)
    if c.trans_x1:
        x1 = np.swapaxes(x1, -1, -2)
    if c.trans_x2:
        x2 = np.swapaxes(x2, -1, -2)
    mm = np.matmul(x1, x2)                                       # (B,M,N) int32

    sc = d["scale"]
    if d["scale_dtype_str"] == "uint64":
        sc = _u64_to_deq_scale(sc)                               # uint64 → fp32 deq_scale (已截断)
    elif d["scale_dtype_str"] == "bfloat16":
        sc = _fp32_to_bf16_sim(sc)                               # fp32 → bf16 截断 (对齐 NPU BF16 scale 精度)
    bdtype = d["bias_dtype"]
    ydt = c.out_dtype

    # int32 输出 = 原始 int32 累加 (不乘 scale), 无论 mode — NPU 对 int32 out 返回 raw matmul (aclnn doc: int32 out 是独立模式)
    if ydt == "int32":
        return _compute_int32(mm, d["bias"], bdtype)

    if c.mode in ("pertensor", "perchannel", "int4sym"):
        do_sg = _needs_scale_generate(x1dt, d["scale_dtype_str"], sc, bdtype)
        bv = _is_bias_vec(x1dt, bdtype)
        return _compute_tc(mm, sc, d["bias"], bdtype, do_sg, bv, ydt)
    if c.mode == "pertoken":
        bv = _is_bias_vec(x1dt, bdtype)
        return _compute_pertoken(mm, sc, d["pertoken_scale"], d["bias"], bdtype, bv, ydt)
    if c.mode == "requant":
        return _compute_requant(mm, sc, d["offset"], d["bias"], bdtype)
    if c.mode == "int32":
        return _compute_int32(mm, d["bias"], bdtype)
    if c.mode == "a8w4":
        # MSD: out = matmul(int8,int4) × x2Scale[N] × x1Scale[M] + yOffset[N]
        x1s = np.expand_dims(d["pertoken_scale"].astype(np.float32), -1)
        yoff = d["offset"].astype(np.float32)
        sc_f = sc.astype(np.float32)
        while sc_f.ndim < mm.ndim:
            sc_f = np.expand_dims(sc_f, -2)
        while yoff.ndim < mm.ndim:
            yoff = np.expand_dims(yoff, -2)
        return _cast_output(mm.astype(np.float32) * sc_f * x1s + yoff, ydt)
    if c.mode == "perblock":
        # G-B perblock: per-tile K matmul × (x1Scale[M,1] × x2Scale[1,N]) 累加; 忠实移植 ops-nn _compute_per_tile_int8 (2D)
        # 转置须 swap x1/x2 + scale (照 ops-nn _compute_per_tile_int8:345-354)
        gM, gN, gK = c.group_sizes if c.group_sizes else (1, 128, 128)
        x1f = d["x1"].astype(np.float32)
        x2f = d["x2"].astype(np.float32)
        x1sf = d["pertoken_scale"].astype(np.float32)
        x2sf = d["scale"].astype(np.float32)
        if c.trans_x1:
            x1f = np.swapaxes(x1f, -1, -2)
            x1sf = np.swapaxes(x1sf, -1, -2)
        if c.trans_x2:
            x2f = np.swapaxes(x2f, -1, -2)
            x2sf = np.swapaxes(x2sf, -1, -2)
        m = x1f.shape[-2]; k = x1f.shape[-1]; n = x2f.shape[-1]
        x2sf_kn = np.repeat(x2sf, gN, axis=-1)[..., :n]
        out = np.zeros(x1f.shape[:-2] + (m, n), dtype=np.float32)
        for kt in range((k + gK - 1) // gK):
            ks = kt * gK; ke = min(ks + gK, k)
            tile_sc = x1sf[..., kt:kt + 1] * x2sf_kn[..., kt:kt + 1, :]
            out = out + np.matmul(x1f[..., ks:ke], x2f[..., ks:ke, :]) * tile_sc
        if d["bias"] is not None:
            out = out + d["bias"].astype(np.float32)
        return _cast_output(out, ydt)
    raise ValueError(f"unknown mode {c.mode}")


# ============================================================
# NPU 调用 — torch.ops.custom.npu_ai_infra_quant_matmul
# ============================================================
_TORCH_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
             "int32": torch.int32, "int8": torch.int8}


def _np_to_npu(arr, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t.npu()


def call_npu(d: dict, c: Case) -> torch.Tensor:
    # x1/x2: int4sym 用 int32 packed; a8w4 用 x1 int8 + x2 int32(N-packed); 其余 int8 logical
    if c.mode == "int4sym":
        x1 = _np_to_npu(d["x1_packed"])
        x2 = _np_to_npu(d["x2_packed"])
    elif c.mode == "a8w4":
        x1 = _np_to_npu(d["x1"].astype(np.int8))
        x2 = _np_to_npu(d["x2_packed"])
    else:
        x1 = _np_to_npu(d["x1"].astype(np.int8))
        x2 = _np_to_npu(d["x2"].astype(np.int8))

    # NZ cast 须在 transpose 之前 (参考 QBMM-Batch: transpose 后再 cast 的 contiguous 抹转置 stride)
    if c.weight_nz:
        x2 = torch_npu.npu_format_cast(x2, 29)
    # transpose: binding transpose1/2 恒 false, 靠 stride 检测; .transpose(-1,-2) 造转置 stride
    if c.trans_x1:
        x1 = x1.transpose(-1, -2)
    if c.trans_x2:
        x2 = x2.transpose(-1, -2)

    kw = dict(x1=x1, x2=x2)
    # scale: uint64 static → int64 (binding 接 int64/uint64 deq_scale); bf16 → torch.bfloat16; fp32 → float
    if d["scale_dtype_str"] == "uint64":
        kw["scale"] = _np_to_npu(d["scale"].astype(np.int64))
    elif d["scale_dtype_str"] == "bfloat16":
        kw["scale"] = _np_to_npu(d["scale"], dtype=torch.bfloat16)
    else:
        kw["scale"] = _np_to_npu(d["scale"])
    # perblock(G-B): x2Scale 是 2D, trans_x2 时须同步转置 (binding is_x_scale_same_transpose 校验 x2↔scale 一致)
    if c.trans_x2 and kw["scale"].dim() >= 2:
        kw["scale"] = kw["scale"].transpose(-1, -2)
    if d["offset"] is not None:
        kw["offset"] = _np_to_npu(d["offset"])
    if d["pertoken_scale"] is not None:
        kw["pertoken_scale"] = _np_to_npu(d["pertoken_scale"])
    if d["bias"] is not None:
        bt = torch.int32 if d["bias_dtype"] == "int32" else torch.float32
        kw["bias"] = _np_to_npu(d["bias"], dtype=bt)
    if c.group_sizes:
        kw["group_sizes"] = c.group_sizes
    kw["output_dtype"] = _TORCH_DT[c.out_dtype]

    out = torch.ops.custom.npu_ai_infra_quant_matmul(**kw)
    torch.npu.synchronize()
    return out


# ============================================================
# 精度判据 — 对齐 quant_batch_matmul_v3 官方 BenchmarkCompareStandard (误差均衡标准)
#   忠实移植 ops-transformer tests/pytests/utils/compare.py::checkResultNew + check_result(new=True)
#   参数取 ops-nn quant_batch_matmul_v3/tests/assets/golden.py 的 tolerance 表 (fp16/bf16/fp32)。
#   大值(|g|≥small_value): 逐指标相对误差 (max/avg) ≤ benchmark_floor × rtol; 小值: |Δ|≤small_value_atol。
#   benchmark_floor = 该 dtype 相对分辨率 2^-mantissa (new=True 用固定地板替代 CPU 标杆的舍入误差)。
#   int32: BinaryMatch 逐元素精确; int8: IsClose rtol=0 atol=1 (golden 表 line 170-171)。
# ============================================================
# 官方 tolerance 表 (ops-nn quant_batch_matmul_v3 golden.py:150-172)
BM_CMP_STD = {
    "float32":  {"avg_re_rtol": 2.0, "max_re_rtol": 5.0,  "rmse_rtol": 2.0, "small_value": 1e-6, "small_value_atol": 1e-9},
    "float16":  {"avg_re_rtol": 2.0, "max_re_rtol": 10.0, "rmse_rtol": 2.0, "small_value": 1e-3, "small_value_atol": 1e-5},
    "bfloat16": {"avg_re_rtol": 2.0, "max_re_rtol": 10.0, "rmse_rtol": 2.0, "small_value": 1e-3, "small_value_atol": 1e-5},
}
# new=True 的 benchmark 相对误差地板 (compare.py:117-120 lo_bound)
BM_FLOOR = {"float32": 2.0 ** -24, "float16": 2.0 ** -11, "bfloat16": 2.0 ** -8}
INT_EXACT = {"int32": True, "int8": False}
INT8_ATOL = 1


def bench_metrics(value, golden, out_dtype):
    # 照 compare.py: float32 比对; NPU 输出与 golden 都拉平到 cpu.float
    v = value.reshape(-1).float().cpu()
    g = golden.reshape(-1).float().cpu()
    n = g.numel()
    absdiff_all = (v - g).abs()

    # 打印参考: cos (方向一致性) + 最差元素 (非有限位置置 0 不入统计)
    zeros = torch.zeros_like(g)
    finite = ~(torch.isinf(g) | torch.isnan(g)) & ~(torch.isinf(v) | torch.isnan(v))
    gf, vf = torch.where(finite, g, zeros), torch.where(finite, v, zeros)
    gn, vn = float(gf.norm()), float(vf.norm())
    cos = float(gf @ vf / (gn * vn)) if gn > 0 and vn > 0 else 1.0
    gabs = gf.abs()
    n_worst = int(min(5, n))
    if n_worst > 0:
        _, idx = torch.topk((vf - gf).abs(), n_worst)
        worst = [(float(gf[i]), float(vf[i]), float((vf - gf).abs()[i]),
                  float((vf - gf).abs()[i] / gabs[i]) if float(gabs[i]) > 0 else float("inf"))
                 for i in idx.tolist()]
    else:
        worst = []
    m = dict(cos=cos, worst=worst, max_abs_diff=float(absdiff_all.max()) if n > 0 else 0.0)

    # ---- 整数: BinaryMatch / IsClose ----
    if out_dtype in INT_EXACT and INT_EXACT[out_dtype]:
        n_mism = int(torch.sum(absdiff_all > 0))
        m.update(kind="int_exact", n_mism=n_mism, err_ratio=float(n_mism) / n if n else 0.0)
        return m
    if out_dtype in INT_EXACT:
        n_mism = int(torch.sum(absdiff_all > INT8_ATOL))
        m.update(kind="int_atol", n_mism=n_mism, err_ratio=float(n_mism) / n if n else 0.0)
        return m

    # ---- 浮点: BenchmarkCompareStandard (new=True) ----
    std = BM_CMP_STD[out_dtype]
    sv = std["small_value"]
    big_mask = g.abs() >= sv
    total_big = int(torch.sum(big_mask))
    # 大值相对误差 (compare.py:227-235: 小值位置 g/v 置 1 使其 diff=0 不干扰大值统计)
    gb = torch.where(big_mask, g, torch.ones_like(g))
    vb = torch.where(big_mask, v, torch.ones_like(v))
    diff_big = (vb - gb).abs()
    ratio_big = diff_big / gb.abs()
    dbr_max = float(ratio_big.max()) if n else 0.0
    dbr_avg = float(ratio_big.sum() / total_big) if total_big else 0.0
    # 小值绝对误差 (compare.py:250-256)
    gs = torch.where(~big_mask, g, torch.ones_like(g))
    vs = torch.where(~big_mask, v, torch.ones_like(v))
    err_small = int(torch.sum((vs - gs).abs() > std["small_value_atol"]))
    # 门 (compare.py:114-145, new=True: benchmark 用地板, 只查 max/avg/err_small)
    floor = BM_FLOOR[out_dtype]
    gate_max = floor * std["max_re_rtol"]
    gate_avg = floor * std["avg_re_rtol"]
    gate_small = 0.0  # new=True benchmark err_small=0 → gate = 0 × avg_re_rtol = 0 (小值须零错)
    m.update(kind="bench", total_big=total_big, dbr_max=dbr_max, dbr_avg=dbr_avg,
             err_small=err_small, gate_max=gate_max, gate_avg=gate_avg, gate_small=gate_small,
             sv=sv, small_atol=std["small_value_atol"])
    return m


def bench_verdict(m, out_dtype):
    reasons = []
    if m["kind"] == "int_exact":
        if m["n_mism"] > 0:
            reasons.append(f"int32 BinaryMatch: {m['n_mism']} 个不精确")
    elif m["kind"] == "int_atol":
        if m["n_mism"] > 0:
            reasons.append(f"int8 IsClose atol={INT8_ATOL}: {m['n_mism']} 个超差")
    else:
        if m["dbr_max"] > m["gate_max"]:
            reasons.append(f"大值最大相对误差 {m['dbr_max']:.4g}>{m['gate_max']:.4g}")
        if m["dbr_avg"] > m["gate_avg"]:
            reasons.append(f"大值平均相对误差 {m['dbr_avg']:.4g}>{m['gate_avg']:.4g}")
        if m["err_small"] > m["gate_small"]:
            reasons.append(f"小值错误数 {m['err_small']}>{m['gate_small']:.0f}")
    return (len(reasons) == 0), reasons


def _g4(x):
    return f"{x:.4g}"


def _fmt_verdict(m, out_dtype, passed, note_extra=""):
    verd = "PASS" if passed else "FAIL"
    if m["kind"] == "int_exact":
        body = f"int32 BinaryMatch(exact) 不匹配={m['n_mism']}"
    elif m["kind"] == "int_atol":
        body = f"int8 IsClose atol={INT8_ATOL} 不匹配={m['n_mism']}"
    else:
        mx_ok = m["dbr_max"] <= m["gate_max"]
        av_ok = m["dbr_avg"] <= m["gate_avg"]
        sm_ok = m["err_small"] <= m["gate_small"]
        mx = (f"大值maxRE {_g4(m['dbr_max'])}/{_g4(m['gate_max'])}=✓({m['dbr_max']/m['gate_max']*100:.1f}%)"
              if mx_ok else f"大值maxRE {_g4(m['dbr_max'])}/{_g4(m['gate_max'])}=✗破(超{m['dbr_max']/m['gate_max']:.3g}×)")
        av = (f"avgRE {_g4(m['dbr_avg'])}/{_g4(m['gate_avg'])}=✓" if av_ok
              else f"avgRE {_g4(m['dbr_avg'])}/{_g4(m['gate_avg'])}=✗破")
        sm = (f"小值err {m['err_small']}/{m['gate_small']:.0f}=✓" if sm_ok
              else f"小值err {m['err_small']}/{m['gate_small']:.0f}=✗破")
        body = f"大值{m['total_big']}个 | {mx} | {av} | {sm}"
    line = f"  -> {verd}  |Δ|max={m['max_abs_diff']:.4g}  {body} | cos={m['cos']:.8f}(仅参考)"
    if not passed and m.get("worst"):
        parts = [f"(g={gv:.4g} npu={nv:.4g} |Δ|={ad:.3g} relΔ={('inf' if rd==float('inf') else f'{rd:.3g}')})"
                 for gv, nv, ad, rd in m["worst"][:3]]
        line += "\n     最差元素: " + " ".join(parts)
    if note_extra:
        line += f"  {note_extra}"
    return line


# ============================================================
# run / main
# ============================================================
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NOT_IMPL = 2

OP_NAME = "npu_ai_infra_quant_matmul"


def _op_registered() -> bool:
    # hasattr 对 torch.ops 命名空间不可靠 (某些 torch 版本 __getattr__ 抛非 AttributeError, hasattr 接不住)。
    # 直接 try 访问, 捕获一切异常。
    try:
        _ = getattr(torch.ops.custom, OP_NAME)
        return True
    except Exception:
        return False


def _get_cann_recent_err():
    """best-effort: 直接调 CANN runtime 的 aclGetRecentErrMsg 拿最后的 EZ 错误 (比 fd2 捕获可靠)。"""
    try:
        import ctypes
        for libname in ("libascendcl.so", "libruntime.so"):
            try:
                lib = ctypes.CDLL(libname)
                lib.aclGetRecentErrMsg.restype = ctypes.c_char_p
                msg = lib.aclGetRecentErrMsg()
                if msg:
                    return msg.decode("utf-8", errors="replace")
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _call_npu_with_cann_err(d, c):
    """调 call_npu, best-effort 捕获 OS stderr(fd2)里的 CANN EZ 错误 (Python redirect_stderr 接不到 C++ stderr)。
    返回 (result, cann_err_str, exc)。ASCEND_LAUNCH_BLOCKING=1 时 EZ 码会同步打到 stderr, 捕得到。"""
    import os, tempfile
    cann_err = ""
    exc = None
    result = None
    tmp = None
    saved_fd = None
    try:
        sys.stdout.flush()
        tmp = tempfile.TemporaryFile(mode="w+b")
        saved_fd = os.dup(2)
        os.dup2(tmp.fileno(), 2)
    except Exception:
        tmp = None
    try:
        result = call_npu(d, c)
    except Exception as e:
        exc = e
    finally:
        if saved_fd is not None:
            try:
                sys.stdout.flush()
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
            except Exception:
                pass
        if tmp is not None:
            try:
                tmp.seek(0)
                cann_err = tmp.read().decode("utf-8", errors="replace")
                tmp.close()
            except Exception:
                pass
    return result, cann_err, exc


def _classify_err(exc_msg, cann_err):
    """据报错文本判哪一层拦截: binding检查 / aclnn checker(EZxx) / host tiling / kernel。"""
    import re
    full = exc_msg + "\n" + cann_err
    ez = re.search(r"EZ\d{3,4}", full)
    if "transpose are not same" in full or "only support 3 elements" in full:
        return "binding检查"
    if ez:
        return f"aclnn checker({ez.group()})"
    if "TilingParse" in full or "DoTiling" in full or "Failed to execute tiling" in full or "GetShapeAttrsInfo" in full:
        return "host tiling"
    if "Kernel Run failed" in full or "launch failed" in full or "Cannot find binary" in full:
        return "kernel/binary"
    if ez is None and ("Invalid_Argument" in full or "ERR01002" in full):
        return "aclnn checker(无EZ码)"
    return "未知层"


def run(c: Case, npu: bool, v: bool) -> dict:
    b_s = "x".join(str(b) for b in c.batch) if c.batch else "none"
    print(f"\n{'=' * 60}")
    print(f"  {c.name}  mode={c.mode}  M={c.M} N={c.N} K={c.K}  out={c.out_dtype} "
          f"scale={c.scale_dtype}  nz={c.weight_nz} tx1={c.trans_x1} tx2={c.trans_x2} "
          f"bias={c.bias_dtype if c.has_bias else 'no'}  batch={b_s}")
    d = gen_data(c)
    try:
        g = golden_ref(d, c)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dict(name=c.name, g_ok=False, npu_ok=False, passed=False, note=f"golden ERR: {e!r}")
    g_t = torch.as_tensor(g, dtype=torch.float32)
    print(f"  golden: shape={tuple(g.shape)} dtype={g.dtype} "
          f"range=[{g_t.min():.4g}, {g_t.max():.4g}]")
    r = dict(name=c.name, g_ok=True, npu_ok=False, passed=None, m=None, max_diff=None,
             cosine=None, note="")
    if not npu:
        return r  # golden-only: 不判 NPU
    if not _op_registered():
        r["note"] = "op未注册(wheel没装)"
        return r
    if c.expect_fail:
        try:
            call_npu(d, c)
            r.update(npu_ok=True, passed=False, note="expected NPU rejection but succeeded")
            print(f"  -> FAIL  (expect_fail 用例反而跑通)")
        except Exception as e:
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            r.update(npu_ok=True, passed=True, note=f"rejected as expected: {msg}")
            print(f"  -> PASS  (expect_fail: NPU 按预期拒绝 -> {msg})")
        return r
    try:
        out, cann_err, exc = _call_npu_with_cann_err(d, c)
    except Exception:
        out, cann_err, exc = None, "", None
    if exc is not None:
        e = exc
        recent = _get_cann_recent_err()
        print(f"  [完整报错] {type(e).__name__}: {e}")
        if recent.strip():
            print(f"  [aclGetRecentErrMsg] {recent.strip()[:500]}")
        if cann_err.strip():
            for ln in cann_err.splitlines():
                ls = ln.strip()
                if ls and any(k in ls for k in ("EZ0", "EZ1", "Invalid_Argument", "Reason", "ERR0", "[ERROR]",
                                                "Fail", "failed", "Tiling", "tiling", "binary", "launch")):
                    print(f"  [CANN] {ls[:200]}")
        layer = _classify_err(str(e), recent + "\n" + cann_err)
        r["note"] = f"[{layer}] {str(e)[:40]}"
        return r
    print(f"  npu:    shape={tuple(out.shape)} dtype={out.dtype}")
    if tuple(out.shape) != tuple(g.shape):
        r["note"] = f"shape mismatch npu={tuple(out.shape)} golden={tuple(g.shape)}"
        print(f"  -> FAIL  ({r['note']})")
        r["passed"] = False
        return r
    m = bench_metrics(out.float().reshape(-1), g_t.reshape(-1), c.out_dtype)
    ok, reasons = bench_verdict(m, c.out_dtype)
    r.update(npu_ok=True, passed=ok, m=m, max_diff=m["max_abs_diff"], cosine=m["cos"],
             note=("" if ok else "; ".join(reasons)))
    print(_fmt_verdict(m, c.out_dtype, ok))
    return r


def select_cases(a) -> List[Case]:
    cases = CASES
    if a.case:
        wanted = set(x.strip() for x in a.case.split(","))
        cases = [c for c in cases if c.name in wanted]
    if a.filter:
        cases = [c for c in cases if a.filter in c.name]
    if a.enable_all:
        return cases
    return [c for c in cases if c.enabled]


def _dw(s):
    # 显示宽度 (CJK/全角=2, ASCII=1) — 终端列对齐用
    return sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))


def _pad(s, w):
    d = _dw(s)
    return str(s) + " " * (w - d) if d < w else str(s)


def _print_summary(pairs):
    # 按 aclnn 路径(V5·ND/WeightNz·NZ) × 核(V3/V4) 分组; CJK 对齐; PASS/FAIL/skip 三分类
    _MD = {"pertensor": "pertensor(T-C)", "perchannel": "perchannel(T-T)",
           "pertoken": "pertoken(K-C)", "requant": "requant(int8)",
           "int32": "int32out", "int4sym": "int4x4对称", "a8w4": "A8W4-int(MSD)",
           "perblock": "G-B perblock"}

    def _group(c):
        return ("WeightNz·NZ" if c.weight_nz else "V5·ND") + " │ " + \
               ("V4核" if c.mode in ("a8w4", "perblock") else "V3核")

    def _variant(c):
        v = [_MD.get(c.mode, c.mode)]
        if c.group_sizes: v.append(f"gs={c.group_sizes}")
        if c.scale_dtype == "int64": v.append("stat")
        elif c.scale_dtype == "bfloat16": v.append("bf16sc")
        if c.has_bias: v.append(c.bias_dtype + "bias")
        if c.trans_x1 or c.trans_x2: v.append("t" + ("1" if c.trans_x1 else "") + ("2" if c.trans_x2 else ""))
        if c.batch: v.append(f"b{len(c.batch)}D")
        return " ".join(v)

    def _short_note(n):
        if not n: return ""
        if "未注册" in n: return "op未注册(wheel没装)"
        return n[:32]

    groups = {}
    for c, r in pairs:
        groups.setdefault(_group(c), []).append((c, r))
    name_w = max((_dw(r["name"]) for _, r in pairs), default=8)
    var_w = max((_dw(_variant(c)) for c, _ in pairs), default=10)

    n_p = n_f = n_s = 0
    print(f"\n{'=' * 76}\n  汇总 (按 aclnn 路径 │ 核 分组)\n{'=' * 76}")
    for gname in sorted(groups):
        items = groups[gname]
        gp = sum(1 for _, r in items if r.get("passed") is True)
        gf = sum(1 for _, r in items if r.get("passed") is False)
        gs = len(items) - gp - gf
        n_p += gp; n_f += gf; n_s += gs
        tag = f"{gp}/{len(items)}" + (" ✓" if gp == len(items) else "") + (f"  ({gs} skip)" if gs else "")
        print(f"\n── {gname} ─  {tag}")
        print(f"  {_pad('name', name_w)}  {_pad('变体', var_w)}  res    max_diff  note")
        for c, r in items:
            p_ = "PASS" if r.get("passed") is True else ("FAIL" if r.get("passed") is False else " -- ")
            d_ = f"{r['max_diff']:.4g}" if r.get("max_diff") is not None else "--"
            print(f"  {_pad(r['name'], name_w)}  {_pad(_variant(c), var_w)}  {p_}  {_pad(d_, 8)}  {_short_note(r.get('note',''))}")

    print(f"\n{'=' * 76}")
    fails = [(g, c, r) for g, items in groups.items() for c, r in items if r.get("passed") is False]
    if fails:
        print("  ★ FAIL 定位:")
        for g, c, r in fails:
            print(f"    [{g}] {r['name']}: {_short_note(r.get('note',''))}")
    tot = len(pairs)
    parts = [f"{n_p}/{tot} PASS"]
    if n_f: parts.append(f"{n_f} FAIL")
    if n_s: parts.append(f"{n_s} skip(wheel没装?)")
    print(f"  总计: {'  '.join(parts)}")
    print(f"{'=' * 76}")


# ============================================================
# self-test: 手算硬编码 tiny case, 离线验 oracle 数学正确 (每个 _compute_* 一个)
# ============================================================
def _self_test() -> bool:
    print(f"\n{'#' * 60}\n  SELF-TEST: oracle 手算硬编码 tiny case (离线验数学)\n{'#' * 60}")
    ok_all = True

    # 公共: x1=[[1,2],[3,4]] x2=[[5,6],[7,8]] → mm=[[19,22],[43,50]] (int32)
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    x2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    mm = np.matmul(x1.astype(np.int32), x2.astype(np.int32))
    exp_mm = np.array([[19, 22], [43, 50]])
    _ok = np.array_equal(mm, exp_mm)
    print(f"  [{'PASS' if _ok else 'FAIL'}] matmul 基线 mm=[[19,22],[43,50]]: got={mm.tolist()}")
    ok_all &= _ok

    # _compute_tc perchannel: scale=[0.5, 0.25] → out = mm * [0.5,0.25]
    sc = np.array([0.5, 0.25], dtype=np.float32)
    tc = _compute_tc(mm, sc, None, None, False, False, "float32")
    exp_tc = (exp_mm * np.array([0.5, 0.25])).astype(np.float32)
    _ok = np.allclose(tc, exp_tc, atol=1e-6)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _compute_tc perchannel: exp={exp_tc.tolist()} got={tc.tolist()}")
    ok_all &= _ok

    # _compute_tc pertensor + scale_generate (int8 + fp32 scale[1] + no bias → do_scale_gen=True)
    # scale_generate(0.5): 0.5 = 0x3F000000 → &0xFFFFE000 = 0x3F000000 (低13位本就0) → 0.5 不变
    sc1 = np.array([0.5], dtype=np.float32)
    _ok_sg = np.float32(_scale_generate(sc1)[0]) == np.float32(0.5)
    print(f"  [{'PASS' if _ok_sg else 'FAIL'}] _scale_generate(0.5)=0.5 (低13位0不变): "
          f"got={_scale_generate(sc1)[0]}")
    ok_all &= _ok_sg
    tc_pt = _compute_tc(mm, sc1, None, None, True, False, "float32")
    exp_pt = (exp_mm * 0.5).astype(np.float32)
    _ok = np.allclose(tc_pt, exp_pt, atol=1e-6)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _compute_tc pertensor+scale_generate: exp={exp_pt.tolist()} got={tc_pt.tolist()}")
    ok_all &= _ok

    # _compute_pertoken: pertoken=[1,2], scale=[0.5,0.25] → mm * [0.5,0.25] * [1,2]^T
    ps = np.array([1.0, 2.0], dtype=np.float32)
    pt = _compute_pertoken(mm, sc, ps, None, None, False, "float32")
    exp_pt = (exp_mm * np.array([0.5, 0.25]) * np.array([[1], [2]])).astype(np.float32)
    _ok = np.allclose(pt, exp_pt, atol=1e-6)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _compute_pertoken: exp={exp_pt.tolist()} got={pt.tolist()}")
    ok_all &= _ok

    # _compute_requant: scale=[0.5,0.5], offset=[1,1]
    # out = f32_2_s9(mm*0.5) = round(mm*0.5) clip[-256,255] = [[10,11],[22,25]]
    # + f32_2_s9(offset) = round(1) = 1 → [[11,12],[23,26]], clip[-128,127]
    rq = _compute_requant(mm, np.array([0.5, 0.5], dtype=np.float32),
                          np.array([1.0, 1.0], dtype=np.float32), None, None)
    exp_rq = np.array([[11, 12], [23, 26]], dtype=np.int8)
    _ok = np.array_equal(rq, exp_rq)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _compute_requant: exp={exp_rq.tolist()} got={rq.tolist()}")
    ok_all &= _ok

    # _compute_int32 + int32 bias [1,1] → mm + [1,1]
    i32 = _compute_int32(mm, np.array([1, 1], dtype=np.int32), "int32")
    exp_i32 = exp_mm + 1
    _ok = np.array_equal(i32, exp_i32)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _compute_int32+ibias: exp={exp_i32.tolist()} got={i32.tolist()}")
    ok_all &= _ok

    # _u64_to_deq_scale: 0.5 的 float32 位模式 0x3F000000 作 uint64 低32位 → 掩码后 view fp32 = 0.5
    u64 = np.array([0x3F000000], dtype=np.uint64)
    dq = _u64_to_deq_scale(u64)
    _ok = np.float32(dq[0]) == np.float32(0.5)
    print(f"  [{'PASS' if _ok else 'FAIL'}] _u64_to_deq_scale(0x3F000000)=0.5: got={dq[0]}")
    ok_all &= _ok

    # pack_int4_lastdim: [1,2,3,4,5,6,7,-1] (8 int4) → int32 = 0x_FEDCBA543... 验逐 nibble
    code = np.array([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.int8).reshape(1, 8)
    pk = pack_int4_lastdim(code)
    # nibble 序: lo=code[0] → bit0-3; -1 两补 = 0xF
    exp_pk = 1 | (2 << 4) | (3 << 8) | (4 << 12) | (5 << 16) | (6 << 20) | (7 << 24) | (0xF << 28)
    # 0xF<<28 = sign bit → int32 负; 转 uint32 比
    _ok = (pk[0].astype(np.int64) & 0xFFFFFFFF) == (exp_pk & 0xFFFFFFFF)
    print(f"  [{'PASS' if _ok else 'FAIL'}] pack_int4_lastdim [1,2,3,4,5,6,7,-1]: "
          f"exp=0x{exp_pk & 0xFFFFFFFF:08X} got=0x{int(pk[0].astype(np.int64) & 0xFFFFFFFF):08X}")
    ok_all &= _ok

    print(f"\n  SELF-TEST {'ALL PASS' if ok_all else 'HAS FAIL'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-only", action="store_true", help="仅 oracle (不调 NPU, 验能跑出有限值)")
    ap.add_argument("--self-test", action="store_true", help="手算硬编码 tiny case 验 oracle 数学")
    ap.add_argument("--list", action="store_true", help="列 case + enabled")
    ap.add_argument("--enable-all", action="store_true", help="无视 enabled, 跑全部")
    ap.add_argument("--filter", default=None, help="name 子串过滤")
    ap.add_argument("--case", default=None, help="精确 name (逗号分隔)")
    a = ap.parse_args()

    if a.self_test:
        return EXIT_PASS if _self_test() else EXIT_FAIL

    if a.list:
        for c in CASES:
            print(f"  [{'x' if c.enabled else ' '}] {c.mode:11s} {c.name}")
        print(f"  {sum(1 for c in CASES if c.enabled)}/{len(CASES)} enabled")
        return EXIT_PASS

    npu_mode = not a.golden_only
    cases = select_cases(a)
    if not cases:
        print("no case selected (检查 --case/--filter/enabled)")
        return EXIT_NOT_IMPL

    op_reg = _op_registered()
    print(f"QBMM-omni V3 golden  cases={len(cases)}  npu={'ON' if npu_mode else 'OFF(golden-only)'}"
          f"  omni_custom_ops_import={_HAS_WHEEL}  op_registered={op_reg}")
    if npu_mode and not op_reg:
        print(f"[WARN] torch.ops.custom.{OP_NAME} 未注册。"
              f"wheel装了仍 False → 多半 import omni_custom_ops 没触发注册 (golden 没 import / wheel 的 op def 没装)。"
              f"可在 python 里 `import omni_custom_ops; print(hasattr(__import__('torch').ops.custom, '{OP_NAME}'))` 自查。")

    res = [run(c, npu_mode, False) for c in cases]
    _print_summary(list(zip(cases, res)))

    if not npu_mode:
        # golden-only: 验 oracle 能跑出有限值 (非 nan/inf) 即认为 oracle 自洽
        g_ok = all(r["g_ok"] for r in res)
        print(f"\n[exit] --golden-only: {'oracle 自洽 (全 case 跑出有限值)' if g_ok else 'oracle 有 case 报错'}")
        return EXIT_NOT_IMPL  # golden-only 不下 PASS/FAIL 结论 (未对 NPU)

    compared = [r for r in res if r.get("npu_ok") and r.get("passed") is not None]
    n_pass = sum(1 for r in compared if r["passed"])
    n_fail = sum(1 for r in compared if not r["passed"])
    if not compared:
        print("\n[exit] NOT_IMPL: 无成功比较 (NPU 全报错/未注册?)")
        return EXIT_NOT_IMPL
    if n_fail:
        print(f"\n[exit] FAIL: {n_fail}/{len(compared)} 不匹配")
        return EXIT_FAIL
    print(f"\n[exit] PASS: {n_pass}/{len(compared)} 全过")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
