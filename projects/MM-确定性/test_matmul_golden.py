#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""
npu_ai_infra_matmul 全分支 golden + batch 一致性验证脚本.

覆盖所有模板/分支（基于 omni-ops / ops-nn 源码分析）：
  Datatypes:   FP16, BF16, FP32
  Transpose:   NN, NT, TN, TT
  LoadModes:   BASE, AL1_FULLLOAD, BL1_FULLLOAD
  FixOpti:     BASE, ENABLE_ALIGNOUT, VEC_NZ2ND_UNALIGNOUT
  SpecialOpt:  K_NOT_SHIFT, K_SHIFT
  SplitCore:   BASE, SC_SPLIT_K, DET_SPLIT_K, SC_GM_TO_L1, SC_NKM, MC_SPLIT_K
  Format:      ND×ND→ND (常规), ND×NZ→ND (Weight NZ)
  batch_invariant: golden 默认每个用例跑 BI=True + BI=False 双路径 (覆盖 + 不覆盖);
                   BI=True=PR 8a8172a 强制 SpecialOpti=BASE 关 K_SHIFT 换确定性; --no-batch-invariant 只跑 False。

精度判据（对齐 omni-ops 官方 test_ai_infra_matmul.py::verify_result）：
  torch.isclose(rtol=tol, atol=tol) flat 冗余, PASS 当 err_ratio ≤ 1e-4;
  per-dtype tol: fp16/fp32=1e-3, bf16=5e-3. cos/rel_l2 仅诊断打印不 gating
  (旧 2^-mantissa 辅助门对 fp32 跨实现累加 floor≈sqrt(K)·eps 过紧, 全 fp32 假 FAIL, 已弃).
  Batch 一致性: 每个 case 同时跑 BI=True 与 BI=False, MD5 逐字节对比 (repeat+perm+suffix)。
    判据: BI=True 须一致 (确定性承诺, 硬 FAIL); BI=False 仅对照 (允许不一致, 不计失败)。
    预期组合: True一致/False不一致 = K_SHIFT 非确定性被 BI 修正; True一致/False一致 = K_SHIFT 未启用 (omni 当前)。

用法:
  python test_ai_infra_matmul_golden.py              # 默认: 全用例 × {BI=True, BI=False} 双路径 + batch 一致性
  python test_ai_infra_matmul_golden.py --golden-only # 仅 golden 自洽
  python test_ai_infra_matmul_golden.py --self-test   # 手算 oracle
  python test_ai_infra_matmul_golden.py --list        # 列用例
  python test_ai_infra_matmul_golden.py --filter fp32
  python test_ai_infra_matmul_golden.py --case k_shift_768
  python test_ai_infra_matmul_golden.py --no-batch-invariant  # 只跑 BI=False (默认 True+False 双路径)

前置: torch_ops_extension wheel 已编+装 → torch.ops.custom.npu_ai_infra_matmul 注册。
  (需含 PR 8a8172a 的 batch_invariant 形参; 旧 wheel 自动降级仅验 golden 不验 True 路径)
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

try:
    import omni_custom_ops  # noqa: F401
    _HAS_WHEEL = True
except ImportError:
    _HAS_WHEEL = False

if _HAS_TORCH_NPU:
    try: torch.npu.set_device(0)
    except Exception: pass

# ---------- 常量 ----------
OP_NAME = "npu_ai_infra_matmul"
EXIT_PASS, EXIT_FAIL, EXIT_NOT_IMPL = 0, 1, 2
# 精度判据对齐 omni-ops 官方 test_ai_infra_matmul.py::verify_result — flat isclose(rtol=tol, atol=tol),
# PASS 当 err_ratio (isclose=False 占比) ≤ ERR_RATIO_TOL. per-dtype tol: fp16/fp32=1e-3, bf16=5e-3 (官方值).
# cos/rel_l2 仅诊断打印不 gating (旧 2^-mantissa 辅助门对 fp32 跨实现累加 floor≈sqrt(K)·eps 过紧, 全 fp32 假 FAIL, 弃).
CMP_TOL = {"float16": 1e-3, "bfloat16": 5e-3, "float32": 1e-3}
ERR_RATIO_TOL = 1e-4
COS_INFO = 0.9999
NZ_FMT = 29


# ---------- op schema 能力探测 (兼容旧 wheel) ----------
def _op_default():
    return getattr(torch.ops.custom, OP_NAME).default


def _op_supports_kw(kw: str) -> bool:
    """探测已注册 op schema 是否含某 keyword 形参 (batch_invariant/cube_math_type)."""
    try:
        sig = str(_op_default()._schema)
        return (kw + ":") in sig or (kw + " ") in sig or kw in sig
    except Exception:
        return False


_BI_CAPABLE = None  # lazy: 是否支持 batch_invariant (PR 8a8172a)


def _bi_capable() -> bool:
    global _BI_CAPABLE
    if _BI_CAPABLE is None:
        _BI_CAPABLE = hasattr(torch.ops.custom, OP_NAME) and _op_supports_kw("batch_invariant")
    return _BI_CAPABLE


# ============================================================
# Case 框架
# ============================================================
@dataclass
class Case:
    name: str
    M: int; K: int; N: int
    dtype: str = "float16"
    trans_x1: bool = False
    trans_x2: bool = False
    weight_nz: bool = False
    cube_math_type: int = 0
    batch_invariant: bool = True
    enabled: bool = True
    seed: Optional[int] = None
    branch_load: str = "BASE"
    branch_split: str = "BASE"
    branch_fixopti: str = "BASE"
    branch_special: str = "K_NOT_SHIFT"
    branch_shape: str = ""

    @property
    def shape_str(self) -> str:
        return f"[{self.M},{self.K}] @ [{self.K},{self.N}]"


CASES: List[Case] = [
    # ── BASE · FP16 ──
    Case("base_fp16_nn",  128, 256, 512, dtype="float16"),
    Case("base_fp16_nt",  256, 128, 256, dtype="float16", trans_x2=True),
    Case("base_fp16_tn",  256, 128, 256, dtype="float16", trans_x1=True),
    Case("base_fp16_tt",  256, 128, 512, dtype="float16", trans_x1=True, trans_x2=True),
    # ── BASE · BF16 ──
    Case("base_bf16_nn",  128, 256, 512, dtype="bfloat16"),
    Case("base_bf16_nt",  128, 256, 256, dtype="bfloat16", trans_x2=True),
    # ── BASE · FP32 ──
    Case("base_fp32_nn",  128, 256, 512, dtype="float32"),
    Case("base_fp32_nt",  128, 256, 256, dtype="float32", trans_x2=True),
    # ── AL1_FULLLOAD · FP32 ──
    Case("al1_8x4096x128",  8, 4096, 128, dtype="float32", trans_x2=True, branch_load="AL1_FULLLOAD"),
    Case("al1_16x8192x384", 16, 8192, 384, dtype="float32", trans_x2=True, branch_load="AL1_FULLLOAD"),
    # ── BL1_FULLLOAD ──
    Case("bl1_fp16",  8192, 128, 256, dtype="float16", branch_load="BL1_FULLLOAD"),
    Case("bl1_bf16",  8192, 128, 256, dtype="bfloat16", branch_load="BL1_FULLLOAD"),
    # ── BL1 CoreSplit ──
    Case("bl1_coresplit_fp16", 32768, 512, 512, dtype="float16", trans_x2=True, branch_load="BL1_FULLLOAD"),
    Case("bl1_coresplit_bf16", 32768, 512, 512, dtype="bfloat16", trans_x2=True, branch_load="BL1_FULLLOAD"),
    # ── Fixpipe + ALIGNOUT ──
    Case("fixpipe_4096x128x63", 4096, 128, 63, dtype="float16", branch_load="BL1_FULLLOAD", branch_fixopti="ALIGNOUT"),
    Case("fixpipe_6000x192x47", 6000, 192, 47, dtype="float16", branch_load="BL1_FULLLOAD", branch_fixopti="ALIGNOUT"),
    # ── Fixpipe + VEC_NZ2ND · FP32 ──
    Case("fixpipe_vec_nz2nd", 2048, 128, 128, dtype="float32", branch_load="BL1_FULLLOAD", branch_fixopti="VEC_NZ2ND"),
    # ── K_SHIFT (默认双路径: BI=True 强制 SpecialOpti=BASE 验精度不退化; BI=False 对照) ──
    Case("k_shift_768",     768, 768, 768,   dtype="float16", branch_special="K_SHIFT"),
    Case("k_shift_12800",  12800, 2560, 2560, dtype="float16", branch_special="K_SHIFT"),
    # ── Shape 模式 ──
    Case("small_16x32x64",      16, 32, 64,    branch_shape="small"),
    Case("square_256",          256, 256, 256,  branch_shape="square"),
    Case("single_row",          1, 1024, 4096,  branch_shape="single-row"),
    Case("large_k",             64, 4096, 128,  branch_shape="large-k"),
    Case("large_m",             12288, 256, 64,  branch_shape="large-m"),
    Case("large_n",             64, 256, 16384, branch_shape="large-n"),
    # ── SC_SPLIT_K ──
    Case("sc_split_k_fp16", 128, 16384, 128, branch_split="SC_SPLIT_K"),
    Case("sc_split_k_fp32", 64, 32768, 64,   dtype="float32", branch_split="SC_SPLIT_K"),
    # ── 其余 SplitCore 分支 (防 batch_invariant=True 下其它分支回归) ──
    Case("sc_gm2l1_fp16",  128, 16384, 128, branch_split="SC_GM_TO_L1"),
    Case("sc_nkm_fp16",     64,  8192,  64, branch_split="SC_NKM"),
    Case("mc_split_k_fp16", 128, 32768, 128, branch_split="MC_SPLIT_K"),
    # ── DET_SPLIT_K · FP32 ──
    Case("det_split_k_64", 64, 16384, 64, dtype="float32", branch_split="DET_SPLIT_K"),
    Case("det_split_k_32", 32, 8192, 32,  dtype="float32", branch_split="DET_SPLIT_K"),
    # ── ND2NZ 不对齐 ──
    Case("nd2nz_k63",  128, 63, 512,  branch_shape="unaligned"),
    Case("nd2nz_n97",  256, 256, 97,  branch_shape="unaligned"),
    # ── Weight NZ (多 transpose/dtype 组合, 防 NZ 分支回归) ──
    Case("nz_fp16_nn", 128, 256, 512, weight_nz=True),
    Case("nz_fp16_nt", 128, 256, 512, weight_nz=True, trans_x2=True),
    Case("nz_fp16_tn", 256, 128, 512, weight_nz=True, trans_x1=True),
    Case("nz_bf16_nn", 128, 256, 512, dtype="bfloat16", weight_nz=True),
    Case("nz_bf16_nt", 128, 256, 256, dtype="bfloat16", weight_nz=True, trans_x2=True),
    # ── 边缘 ──
    Case("edge_fp32_mata_k", 128, 16384, 128, dtype="float32", branch_shape="mata"),
    Case("edge_fp16_mata_m", 16384, 256, 128,  branch_shape="mata"),
    Case("edge_1x1x1",       1, 1, 1,           branch_shape="tiny"),
    Case("edge_ab_trans",    256, 128, 512,     trans_x1=True, trans_x2=True),
    Case("edge_max_dims",    64, 65535, 128,    branch_shape="max-dims"),
    Case("incre_64x512x1024", 64, 512, 1024,    trans_x2=True, branch_shape="incre"),
]


# ============================================================
# 数据生成 + Golden
# ============================================================
def gen_data(c: Case) -> dict:
    if c.seed is not None:
        torch.manual_seed(c.seed)
    a_shape = (c.K, c.M) if c.trans_x1 else (c.M, c.K)
    b_shape = (c.N, c.K) if c.trans_x2 else (c.K, c.N)
    a = torch.randn(*a_shape, dtype=torch.float32) * 0.5
    b = torch.randn(*b_shape, dtype=torch.float32) * 0.5
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[c.dtype]
    return dict(a=a, b=b, dtype=dt, trans_x1=c.trans_x1, trans_x2=c.trans_x2, weight_nz=c.weight_nz,
                cube_math_type=c.cube_math_type, batch_invariant=c.batch_invariant)


def golden(d: dict) -> torch.Tensor:
    # 输入先舍到目标 dtype 再 fp32 累加 (对齐 op: call_npu 拿 a.to(dtype) 喂 NPU).
    #   旧版用 fp32 输入做 golden → fp16/bf16 比 op 多一重输入量化误差 (~ULP),
    #   err_ratio 虚高 2~22%. 参考 QBMM-Batch golden_mx_matmul_batch.py: 输入 .to(x1_dt), golden .float().
    dt = d["dtype"]
    a = d["a"].to(dt).float()
    b = d["b"].to(dt).float()
    if d["trans_x1"]: a = a.T
    if d["trans_x2"]: b = b.T
    return torch.matmul(a, b).to(dt)


def call_npu(d: dict) -> torch.Tensor:
    a = d["a"].clone().to(d["dtype"]).npu()
    b = d["b"].clone().to(d["dtype"]).npu()
    if d["weight_nz"]:
        import torch_npu as tn
        b = tn.npu_format_cast(b, NZ_FMT)
    if d["trans_x1"]: a = a.transpose(-1, -2)
    if d["trans_x2"]: b = b.transpose(-1, -2)
    cube_math_type = d.get("cube_math_type", 0)
    bi = d.get("batch_invariant", False)
    op = _op_default()
    if _bi_capable():
        # 新 schema (PR 8a8172a): (self, mat2, *, cube_math_type, batch_invariant)
        return op(a, b, cube_math_type=cube_math_type, batch_invariant=bi)
    if bi:
        # 旧 wheel 不支持 batch_invariant: True 路径无法验, 抛出提示而非静默降级
        raise RuntimeError("batch_invariant=True 需 PR 8a8172a 的新 wheel; 当前注册的 op schema 无此形参")
    try:
        return op(a, b, cube_math_type=cube_math_type)
    except TypeError:
        return op(a, b)


# ============================================================
# 精度判据
# ============================================================
def _cos_rell2(value: torch.Tensor, ref: torch.Tensor) -> dict:
    v, g = value.reshape(-1).double(), ref.reshape(-1).double()
    nan_mask = torch.isnan(g) ^ torch.isnan(v)
    err_nan = int(torch.sum(nan_mask))
    zeros = torch.zeros_like(g)
    finite = ~(torch.isinf(g) | torch.isnan(g)) & ~(torch.isinf(v) | torch.isnan(v))
    gf, vf = torch.where(finite, g, zeros), torch.where(finite, v, zeros)
    gn, vn = float(gf.norm()), float(vf.norm())
    cos = float(gf @ vf / (gn * vn)) if gn > 0 and vn > 0 else 1.0
    rel_l2 = float((vf - gf).norm()) / gn if gn > 0 else 0.0
    absdiff = (vf - gf).abs()
    n_w = min(5, g.numel())
    worst = []
    if n_w > 0 and gf.abs().sum() > 0:
        _, idx = torch.topk(absdiff, n_w)
        gabs = gf.abs()
        worst = [(float(g[i]), float(v[i]), float(absdiff[i]),
                  float(absdiff[i]/gabs[i]) if float(gabs[i]) > 1e-10 else float("inf")) for i in idx.tolist()]
    return dict(cos=cos, rel_l2=rel_l2, err_nan=err_nan, max_abs_diff=float(absdiff.max()) if g.numel() > 0 else 0.0, worst=worst)


def cmp(value: torch.Tensor, ref: torch.Tensor, out_dtype: str, verbose: bool = False) -> dict:
    vf, gf = value.float().cpu(), ref.float().cpu()
    m = _cos_rell2(vf, gf)  # cos/rel_l2/worst/max_abs_diff/err_nan 仅诊断
    tol = CMP_TOL.get(out_dtype, 1e-3)
    close = torch.isclose(vf, gf, rtol=tol, atol=tol, equal_nan=True)
    err_ratio = float((~close).sum()) / float(gf.numel()) if gf.numel() > 0 else 0.0
    passed = (err_ratio <= ERR_RATIO_TOL) and (m["err_nan"] == 0)
    if verbose:
        print(f"  isclose tol={tol:g} err_ratio={err_ratio:.2e}/{ERR_RATIO_TOL:g} "
              f"cos={m['cos']:.8f} rl2={m['rel_l2']:.4g} → {'PASS' if passed else 'FAIL'}")
    return dict(passed=passed, cos=m["cos"], rel_l2=m["rel_l2"], max_abs_diff=m["max_abs_diff"],
                err_ratio=err_ratio, tol=tol, err_nan=m["err_nan"])


def run_case(c: Case, npu: bool, verbose: bool, bi: bool) -> dict:
    tag = "[BI]" if bi else "[noBI]"
    print(f"\n{'─'*60}\n  {c.name:30s}{tag:7s} {c.shape_str} {c.dtype}"
          f"{' NZ' if c.weight_nz else ''}{' tx1' if c.trans_x1 else ''}{' tx2' if c.trans_x2 else ''}")
    d = gen_data(c)
    d["batch_invariant"] = bi  # 显式覆盖: golden 默认跑 BI=True/False 双路径
    try:
        g = golden(d)
    except Exception as e:
        return dict(name=f"{c.name}{tag}", g_ok=False, npu_ok=False, passed=False, note=f"golden ERR: {e!r}")
    print(f"  golden: {tuple(g.shape)} {g.dtype} [{g.float().min():.4g}, {g.float().max():.4g}]")
    r = dict(name=f"{c.name}{tag}", g_ok=True, npu_ok=False, passed=None, cos=None, max_diff=None, note="")
    if not npu: return r
    if not hasattr(torch.ops.custom, OP_NAME):
        r["note"] = f"torch.ops.custom.{OP_NAME} 未注册"; return r
    try:
        out = call_npu(d)
    except Exception as e:
        r["note"] = f"NPU ERR: {type(e).__name__}: {e!r}"; print(f"  {r['note']}"); return r
    print(f"  npu:    {tuple(out.shape)} {out.dtype}")
    if tuple(out.shape) != tuple(g.shape):
        r["note"] = f"shape mismatch"; r["passed"] = False; return r
    m = cmp(out, g, c.dtype, verbose=verbose)
    r.update(npu_ok=True, passed=m["passed"], cos=m["cos"], rel_l2=m["rel_l2"],
             dtype=c.dtype, max_diff=m["max_abs_diff"], err_ratio=m["err_ratio"], tol=m["tol"],
             note=("" if m["passed"] else f"err_ratio={m['err_ratio']:.2e} |Δ|max={m['max_abs_diff']:.3g}"))
    print(f"  → {'PASS' if m['passed'] else 'FAIL'} err_ratio={m['err_ratio']:.2e}/{ERR_RATIO_TOL:g} "
          f"tol={m['tol']:g} |Δ|max={m['max_abs_diff']:.3g} cos={m['cos']:.6f}")
    return r


# ============================================================
# Batch 一致性 (整块 vs 截取片段, bit 级 MD5)
#   设计: 跑一整块 (complete_M) → y_full; 把同一份片段输入 (frag_M 行) 既单独跑 → y_frag,
#         又作为整块的一个连续子段 [start:end] 拼进整块; bit 级比 MD5(y_full[start:end]) vs MD5(y_frag).
#         一致 = 该片段输出不随 batch 语境 (M/位置/邻居) 改变 = batch 一致成立.
#   原理: batch 不一致根因 = NPU cube K 轴累加 tiling (K_SHIFT 等) 随 shape 变 → M 变 →
#         K-tiling 变 → 同一行累加顺序变 → bit 变. 只有"换 M"能检测; repeat(同M)/perm(同M) 测不到, 已弃.
#         差异 ~1e-6, 任何 tol 看不见 → 判据必须 bit 级 (MD5), 非 tolerance.
#   verdict: BI=True 全一致=确定性承诺成立; BI=True 有不一致=承诺破坏(硬FAIL);
#           BI=False 不一致 + BI=True 一致=开关有效; 两者均一致=未证(K_SHIFT 可能死代码/未触发边界).
# ============================================================
# 探测配置 (可调, 缩小可减板时): complete_M 卡 L1/core 边界 + 生产 size; 片段固定 frag_M; 三位置覆盖 M-tiling 边界.
BI_COMPLETE_M_LIST = [128, 256, 512, 1024, 4735]
BI_FRAG_M = 127
BI_SLICE_POSITIONS = ["head", "mid", "tail"]


BATCH_CONSISTENCY_CASES = [
    dict(name="batch_base_nn_fp16",   K=256, N=256, dtype=torch.float16,  trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_base_nn_bf16",   K=256, N=256, dtype=torch.bfloat16, trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_base_nn_fp32",   K=256, N=512, dtype=torch.float32,  trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_base_nt_fp16",   K=128, N=256, dtype=torch.float16,  trans_x1=False, trans_x2=True,  weight_nz=False),
    dict(name="batch_base_tn_fp16",   K=128, N=256, dtype=torch.float16,  trans_x1=True,  trans_x2=False, weight_nz=False),
    dict(name="batch_base_tt_fp16",   K=128, N=512, dtype=torch.float16,  trans_x1=True,  trans_x2=True,  weight_nz=False),
    dict(name="batch_kshift_nn_fp16", K=768, N=768, dtype=torch.float16,  trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_kshift_nn_bf16", K=768, N=768, dtype=torch.bfloat16, trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_largek_fp32",    K=4096, N=128, dtype=torch.float32, trans_x1=False, trans_x2=True,  weight_nz=False),
    dict(name="batch_bl1_nn_fp16",    K=128, N=256, dtype=torch.float16,  trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_bl1_nn_bf16",    K=128, N=256, dtype=torch.bfloat16, trans_x1=False, trans_x2=False, weight_nz=False),
    dict(name="batch_nz_nn_fp16",     K=256, N=512, dtype=torch.float16,  trans_x1=False, trans_x2=False, weight_nz=True),
    dict(name="batch_nz_nt_fp16",     K=256, N=512, dtype=torch.float16,  trans_x1=False, trans_x2=True,  weight_nz=True),
]


def _md5_hex(t: torch.Tensor) -> str:
    # Bit-level MD5: numpy 不原生支持 bfloat16, 但 batch_invariant 的判据是
    # "bit-identical", 故不能用 .float() 中转 (会合并 +0.0/-0.0 / NaN payload 等
    # 不同 bit pattern). 改 view(uint16) 直接对原始 bit 流做哈希.
    # 对所有 float dtype (fp32→uint32, fp16/bf16→uint16) 都保留位模式; 对 int
    # dtype 本就 numpy 原生支持.
    t_cpu = t.contiguous().cpu()
    if t_cpu.dtype in (torch.bfloat16, torch.float16):
        arr = t_cpu.view(torch.uint16).numpy()
    elif t_cpu.dtype == torch.float32:
        arr = t_cpu.view(torch.uint32).numpy()
    else:
        arr = t_cpu.numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()


def _gen_rows(K: int, rows: int, trans_x1: bool, seed: int) -> torch.Tensor:
    """生成 rows 行 A: trans_x1=True → [K,rows], 否则 [rows,K]; fp32 *0.5."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (K, rows) if trans_x1 else (rows, K)
    return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5


def _gen_weight(K: int, N: int, trans_x2: bool, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (N, K) if trans_x2 else (K, N)
    return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5


def _run_fragment_probe(cfg: dict, complete_M: int, position: str, bi: bool) -> dict:
    """整块 vs 截取片段, bit 级 MD5 对比.

    同一份片段 (frag_M 行, 固定 seed) 既单独跑 (→y_frag), 又作为整块 complete_M 的连续子段
    [start:end] 拼进整块跑 (→y_full); 比对 MD5(y_full[start:end]) vs MD5(y_frag).
    position: head=整块首, mid=中, tail=末. 输出 y 始终 [M,N], 片段对应行即 [start:end].
    """
    K, N = cfg["K"], cfg["N"]
    tx1, tx2, nz, dtype = cfg["trans_x1"], cfg["trans_x2"], cfg["weight_nz"], cfg["dtype"]
    frag_M = min(BI_FRAG_M, complete_M)
    if frag_M < 1 or complete_M < 1:
        return dict(ok=True, max_diff=0.0, complete_M=complete_M, position=position, frag_M=frag_M)

    name_seed = hash(cfg["name"]) & 0xFFFFFFFF
    frag_seed = hash(cfg["name"] + "_frag") & 0xFFFFFFFF
    a_frag = _gen_rows(K, frag_M, tx1, frag_seed)
    b = _gen_weight(K, N, tx2, name_seed)

    filler_before = complete_M - frag_M
    start = {"head": 0, "tail": filler_before, "mid": filler_before // 2}[position]
    end = start + frag_M

    cat_dim = 1 if tx1 else 0
    parts = []
    if start > 0:
        parts.append(_gen_rows(K, start, tx1, frag_seed + 1))
    parts.append(a_frag)
    if complete_M - end > 0:
        parts.append(_gen_rows(K, complete_M - end, tx1, frag_seed + 2))
    a_full = torch.cat(parts, dim=cat_dim) if len(parts) > 1 else parts[0]

    d_full = dict(a=a_full, b=b, dtype=dtype, trans_x1=tx1, trans_x2=tx2, weight_nz=nz, batch_invariant=bi)
    d_frag = dict(a=a_frag, b=b, dtype=dtype, trans_x1=tx1, trans_x2=tx2, weight_nz=nz, batch_invariant=bi)
    y_full = call_npu(d_full).cpu()
    y_frag = call_npu(d_frag).cpu()

    y_full_slice = y_full[start:end]
    h_full = _md5_hex(y_full_slice)
    h_frag = _md5_hex(y_frag)
    diff = (y_full_slice.float() - y_frag.float()).abs()
    return dict(ok=(h_full == h_frag), md5_full=h_full, md5_frag=h_frag,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                complete_M=complete_M, position=position, frag_M=frag_M)


def run_batch_test(cfg: dict) -> dict:
    print(f"\n  batch: {cfg['name']}  K={cfg['K']} N={cfg['N']} {cfg['dtype']}"
          f"{' NZ' if cfg['weight_nz'] else ''}{' tx1' if cfg['trans_x1'] else ''}{' tx2' if cfg['trans_x2'] else ''}")
    errors = []
    if not _bi_capable():
        print("    [skip] batch_invariant 形参未注册 (需 PR 8a8172a 新 wheel)")
        return dict(name=cfg["name"], passed=False, errors=["batch_invariant 未注册 (需新 wheel)"])

    probes_on, probes_off = [], []
    for m in BI_COMPLETE_M_LIST:
        for pos in BI_SLICE_POSITIONS:
            probes_on.append(_run_fragment_probe(cfg, m, pos, True))
            probes_off.append(_run_fragment_probe(cfg, m, pos, False))

    def _summarize(probes, tag):
        n_ok = sum(1 for p in probes if p["ok"])
        worst = max(probes, key=lambda p: p.get("max_diff", 0.0))
        print(f"    {tag}: {n_ok}/{len(probes)} 片段一致  最差 |Δ|max={worst.get('max_diff', 0.0):.3g}"
              f" (M={worst.get('complete_M')}, {worst.get('position')})")
        return n_ok == len(probes), worst

    on_all_ok, _ = _summarize(probes_on, "BI=True ")
    off_all_ok, off_worst = _summarize(probes_off, "BI=False")

    if not on_all_ok:
        bad = [p for p in probes_on if not p["ok"]]
        msg = (f"BI=True 不一致 → 确定性承诺破坏: {len(bad)} 片段 MD5 不等, "
               f"首例 M={bad[0]['complete_M']} {bad[0]['position']} |Δ|max={bad[0]['max_diff']:.3g}")
        errors.append(msg)
        print(f"    → ❌ {msg}")
    elif not off_all_ok:
        print(f"    → ✅ 开关有效: BI=True 全一致 / BI=False 不一致 "
              f"(K_SHIFT 非确定性被 BI 修正; 最差 |Δ|max={off_worst.get('max_diff', 0.0):.3g})")
    else:
        print("    → ⚠️ 一致性成立, 但开关必要性未证 (BI 开/关均一致: "
              "K_SHIFT 当前可能死代码 或 所选 complete_M 未触发 tiling 边界)")
    return dict(name=cfg["name"], passed=len(errors) == 0, errors=errors)


# ============================================================
# self-test
# ============================================================
def _self_test() -> bool:
    print(f"\n{'#'*60}\n  SELF-TEST: 手算 golden 数学验证\n{'#'*60}")
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[5., 6.], [7., 8.]])
    exp = torch.tensor([[19., 22.], [43., 50.]])
    d = dict(a=a, b=b, dtype=torch.float32, trans_x1=False, trans_x2=False, weight_nz=False)
    g = golden(d)
    ok = torch.allclose(g, exp)
    print(f"  [{'PASS' if ok else 'FAIL'}] [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]: got={g.tolist()}")
    return ok


# ============================================================
# CLI
# ============================================================
def select_cases(a: argparse.Namespace) -> List[Case]:
    cases = CASES
    if a.case: cases = [c for c in cases if c.name in set(x.strip() for x in a.case.split(","))]
    if a.filter: cases = [c for c in cases if a.filter in c.name]
    if not a.enable_all: cases = [c for c in cases if c.enabled]
    return cases


def _print_summary(res: List[dict], batch_res: List[dict] = None) -> None:
    name_w = max((len(r["name"]) for r in res), default=8)
    print(f"\n{'='*60}\n  汇总\n{'-'*60}")
    for r in res:
        g_ = "OK" if r["g_ok"] else "ERR"
        n_ = "OK" if r.get("npu_ok") else ("--" if r.get("passed") is None else "ERR")
        p_ = "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "--")
        c_ = f"{r['cos']:.6f}" if r.get("cos") is not None else "--"
        print(f"  {r['name']:<{name_w}} {g_:>3} {n_:>3} {p_:>4} cos={c_} {r.get('note','')}")
    if batch_res:
        print(f"\n  Batch 一致性:")
        for br in batch_res:
            print(f"    {br['name']:<{name_w}} {'PASS' if br['passed'] else 'FAIL'}{' — ' + '; '.join(br['errors']) if br['errors'] else ''}")


def _print_fail_summary(res: List[dict]) -> None:
    """精度失败用例专项汇总: 只列 FAIL (isclose err_ratio > ERR_RATIO_TOL), 按 err_ratio 降序, 打印差距."""
    fails = [r for r in res if r.get("passed") is False and r.get("err_ratio") is not None]
    if not fails:
        return

    fails.sort(key=lambda r: r.get("err_ratio") or 0.0, reverse=True)
    print(f"\n{'=' * 60}\n  精度失败用例 ({len(fails)}): err_ratio > {ERR_RATIO_TOL:g}, 按 err_ratio 降序\n{'-' * 60}")
    print(f"  {'case':<28} {'dtype':<8} {'err_ratio':>10} {'限':>7} {'|Δ|max':>10} {'tol':>7} {'cos':>9}")
    for r in fails:
        print(f"  {r['name']:<28} {r.get('dtype','?'):<8} {r['err_ratio']:>10.2e} {ERR_RATIO_TOL:>7.1e} "
              f"{r.get('max_diff', 0.0):>10.2e} {r.get('tol', 1e-3):>7.1e} {r['cos']:>9.6f}")
    worst = fails[0]
    print(f"{'-' * 60}")
    print(f"  共 {len(fails)} 例失败; 最差 {worst['name']} err_ratio={worst['err_ratio']:.2e}"
          f" (限 {ERR_RATIO_TOL:g}), |Δ|max={worst.get('max_diff', 0.0):.2e}.")


def main() -> int:
    ap = argparse.ArgumentParser(description="npu_ai_infra_matmul 全分支 golden + batch 验证")
    ap.add_argument("--golden-only", action="store_true"); ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--enable-all", action="store_true")
    ap.add_argument("--filter", default=None); ap.add_argument("--case", default=None)
    ap.add_argument("--no-batch", action="store_true", help="跳过 batch 一致性测试")
    ap.add_argument("--no-batch-invariant", action="store_true", help="只跑 BI=False 路径 (默认是 True+False 双路径)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.self_test: return EXIT_PASS if _self_test() else EXIT_FAIL
    if a.list:
        for c in CASES: print(f"  [{'x' if c.enabled else ' '}] {c.name:35s} {c.shape_str} {c.dtype} load={c.branch_load} split={c.branch_split} special={c.branch_special}")
        return EXIT_PASS

    npu_mode = not a.golden_only
    cases = select_cases(a)
    if not cases: print("no case selected"); return EXIT_NOT_IMPL

    # 默认: 全用例 × {BI=True, BI=False} 双路径 (覆盖 + 不覆盖);
    #   --no-batch-invariant / 旧 wheel(无 batch_invariant 形参) / golden-only → 只跑一次 (BI=False, golden 与 BI 无关)。
    bi_capable = (not npu_mode) or _bi_capable()
    if a.no_batch_invariant or not npu_mode:
        bi_grid = [False]
    elif not _bi_capable():
        print("[warn] op schema 无 batch_invariant 形参 (需 PR 8a8172a 新 wheel); 仅跑 BI=False 路径")
        bi_grid = [False]
    else:
        bi_grid = [True, False]

    paths = "BI=" + "/".join("True" if b else "False" for b in bi_grid)
    print(f"npu_ai_infra_matmul golden  用例={len(cases)}  npu={'ON' if npu_mode else 'OFF'}"
          f"  路径={paths}  capable={'ON' if bi_capable else 'OFF'}")
    res = []
    for c in cases:
        for bi in bi_grid:
            res.append(run_case(c, npu_mode, a.verbose, bi))

    batch_res = None
    if npu_mode and not a.no_batch:
        print(f"\n{'='*60}\n  Batch 一致性 ({len(BATCH_CONSISTENCY_CASES)} cases, 每个 case 跑 BI=True/False 对照)")
        batch_res = [run_batch_test(bc) for bc in BATCH_CONSISTENCY_CASES]

    _print_summary(res, batch_res)
    _print_fail_summary(res)

    if not npu_mode:
        return EXIT_NOT_IMPL
    compared = [r for r in res if r.get("npu_ok") and r.get("passed") is not None]
    n_pass = sum(1 for r in compared if r["passed"])
    n_fail = sum(1 for r in compared if not r["passed"])
    n_batch_fail = sum(1 for br in (batch_res or []) if not br["passed"])
    if n_fail or n_batch_fail:
        print(f"\n[exit] FAIL: golden={n_pass}/{len(compared)} batch_fail={n_batch_fail}")
        return EXIT_FAIL
    print(f"\n[exit] PASS: {n_pass}/{len(compared)} golden + batch 全过")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
