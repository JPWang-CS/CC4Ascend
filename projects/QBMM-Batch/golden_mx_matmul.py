#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MX量化Matmul Golden验证脚本 (无batch版本)

验证 quantBatchMatmulV3 / aclnnQuantMatmulV5 MX全量化精度。
Scale shape 遵循 需求.md 和 aclnnQuantMatmulWeightNz.md:
  x1Scale (pertoken_scale): (M, ceil(K/64), 2)   — E8M0
  x2Scale (scale):          (ceil(K/64), N, 2)    — E8M0

Golden 模型 (对齐 ops-nn/tests/assets/golden.py L64-108):
  1. x1_f = x1.to(float32); x2_f = x2.to(float32)
  2. pertoken_2d = pertoken_scale.reshape(M, ceilK*2)
  3. scale_2d    = scale.transpose(-1,-2).reshape(ceilK*2, N)
  4. if ceil(K/32) % 2 != 0: 截断末尾
  5. pertoken_bc = repeat(pertoken_2d, 32, dim=-1)  → (M, K_padded)
  6. scale_bc    = repeat(scale_2d, 32, dim=-2)     → (K_padded, N)
  7. pad x1/x2; x1 *= pertoken_bc; x2 *= scale_bc; out = x1 @ x2

用法:
  python golden_mx_matmul.py                # 默认: golden + NPU
  python golden_mx_matmul.py -v             # 详细
  python golden_mx_matmul.py --golden-only  # 仅golden
"""

import argparse
import sys
import torch
import torch_npu
from dataclasses import dataclass
from typing import Optional, List

# ============================================================
# 常量 — 与 C++ 代码对齐 (MXFP_DIVISOR_SIZE=64, MXFP_MULTI_BASE=2)
# ============================================================
MXFP_DIVISOR = 64   # K方向每64个元素一组
MXFP_BASE    = 2    # 每组2个scale值
MXFP_INNER   = 32   # 每个scale值覆盖32个K元素 (64/2)


def ceil_div(a, b):
    return (a + b - 1) // b


def e8m0_to_float32(e8m0: torch.Tensor) -> torch.Tensor:
    """uint8 E8M0 → float32:  2^(e - 127),  e=0 → 0.0"""
    e = e8m0.float()
    val = torch.pow(2.0, e - 127.0)
    return torch.where(e8m0 == 0, torch.zeros_like(val), val)


# ============================================================
# 测试用例
# ============================================================

@dataclass
class Case:
    name: str
    M: int; N: int; K: int
    x_dtype: str       # 'fp8_e4m3fn' | 'fp4_e2m1'
    out_dtype: str      # 'float16' | 'bfloat16' | 'float32'
    trans_x1: bool = False
    trans_x2: bool = False
    has_bias: bool = False
    seed: Optional[int] = None


CASES: List[Case] = [
    # ---- FP8 基础 shape ----
    Case("fp8_basic",       M=128, N=256, K=512,   x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_odd_K",       M=185, N=1880, K=480,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_even_K",      M=112, N=184, K=944,   x_dtype="fp8_e4m3fn", out_dtype="bfloat16"),
    # ---- FP8 小 shape ----
    Case("fp8_small",       M=32,  N=32,  K=64,    x_dtype="fp8_e4m3fn", out_dtype="float32"),
    Case("fp8_K128",        M=64,  N=128, K=128,   x_dtype="fp8_e4m3fn", out_dtype="float16"),
    # ---- FP8 非对齐 K ----
    Case("fp8_K200",        M=64,  N=128, K=200,   x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_K120",        M=32,  N=64,  K=120,   x_dtype="fp8_e4m3fn", out_dtype="float16"),
    # ---- FP8 bias ----
    Case("fp8_bias",        M=128, N=256, K=512,   x_dtype="fp8_e4m3fn", out_dtype="float16", has_bias=True),
    # ---- FP8 输出类型 ----
    Case("fp8_out_bf16",    M=128, N=256, K=512,   x_dtype="fp8_e4m3fn", out_dtype="bfloat16"),
    Case("fp8_out_fp32",    M=64,  N=128, K=256,   x_dtype="fp8_e4m3fn", out_dtype="float32"),
    # ---- FP8 大 shape ----
    Case("fp8_large",       M=512, N=1024, K=1024,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    # ---- FP8 泛化: 随机 M/N/K ----
    Case("fp8_gen_96x64x256",   M=96,  N=64,  K=256,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_256x512x384", M=256, N=512, K=384,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_64x64x128",   M=64,  N=64,  K=128,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_384x256x768", M=384, N=256, K=768,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_192x128x320", M=192, N=128, K=320,  x_dtype="fp8_e4m3fn", out_dtype="bfloat16"),
    Case("fp8_gen_256x128x512_bf16_bias",
                                 M=256, N=128, K=512,  x_dtype="fp8_e4m3fn", out_dtype="bfloat16", has_bias=True),
    Case("fp8_gen_128x256x640_fp32",
                                 M=128, N=256, K=640,  x_dtype="fp8_e4m3fn", out_dtype="float32"),
    Case("fp8_gen_1x32x64",     M=1,   N=32,  K=64,   x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_32x1x128",    M=32,  N=1,   K=128,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    Case("fp8_gen_64x1024x512", M=64,  N=1024, K=512,  x_dtype="fp8_e4m3fn", out_dtype="float16"),
    # ---- FP8 transpose ----
    # NOTE: torch层 MX transpose check 有bug (dim_x2_scale=0), 需先修 2.2节再放开
    # Case("fp8_trans_x2",  M=128, N=256, K=512,  x_dtype="fp8_e4m3fn", out_dtype="float16", trans_x2=True),
    # ---- FP4 (x2固定transposed, 需单独适配) ----
    # Case("fp4_notrans",   M=112, N=184, K=944,  x_dtype="fp4_e2m1", out_dtype="bfloat16"),
]


# ============================================================
# 数据生成 — 3D scale
# ============================================================

def gen_data(c: Case) -> dict:
    if c.seed is not None:
        torch.manual_seed(c.seed)
    M, N, K = c.M, c.N, c.K
    ceilK = ceil_div(K, MXFP_DIVISOR)

    # x1, x2 — NPU 存储格式:
    #   trans_x1=False → x1(M,K);  trans_x1=True → x1(K,M)
    #   trans_x2=False → x2(K,N);  trans_x2=True → x2(N,K)
    x1_shape = (K, M) if c.trans_x1 else (M, K)
    x2_shape = (N, K) if c.trans_x2 else (K, N)
    if c.x_dtype == "fp8_e4m3fn":
        x1 = torch.randint(-5, 5, x1_shape).to(torch.float8_e4m3fn)
        x2 = torch.randint(-5, 5, x2_shape).to(torch.float8_e4m3fn)
    else:  # fp4_e2m1
        x1 = torch.randint(-5, 5, x1_shape, dtype=torch.int8)
        x2 = torch.randint(-5, 5, x2_shape, dtype=torch.int8)

    # 3D scale — 文档约定:
    #   pertoken_scale: trans_x1=False → (M, ceilK, 2);  True → (ceilK, M, 2)
    #   scale:          trans_x2=False → (ceilK, N, 2);   True → (N, ceilK, 2)
    if c.trans_x1:
        pertoken_scale = torch.randint(125, 131, (ceilK, M, MXFP_BASE), dtype=torch.uint8)
    else:
        pertoken_scale = torch.randint(125, 131, (M, ceilK, MXFP_BASE), dtype=torch.uint8)

    if c.trans_x2:
        scale = torch.randint(125, 131, (N, ceilK, MXFP_BASE), dtype=torch.uint8)
    else:
        scale = torch.randint(125, 131, (ceilK, N, MXFP_BASE), dtype=torch.uint8)

    bias = torch.randn(N) * 0.5 if c.has_bias else None
    return dict(x1=x1, x2=x2, pertoken_scale=pertoken_scale, scale=scale, bias=bias)


# ============================================================
# Golden (torch CPU)
# ============================================================

def golden(d: dict, c: Case, verbose: bool = False) -> torch.Tensor:
    """
    MX Golden — 3D scale, 对齐 golden.py L64-108.
    统一转到 (M,K) × (K,N) 格式后再计算.
    """
    # ---- float32 + 转到逻辑 (M,K)/(K,N) ----
    x1 = d["x1"].float()
    x2 = d["x2"].float()
    ps  = e8m0_to_float32(d["pertoken_scale"])
    sc  = e8m0_to_float32(d["scale"])
    if c.trans_x1:
        x1 = x1.transpose(-1, -2)   # 存储(K,M) → 逻辑(M,K)
    if c.trans_x2:
        x2 = x2.transpose(-1, -2)   # 存储(N,K) → 逻辑(K,N)

    if verbose:
        print(f"  [g] input: x1={tuple(x1.shape)} x2={tuple(x2.shape)} "
              f"ps={tuple(ps.shape)} sc={tuple(sc.shape)}")

    # ---- reshape 3D scale → 2D ----
    # 目标: ps → (M, ceilK*2),  sc → (ceilK*2, N)
    # 注意: 必须 split 成两行赋值, 否则 reshape() 里的 .shape 引用的是旧值!
    if c.trans_x1:
        ck, m, _ = ps.shape   # (ceilK, M, 2)
        ps = ps.transpose(-1, -2)          # → (ceilK, 2, M)
        ps = ps.reshape(ck * 2, m)         # → (ceilK*2, M)
        ps = ps.transpose(-1, -2)          # → (M, ceilK*2)
    else:
        ps = ps.reshape(ps.shape[0], ps.shape[1] * ps.shape[2])  # (M, ceilK*2)

    if c.trans_x2:
        n, ck, _ = sc.shape   # (N, ceilK, 2)
        sc = sc.transpose(0, 1)            # → (ceilK, N, 2)
        sc = sc.transpose(-1, -2)          # → (ceilK, 2, N)
        sc = sc.reshape(ck * 2, n)         # → (ceilK*2, N)
    else:
        sc = sc.transpose(-1, -2)          # (ceilK, N, 2) → (ceilK, 2, N)
        sc = sc.reshape(sc.shape[0] * sc.shape[1], sc.shape[2])  # (ceilK*2, N)

    if verbose:
        print(f"  [g] reshape: ps={tuple(ps.shape)} sc={tuple(sc.shape)}")

    # ---- 截断 ----
    k_dim = x1.shape[-1]
    if ceil_div(k_dim, MXFP_INNER) % 2 != 0:
        ps = ps[:, :-1]
        sc = sc[:-1, :]
        if verbose:
            print(f"  [g] truncated: ceil(K/32)={ceil_div(k_dim, MXFP_INNER)} odd")

    # ---- broadcast ×32 ----
    ps_bc = ps.repeat_interleave(MXFP_INNER, dim=-1)   # (M, K_padded)
    sc_bc = sc.repeat_interleave(MXFP_INNER, dim=-2)   # (K_padded, N)

    # ---- pad ----
    x1 = torch.nn.functional.pad(x1, (0, max(ps_bc.shape[-1] - x1.shape[-1], 0)))
    x2 = torch.nn.functional.pad(x2, (0, 0, 0, max(sc_bc.shape[-2] - x2.shape[-2], 0)))

    # ---- apply + matmul ----
    out = torch.matmul(x1 * ps_bc, x2 * sc_bc)

    if verbose:
        print(f"  [g] out={tuple(out.shape)} range=[{out.min():.4f}, {out.max():.4f}]")

    # ---- bias + cast ----
    # 对齐仓库 golden (ops-nn/tests/assets/golden.py L210-213)
    # 注: NPU 内部 bias 在第一个 K-tile 累加(float32), golden 在 matmul 后加(float32)
    # float32 多次舍入差异可导致 1 ULP 偏差, 属于已知可接受精度范围
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if d["bias"] is not None:
        out = out + d["bias"].float()
    return out.to(dt[c.out_dtype])


# ============================================================
# NPU 调用
# ============================================================

def call_npu(d: dict, c: Case, verbose: bool = False) -> torch.Tensor:
    x1 = d["x1"].clone()
    x2 = d["x2"].clone()
    ps  = d["pertoken_scale"].clone()
    sc  = d["scale"].clone()
    bias = d["bias"].clone() if d["bias"] is not None else None

    # dtype args
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
        group_sizes=[1, 1, 32],
    )
    if x1_dt: kw["x1_dtype"] = x1_dt
    if x2_dt: kw["x2_dtype"] = x2_dt
    if bias is not None: kw["bias"] = bias.float().npu()  # API要求bias必须为DT_FLOAT（checker L1494-1496）

    # transpose: .t() 让 NPU 通过 stride 检测转置
    if c.trans_x1: kw["x1"] = kw["x1"].t()
    if c.trans_x2: kw["x2"] = kw["x2"].t()

    out = torch_npu.npu_quant_matmul(**kw)
    return out.float().cpu()


# ============================================================
# 比较
# ============================================================

def cmp(g: torch.Tensor, n: torch.Tensor, out_dtype: str, v=False) -> dict:
    g, n = g.float(), n.float()
    diff = (g - n).abs()
    mx = diff.max().item()
    cs = (g.flatten() @ n.flatten() / (g.norm() * n.norm())).item() if g.norm() > 0 else 1.0
    tol = {"bfloat16": (0.01, 0.01), "float16": (0.001, 0.001), "float32": (0.001, 0.001)}
    rt, at = tol.get(out_dtype, (0.001, 0.001))
    ok = torch.allclose(g, n, rtol=rt, atol=at)
    if v:
        print(f"  [cmp] max_diff={mx:.6f} cosine={cs:.6f} → {'PASS' if ok else 'FAIL'}")
    return dict(passed=ok, max_diff=mx, cosine=cs)


# ============================================================
# main
# ============================================================

def run(c: Case, npu: bool, v: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"  {c.name}  M={c.M} N={c.N} K={c.K}  x={c.x_dtype} out={c.out_dtype}")
    d = gen_data(c)
    g = golden(d, c, v)
    print(f"  golden: {tuple(g.shape)} {g.dtype}  [{g.float().min():.2f}, {g.float().max():.2f}]")

    r = dict(name=c.name, g_ok=True, npu_ok=False, passed=None)
    if npu:
        try:
            n = call_npu(d, c, v)
            print(f"  npu:    {tuple(n.shape)} {n.dtype}  [{n.min():.2f}, {n.max():.2f}]")
            c_ = cmp(g, n, c.out_dtype, v)
            r.update(npu_ok=True, **c_)
            t = "PASS" if c_["passed"] else "FAIL"
            print(f"  → {t}  diff={c_['max_diff']:.6f}  cos={c_['cosine']:.6f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            r["err"] = str(e)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    npu = not a.golden_only

    print(f"MX Golden+NPU  cases={len(CASES)}  npu={'ON' if npu else 'OFF'}")
    res = [run(c, npu, a.verbose) for c in CASES]

    print(f"\n{'='*60}")
    print(f"{'name':<20} {'golden':>6} {'npu':>6} {'result':>6} {'max_diff':>10} {'cosine':>8}")
    print("-" * 62)
    for r in res:
        g_ = "OK" if r["g_ok"] else "ERR"
        n_ = "OK" if r.get("npu_ok") else ("ERR" if "err" in r else "--")
        p_ = "PASS" if r.get("passed") else ("FAIL" if r.get("passed") is False else "--")
        d_ = f"{r['max_diff']:.6f}" if r.get("npu_ok") else "--"
        c_ = f"{r['cosine']:.6f}" if r.get("npu_ok") else "--"
        print(f"{r['name']:<20} {g_:>6} {n_:>6} {p_:>6} {d_:>10} {c_:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
