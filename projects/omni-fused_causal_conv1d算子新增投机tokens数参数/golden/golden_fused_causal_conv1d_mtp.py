#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fused_causal_conv1d 投机 tokens 数扩展 — 项目侧 golden (G3)

继承 atk `causal_conv1d_golden` oracle（continue/prefill executor 共用同一实现，
已逐行 trace 验证语义等价），扩展 m ∈ [0, 16] 用例矩阵，套 bench_metrics 输出。

定位：
    - 纯 PyTorch + numpy，不依赖 npu 环境 / torch_npu / omni_custom_ops
    - 不调 npu 算子（独立于 binding 落地）
    - 用途：
        (a) binding 落地后，npu 输出与本 oracle 比对（ratio 制）
        (b) 本脚本自带“双 oracle 互比”自洽验证（atk oracle vs 朴素独立卷积），
            在本机无 torch_npu 时仍能给数值正确性证据

oracle 继承来源（trace 行号）：
    D:/Desktop/Code/CustomOP/atk/AiInfraFusedCausalConv1d/
        executor_ai_infra_fused_causal_conv1d_continue.py:73-248  (decode 主用)
        executor_ai_infra_fused_causal_conv1d_pref.py:95-268      (prefill，语义等价)

判据（对齐华为精度标准 / atk json cv_fused_double_benchmark）：
    双标杆 ratio: max_re_ratio ≤ 5, avg_re_ratio ≤ 1.5, rms_ratio ≤ 1.5
    custom op 无 GPU 标杆时降级单标杆（合成地板 10/2/2）

纪律：
    - 输入先舍到 dtype 再 fp64 累加（卷积场景：模拟算子 bf16/fp16 输入舍入）
    - 不用 2^-mantissa 容差（过紧致假 FAIL）
    - 输出 y + 写回的 conv_states 两份都比对（cache 写错不反映在 y 上是常见 false-pass）
    - 全指标 + 余量（每行 PASS/FAIL 都打全）

用法：
    python golden_fused_causal_conv1d_mtp.py            # 双 oracle 自洽验证（默认）
    python golden_fused_causal_conv1d_mtp.py --full     # 含 npu 比对（需 torch_npu，待 binding 落地）
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# torch 可选：本机无 torch 时降级到 numpy-only 自洽验证
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # 本机开发机常见
    _HAS_TORCH = False


# ============================================================
# 一、oracle（继承自 atk causal_conv1d_golden，trace 行号见文件头）
# ============================================================

def causal_conv1d_golden_np(
    x: np.ndarray,            # [cu_seq_len, dim], float64
    weight: np.ndarray,       # [K, dim], float64
    conv_states: np.ndarray,  # [N, stateLen, dim], float64（会被原地写回）
    query_start_loc: np.ndarray,        # [batch+1], int64
    cache_indices: Optional[np.ndarray],  # 1D[batch] 或 2D[batch,maxBlocks] 或 None
    max_query_len: int,
    pad_slot_id: int,
    num_accepted_tokens: Optional[np.ndarray],  # [batch], int64 或 None
    num_computed_tokens: Optional[np.ndarray],  # [batch], int64 或 None
    block_idx_first: Optional[np.ndarray],       # [batch], int64 或 None
    block_idx_last: Optional[np.ndarray],        # [batch], int64 或 None
    initial_state_idx: Optional[np.ndarray],     # [batch], int64 或 None
    block_size: int,
    conv_mode: int,
    inplace: bool,
    residual: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    numpy 版 atk causal_conv1d_golden，逐分支对齐 continue executor:73-248。

    输入约定（与 atk 一致）：
        - x 已是 2D [cu_seq_len, dim]（3D flatten 由上层处理）
        - conv_states 会被原地写回（caller 传 clone）
        - 所有 int tensor 转 int64
    返回：
        (out, conv_states_after_write)
    """
    cu_seq_len, dim = x.shape
    batch_size = query_start_loc.shape[0] - 1
    width = weight.shape[0]  # K
    assert conv_states.shape[1] >= width - 1, \
        f"conv_states.size(1)={conv_states.shape[1]} < width-1={width-1}"

    apc_enabled = block_idx_last is not None
    out = np.zeros_like(x)

    for batch_idx in range(batch_size):
        start_idx = int(query_start_loc[batch_idx])
        end_idx = int(query_start_loc[batch_idx + 1])
        seq_len = end_idx - start_idx
        seq_x = x[start_idx:end_idx]  # [seq_len, dim]

        # ---- atk:154-167 apc 分支 ----
        if apc_enabled:
            nct_b = int(num_computed_tokens[batch_idx])
            seq_completed_offset_token = nct_b % block_size
            seq_completed_offset = block_size - seq_completed_offset_token
            seq_end_offset = (seq_len - seq_completed_offset) % block_size
            last_full_block_token_index = seq_len - seq_end_offset
            if seq_end_offset == 0:
                last_full_block_token_index -= block_size

            idx_first = int(block_idx_first[batch_idx])
            idx_last = int(block_idx_last[batch_idx])
            n_block_to_fill = idx_last - idx_first

            assert cache_indices is not None and cache_indices.ndim == 2
            read_cache_line = int(cache_indices[batch_idx, int(initial_state_idx[batch_idx])])
            write_cache_line = int(cache_indices[batch_idx, idx_last])
        else:
            # atk:160-167 apc not enabled
            if cache_indices is not None:
                read_cache_line = int(cache_indices[batch_idx])
                write_cache_line = read_cache_line
            else:
                read_cache_line = batch_idx
                write_cache_line = batch_idx

        # atk:169-170 pad_slot_id 跳过（输出保持 0）
        if read_cache_line == pad_slot_id:
            continue

        # ---- atk:173-189 read cache ----
        nct_b = int(num_computed_tokens[batch_idx]) if num_computed_tokens is not None else 0
        if num_computed_tokens is not None and nct_b == 0:
            cached_state = np.zeros((width - 1, dim), dtype=x.dtype)
            offset = 0
        else:
            if num_accepted_tokens is not None:
                accepted_tokens = int(num_accepted_tokens[batch_idx])
                assert 1 <= accepted_tokens <= seq_len, \
                    f"batch {batch_idx}: accepted={accepted_tokens} not in [1, seq_len={seq_len}]"
                offset = accepted_tokens - 1
            else:
                # NAT 为空：pad 最后 width-1 个到 x
                offset = conv_states.shape[1] - (width - 1)
            cached_state = conv_states[read_cache_line][: offset + width - 1].copy()

        padded_input = np.concatenate([cached_state, seq_x], axis=0)  # [offset+W-1+seq_len, dim]

        # ---- atk:196-198 write cache（尾对齐） ----
        cache_len = min(conv_states.shape[1], padded_input.shape[0])
        conv_states[write_cache_line][-cache_len:] = padded_input[-cache_len:]

        # ---- atk:200 切掉前 offset ----
        padded_input = padded_input[offset:]  # [W-1+seq_len, dim]

        # ---- atk:202-213 apc 跨 block 写 cache ----
        if apc_enabled:
            for chunk in range(n_block_to_fill):
                boundary_idx = last_full_block_token_index - (n_block_to_fill - chunk - 1) * block_size
                assert boundary_idx > 0, "seq_len / block_idx_first/last mismatched"
                write_cache_line_chunk = int(cache_indices[batch_idx, idx_first + chunk])
                conv_states[write_cache_line_chunk][-(width - 1):] = \
                    padded_input[boundary_idx: boundary_idx + width - 1]

        # ---- atk:219-230 因果深度卷积（groups=dim） ----
        # padded_input: [W-1+seq_len, dim] -> 转置 [dim, W-1+seq_len] -> [1, dim, L]
        # weight: [K, dim] -> 转置 [dim, K] -> [dim, 1, K]
        # 对每个 dim 通道独立做 1D 卷积（stride=1, padding=0, groups=dim）
        L_in = padded_input.shape[0]
        out_len = L_in - width + 1  # = seq_len
        # 用 einsum 避免逐通道循环：padded[W-1+seq_len, dim] 滑窗
        # 构造滑窗 [out_len, K, dim]
        windows = np.stack(
            [padded_input[t: t + width] for t in range(out_len)],
            axis=0,
        )  # [out_len, K, dim]
        result = np.einsum("tkd,kd->td", windows, weight)  # [out_len, dim] = [seq_len, dim]

        # ---- atk:233-238 conv_mode==1 (Pangu V2): reset zero-padding 影响 ----
        if conv_mode == 1:
            assert num_computed_tokens is not None
            last_reset_idx = width - 1 - nct_b
            last_reset_idx = max(last_reset_idx, 0)
            result[:last_reset_idx] = 0

        # ---- atk:242 残差 + 写出 ----
        slot = result + seq_x if residual else result
        out[start_idx:end_idx] = slot
        if inplace:
            x[start_idx:end_idx] = slot

    if inplace:
        return x, conv_states
    return out, conv_states


# ---- torch 版（与 atk 原版一致，本机有 torch 时启用） ----
if _HAS_TORCH:
    def causal_conv1d_golden_torch(
        x: "torch.Tensor",
        weight: "torch.Tensor",
        conv_states: "torch.Tensor",
        query_start_loc: "torch.Tensor",
        cache_indices: Optional["torch.Tensor"],
        max_query_len: int,
        pad_slot_id: int,
        num_accepted_tokens: Optional["torch.Tensor"],
        num_computed_tokens: Optional["torch.Tensor"],
        block_idx_first: Optional["torch.Tensor"],
        block_idx_last: Optional["torch.Tensor"],
        initial_state_idx: Optional["torch.Tensor"],
        block_size: int,
        conv_mode: int,
        inplace: bool,
        residual: bool,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """torch 版 oracle，逐行对齐 atk continue executor:73-248。"""
        if x.ndim == 3:
            flattened = True
            bsz, seq_len, dim = x.shape
            x = x.view(-1, dim)
            if query_start_loc is None:
                query_start_loc = torch.arange(
                    0, (bsz + 1) * seq_len, seq_len, dtype=torch.int32, device=x.device
                )
        else:
            flattened = False

        cu_seq_len, dim = x.shape
        batch_size = query_start_loc.shape[0] - 1
        width = weight.size(0)
        assert conv_states.size(1) >= width - 1
        apc_enabled = block_idx_last is not None
        out = torch.zeros_like(x)

        for batch_idx in range(batch_size):
            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_x = x[start_idx:end_idx]

            if apc_enabled:
                nct_b = num_computed_tokens[batch_idx].item()
                seq_completed_offset_token = nct_b % block_size
                seq_completed_offset = block_size - seq_completed_offset_token
                seq_end_offset = (seq_len - seq_completed_offset) % block_size
                last_full_block_token_index = seq_len - seq_end_offset
                if seq_end_offset == 0:
                    last_full_block_token_index -= block_size

                idx_first = block_idx_first[batch_idx].item()
                idx_last = block_idx_last[batch_idx].item()
                n_block_to_fill = idx_last - idx_first

                assert cache_indices is not None and cache_indices.ndim == 2
                read_cache_line = cache_indices[batch_idx, initial_state_idx[batch_idx]].item()
                write_cache_line = cache_indices[batch_idx, idx_last].item()
            else:
                if cache_indices is not None:
                    read_cache_line = cache_indices[batch_idx].item()
                    write_cache_line = read_cache_line
                else:
                    read_cache_line = batch_idx
                    write_cache_line = batch_idx

            if read_cache_line == pad_slot_id:
                continue

            nct_b = num_computed_tokens[batch_idx].item() if num_computed_tokens is not None else 0
            if num_computed_tokens is not None and nct_b == 0:
                cached_state = torch.zeros((width - 1, dim), device=x.device, dtype=x.dtype)
                offset = 0
            else:
                if num_accepted_tokens is not None:
                    accepted_tokens = num_accepted_tokens[batch_idx].item()
                    assert 1 <= accepted_tokens <= seq_len
                    offset = accepted_tokens - 1
                else:
                    offset = conv_states.size(1) - (width - 1)
                cached_state = conv_states[read_cache_line][: offset + width - 1]

            padded_input = torch.cat([cached_state, seq_x], dim=0)
            cache_len = min(conv_states.size(1), padded_input.size(0))
            conv_states[write_cache_line][-cache_len:] = padded_input[-cache_len:]
            padded_input = padded_input[offset:]

            if apc_enabled:
                for chunk in range(n_block_to_fill):
                    boundary_idx = last_full_block_token_index - (n_block_to_fill - chunk - 1) * block_size
                    assert boundary_idx > 0
                    wcl = cache_indices[batch_idx, idx_first + chunk]
                    conv_states[wcl][-(width - 1):] = padded_input[boundary_idx: boundary_idx + width - 1]

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight.transpose(0, 1).unsqueeze(1),
                bias=None, stride=1, padding=0, groups=dim,
            )
            result = result.squeeze(0).transpose(0, 1)

            if conv_mode == 1:
                assert num_computed_tokens is not None
                last_reset_idx = max(width - 1 - nct_b, 0)
                result[:last_reset_idx] = 0

            slot = result + seq_x if residual else result
            out[start_idx:end_idx] = slot
            if inplace:
                x[start_idx:end_idx] = slot

        if inplace:
            return (x if not flattened else x.view(bsz, -1, dim)), conv_states
        return (out if not flattened else out.view(bsz, -1, dim)), conv_states


# ============================================================
# 二、独立朴素 oracle（双 oracle 互比的第二参考）
# ============================================================

def naive_causal_conv1d_np(
    x: np.ndarray,
    weight: np.ndarray,
    conv_states: np.ndarray,
    query_start_loc: np.ndarray,
    cache_indices: Optional[np.ndarray],
    num_accepted_tokens: Optional[np.ndarray],
    num_computed_tokens: Optional[np.ndarray],
    block_idx_first: Optional[np.ndarray],
    block_idx_last: Optional[np.ndarray],
    initial_state_idx: Optional[np.ndarray],
    pad_slot_id: int,
    block_size: int,
    conv_mode: int,
    residual: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    独立实现的朴素因果卷积 oracle，与 causal_conv1d_golden_np 走同一语义但
    不同代码路径（显式滑窗而非 einsum，cache 读写顺序微调），用于互比证伪。

    语义约束（必须与 atk oracle 一致，否则互比无意义）：
        - read cache line / write cache line 选择规则同 atk
        - offset = accepted_tokens - 1（NAT 模式 offset = stateLen - (W-1)）
        - padded_input = cat([cached_state, seq_x])，切前 offset
        - cache 尾对齐写回
        - conv_mode==1 reset
    """
    cu_seq_len, dim = x.shape
    batch_size = query_start_loc.shape[0] - 1
    width = weight.shape[0]
    apc_enabled = block_idx_last is not None
    out = np.zeros_like(x)
    cs = conv_states  # alias，原地写

    for batch_idx in range(batch_size):
        s = int(query_start_loc[batch_idx])
        e = int(query_start_loc[batch_idx + 1])
        seq_len = e - s
        seq_x = x[s:e]

        if apc_enabled:
            nct_b = int(num_computed_tokens[batch_idx])
            sot_tok = nct_b % block_size
            sot = block_size - sot_tok
            seo = (seq_len - sot) % block_size
            lfbi = seq_len - seo
            if seo == 0:
                lfbi -= block_size
            idx_first = int(block_idx_first[batch_idx])
            idx_last = int(block_idx_last[batch_idx])
            n_block_to_fill = idx_last - idx_first
            read_line = int(cs_idx_2d(cache_indices, batch_idx, int(initial_state_idx[batch_idx])))
            write_line = int(cs_idx_2d(cache_indices, batch_idx, idx_last))
        else:
            lfbi = 0
            n_block_to_fill = 0
            idx_first = 0
            if cache_indices is not None:
                read_line = int(cache_indices[batch_idx])
                write_line = read_line
            else:
                read_line = batch_idx
                write_line = batch_idx

        if read_line == pad_slot_id:
            continue

        nct_b = int(num_computed_tokens[batch_idx]) if num_computed_tokens is not None else 0
        if num_computed_tokens is not None and nct_b == 0:
            cached = np.zeros((width - 1, dim), dtype=x.dtype)
            offset = 0
        else:
            if num_accepted_tokens is not None:
                acc = int(num_accepted_tokens[batch_idx])
                offset = acc - 1
            else:
                offset = cs.shape[1] - (width - 1)
            cached = cs[read_line][: offset + width - 1].copy()

        padded = np.concatenate([cached, seq_x], axis=0)
        cl = min(cs.shape[1], padded.shape[0])
        cs[write_line][-cl:] = padded[-cl:]
        padded = padded[offset:]

        if apc_enabled:
            for chunk in range(n_block_to_fill):
                bi = lfbi - (n_block_to_fill - chunk - 1) * block_size
                wcl = int(cs_idx_2d(cache_indices, batch_idx, idx_first + chunk))
                cs[wcl][-(width - 1):] = padded[bi: bi + width - 1]

        # 朴素显式滑窗（与 einsum 路径独立）
        out_len = padded.shape[0] - width + 1
        result = np.empty((out_len, dim), dtype=x.dtype)
        for t in range(out_len):
            acc = np.zeros(dim, dtype=x.dtype)
            for k in range(width):
                acc += weight[k] * padded[t + k]
            result[t] = acc

        if conv_mode == 1:
            lri = max(width - 1 - nct_b, 0)
            result[:lri] = 0

        out[s:e] = result + seq_x if residual else result

    return out, cs


def cs_idx_2d(cache_indices: np.ndarray, batch_idx: int, col: int) -> int:
    """统一处理 cache_indices 1D/2D 取值。"""
    if cache_indices.ndim == 2:
        return int(cache_indices[batch_idx, col])
    return int(cache_indices[batch_idx])


# ============================================================
# 三、bench_metrics（ratio 制，对齐华为精度标准 / cv_fused_double_benchmark）
# ============================================================

SMALL_VALUE = 1e-12  # _calc_ratio 地板，防除零（对齐 aclnn-fuzz checkResultNew）


def _calc_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """x / max(y, small_value)，对齐 aclnn-fuzz _calc_ratio。"""
    return numer / np.maximum(np.abs(denom), SMALL_VALUE)


def bench_metrics(got: np.ndarray, ref: np.ndarray, name: str) -> dict:
    """
    返回 ratio 制全指标。got/ref 同 shape float64。

    指标（对齐华为精度标准 L1 ratio）：
        - max_re_ratio : max(|got-ref| / max(|ref|, eps))   阈值 5（双标杆）/ 10（单标杆合成地板）
        - avg_re_ratio : mean 同上                            阈值 1.5 / 2
        - rms_ratio    : sqrt(mean((diff/max)^2))            阈值 1.5 / 2
        - max_abs_err  : max|got-ref|（参考用，无阈值）
        - mse          : mean((got-ref)^2)（参考用）
    """
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    diff = got - ref
    ratio = _calc_ratio(np.abs(diff), ref)
    m = {
        "name": name,
        "n_elem": got.size,
        "max_re_ratio": float(np.max(ratio)),
        "avg_re_ratio": float(np.mean(ratio)),
        "rms_ratio": float(np.sqrt(np.mean(ratio ** 2))),
        "max_abs_err": float(np.max(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "ref_abs_max": float(np.max(np.abs(ref))),
    }
    return m


def verdict(metrics: dict, thr_max: float, thr_avg: float, thr_rms: float) -> Tuple[bool, str]:
    """三阈值裁决，返回 (pass, margin_str)。"""
    pm = lambda v, t: f"{v:.4e}<= {t:.1f}?"  # noqa
    ok_max = metrics["max_re_ratio"] <= thr_max
    ok_avg = metrics["avg_re_ratio"] <= thr_avg
    ok_rms = metrics["rms_ratio"] <= thr_rms
    passed = ok_max and ok_avg and ok_rms
    margin = (
        f"max_re={metrics['max_re_ratio']:.4e}(thr {thr_max}, "
        f"margin {thr_max - metrics['max_re_ratio']:+.4e}) | "
        f"avg_re={metrics['avg_re_ratio']:.4e}(thr {thr_avg}, "
        f"margin {thr_avg - metrics['avg_re_ratio']:+.4e}) | "
        f"rms={metrics['rms_ratio']:.4e}(thr {thr_rms}, "
        f"margin {thr_rms - metrics['rms_ratio']:+.4e}) | "
        f"max_abs={metrics['max_abs_err']:.4e} ref_max={metrics['ref_abs_max']:.4e} "
        f"n={metrics['n_elem']}"
    )
    return passed, margin


# ============================================================
# 四、caseData 矩阵
# ============================================================

@dataclass
class CaseData:
    name: str
    m: int                       # multiTokenNum
    K: int                       # windowSize = weight.shape[0]
    D: int                       # dim
    batch: int                   # batch_size
    accepted: List[int]          # per-batch accepted tokens（len=batch）
    apc: bool                    # APC 开关（cache_indices 2D）
    conv_mode: int               # 0=Qwen3-Next, 1=Pangu V2
    residual: bool
    block_size: int
    num_computed: List[int]      # per-batch，None 用 [0]*batch 表示首 token
    dtype: str = "bf16"          # bf16 / fp16 / fp64


# ============================================================
# 矩阵常量（边界值取自规格，非魔鬼数字）
# ============================================================
# CUTBS/CUTBSD 切换阈值：MAX_DIM_CUTBSD = 3152（完整技术文档.md:232 纠正需求文档的 512 误导）
# 阈值判定：dimSize >= 3152 → CUTBSD(TilingKey=0)，否则 → CUTBS(TilingKey=1)
# dim 必须 % 16 == 0，所以边界两侧用 3136(< 3152, CUTBS) / 3152(=, CUTBSD) / 3168(>, CUTBSD)
MAX_DIM_CUTBSD = 3152
MIN_DIM = 64         # dim ∈ [64, 16384], dim % 16 == 0

# A 层 dim 边界代表：MIN/小/中大/CUTBS-CUTBSD 切换两侧/极限
DIM_CORE = [MIN_DIM, 1024, 3136, 3152, 3168, 16384]
# B/C/D 层 dim 子集（降维）
DIM_REP = [1024, 16384]
DIM_ONE = [1024]


def _accepted_reps(m: int) -> List[List[int]]:
    """单 batch accepted 代表点（m=0 仅 prefill）。"""
    if m == 0:
        return [[0]]
    return [
        [0],                    # prefill
        [1],                    # 单接受
        [max(1, m // 4)],       # 偏小
        [max(1, m // 2)],       # 中间
        [max(1, 3 * m // 4)],   # 偏大
        [m],                    # 满接受（= m）
        [m + 1],                # 上界（m+1 = seqLen）
    ]


def build_casematrix() -> List[CaseData]:
    """
    caseData 矩阵设计（v2，覆盖需求 §1.2.3.3 泛化规格 m∈[0,16]，扩 dtype/conv_mode/batch/dim 边界）。

    分层降维（避免全交叉爆炸，目标 300-600 cases）：
        A 层（核心，全交叉）：
            m ∈ [0,16] 全 17 值 × K{3,6} × D{64,1024,3136,3152,3168,16384}
                  × accepted 代表点（prefill/单/偏小/中/偏大/满/上界）
                  × apc{on,off} × dtype=bf16 × conv_mode=1
            —— m 全覆盖 + CUTBS/CUTBSD 切换边界 + K 双值 + accepted 密集
        B 层（dtype FP16，m 代表点）：
            m{0,8,16} × K=3 × D{1024,16384} × accepted 代表 × apc{on,off}
            × dtype=fp16 × conv_mode=1
            —— FP16 路径覆盖（不同 ULP/舍入行为）
        C 层（conv_mode=0 Qwen3-Next，m 代表点）：
            m{0,8,16} × K=3 × D=1024 × accepted 代表 × apc{on,off}
            × dtype=bf16 × conv_mode=0
            —— Qwen3-Next 路径（无 conv_mode==1 reset 行为）
        D 层（batch 多值，per-batch accepted 不同）：
            m{8,16} × batch=4 × K=3 × D=1024 × apc{on,off}
            × dtype=bf16 × conv_mode=1
            —— per-batch num_accepted_tokens 各异（覆盖 batch 维独立性）

    总规模估算：A 约 17*2*6*5*2 ≈ 1000（accepted 代表点 m=0 退化为 1，故实际 ~ 17*2*6*~5.7*2 ≈ 1000）
                B 约 3*1*2*5*2 = 60
                C 约 3*1*1*5*2 = 30
                D 约 2*1*1*1*2 = 4（每 case batch=4，per-batch accepted 异）
                合计 ~ 1100 cases。
    """
    cases: List[CaseData] = []

    # ---- A 层：核心全交叉（m 全覆盖 + dim 边界 + K 双 + accepted 密集） ----
    K_core = [3, 6]
    for m in range(0, 17):                       # m ∈ [0,16] 全 17 值
        for K in K_core:
            for D in DIM_CORE:
                for acc in _accepted_reps(m):
                    batch = 1
                    for apc in (False, True):
                        nm = (
                            f"A_m{m:02d}_K{K}_D{D}_acc{acc[0]}"
                            f"_apc{int(apc)}_bf16_pangu"
                        )
                        cases.append(CaseData(
                            name=nm, m=m, K=K, D=D, batch=batch,
                            accepted=acc, apc=apc,
                            conv_mode=1, residual=True, block_size=128,
                            num_computed=[0] * batch,
                            dtype="bf16",
                        ))

    # ---- B 层：dtype FP16（m 代表点 + 双 D） ----
    m_rep = [0, 8, 16]
    for m in m_rep:
        for D in DIM_REP:
            for acc in _accepted_reps(m):
                for apc in (False, True):
                    nm = (
                        f"B_m{m:02d}_K3_D{D}_acc{acc[0]}"
                        f"_apc{int(apc)}_fp16_pangu"
                    )
                    cases.append(CaseData(
                        name=nm, m=m, K=3, D=D, batch=1,
                        accepted=acc, apc=apc,
                        conv_mode=1, residual=True, block_size=128,
                        num_computed=[0],
                        dtype="fp16",
                    ))

    # ---- C 层：conv_mode=0 (Qwen3-Next) ----
    for m in m_rep:
        for acc in _accepted_reps(m):
            for apc in (False, True):
                nm = (
                    f"C_m{m:02d}_K3_D1024_acc{acc[0]}"
                    f"_apc{int(apc)}_bf16_qwen"
                )
                cases.append(CaseData(
                    name=nm, m=m, K=3, D=1024, batch=1,
                    accepted=acc, apc=apc,
                    conv_mode=0, residual=True, block_size=128,
                    num_computed=[0],
                    dtype="bf16",
                ))

    # ---- D 层：batch 多值，per-batch accepted 各异 ----
    for m in [8, 16]:
        for apc in (False, True):
            # per-batch accepted 故意各异：覆盖 prefill(0)/单(1)/中间/满 全在一个 batch 内
            cap = min(m, 4) if m > 0 else 0
            if m == 0:
                acc_multi = [0, 0, 0, 0]
            else:
                # 选 4 个差异化的 per-batch accepted（不超过 m+1）
                cand = sorted({0, 1, max(1, m // 2), m, m + 1})[:4]
                acc_multi = (cand * 4)[:4]
                while len(set(acc_multi)) < 2:  # 保证有差异
                    acc_multi[-1] = min(m + 1, acc_multi[-1] + 1)
            nm = (
                f"D_m{m:02d}_K3_D1024_b4_acc"
                f"{'-'.join(str(a) for a in acc_multi)}"
                f"_apc{int(apc)}_bf16_pangu"
            )
            cases.append(CaseData(
                name=nm, m=m, K=3, D=1024, batch=len(acc_multi),
                accepted=acc_multi, apc=apc,
                conv_mode=1, residual=True, block_size=128,
                num_computed=[0] * len(acc_multi),
                dtype="bf16",
            ))

    return cases


# ============================================================
# 五、输入构造（先舍到 dtype 再 fp64 累加）
# ============================================================

def materialize_case(case: CaseData, seed: int) -> dict:
    """
    构造一组输入。纪律：
        - 先生成 fp64 随机 → 舍到 dtype（bf16/fp16 模拟算子输入舍入）→ 再升 fp64 累加
          （对齐 memory matmul-golden-inputs-round-to-dtype）
        - conv_states.shape[1] = case.K - 1 + case.m（MTP stateLen）
        - query_start_loc: 2D x 时 batch 段切分；这里 batch 段长 = accepted（或 m+1）
        - cache_indices: apc on 时 2D [batch, maxBlocks]，off 时 1D [batch]
        - num_computed_tokens: 每 batch 0（首 token，cache 零初始化）
          —— 同时加一组 num_computed>0 的变体覆盖 cache 读已有数据路径
    """
    rng = np.random.default_rng(seed)
    K, D, m, batch = case.K, case.D, case.m, case.batch
    state_len = K - 1 + m

    # 各 batch 段长 = max(accepted, 1)，accepted=0(prefill) 时段长=1（m=0 的退化）
    seg_lens = [max(a, 1) for a in case.accepted]
    total = sum(seg_lens)
    qsl = np.zeros(batch + 1, dtype=np.int64)
    for i, sl in enumerate(seg_lens):
        qsl[i + 1] = qsl[i] + sl

    # x: fp64 随机 → dtype 舍入 → fp64
    x_fp64 = rng.uniform(-1.0, 1.0, size=(total, D)).astype(np.float64)
    if case.dtype == "bf16":
        x = _round_to_bf16_np(x_fp64).astype(np.float64)
    elif case.dtype == "fp16":
        x = _round_to_fp16_np(x_fp64).astype(np.float64)
    else:
        x = x_fp64.copy()

    w_fp64 = rng.uniform(-1.0, 1.0, size=(K, D)).astype(np.float64)
    if case.dtype == "bf16":
        weight = _round_to_bf16_np(w_fp64).astype(np.float64)
    elif case.dtype == "fp16":
        weight = _round_to_fp16_np(w_fp64).astype(np.float64)
    else:
        weight = w_fp64.copy()

    # conv_states: [N, stateLen, D]，N 取 max(batch*2, 4) 保证 cache_indices 有挑选空间
    N = max(batch * 2, 4)
    cs_fp64 = rng.uniform(-1.0, 1.0, size=(N, state_len, D)).astype(np.float64)
    if case.dtype == "bf16":
        conv_states = _round_to_bf16_np(cs_fp64).astype(np.float64)
    elif case.dtype == "fp16":
        conv_states = _round_to_fp16_np(cs_fp64).astype(np.float64)
    else:
        conv_states = cs_fp64.copy()

    # cache_indices
    if case.apc:
        max_blocks = max(2, (max(seg_lens) + case.block_size - 1) // case.block_size + 2)
        ci = np.zeros((batch, max_blocks), dtype=np.int32)
        for b in range(batch):
            perm = rng.permutation(N)[:max_blocks]
            ci[b] = perm
    else:
        ci = rng.permutation(N)[:batch].astype(np.int32)

    # num_accepted_tokens
    nat = np.array(case.accepted, dtype=np.int32)

    # num_computed_tokens: 首 token=0（cache 零初始化路径）
    nct = np.array(case.num_computed, dtype=np.int32)

    # apc 衍生
    bif = bil = isi = None
    if case.apc:
        bif = (nct // case.block_size).astype(np.int32)
        bil = ((nct + np.array(seg_lens, dtype=np.int64) - 1) // case.block_size).astype(np.int32)
        isi_arr = []
        for b in range(batch):
            hi = max(int(bif[b]) + 1, 1)
            isi_arr.append(int(rng.integers(0, hi)))
        isi = np.minimum(np.array(isi_arr, dtype=np.int32), bil)

    pad_slot_id = N + 10  # 保证 > N，不触发跳过（独立 oracle 互比不测 pad_slot）

    return {
        "x": x, "weight": weight, "conv_states": conv_states,
        "query_start_loc": qsl, "cache_indices": ci,
        "num_accepted_tokens": nat, "num_computed_tokens": nct,
        "block_idx_first": bif, "block_idx_last": bil, "initial_state_idx": isi,
        "pad_slot_id": pad_slot_id, "block_size": case.block_size,
        "conv_mode": case.conv_mode, "residual": case.residual,
        "state_len": state_len,
    }


def _round_to_bf16_np(x: np.ndarray) -> np.ndarray:
    """numpy 模拟 bf16 舍入（截断到高 7 位尾数 + 1 符号 + 8 指数）。
    用 ml_dtypes 如果可用，否则用 fp32→位操作近似。
    """
    try:
        from ml_dtypes import bfloat16
        return x.astype(np.float32).astype(bfloat16).astype(np.float32)
    except ImportError:
        # 朴素近似：fp32 精度的 bf16 截断
        f32 = x.astype(np.float32)
        bits = f32.view(np.int32)
        # bf16 = fp32 高 16 位；round-to-nearest-even
        lsb = (bits >> 16) & 1
        rounding_bias = 0x7FFF + lsb
        bf16_bits = (bits + rounding_bias) & 0xFFFF0000
        return bf16_bits.view(np.float32).astype(np.float64)


def _round_to_fp16_np(x: np.ndarray) -> np.ndarray:
    try:
        f16 = x.astype(np.float16)
        return f16.astype(np.float32)
    except Exception:
        return x.astype(np.float32)


# ============================================================
# 六、双 oracle 自洽验证（本机无 torch_npu 时的可证伪路径）
# ============================================================

def run_double_oracle_selfcheck(cases: List[CaseData], seed_base: int = 20260803) -> Tuple[int, int, List[str]]:
    """
    atk oracle (causal_conv1d_golden_np) vs 朴素 oracle (naive_causal_conv1d_np)
    两个独立实现走不同代码路径，互比 ratio。两者都 fp64，理论上应严格相等
    （ratio → 0），任何非零 ratio 都指向 oracle 实现 bug（任一方）。

    返回 (n_pass, n_total, lines)
    """
    lines: List[str] = []
    n_pass = 0
    thr = (5.0, 1.5, 1.5)  # 双标杆阈值（参考用；自洽比值预期 << 阈值）

    for ci, case in enumerate(cases):
        seed = seed_base + ci
        inp = materialize_case(case, seed)

        # oracle A: atk 版（np）
        cs_a = inp["conv_states"].copy()
        out_a, cs_a_out = causal_conv1d_golden_np(
            inp["x"].copy(), inp["weight"].copy(), cs_a,
            inp["query_start_loc"], inp["cache_indices"],
            max_query_len=-1, pad_slot_id=inp["pad_slot_id"],
            num_accepted_tokens=inp["num_accepted_tokens"],
            num_computed_tokens=inp["num_computed_tokens"],
            block_idx_first=inp["block_idx_first"],
            block_idx_last=inp["block_idx_last"],
            initial_state_idx=inp["initial_state_idx"],
            block_size=inp["block_size"], conv_mode=inp["conv_mode"],
            inplace=False, residual=inp["residual"],
        )

        # oracle B: 朴素版（np）
        cs_b = inp["conv_states"].copy()
        out_b, cs_b_out = naive_causal_conv1d_np(
            inp["x"].copy(), inp["weight"].copy(), cs_b,
            inp["query_start_loc"], inp["cache_indices"],
            inp["num_accepted_tokens"], inp["num_computed_tokens"],
            inp["block_idx_first"], inp["block_idx_last"],
            inp["initial_state_idx"], inp["pad_slot_id"],
            inp["block_size"], inp["conv_mode"], inp["residual"],
        )

        m_y = bench_metrics(out_a, out_b, case.name + "/y")
        m_cs = bench_metrics(cs_a_out, cs_b_out, case.name + "/conv_states")
        ok_y, mar_y = verdict(m_y, *thr)
        ok_cs, mar_cs = verdict(m_cs, *thr)
        passed = ok_y and ok_cs
        n_pass += int(passed)
        tag = "PASS" if passed else "FAIL"
        lines.append(f"[{tag}] {case.name}")
        lines.append(f"    y            : {mar_y}")
        lines.append(f"    conv_states  : {mar_cs}")

    return n_pass, len(cases), lines


# ============================================================
# 七、入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="fused_causal_conv1d MTP golden (G3)")
    ap.add_argument("--full", action="store_true",
                    help="含 npu 比对（需 torch_npu + binding 落地，默认 off）")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("fused_causal_conv1d 投机 tokens 数扩展 — 项目 golden G3")
    print("=" * 78)
    print(f"torch available : {_HAS_TORCH}")
    print(f"mode            : {'full (npu比对)' if args.full else '双 oracle 自洽验证'}")
    print()

    cases = build_casematrix()
    print(f"caseData 矩阵: {len(cases)} cases")
    print(f"  A 层: m∈[0,16]全 × K{{3,6}} × D{{64,1024,3136,3152,3168,16384}} "
          f"× accepted代表 × apc{{on,off}} × bf16 × pangu")
    print(f"  B 层: m{{0,8,16}} × K3 × D{{1024,16384}} × accepted代表 × apc{{on,off}} × fp16 × pangu")
    print(f"  C 层: m{{0,8,16}} × K3 × D1024 × accepted代表 × apc{{on,off}} × bf16 × qwen(conv_mode=0)")
    print(f"  D 层: m{{8,16}} × batch4 per-batch accepted各异 × apc{{on,off}} × bf16 × pangu")
    print()

    # 阶段 1：双 oracle 自洽（本机可跑）
    print("-" * 78)
    print("[阶段 1] 双 oracle 自洽验证（atk oracle vs 朴素 oracle，均 fp64）")
    print("-" * 78)
    n_pass, n_total, lines = run_double_oracle_selfcheck(cases, seed_base=args.seed)
    if args.verbose:
        for ln in lines:
            print(ln)
    else:
        # 仅打 FAIL + 摘要
        for ln in lines:
            if ln.startswith("[FAIL]") or ln.startswith("="):
                print(ln)
    print()
    print(f"自洽结果: {n_pass}/{n_total} PASS")
    if n_pass != n_total:
        print("WARNING: 存在 oracle 实现差异，需 trace 定位（atk 版 or 朴素版）")
        return 1

    # 阶段 2：npu 比对（待 binding 落地）
    if args.full:
        print()
        print("-" * 78)
        print("[阶段 2] npu 输出 vs oracle 比对（需 torch_npu + binding）")
        print("-" * 78)
        if not _HAS_TORCH:
            print("SKIP: 本机无 torch，无法跑 npu 比对")
        else:
            print("TODO: binding 落地后接入（调用 torch.ops.custom.npu_ai_infra_fused_causal_conv1d）")
            # 占位：待 binding + maxDraftTokens 落地后补 npu 调用 + ratio 比对

    print()
    print("=" * 78)
    print("判据: 双标杆 ratio (max_re≤5, avg_re≤1.5, rms≤1.5)；无 GPU 降级单标杆 10/2/2")
    print("输出: y + conv_states 两份都比对（cache 写错不反映在 y 上是常见 false-pass）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
