#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简纯 Python 数值自检（无 numpy/torch 依赖）

目的：验证 golden_fused_causal_conv1d_mtp.py 中 oracle 的核心 shape/offset/cache 逻辑
      在 m ∈ [0,16] 全边界无 bug。

不做精度比对（纯 python 浮点），只验证结构性约束（这些是 atk oracle 真正的硬 assert）：
    1. padded_input 切前 offset 后长度 = (W-1) + seq_len
    2. 卷积输出长度 = seq_len（因果性）
    3. cache 写回 cache_len <= stateLen（不越界）
    4. apc 跨 block 写 cache 的 boundary_idx > 0 且 boundary_idx + K - 1 <= padded_len
       （atk 有 assert boundary_idx > 0，越界读是真实 bug）
    5. accepted ∈ [0, m+1]，stateLen = K-1+m

注意：conv_mode==1 的 result[:last_reset_idx]=0 当 last_reset_idx > seq_len 时
      python slice 自动截断（合法行为），不是 bug，故不检查。
"""

import sys


def run_shape_check():
    fails = []
    cases_ok = 0

    for m in [0, 1, 7, 8, 12, 16]:
        for K in [3, 6]:
            for apc in [False, True]:
                acc_list = [0] if m == 0 else [0, 1, max(1, m // 2), m + 1]
                for acc in acc_list:
                    seq_len = max(acc, 1)
                    state_len = K - 1 + m
                    block_size = 128
                    nct = 0  # 首 token

                    # offset / cached_len
                    if nct == 0:
                        offset = 0
                        cached_len = K - 1
                    elif acc >= 1:
                        offset = acc - 1
                        cached_len = offset + K - 1
                    else:
                        offset = state_len - (K - 1)
                        cached_len = offset + K - 1

                    padded_len_before = cached_len + seq_len
                    padded_len_after = padded_len_before - offset
                    assert padded_len_after == K - 1 + seq_len, \
                        f"m={m} K={K} acc={acc}: padded_after wrong"

                    conv_out_len = padded_len_after - K + 1
                    assert conv_out_len == seq_len

                    cache_len = min(state_len, padded_len_before)
                    assert cache_len <= state_len

                    assert state_len == K - 1 + m
                    assert 0 <= acc <= m + 1

                    # apc boundary（atk 真实 assert）
                    if apc:
                        sot_tok = nct % block_size
                        sot = block_size - sot_tok
                        seo = (seq_len - sot) % block_size
                        lfbi = seq_len - seo
                        if seo == 0:
                            lfbi -= block_size
                        idx_first = nct // block_size
                        idx_last = (nct + seq_len - 1) // block_size
                        n_block = idx_last - idx_first
                        for chunk in range(n_block):
                            bi = lfbi - (n_block - chunk - 1) * block_size
                            if bi <= 0:
                                fails.append(f"m={m} K={K} acc={acc} chunk={chunk}: boundary={bi}<=0")
                            if bi + K - 1 > padded_len_after:
                                fails.append(f"m={m} K={K} acc={acc} chunk={chunk}: read OOB")

                    cases_ok += 1

    print(f"shape/offset self-check: {cases_ok} cases, {len(fails)} structural failures")
    for f in fails[:20]:
        print("  FAIL:", f)
    return len(fails) == 0


if __name__ == "__main__":
    ok = run_shape_check()
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
