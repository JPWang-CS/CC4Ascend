"""npu_quant_matmul (W8A8) batch 一致性自检 —— WeightNZ 版本。

与 check_batch_consistency.py 相同逻辑，但权重使用 NZ 格式 (npu_trans_quant_param)。
适用于线上实际走 WeightNZ 路径的场景。
"""
import hashlib

import torch
import torch_npu

K = 1536
N = 768
M_LIST = [113, 127, 128, 129, 256, 257, 512, 511, 1024, 1023, 4735, 9457]
SUFFIX_M = 113
OUTER_LIST = [127, 128, 256, 512, 1024, 4735, 9457]
ACT_SEED = 100
WEIGHT_SEED = 200
PERM_SEED = 300
NPU_DEVICE = 5
DTYPE = torch.bfloat16
WORST_N = 5
torch.npu.set_device(NPU_DEVICE)


def make_activation_cpu(m, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, k, generator=g, dtype=torch.float64) * 8.0
    amax = x.abs().amax(dim=1).clamp_min(1e-8)
    pertoken_scale = (amax / 127.0).to(torch.float32)
    x_int8 = torch.round(x / amax.unsqueeze(1) * 127.0).clamp(-127.0, 127.0).to(torch.int8)
    return x_int8, pertoken_scale


def make_weight_nz(k, n, seed):
    """生成权重并用 npu_format_cast 转为 FRACTAL_NZ(29) 格式。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(k, n, generator=g, dtype=torch.float64) * 0.1
    amax = w.abs().amax(dim=0).clamp_min(1e-8)
    scale = (amax / 127.0).to(DTYPE)
    w_int8 = torch.round(w / amax * 127.0).clamp(-127.0, 127.0).to(torch.int8)
    # npu_format_cast(tensor, 29) 将 ND 格式转为 FRACTAL_NZ
    w_nz = torch_npu.npu_format_cast(w_int8.npu(), 29)
    return w_nz, scale


def call_op(x_int8, pertoken_scale, w_nz, w_scale):
    y = torch_npu.npu_quant_matmul(
        x1=x_int8.npu(),
        x2=w_nz,
        scale=w_scale.npu(),
        pertoken_scale=pertoken_scale.npu(),
        bias=None,
        output_dtype=DTYPE,
    )
    torch.npu.synchronize()
    return y


def md5_hex(t):
    raw = t.contiguous().cpu().view(torch.int32).numpy().tobytes()
    return hashlib.md5(raw).hexdigest()


def diff_stats(a, b):
    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return diff.max().item(), diff.max(dim=1).values


def worst_rows(row_diff, n):
    n = min(n, row_diff.numel())
    vals, idx = row_diff.topk(n)
    return list(zip(idx.tolist(), vals.tolist()))


def main():
    w_nz, w_scale = make_weight_nz(K, N, WEIGHT_SEED)
    print(f"权重(NZ格式): scale {tuple(w_scale.shape)}  "
          f"w_scale_sum={w_scale.to(torch.float32).sum().item():.4e}", flush=True)
    print(f"shape: x=[M,{K}] × weight_nz[{K},{N}]，M ∈ {M_LIST}", flush=True)
    print(f"重点复现日志错误 shape: x=[113,{K}] vs x=[9457,{K}]，weight=[{K},{N}]", flush=True)
    print(f"日志关注 shape(qbmm_batch_nz): x.shape ∈ {[ (m, K) for m in M_LIST ]}，weight.shape=({K}, {N})", flush=True)
    print("对比方式: 两次 NPU 前向结果按打乱规律对齐后，对 raw bytes 取 MD5 直接比哈希。\n", flush=True)

    results = []

    print("=" * 96)
    print("【A】行置换一致性 —— 同一输入打乱行顺序跑两次 NPU，输出(按打乱规律对齐)MD5 必须相同")
    print("=" * 96, flush=True)
    for m in M_LIST:
        x_int8, pertoken_scale = make_activation_cpu(m, K, ACT_SEED)
        print(f"  M={m:<5} 运行...", end=" ", flush=True)
        y = call_op(x_int8, pertoken_scale, w_nz, w_scale).cpu()
        g = torch.Generator(device="cpu").manual_seed(PERM_SEED)
        perm = torch.randperm(m, generator=g)
        y_p = call_op(x_int8[perm], pertoken_scale[perm], w_nz, w_scale).cpu()
        h_orig = md5_hex(y[perm])
        h_perm = md5_hex(y_p)
        ok = (h_orig == h_perm)
        results.append(("A", m, ok))
        if ok:
            print(f"通过(MD5一致)  md5={h_orig}", flush=True)
        else:
            max_abs, row_diff = diff_stats(y_p, y[perm])
            print(f"不通过(MD5不一致)  max_abs={max_abs:.3e}", flush=True)
            print(f"        md5原序对齐={h_orig}", flush=True)
            print(f"        md5打乱    ={h_perm}", flush=True)
            for r, v in worst_rows(row_diff, WORST_N):
                print(f"        置换后行={r:<5} 原始行={int(perm[r]):<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【B】M 后缀一致性 —— 同一组 127 行，单独跑 vs 塞进更大 M 末尾跑，末尾 127 行 MD5 必须相同")
    print("=" * 96, flush=True)
    d_x, d_ps = make_activation_cpu(SUFFIX_M, K, ACT_SEED + 1)
    y_small = call_op(d_x, d_ps, w_nz, w_scale).cpu()
    for outer in OUTER_LIST:
        if outer < SUFFIX_M:
            continue
        print(f"  OUTER={outer:<5} 运行...", end=" ", flush=True)
        filler_x, filler_ps = make_activation_cpu(outer - SUFFIX_M, K, ACT_SEED + outer)
        big_x = torch.cat([filler_x, d_x], dim=0)
        big_ps = torch.cat([filler_ps, d_ps], dim=0)
        y_big = call_op(big_x, big_ps, w_nz, w_scale).cpu()
        h_small = md5_hex(y_small)
        h_suffix = md5_hex(y_big[-SUFFIX_M:])
        ok = (h_small == h_suffix)
        results.append(("B", outer, ok))
        if ok:
            print(f"通过(MD5一致)  md5={h_small}", flush=True)
        else:
            max_abs, row_diff = diff_stats(y_big[-SUFFIX_M:], y_small)
            print(f"不通过(MD5不一致)  max_abs={max_abs:.3e}", flush=True)
            print(f"        md5单独127行   ={h_small}", flush=True)
            print(f"        md5末尾127行   ={h_suffix}", flush=True)
            for r, v in worst_rows(row_diff, WORST_N):
                print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项复现 —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", "113_vs_9457_suffix113", sp_ok))
    if sp_ok:
        print(f"  通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"  不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项复现 —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", 9457, sp_ok))
    if sp_ok:
        print(f"  通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"  不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项复现 —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", 9457, sp_ok))
    if sp_ok:
        print(f"通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项复现 —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", 9457, sp_ok))
    if sp_ok:
        print(f"通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项复现 —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", 9457, sp_ok))
    if sp_ok:
        print(f"通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    print("\n" + "=" * 96, flush=True)
    print("【C】日志专项 case —— 只测 113 vs 9457 后缀 113 行")
    print("    对标日志: x=[113,1536] vs x=[9457,1536]，比较 9457 的末尾 113 行与单独 113 行是否逐字节一致")
    print("=" * 96, flush=True)
    sp_x, sp_ps = make_activation_cpu(113, K, ACT_SEED + 113)
    sp_y_small = call_op(sp_x, sp_ps, w_nz, w_scale).cpu()
    filler_x, filler_ps = make_activation_cpu(9457 - 113, K, ACT_SEED + 9457)
    sp_big_x = torch.cat([filler_x, sp_x], dim=0)
    sp_big_ps = torch.cat([filler_ps, sp_ps], dim=0)
    sp_y_big = call_op(sp_big_x, sp_big_ps, w_nz, w_scale).cpu()
    sp_h_small = md5_hex(sp_y_small)
    sp_h_suffix = md5_hex(sp_y_big[-113:])
    sp_ok = (sp_h_small == sp_h_suffix)
    results.append(("C", 9457, sp_ok))
    if sp_ok:
        print(f"通过(MD5一致)  md5={sp_h_small}", flush=True)
    else:
        sp_max_abs, sp_row_diff = diff_stats(sp_y_big[-113:], sp_y_small)
        print(f"不通过(MD5不一致)  max_abs={sp_max_abs:.3e}", flush=True)
        print(f"        md5单独113行   ={sp_h_small}", flush=True)
        print(f"        md5末尾113行   ={sp_h_suffix}", flush=True)
        for r, v in worst_rows(sp_row_diff, WORST_N):
            print(f"        后缀行={r:<5} 该行最大绝对差={v:.3e}")

    n_pass = sum(1 for _, _, ok in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 96, flush=True)
    print("【汇总】")
    print(f"  batch 一致性(WeightNZ): 共 {len(results)} 项 → 通过 {n_pass}，不通过 {n_fail}")
    if n_fail == 0:
        print("  结论: WeightNZ 路径下，打乱 batch / 改变 M 后两次 NPU 输出 MD5 完全相同。")
    else:
        print("  结论: WeightNZ 路径下出现 MD5 不一致 —— batch 不一致 bug 存在。")
        for tag, size, ok in results:
            if not ok:
                print(f"  -> 不通过: [{tag}] size={size}")


if __name__ == "__main__":
    main()
