"""npu_quant_matmul (W8A8) batch 一致性自检 —— 两次 NPU 前向结果 MD5 直接对比。

同一份输入打乱行顺序（或把同一组 127 行塞进更大 M）后，再跑一次 NPU；
两次 NPU 输出按相同规律对齐后，对各自 raw bytes 取 MD5，PASS 当且仅当两个 MD5 完全相同。
W8A8 是 int8×int8→int32 精确整数累加，逐行确定，行顺序 / 周围 M 不应改变任何输出字节。
全程 NPU-vs-NPU，不引入 CPU golden。日志报错 shape: x=[M,1536] × weight[1536,768]，M∈{127,4735}。
"""
import hashlib

import torch
import torch_npu

K = 1536
N = 768
M_LIST = [127, 128, 129, 256, 257, 512, 511, 1024, 1023, 4735]
SUFFIX_M = 127
OUTER_LIST = [128, 256, 512, 1024, 4735]
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
    amax = x.abs().amax(dim=1).clamp_min(1e-8)  # [M] 1D
    pertoken_scale = (amax / 127.0).to(torch.float32)  # [M] 1D fp32：aclnnQuantMatmulV5 要求 x1Scale=DT_FLOAT 且非 G-B/B-B 须 1D
    x_int8 = torch.round(x / amax.unsqueeze(1) * 127.0).clamp(-127.0, 127.0).to(torch.int8)
    return x_int8, pertoken_scale


def make_weight_cpu(k, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(k, n, generator=g, dtype=torch.float64) * 0.1
    amax = w.abs().amax(dim=0).clamp_min(1e-8)
    scale = (amax / 127.0).to(DTYPE)
    w_int8 = torch.round(w / amax * 127.0).clamp(-127.0, 127.0).to(torch.int8)
    return w_int8, scale


def call_op(x_int8, pertoken_scale, w_int8, w_scale):
    y = torch_npu.npu_quant_matmul(
        x1=x_int8.npu(),
        x2=w_int8.npu(),
        scale=w_scale.npu(),
        pertoken_scale=pertoken_scale.npu(),
        bias=None,
        output_dtype=DTYPE,
    )
    torch.npu.synchronize()
    return y


def md5_hex(t):
    raw = t.contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()


def diff_stats(a, b):
    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return diff.max().item(), diff.max(dim=1).values


def worst_rows(row_diff, n):
    n = min(n, row_diff.numel())
    vals, idx = row_diff.topk(n)
    return list(zip(idx.tolist(), vals.tolist()))


def main():
    w_int8, w_scale = make_weight_cpu(K, N, WEIGHT_SEED)
    print(f"权重: int8 {tuple(w_int8.shape)} scale {tuple(w_scale.shape)}  "
          f"w_int8_sum={w_int8.to(torch.int32).sum().item()}  "
          f"w_scale_sum={w_scale.to(torch.float32).sum().item():.4e}", flush=True)
    print(f"日志报错 shape: x=[M,{K}] × weight[{K},{N}]，M ∈ (127, 4735)", flush=True)
    print("对比方式: 两次 NPU 前向结果按打乱规律对齐后，对 raw bytes 取 MD5 直接比哈希；不引入 CPU golden。\n", flush=True)

    results = []

    print("=" * 96)
    print("【A】行置换一致性 —— 同一输入打乱行顺序跑两次 NPU，输出(按打乱规律对齐)MD5 必须相同")
    print("    原理: W8A8 int8×int8→int32 精确累加，逐行确定；行顺序改变不应改变任何输出字节")
    print("=" * 96, flush=True)
    for m in M_LIST:
        x_int8, pertoken_scale = make_activation_cpu(m, K, ACT_SEED)
        print(f"  M={m:<5} 运行...", end=" ", flush=True)
        y = call_op(x_int8, pertoken_scale, w_int8, w_scale).cpu()
        g = torch.Generator(device="cpu").manual_seed(PERM_SEED)
        perm = torch.randperm(m, generator=g)
        y_p = call_op(x_int8[perm], pertoken_scale[perm], w_int8, w_scale).cpu()
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
    print("    (OUTER=4735 即复刻日志报错场景: 4735 末尾 127 行 vs 单独 127 行)")
    print("=" * 96, flush=True)
    d_x, d_ps = make_activation_cpu(SUFFIX_M, K, ACT_SEED + 1)
    y_small = call_op(d_x, d_ps, w_int8, w_scale).cpu()
    for outer in OUTER_LIST:
        if outer < SUFFIX_M:
            continue
        print(f"  OUTER={outer:<5} 运行...", end=" ", flush=True)
        filler_x, filler_ps = make_activation_cpu(outer - SUFFIX_M, K, ACT_SEED + outer)
        big_x = torch.cat([filler_x, d_x], dim=0)
        big_ps = torch.cat([filler_ps, d_ps], dim=0)
        y_big = call_op(big_x, big_ps, w_int8, w_scale).cpu()
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

    n_pass = sum(1 for _, _, ok in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 96, flush=True)
    print("【汇总】")
    print(f"  batch 一致性: 共 {len(results)} 项 → 通过 {n_pass}，不通过 {n_fail}")
    if n_fail == 0:
        print("  结论: 所测 shape 内，打乱 batch / 改变 M 后两次 NPU 输出 MD5 完全相同 —— 未复现 batch 不一致。")
    else:
        print("  结论: 出现 MD5 不一致 —— 两次 NPU 同数据输出非逐字节相同，坐实算子 batch 不一致(真 bug)，")
        print("        上面打印的行号/置换对应即定位 tiling 边界的入口。")
        for tag, size, ok in results:
            if not ok:
                print(f"  -> 不通过: [{tag}] size={size}")


if __name__ == "__main__":
    main()
