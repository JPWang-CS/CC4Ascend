#!/usr/bin/env python3
# A1 验证：CPU 上 torch.matmul(fp16, fp16) 实际用什么精度累加？
#   匹配 fp32 累加 → benchmark 会过精 → Ratio 失真 → 需手动 K-chunk fp16 累加 (C7)
#   匹配 fp16 累加 → benchmark 正确，可直接用
#
# 跑法：python verify_a1_matmul_accum.py
# 贴 stdout 给 Claude 判定。
import torch

torch.set_grad_enabled(False)

def max_rel_diff(a, b, eps=1e-7):
    return float(((a.float() - b.float()).abs() / (b.float().abs() + eps)).max())

def kchunk_fp16(a16, b16, k_chunk=16):
    # 手动 K-chunk fp16 累加：每块 fp16 matmul 后 .half() 再 fp16 累加
    M, K = a16.shape
    K2, N = b16.shape
    assert K == K2
    acc = torch.zeros(M, N, dtype=torch.float16)
    for i in range(0, K, k_chunk):
        a_blk = a16[:, i:i+k_chunk]
        b_blk = b16[i:i+k_chunk, :]
        partial = torch.matmul(a_blk, b_blk)  # fp16 in, fp16 out (小 K 误差小)
        acc = (acc.float() + partial.float()).half()  # fp16 累加域
    return acc

print(f"torch {torch.__version__}")
print(f"num_threads {torch.get_num_threads()}")
print("=" * 70)

for (M, K, N) in [(64, 256, 64), (128, 512, 128), (128, 1024, 128)]:
    torch.manual_seed(42)
    a16 = (torch.randn(M, K) * 0.1).half()
    b16 = (torch.randn(K, N) * 0.1).half()

    # 1) 默认 torch.matmul(fp16, fp16) —— benchmark 会用的
    default = torch.matmul(a16, b16)

    # 2) fp32 参考累加 (a.float @ b.float, 再 half) —— 已知最精
    fp32_ref = (a16.float() @ b16.float()).half()

    # 3) 手动 K-chunk fp16 累加 —— 强制 fp16 累加域
    kchunk = kchunk_fp16(a16, b16, k_chunk=16)

    d_default_vs_fp32 = max_rel_diff(default, fp32_ref)
    d_default_vs_kchunk = max_rel_diff(default, kchunk)
    d_kchunk_vs_fp32 = max_rel_diff(kchunk, fp32_ref)

    print(f"\n[shape M={M}, K={K}, N={N}]")
    print(f"  default vs fp32_ref : max_rel = {d_default_vs_fp32:.6e}")
    print(f"  default vs kchunk16 : max_rel = {d_default_vs_kchunk:.6e}")
    print(f"  kchunk16 vs fp32_ref : max_rel = {d_kchunk_vs_fp32:.6e}")

    if d_default_vs_fp32 < d_default_vs_kchunk:
        verdict = "default 更接近 fp32_ref → CPU torch.matmul(fp16) 用 fp32 累加 (A1 FAIL: benchmark 过精, 需 K-chunk)"
    else:
        verdict = "default 更接近 kchunk16 → CPU torch.matmul(fp16) 用 fp16 累加 (A1 PASS: benchmark 可直接用)"
    print(f"  判定: {verdict}")

print("\n" + "=" * 70)
print("bf16 同测:")
for (M, K, N) in [(128, 512, 128)]:
    torch.manual_seed(42)
    ab = (torch.randn(M, K) * 0.1).to(torch.bfloat16)
    bb = (torch.randn(K, N) * 0.1).to(torch.bfloat16)
    default_bf = torch.matmul(ab, bb)
    fp32_ref_bf = (ab.float() @ bb.float()).to(torch.bfloat16)
    print(f"  [bf16 M={M},K={K},N={N}] default vs fp32_ref max_rel = {max_rel_diff(default_bf, fp32_ref_bf):.6e}")
