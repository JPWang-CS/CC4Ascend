#!/usr/bin/env python3
# A4 验证：NPU 上 torch_npu 怎么做 int8 matmul？是否真走 int8×int8→int32 L0C 累加？
#   匹配 CPU int32 累加（bitwise 或近似）→ benchmark 可用该 op，方案成立
#   不匹配 / upcast fp32 → benchmark 过精，需找专用 int8 matmul op 或退 CPU 拼接
#
# 跑法：python verify_a4_npu_int8_matmul.py
# 需 torch_npu + NPU 设备可用。贴 stdout 给 Claude。
import torch
import numpy as np

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except Exception as e:
    HAS_NPU = False
    print(f"torch_npu 不可用: {e}")
    print("A4 无法在本地验，需在 build 机（有 NPU）跑")
    raise SystemExit(0)

torch.set_grad_enabled(False)
DEV = "npu"
print(f"torch {torch.__version__}, torch_npu 可用, device={DEV}")
print("=" * 70)

for (M, K, N) in [(64, 128, 64), (128, 256, 128)]:
    torch.manual_seed(42)
    # int8 输入（-128..127），控制范围避免溢出
    a_i8 = (torch.randint(-16, 16, (M, K))).to(torch.int8)
    b_i8 = (torch.randint(-16, 16, (K, N))).to(torch.int8)

    # CPU int32 参考累加（精确整数算术）
    ref_i32 = torch.matmul(a_i8.to(torch.int32), b_i8.to(torch.int32))  # int32 精确

    print(f"\n[shape M={M}, K={K}, N={N}]  ref_i32 dtype={ref_i32.dtype}, range=[{ref_i32.min()},{ref_i32.max()}]")

    # 试法 1: torch.matmul(int8, int8) 直接在 NPU
    try:
        a_n = a_i8.to(DEV); b_n = b_i8.to(DEV)
        r1 = torch.matmul(a_n, b_n)
        diff1 = (r1.cpu().to(torch.int32) - ref_i32).abs().max().item()
        print(f"  试法1 torch.matmul(int8.npu, int8.npu): OK, out dtype={r1.dtype}, vs ref_i32 max_diff={diff1}")
        if r1.dtype == torch.int32 and diff1 == 0:
            print(f"    → ✅ NPU int8 matmul 走 int8→int32 累加, bitwise 精确 (A4 PASS)")
        elif diff1 == 0:
            print(f"    → ✅ 结果精确匹配 int32, dtype={r1.dtype} (A4 PASS)")
        else:
            print(f"    → ⚠️ 结果与 int32 ref 不一致 (diff={diff1}), 可能 upcast 或其他路径")
    except Exception as e:
        print(f"  试法1 torch.matmul(int8.npu, int8.npu): FAIL — {e}")

    # 试法 2: int8 → int32 后 NPU matmul
    try:
        a_n = a_i8.to(torch.int32).to(DEV); b_n = b_i8.to(torch.int32).to(DEV)
        r2 = torch.matmul(a_n, b_n)
        diff2 = (r2.cpu().to(torch.int32) - ref_i32).abs().max().item()
        print(f"  试法2 torch.matmul(int32.npu, int32.npu): OK, out dtype={r2.dtype}, vs ref_i32 max_diff={diff2}")
    except Exception as e:
        print(f"  试法2 torch.matmul(int32.npu, int32.npu): FAIL — {e}")

    # 试法 3: int8 → fp32 NPU matmul (已知会 upcast, 作对照)
    try:
        a_n = a_i8.to(torch.float32).to(DEV); b_n = b_i8.to(torch.float32).to(DEV)
        r3 = torch.matmul(a_n, b_n)
        diff3 = (r3.cpu().to(torch.int32) - ref_i32.to(torch.float32)).abs().max().item()
        print(f"  试法3 torch.matmul(fp32.npu, fp32.npu) [对照]: OK, vs ref_i32 max_diff={diff3:.1f} (fp32 精确表示 int8→同)")
    except Exception as e:
        print(f"  试法3 torch.matmul(fp32.npu, fp32.npu): FAIL — {e}")

    # 试法 4: 有没有 torch_npu 专用 int8/quant matmul op
    quant_ops = [x for x in dir(torch_npu) if "quant" in x.lower() or "int8" in x.lower() or "matmul" in x.lower()]
    print(f"  试法4 torch_npu 里 quant/int8/matmul 相关: {quant_ops[:15]}")

print("\n" + "=" * 70)
print("判定: 试法1 若 dtype=int32 且 diff=0 → A4 PASS, benchmark 直接用 torch.matmul(int8.npu)")
print("      试法1 FAIL/upcast → 看试法4 有无专用 op, 或退 CPU int32 拼接")
