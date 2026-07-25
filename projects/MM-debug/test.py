import torch
import torch_npu
import numpy as np
import random
import omni_custom_ops
torch.npu.set_device(5)
torch_npu.npu.config.allow_internal_format = True

# ========== 固定随机数种子 ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed(SEED)
torch.npu.manual_seed_all(SEED)

M, K, N = 128, 8192, 128

# 同一份基准 a/b（cpu fp32），bf16 / fp32 共用
a_base = torch.randn(M, K)
b_base = torch.randn(K, N)


def show_diff(tag, out, golden):
    d = (out - golden).abs()
    print("  %-24s max=%.4f  mean=%.4f  超差(>0.05)=%d/%d" %
          (tag, d.max().item(), d.mean().item(), int((d > 0.05).sum().item()), M * N))


def run(dtype, tag):
    print("\n================= %s =================" % tag)
    a = a_base.to(dtype).npu()
    b = b_base.to(dtype).npu()
    b_nz = torch_npu.npu_format_cast(b, 29)  # ND -> NZ
    golden = torch.matmul(a.cpu().float(), b.cpu().float())

    # custom op：cube=0(KEEP_DTYPE 原生) / cube=1(ALLOW_FP32_DOWN_P，910B 仍走 FP32/HF32) /
    #            cube=2(USE_FP16，把权重 cast 成 fp16 → 走 c0=16 分支)
    for cube in (0, 1, 2):
        try:
            o = torch.ops.custom.npu_ai_infra_matmul(a, b_nz, cube_math_type=cube).cpu().float()
            show_diff("custom op cube=%d" % cube, o, golden)
        except Exception as e:
            print("  custom op cube=%d          ERROR: %s" % (cube, str(e)[:110]))

    # torch.matmul(a, b_nz)：fp32 因 is_nz_dtype_valid=False 不走 WeightNz，回退 acl_op（参考正确值）
    try:
        om = torch.matmul(a, b_nz).cpu().float()
        show_diff("torch.matmul(a,b_nz)", om, golden)
    except Exception as e:
        print("  torch.matmul              ERROR: %s" % str(e)[:110])


run(torch.bfloat16, "bf16 (对照 · 已知正确)")
run(torch.float32,  "fp32 (问题组)")

# 预期（910B / DAV_2201）:
#   bf16 : cube=0/1/2、torch.matmul 都 max≈1(bf16 精度，对)
#   fp32 : cube=0 -> max≈543(垃圾, 原生 fp32 kernel 按 c0=8 读 c0=16 权重)
#          cube=1 -> max≈543(垃圾, 910B 表里 ALLOW_FP32_DOWN_P 仍是 FP32，不 cast fp16)
#          cube=2 -> max≈1  (对, 权重 cast fp16 → c0=16, 与 npu_format_cast(29) 对齐)
#          torch.matmul -> 对(fp32 走 acl_op 回退，绕开 WeightNz)
