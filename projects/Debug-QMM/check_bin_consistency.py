"""npu_quant_matmul (W8A8) batch 一致性自检 —— 对比框架输出 vs 我们重跑的输出。

目录结构:
  dumped_kwargs/          -- APC 命中 (M=127)
    x.pt, x_scale.pt, layer.weight.pt, layer.weight_scale.pt, y.pt, metadata.json
  dumped_kwargs_no_APC/   -- 完整 batch (M=4735)
    x.pt, x_scale.pt, layer.weight.pt, layer.weight_scale.pt, y.pt, metadata.json

检测项:
  【A】框架输出 batch 一致性: 框架的 APC y vs 框架的 no_APC y 末尾 127 行
  【B】我们重跑 batch 一致性: 我们跑 APC vs 我们跑 no_APC 末尾 127 行
  【C】框架 vs 我们: APC 框架输出 vs 我们重跑输出
  【D】框架 vs 我们: no_APC 框架输出 vs 我们重跑输出
"""
import hashlib
import json
import os
import sys

import torch
import torch_npu

WORST_N = 10
DUMP_DIR = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.bfloat16


def load_dump(subdir):
    path = os.path.join(DUMP_DIR, subdir)
    with open(os.path.join(path, "metadata.json"), "r") as f:
        meta = json.load(f)
    x = torch.load(os.path.join(path, "x.pt"), map_location="cpu")
    x_scale = torch.load(os.path.join(path, "x_scale.pt"), map_location="cpu")
    w = torch.load(os.path.join(path, "layer.weight.pt"), map_location="cpu")
    w_scale = torch.load(os.path.join(path, "layer.weight_scale.pt"), map_location="cpu")
    y = torch.load(os.path.join(path, "y.pt"), map_location="cpu")
    return x, x_scale, w, w_scale, y, meta


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


def compare(a, b, label_a, label_b):
    """对比两个 tensor，打印 MD5 和差异统计。"""
    h_a = md5_hex(a)
    h_b = md5_hex(b)
    match = (h_a == h_b)

    print(f"  {label_a} md5 = {h_a}")
    print(f"  {label_b} md5 = {h_b}")

    if match:
        print(f"  → MD5 一致 ✓")
    else:
        diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        nonzero = (diff > 0).sum().item()
        total = diff.numel()
        row_max = diff.max(dim=1).values
        n = min(WORST_N, row_max.numel())
        vals, idx = row_max.topk(n)

        nz = (diff > 0).nonzero(as_tuple=False)
        first = nz[0]
        row = first[0].item()
        col = first[1].item()
        va = a[row, col].to(torch.float32).item()
        vb = b[row, col].to(torch.float32).item()

        print(f"  → MD5 不一致!")
        print(f"    max_abs_diff={max_diff:.6e}  mean_abs_diff={mean_diff:.6e}")
        print(f"    不一致元素: {nonzero}/{total} ({100*nonzero/total:.2f}%)")
        print(f"    首个不一致位置: row={row}, col={col}, {label_a}={va:.9e}, {label_b}={vb:.9e}, diff={abs(va-vb):.9e}")
        print(f"    差异最大的行:")
        for i in range(n):
            print(f"      行={idx[i].item():<5} max_diff={vals[i].item():.6e}")

    print()
    return match


def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.npu.set_device(device_id)

    print(f"NPU device: {device_id}")
    print(f"Dump 目录: {DUMP_DIR}\n", flush=True)

    # 加载
    x_apc, xs_apc, w_apc, ws_apc, y_apc_fw, _ = load_dump("dumped_kwargs")
    x_full, xs_full, w_full, ws_full, y_full_fw, _ = load_dump("dumped_kwargs_no_APC")

    M_apc = x_apc.shape[0]
    M_full = x_full.shape[0]
    K = x_apc.shape[1]
    N = w_apc.shape[1]

    print(f"APC:    x={tuple(x_apc.shape)}  y={tuple(y_apc_fw.shape)}")
    print(f"no_APC: x={tuple(x_full.shape)}  y={tuple(y_full_fw.shape)}")
    print(f"weight: {tuple(w_apc.shape)}  w_scale={tuple(ws_apc.shape)}")
    print(f"M_apc={M_apc}, M_full={M_full}, K={K}, N={N}")
    print(f"新增线上日志关注 shape: qbmm_batch_nz x.shape in {{{M_apc}, {M_full}}}，当前为 [{M_apc},{K}] / [{M_full},{K}]，weight=[{K},{N}]\n", flush=True)

    # 检查输入一致性
    tail_x = x_full[-M_apc:]
    tail_xs = xs_full[-M_apc:]
    input_match = (tail_x.equal(x_apc) and tail_xs.equal(xs_apc))
    weight_match = (w_apc.equal(w_full) and ws_apc.equal(ws_full))
    print(f"no_APC 末尾 {M_apc} 行输入与 APC dump 一致: {input_match}")
    print(f"两组 dump 权重一致: {weight_match}\n", flush=True)

    # 我们重跑
    w_nz = torch_npu.npu_format_cast(w_apc.npu(), 29)

    print("重跑 npu_quant_matmul ...", flush=True)
    y_apc_ours = call_op(x_apc, xs_apc, w_nz, ws_apc).cpu()
    y_full_ours = call_op(x_full, xs_full, w_nz, ws_full).cpu()
    print("完成\n", flush=True)

    results = []

    # 【A】框架输出 batch 一致性
    print("=" * 96)
    print(f"【A】框架输出 batch 一致性: 框架 APC y vs 框架 no_APC y 末尾 {M_apc} 行")
    print("=" * 96)
    ok = compare(y_apc_fw, y_full_fw[-M_apc:], "框架 APC", f"框架 no_APC 末尾{M_apc}行")
    results.append(("A", "框架 batch 一致性", ok))

    # 【B】我们重跑 batch 一致性
    print("=" * 96)
    print(f"【B】我们重跑 batch 一致性: 我们 APC vs 我们 no_APC 末尾 {M_apc} 行")
    print("=" * 96)
    ok = compare(y_apc_ours, y_full_ours[-M_apc:], "我们 APC", f"我们 no_APC 末尾{M_apc}行")
    results.append(("B", "我们 batch 一致性", ok))

    # 【C】APC: 框架 vs 我们
    print("=" * 96)
    print(f"【C】APC (M={M_apc}): 框架输出 vs 我们重跑输出")
    print("=" * 96)
    ok = compare(y_apc_fw, y_apc_ours, "框架 APC", "我们 APC")
    results.append(("C", "APC 框架vs我们", ok))

    # 【D】no_APC: 框架 vs 我们
    print("=" * 96)
    print(f"【D】no_APC (M={M_full}): 框架输出 vs 我们重跑输出")
    print("=" * 96)
    ok = compare(y_full_fw, y_full_ours, "框架 no_APC", "我们 no_APC")
    results.append(("D", "no_APC 框架vs我们", ok))

    # 汇总
    n_pass = sum(1 for _, _, ok in results if ok)
    n_fail = len(results) - n_pass
    print("=" * 96)
    print("【汇总】")
    print(f"  共 {len(results)} 项 → 通过 {n_pass}，不通过 {n_fail}")
    for tag, desc, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {status}: {desc}")

    if n_fail > 0:
        print("\n【分析】")
        a_ok = results[0][2]
        b_ok = results[1][2]
        c_ok = results[2][2]
        d_ok = results[3][2]

        if a_ok and not (c_ok and d_ok):
            print("  框架自身 batch 一致，但和我们重跑不一致。")
            print("  可能原因: 框架调用 npu_quant_matmul 的参数/路径和我们不同 (见下方排查点)")
        elif not a_ok and b_ok:
            print("  框架 batch 不一致，我们重跑一致。")
            print("  说明算子本身无 bug，问题在框架调用层!")
        elif not a_ok and not b_ok:
            print("  两者都 batch 不一致 → 算子本身有 bug")
        elif a_ok and b_ok and c_ok and d_ok:
            print("  全部一致，无法复现问题。")

        print("\n【排查方向】")
        print("  若框架输出与我们不一致，可能原因:")
        print("  1. weight 在框架中不是走 npu_format_cast(..., 29) 而是其他 NZ 转换路径")
        print("  2. 框架传了 bias (我们传 None)")
        print("  3. 框架的 output_dtype 不是 bfloat16")
        print("  4. 框架对 x/x_scale 做了额外预处理 (如 padding、对齐)")
        print("  5. 框架的权重经过了 transpose (走 trans_x2=True 路径)")
        print("  6. 框架使用了 antiquant 路径而非 quant_matmul")
        print("  7. dump 的 y 不是 npu_quant_matmul 的直接输出 (中间经过了其他算子)")


if __name__ == "__main__":
    main()
