"""
nn 仓 matmul fp32+NZ 在 ACLGraph（aclnn 图模式）下能不能跑通。
验证：capture + replay，看通不通、结果对不对。
对照 allow_internal_format True/False（报错提示设 False 可能让 op-plugin 走 aclnn ND）。

板上运行：
  python test_nn_matmul_fp32nz_aclgraph.py
"""

import torch
import torch_npu
import numpy as np

torch.npu.set_device(1)


def run_case(allow_internal_format, label):
    torch.npu.config.allow_internal_format = allow_internal_format
    print(f"\n===== {label} (allow_internal_format={allow_internal_format}) =====")
    M, K, N = 128, 256, 64

    a = torch.randn(M, K, dtype=torch.float32).npu()
    b_nd = torch.randn(K, N, dtype=torch.float32).npu()
    b_nz = torch_npu.npu_format_cast(b_nd, 29)  # 29 = ACL_FORMAT_FRACTAL_NZ
    golden = (a.cpu().float() @ b_nd.cpu().float()).numpy()

    # --- eager ---
    print("--- eager fp32+NZ ---")
    try:
        out = torch.matmul(a, b_nz)
        err = np.max(np.abs(out.cpu().numpy() - golden))
        print(f"PASS max_err={err:.6e}")
    except Exception as e:
        print(f"FAIL {e}")

    # --- ACLGraph ---
    print("--- ACLGraph fp32+NZ ---")
    try:
        a_g = a.clone()
        b_g = b_nz.clone()

        class MM(torch.nn.Module):
            def forward(self, x, w):
                return torch.matmul(x, w)

        model = MM().npu()
        g = torch_npu.npu.NPUGraph()
        with torch.npu.graph(g):
            out = model(a_g, b_g)
        g.replay()
        err = np.max(np.abs(out.cpu().numpy() - golden))
        print(f"PASS max_err={err:.6e}")
    except Exception as e:
        print(f"FAIL {e}")


def run():
    run_case(True, "默认（allow_internal_format=True）")
    run_case(False, "报错提示（allow_internal_format=False）")


if __name__ == "__main__":
    run()
