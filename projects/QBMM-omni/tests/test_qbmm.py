#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# QBMM-omni 当前状态探针 (test_qbmm.py) —— 验证 V3 各分支能否"正确进入"算子。
# 不是数值验证 (数值正确性交 golden.py); 只验: wheel→op 注册→call 分派→aclnn 命中→出张量。
# 每个分支构造最小合法入参 (按 aclnn V5/torch 文档约束), 调一次, 报 ENTRY OK/FAIL + 异常定位。
#
# 用法: python test_qbmm.py
# 前置: torch_ops_extension wheel 已编+装 (S8 完成后)。本仓未装时探针在 ①② 步即报 "需先编 wheel"。
#
# 探针分 3 级:
#   ① wheel 装好没 (import omni_custom_ops)
#   ② op 注册没 (torch.ops.custom.npu_ai_infra_quant_matmul)
#   ③ 每个分支能否进入 (call 不抛 = ENTRY OK; 抛 = 按异常信息定位卡在哪)
#      异常分类:
#        - "not registered" / "could not find"  → schema/binding 没加载 (wheel 问题)
#        - "undefined symbol"                   → binding 链接缺符号 (op_api .so 没装/没编)
#        - "EZ00\d+" / "161002" / "ACLNN_ERR"   → op checker 拒绝 (入参违反约束, 看具体码)
#        - "TransQuantParam"                    → aclnnTransQuantParamV3 缺 (CANN 自带应不缺)
#        - "QuantMatmulV5" / "WeightNz" 缺      → V5/WeightNz .so 没编 (P2/S8 未完成)
#        - 其它                                  → 贴完整 traceback

import sys
import traceback

# ---- 探针的 case 子集: V3 每分支一个代表 (覆盖 mode × ND/NZ/transpose 入口路径) ----
# 复用 golden.py 的 Case/gen_data/call_npu (保持与全量验证同源, 不重复构造逻辑)
from golden import Case, gen_data, call_npu  # noqa: E402

ENTRY_PROBE_CASES = [
    # ---- V3 核心 5 模式 + int4sym (ND) ----
    Case("probe_pertensor_dyn",  "pertensor",  64, 64, 128, out_dtype="float16", scale_dtype="float32"),
    Case("probe_pertensor_stat", "pertensor",  64, 64, 128, out_dtype="float16", scale_dtype="int64"),
    Case("probe_perchannel",     "perchannel", 64, 64, 128, out_dtype="float16"),
    Case("probe_pertoken",       "pertoken",   64, 64, 128, out_dtype="bfloat16"),
    Case("probe_requant",        "requant",    64, 64, 128, out_dtype="int8"),
    Case("probe_int32",          "int32",      64, 64, 128, out_dtype="int32"),
    Case("probe_int4sym",        "int4sym",    64, 64, 128, out_dtype="float16"),   # K,N%8==0
    # ---- NZ 入口 (走 aclnnAiInfraQuantMatmulWeightNz) — V3 每模式 × NZ 各一, 与 ND 对称 ----
    Case("probe_pertensor_dyn_nz", "pertensor",  64, 64, 128, out_dtype="float16", weight_nz=True),
    Case("probe_pertoken_nz",      "pertoken",   64, 64, 128, out_dtype="bfloat16", weight_nz=True),
    Case("probe_requant_nz",       "requant",    64, 64, 128, out_dtype="int8",    weight_nz=True),
    Case("probe_int32_nz",         "int32",      64, 64, 128, out_dtype="int32",   weight_nz=True),
    Case("probe_int4sym_nz",       "int4sym",    64, 64, 128, out_dtype="float16", weight_nz=True),
    Case("probe_perchannel_nz",    "perchannel", 64, 64, 128, out_dtype="float16", weight_nz=True),
    # ---- trans_quant_param 入口 (fp32 scale + 无 pertoken + out≠bf16/int32 → 走 aclnnTransQuantParamV3) ----
    # 已由 probe_pertensor_dyn 覆盖 (fp32 scale pertensor)
    # ---- transpose 入口 (binding transpose1/2 恒 false, 靠 stride 焙进去) ----
    Case("probe_perchannel_tx2", "perchannel", 64, 64, 128, out_dtype="float16", trans_x2=True),
]


def _classify_exc(e: Exception):
    """归类入口失败点。返回 (tag, hint)。
    ⚠ 报错信息里出现算子名 (如 aclnnAiInfraQuantMatmulV5) 不等于 .so 缺失 —— checker 拒绝时也会带算子名。
    靠更具体的信号区分; 真因一律以 raw 报错文本为准 (主循环会打全)。"""
    import re
    msg = repr(e)
    # 符号/so 加载层 (op .so 没装 / 符号未解析 / 名字不匹配 binding 调 AiInfra 但装的还是旧名)
    if "undefined symbol" in msg or "cannot open shared object" in msg or "dlopen" in msg \
            or "symbol lookup error" in msg:
        return "SYMBOL_LOAD_FAIL", "op .so 符号未解析/未装/名字不匹配 (查 .run 是否改名后重编重装)"
    # aclnn 框架层找不到算子 (kernel 未注册)
    if "GetAclnnKernel" in msg or "not registered" in msg or "not found in" in msg \
            or "not support" in msg or "UnregisteredOp" in msg:
        return "KERNEL_NOT_FOUND", "aclnn 框架找不到算子 (.run 没装/名字不匹配/stale .so)"
    # checker 拒绝 (入参违反约束: dtype/shape/format)
    if re.search(r"EZ\d{3,4}", msg) or "161001" in msg or "161002" in msg \
            or "ACLNN_ERR_PARAM" in msg:
        return "CHECKER_REJECT", "op checker 拒绝入参 (dtype/shape/format 违反约束; 看 EZ 码定位)"
    # trans_quant_param 子算子
    if "TransQuantParam" in msg:
        return "TRANS_QUANT_PARAM_FAIL", "aclnnTransQuantParamV3 失败 (CANN 自带; 查版本/fp32 scale 入参)"
    return "OTHER", "未分类 — 看完整 raw 报错"


def _probe_one(c: Case) -> dict:
    """构造最小入参, 调一次 op, 报入口状态 + 完整 raw 报错 (不截断, 真因以 raw 为准)。"""
    try:
        d = gen_data(c)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return dict(name=c.name, status="GEN_FAIL", tag="DATA_GEN", raw=repr(e), tb=tb)
    try:
        out = call_npu(d, c)
    except Exception as e:
        tag, _ = _classify_exc(e)
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return dict(name=c.name, status="ENTRY_FAIL", tag=tag, raw=repr(e), tb=tb)
    return dict(name=c.name, status="ENTRY_OK", tag="-",
                raw=f"shape={tuple(out.shape)} dtype={out.dtype} "
                    f"sample={out.flatten()[:3].cpu().tolist()}", tb="")


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(1)
    # NZ 探针依赖 npu_format_cast(x2, FRACTAL_NZ); 不开则 cast 被拒、x2 退回 base(ND),
    # binding is_nz_format 返回 false → 误走 V5 分支
    torch.npu.config.allow_internal_format = True

    print("=" * 60)
    print("  QBMM-omni V3 入口探针 (正确进入验证; 非数值)")
    print("=" * 60)

    # ---- ① wheel ----
    print("\n[① wheel]")
    try:
        import omni_custom_ops  # noqa: F401  (改成本仓 wheel 真实模块名)
        print("  OK  import omni_custom_ops")
    except Exception as e:
        print(f"  [FATAL] import omni_custom_ops 失败: {e!r}")
        print("          → wheel 没编/没装。先完成 S8 (build.sh -c ascend910b) + 装 wheel。")
        return 1

    # ---- ② op 注册 ----
    print("\n[② op 注册]")
    OP = "npu_ai_infra_quant_matmul"
    registered = hasattr(torch.ops.custom, OP)
    print(f"  {'OK ' if registered else 'FAIL'}  torch.ops.custom.{OP} "
          f"{'registered' if registered else 'NOT registered'}")
    if not registered:
        print(f"          → m.def 没生效 (查 csrc_base/ops_def_registration.cpp 是否含 {OP})")
        return 1

    # ---- ③ 每分支入口 ----
    print(f"\n[③ 分支入口] ({len(ENTRY_PROBE_CASES)} 条入口路径: ND {sum(1 for c in ENTRY_PROBE_CASES if not c.weight_nz)} + NZ {sum(1 for c in ENTRY_PROBE_CASES if c.weight_nz)})")
    results = [_probe_one(c) for c in ENTRY_PROBE_CASES]
    name_w = max(len(r["name"]) for r in results)
    print(f"  {'name':<{name_w}}  {'status':>10}  {'tag':>20}  raw (截断 120 字符; 完整见下方)")
    print("  " + "-" * (name_w + 100))
    n_ok = 0
    for r in results:
        if r["status"] == "ENTRY_OK":
            n_ok += 1
        raw_short = r["raw"][:120].replace("\n", " ")
        print(f"  {r['name']:<{name_w}}  {r['status']:>10}  {r['tag']:>20}  {raw_short}")

    # 失败的逐条打完整 raw 报错 (不截断 — 真因以 raw 为准); 首条加完整 traceback
    fails = [r for r in results if r["status"] != "ENTRY_OK"]
    if fails:
        print(f"\n  ---- 失败用例完整报错 ({len(fails)} 条) ----")
        for i, r in enumerate(fails):
            print(f"\n  [{r['name']}] tag={r['tag']}")
            print(f"  raw: {r['raw']}")
            if i == 0 and r["tb"]:
                print(f"  traceback (首条):\n{r['tb']}")

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    print(f"  入口汇总: {n_ok}/{len(results)} ENTRY_OK")
    fails = [r for r in results if r["status"] != "ENTRY_OK"]
    if not fails:
        print("  全部分支正确进入 — 数值正确性验证用 golden.py")
        return 0
    # 按失败标签分桶
    from collections import Counter
    buckets = Counter(r["tag"] for r in fails)
    print(f"  失败分桶: {dict(buckets)}")
    blocking = [t for t in buckets if t in ("SYMBOL_LOAD_FAIL", "KERNEL_NOT_FOUND",
                                            "TRANS_QUANT_PARAM_FAIL")]
    if blocking:
        print(f"  ★ 阻塞性失败 (.so/符号/子算子未就绪): {blocking} — 多半 .run 没改名后重编重装")
    checker = [t for t in buckets if t == "CHECKER_REJECT"]
    if checker:
        print(f"  ▲ checker 拒绝 (op 进去了但入参违章): {checker} — 看 EZ 码 + raw 报错定位约束")
    return 0 if n_ok == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
