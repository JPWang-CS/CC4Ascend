# tiling 更新算子额外接口

源：wiki WIKI2026040910715297 §附。适用于 **host 侧 tiling 更新算子**（主要是 FIA 类非 AICPU 下沉算子）。

## 为什么需要

ACLGraph Capture 后 Replay 时，某些算子的 tiling 可能与 Capture 时不同（如 actual_seq_qlen 变化）。普通算子 Capture 时 tiling 固化即可；但 **tiling 会动态变化的算子**，须额外接口保证 Replay 时 tiling/workspace 正确、输出地址不变。

## 额外接口（FIA 类为例 `npu_fused_infer_attention_score_v2`）

除 meta 函数外，还要实现 3 个接口：

### 1. `.out` 接口（replay 实际执行的原地更新）
capture 后 replay 时走 `.out`，原地更新输出（输出地址在 capture 时已固定）：
```
npu_fused_infer_attention_score_v2.out(..., Tensor? workspace=None,
    Tensor(a!) attention_out, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))
```

### 2. `_get_max_workspace` 接口（capture 时获取最大 workspace）
保证后续 replay 时内存地址不变：
```
_npu_fused_infer_attention_score_v2_get_max_workspace(...) -> Tensor
```

### 3. `_infer_output` 接口（capture 时申请 .out 的输出）
```
_npu_fused_infer_attention_score_v2_infer_output(...) -> (Tensor, Tensor)
```

## npugraph_ex 注册 tiling 更新

增加上述接口后，向 ACLGraph 注册 tiling 更新。若经 npugraph_ex 完整后端入 ACLGraph，在 npugraph_ex 注册：

```python
from npugraph_ex._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
if hasattr(torch.ops.custom, "npu_fused_infer_attention_sink"):
    _REPLACE_FUNC_MAP.update({
        torch.ops.custom.npu_fused_infer_attention_sink.default: StaticWorkspaceReplaceFunc(
            get_workspace=torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace.default,
            out_operator=torch.ops.custom.npu_fused_infer_attention_sink.out,
            workspace_keys=["workspace"],
            output_keys=["attention_out", "softmax_lse"],
            updated_param_keys=["actual_seq_qlen", "actual_seq_kvlen"],  # 动态变化参数
        )
    })
```

## 哪些算子需要
- **host 侧 tiling 更新算子**（主要是 FIA 类、非 AICPU 下沉的算子）
- 判据：Replay 时 tiling 可能与 Capture 时不同（有 actual_seq_qlen/kvlen 等动态参数）
- 普通 tiling 固化的算子不需要这套接口

## 相关
- [entry-and-meta.md](entry-and-meta.md)（基础 meta 接口）
- 训练算子支持 torch.compile 接入 autoFuse 也需 meta（见 wiki §3）