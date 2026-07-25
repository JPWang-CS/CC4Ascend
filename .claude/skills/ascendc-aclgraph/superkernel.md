# SuperKernel 融合

源：wiki WIKI2026040910715297 §2.2.5。入 ACLGraph 默认要求支持 SuperKernel（SK），性能不能劣化。

## 机制

SuperKernel 把多个子算子融合成一个可调度的 SuperKernel，减 kernel launch 开销。需在模型里圈定融合范围，capture 后开启。

## 启用

```python
class XxxModel(torch.nn.Module):
    def forward(self, input, indices, update):
        torch.npu.super_kernel_scope_begin("sk1")   # 圈定融合范围 begin
        torch.ops.custom.npu_ai_infra_xxx_(input, indices, update)
        torch.npu.super_kernel_scope_end("sk1")     # end
        return input

model_compiled = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True,
    options={"force_eager": True, "static_kernel_compile": True,
             "super_kernel_optimize": True,
             "super_kernel_optimize_options": {"dcci_after_kernel_end": [".*"]}})
res = model_compiled(input, ...)  # 预执行完成编译

g = torch_npu.npu.NPUGraph()
with torch.npu.graph(g):
    res = model_compiled(input, ...)
g.super_kernel_optimize(
    optimize_options={"dcci_after_kernel_end": [".*"]},
    debug_options={"debug_per_op_max_core_num": 1})  # 涉及全核同步的算子需此开关
g.replay()
```

## optimize_options

| 参数 | 说明 |
|---|---|
| `dcci_after_kernel_end` | 指定算子内部 GetValue/SetValue 不插缓存刷新，SK 调用该算子**后**插 DataCacheCleanAndInvalid。格式 `[".*op1_type.*",".*op2_type.*"]`（匹配算子符号名） |
| `dcci_before_kernel_start` | 同上，SK 调用该算子**前**插 DCCI |

## debug_options

| 参数 | 说明 |
|---|---|
| `debug_per_op_max_core_num` | 单算子满核验证模式（0 关 / 1 开）。开启后融合范围每个算子独立成 SK，以设备最大核数启动，覆盖核数边界；隐式启用 debug_cross_core_sync_check。涉及全核同步的算子必须开 |

## 约束限制

- 不支持静态 kernel 的算子，不支持 SK 融合
- 算子内有**全核同步**且实际核数 < SK 整体核数 → 融合中断（scheduleMode=1）
- **A5 上不支持 SIMT 算子**融合
- 禁用 `get_block_idx/get_block_num/get_task_ration/block_idx/block_num`，须用 `AscendC::GetBlockIdx()` 等
- **KERNEL_TYPE_MIX_AIC_1_1** 算子（1:1 启动）须确保能适应 SK 的 1:2 启动比例；用 CrossCoreSetFlag/WaitFlag 时**所有 AIV 核都要调**，否则硬同步数量不匹配卡死
- 算子内 scalar 访问 GM 须用 GetValue/SetValue，否则 DCCI 优化不生效

## 不可融原因（sk_fusion_fail_reasons.log）

| 原因 | 说明 |
|---|---|
| NOT_IN_SCOPE | 不在用户标定范围 |
| IN_UNFUSIBLE_SCOPE | 被标为不融合 |
| EXCEED_CORE_MAX | 所需核数超设备最大 |
| EXCEED_SCOPE_MAX | 超 1024 个 Scope |
| **OP_UNSUPPORT** | aclnn 算子不支持 SK 融合（TBE/TIK/非 AscendC AICORE）；或算子没适配 SK |
| DYNAMIC_TASK_UNSUPPORT | 运行时动态刷新 Task |
| **SIMT_OP_UNSUPPORT** | SIMT 算子不支持融合 |

## Scope 切分原因

| 原因 | 说明 |
|---|---|
| UNFUSIBLE_NODE | Scope 内有不可融节点 |
| DEADLOCK_DETECTED | 有死锁风险 |
| SYNCALL_OP_DROP | 全核同步算子且其他算子核数更大 |
| DEBUG_PER_OP_MAX_CORE | 单算子满核调试模式 |

## 验证 SK 生效

### 方式一：profiling
`task_time_xxx.csv` 的 Type 列含 `SuperKernel`。Name 含 `sk_` 前缀 + Scope 标识 + start/end 子算子。

### 方式二：sk_meta log
```bash
export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1
```
看 `sk_meta/sk_fusion_fail_reasons.log`（融合失败原因）+ `sk_fused_nodes.log`（最终融合结果）。

性能劣化时：
```bash
export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1   # 打开 sk meta 和静态编译产物
export ASCEND_PROF_SK_ON=1                     # 打开每个 aicore 运行追踪
```