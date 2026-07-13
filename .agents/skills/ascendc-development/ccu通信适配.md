# CCU 通信适配（A5 专属）

## 概述

Ascend 950 引入 **CCU1.0**（集合通信加速器），替代 A2A3 的 AICPU 通信方式，降低访存需求，减少调度时延。

## 通信方式对比

| | A2A3 | A5 |
|------|------|-----|
| 通信引擎 | AICPU | CCU1.0 |
| Eager 模式 | `NNOPBASE_HCCL_SERVER_AICPU` | `NNOPBASE_HCCL_SERVER_TYPE_CCU` |
| Graph Task 类型 | `aicpu kfc server` + `kfc_stream` | `ccu server` + `ccu_stream` |

## Eager 模式适配

在 aclnn 第二段接口中设置通信类型：

```cpp
aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            // A5: 使用 CCU
            NnopbaseSetHcclServerType(executor,
                NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    return ACLNN_SUCCESS;
}
```

## Graph 模式适配

### CalcParamFunc（资源计算）

```cpp
ge::Status CalcParamFunc(gert::ExeResGenerationContext *context) {
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // A5: CCU
        return CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    // A2: AICPU
    return CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}
```

### GenerateTask（创建通信 Task）

```cpp
// A2A3: AICPU Task
ge::KernelLaunchInfo aicpu_task =
    ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME, KERNEL_NAME);

// A5: CCU Task
ge::KernelLaunchInfo ccuTask =
    ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
```

## 核间同步变更

A5 上 `CrossCoreWaitFlag` 和 `CrossCoreSetFlag` 数量**必须严格匹配**，且建议"先生产后消费"成对设计。

A2A3 的兜底机制（HWT S处理多余 Set）在 A5 上不再存在，不匹配会导致**必现卡死/死锁**。

### 排查重点

1. 异常分支提前返回 → `Set` 了但未 `Wait`（或反之）
2. 多 stage 复用同一 `flagId` 但生命周期重叠 → 跨阶段串扰
3. 循环内条件触发同步但边界未对齐 → 迭代次数不一致

### 新增同步模式

```cpp
// mode 3: 新增的核间同步模式
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
```

## 来源
- `ops-transformer_AI/docs/zh/develop/cross_platform_migration_guide.md` §集合通信 & §核间同步
