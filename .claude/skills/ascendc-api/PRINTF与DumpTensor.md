# PRINTF / DumpTensor 调试 API

## PRINTF

**用途**：Kernel 内部打印标量和字符串，跟踪执行流程和变量值。

```cpp
#include "kernel_operator.h"

// 打印整型/浮点
AscendC::printf("Step %d: val = %f\n", step, val);

// 带核号过滤（必须限流！）
if (GetBlockIdx() == 0) {
    PRINTF("Core 0: max_val = %f, iter = %d\n", maxVal, i);
}
```

**注意事项**：
- 务必 `if (GetBlockIdx() == 0)` 限流，避免多核日志风暴
- PRINTF 是异步的，输出顺序可能和执行顺序不一致
- 性能测试时必须删除
- 关闭打印：编译选项 `-DASCENDC_DUMP=0`

## DumpTensor

**用途**：打印整个 Tensor 的数据内容（UB/L1/L0C/GM 上的 Tensor）。

```cpp
// 不带形状
AscendC::DumpTensor(srcLocal, 5, dataLen);
// 参数：Tensor, 自定义附加信息(如行号), 元素个数

// 带形状（按维度排版）
uint32_t array[] = {8, 8};
ShapeInfo shapeInfo(2, array);
DumpTensor(x, 2, 64, shapeInfo);  // 按 8×8 输出
```

**输出样例**：
```
DumpHead: block_id=0, total_block_num=16
DumpTensor: desc=5, addr=0, data_type=DT_FLOAT16, position=UB
[40, 82, 60, 11, 24, 55, 52, 60, 31, 86, ...]
```

**约束**：
- 只支持 NPU 上板调试（不支持 CPU 仿真）
- dump 元素总长度必须 **32 字节对齐**
- 每核 printf + DumpTensor + assert + 框架 dump 总空间 ≤ **1MB**
- 支持 dtype：uint8/int8/int16/uint16/int32/uint32/int64/uint64/float/half/bfloat16

## MetricsProf API（性能打点）

```cpp
MetricsProfStart();   // 开始性能分析
Compute(...);
MetricsProfStop();    // 结束
```

## 调试路线图

| 阶段 | 工具 | 解决问题 |
|------|------|----------|
| 开发阶段 | CPU 孪生调试 + GDB + ASan | 逻辑错误、内存越界 |
| 联调阶段 | PRINTF（带核号过滤）、DumpTensor | 数值校验、关键状态 |
| 疑难杂症 | CAModel / Timeline | 硬件微观行为、指令竞争 |
| 性能调优 | msprof + MetricsProf + MindStudio Insight | 吞吐瓶颈、内存优化 |

先 CPU 域 `bash run.sh -r cpu` 跑通逻辑 → NPU 域 PRINTF/DumpTensor 定位数值差异 → msprof 性能调优。
