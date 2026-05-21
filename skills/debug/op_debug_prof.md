# 算子调试与性能调优

## 调试定位

### Host 侧日志

```bash
# 默认日志路径
$HOME/ascend/log/debug/plog/plog-pid_*.log

# 打屏显示
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# aclnn 异常错误
printf(aclGetRecentErrMsg());
# 输出样例: [PID:646612] AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null...
```

### Kernel 调试

**printf 打印标量**：
```cpp
AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
```

**DumpTensor 打印张量**：
```cpp
AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
DumpTensor(zLocal, 0, 128);
```

**单步调试**：复杂场景（卡死、越界）使用 [msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)

## 性能调优

### 方式一：上板性能采集

真实 NPU 硬件运行，快速获取整体指标：

```bash
msprof op ./test_aclnn_add_example
```

输出关键指标：
- `Task Duration(us)` — Kernel 耗时
- `Block Dim` — 执行核数
- `Current Freq` — 当前频率
- `ArithmeticUtilization` — 各流水占比（Vector/Cube/MTE 等）

### 方式二：流水图仿真

**A5 (Ascend 950PR)**：使用 CANN Simulator
```bash
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
# trace_core0.json → chrome://tracing 查看
```

**A2A3 (Atlas A2/A3)**：使用 msProf simulator
```bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
# visualize_data.bin → MindStudio Insight 查看
```

## 调试工具速查

| 工具 | 用途 | 适用芯片 |
|------|------|----------|
| msProf op | 上板性能采集 | A2A3 + A5 |
| CANN Simulator | A5 仿真流水图 | A5 only |
| msProf simulator | A2A3 仿真流水图 | A2A3 only |
| msDebug | 单步调试 | A2A3 + A5 |
| MindStudio Insight | 流水图可视化 | A2A3 + A5 |

## 来源
- `ops-transformer_AI/docs/zh/debug/op_debug_prof.md`
