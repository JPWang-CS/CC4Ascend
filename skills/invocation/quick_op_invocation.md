# 算子调用方式

## 三种调用方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **PyTorch API** | 算子 Kernel 注册到 PyTorch 原生框架 | Torch 推理/训练 |
| **aclnn API** | C 语言 API（前缀 aclnn），无需 IR 定义 | 快速调用、单算子验证 |
| **GE 图模式** | 通过算子 IR 构图调用 | 图编译优化、多算子融合 |

## 快速调用（build.sh）

```bash
# aclnn 调用
bash build.sh --run_example ${op} eager cust [--vendor_name=custom] [--soc=ascend950]

# 图模式调用
bash build.sh --run_example ${op} graph [--soc=ascend950]
```

- `${op}`: 算子名（小写下划线，如 `flash_attention_score`）
- `${soc}`: NPU 型号（`ascend910b`/`ascend910_93`/`ascend950`）

## aclnn API 调用流程

```
Init(device, stream) → CreateAclTensor(inputs) →
GetWorkspaceSize(...) → aclrtMalloc(workspace) →
aclxxXxx(workspace, executor, stream) →
aclrtSynchronizeStream(stream) →
ReadOutput → Destroy → Finalize
```

```cpp
// 1. 初始化
int32_t deviceId = 0;
aclrtStream stream;
Init(deviceId, &stream);

// 2. 构造输入
aclTensor* selfX = nullptr;
void* selfXDeviceAddr = nullptr;
std::vector<int64_t> selfXShape = {32, 4, 4, 4};
CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, ACL_FLOAT, &selfX);

// 3. 一阶段：获取 workspace size
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);

// 4. 申请 workspace
void* workspaceAddr = nullptr;
if (workspaceSize > 0)
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 5. 二阶段：执行
aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);

// 6. 同步 & 取结果
aclrtSynchronizeStream(stream);
PrintOutResult(outShape, &outDeviceAddr);

// 7. 释放
aclDestroyTensor(selfX);
aclrtFree(selfXDeviceAddr);
aclrtFree(workspaceAddr);
aclFinalize();
```

## GE 图模式调用流程

```cpp
// 1. 创建图
Graph graph("graphName");
ge::GEInitialize(globalOptions);

// 2. 创建算子实例
auto add1 = op::AddExample("add1");

// 3. 设置输入输出
graph.SetInputs(inputs).SetOutputs(outputs);

// 4. Session 运行
ge::Session* session = new Session(buildOptions);
session->AddGraph(graphId, graph, graphOptions);
session->RunGraph(graphId, input, output);
```

## PyTorch 集成

将算子 Kernel 注册到 PyTorch 框架，以 Torch 原生 API 方式调用。参见 `examples/fast_kernel_launch_example/README.md`。

## 来源
- `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md`
