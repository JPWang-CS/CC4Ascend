# 算子调用通路与真实来源

## 1. 调用通路与真实来源

本次重构后，调用通路不再只粗分成“PyTorch / aclnn / GE 图模式”三类，而是按当前工作区需要**显式区分**：

| 通路 | 当前真实来源 | 说明 |
|---|---|---|
| **PyTorch binding** | `ops-transformer_AI/torch_extension/README.md`, `op-plugin/examples/README.md`, `ops-transformer_AI/torch_extension/cann_ops_transformer/ops/` | PyTorch / torch_npu / C++ wrapper / JIT builder / TorchAir graph 适配入口 |
| **aclnn eager** | `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md`, `ops-nn/docs/QUICKSTART.md`, `test_aclnn_*.cpp` | Host 侧两段式调用；适合单算子验证与样例执行 |
| **aclnn graph** | 方案阶段必须单独判断；在当前仓里常与“图模式”讨论交织出现 | 不要默认把 aclnn graph 等同 aclnn eager 或 GE graph |
| **GE graph** | `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md`, `ops-transformer_AI/docs/zh/develop/graph_develop_guide.md`, `op_graph/`, `test_geir_*.cpp` | 通过 GE / IR / op_graph 交付件实现图模式调用 |
| **验证入口** | `ops-transformer_AI/build.sh --run_example <op> eager|graph` | 现成样例最短验证路径 |

> 说明：官方文档经常把 graph 路径统称“图模式”。在本工作区做 host 设计和实现时，必须在方案里显式写：**本次覆盖 PyTorch binding / aclnn eager / aclnn graph / GE graph 中的哪几条。**

---

## 2. `build.sh` 快速验证入口

### 2.1 `ops-transformer_AI/build.sh`

真实入口来自：
- `ops-transformer_AI/build.sh:264-275`
- `ops-transformer_AI/build.sh:530-566`
- `ops-transformer_AI/build.sh:643-680`

关键点：
- `--run_example <op> eager` → 编译并执行 `test_aclnn_xxx.cpp`
- `--run_example <op> graph` → 编译并执行 `test_geir_xxx.cpp`
- `graph` 模式下不指定 `pkg_mode` / `vendor_name`
- `eager` 模式支持 simulator 选项；`graph` 模式不走同样的 simulator 逻辑

### 2.2 官方文档对应说明

`ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md` 已经明确：
- `mode=eager` 表示 **aclnn 调用**
- `mode=graph` 表示 **图模式调用**
- graph 示例对应 `test_geir_xxx.cpp`

### 2.3 `ops-nn` 的 eager 样例路径

`ops-nn/docs/QUICKSTART.md` 给了工程侧最直观的 `test_aclnn_*.cpp` 样例和 `build.sh --run_example ... eager ...` 运行方式，是当前 host/eager 验证的重要补充来源。

---

## 3. aclnn eager 两段式调用

### 3.1 当前真实说明源

优先看：
- `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md`
- `ops-nn/docs/QUICKSTART.md`

### 3.2 基本调用顺序

```text
Init(device, stream)
→ CreateAclTensor(inputs/outputs)
→ GetWorkspaceSize(..., &workspaceSize, &executor)
→ aclrtMalloc(workspace)
→ aclnnXxx(workspace, workspaceSize, executor, stream)
→ aclrtSynchronizeStream(stream)
→ Read / Check output
→ Destroy tensors / free memory / Finalize
```

### 3.3 什么时候优先走 eager

适合：
- 单算子验证
- Host 侧调用链调试
- 快速确认 checker / workspace / 两段式接口是否通
- 用 `test_aclnn_*.cpp` 样例快速复现

### 3.4 注意点

- “能编译”不等于 eager 通路真的正确
- 要区分：
  - 接口签名对不对
  - workspace/executor 路径对不对
  - 实际跑到的是不是新包/新 `.so`
- 若出现 checker / stale package / 未生效问题，转看 `ascendc-build-errors`（落地后回链）

---

## 4. GE graph 通路

### 4.1 当前真实说明源

优先看：
- `ops-transformer_AI/docs/zh/invocation/quick_op_invocation.md` 的 GE 图模式章节
- `ops-transformer_AI/docs/zh/develop/graph_develop_guide.md`
- 相关算子目录下的 `op_graph/`

### 4.2 真实结构信号

图模式相关真实结构包括：
- `op_graph/` 目录
- `${op_name}_graph_infer.cpp`
- `${op_name}_infershape.cpp`
- GE 注册 / `REG_OP`
- `test_geir_*.cpp`

### 4.3 什么时候要看 GE graph

适合：
- 方案中要求支持图模式
- 要判断该算子是否具备 `op_graph` 交付件
- 要理解 graph 路径的 shape/type infer 面
- 要排查 graph 模式和 eager 行为不一致

### 4.4 注意点

- `docs/zh/install/dir_structure.md` 已明确：若缺少 `op_graph/`，说明该算子暂不支持图模式调用
- 图模式交付件和 aclnn eager 交付件不是一回事，不能混为一谈

---

## 5. PyTorch binding / TorchAir graph

### 5.1 当前真实说明源

优先看：
- `ops-transformer_AI/torch_extension/README.md`
- `ops-transformer_AI/docs/zh/torch_api_list.md`
- `ops-transformer_AI/torch_extension/cann_ops_transformer/ops/`
- `ops-transformer_AI/torch_extension/cann_ops_transformer/ops/graph_convert/`
- `op-plugin/examples/README.md`

### 5.2 真实结构信号

PyTorch 侧当前至少有这些结构：
- C++ backend wrapper
- Python/JIT builder
- `torch.library.impl` / `PYBIND11_MODULE` / `TORCH_LIBRARY_IMPL` 等接入方式
- TorchAir graph mode 相关 `graph_convert/` / `register_fx_node_ge_converter` / `allow_in_graph`

### 5.3 什么时候要看 PyTorch / TorchAir

适合：
- 面向 PyTorch 用户侧接入
- 想知道“这个算子怎么暴露成 Python API”
- 想区分 eager 调用和 TorchAir graph mode 支持面
- 要理解 graph convert / metadata / allow_in_graph 相关路径

### 5.4 注意点

- PyTorch binding 和 aclnn eager 不是同一层事情
- TorchAir graph mode 也不等于 GE graph 样例；它们在宿主框架、转换路径、交付件上都可能不同
- 方案里必须显式说明本次到底要打通哪条 PyTorch/graph 通路

---

## 6. 本 skill 的使用方式

### 如果你要做……

- **快速验证现有算子能不能跑通**
  - 先看第 2 节 `build.sh --run_example`

- **自己写 C/C++ 单算子调用**
  - 先看第 3 节 `aclnn eager`

- **确认图模式交付件和 GE 路径**
  - 先看第 4 节 `GE graph`

- **接入 PyTorch / TorchAir**
  - 先看第 5 节 `PyTorch binding / TorchAir graph`

- **判断这次方案该覆盖哪些调用通路**
  - 先回第 1 节看“调用通路矩阵”

---

## 7. 边界

- 编译 / 安装 / build.sh 细节 → `ascendc-install`
- checker / stale package / 未生效排查 → `ascendc-build-errors`（落地后回链）
- dtype / quant / transpose / broadcast 语义 → `ascendc-data-context`
- 方案阶段判断本次支持哪些通路 → `ascendc-architect` / `ascendc-host-engineer`

本文件只回答：
> **当前到底有哪些调用通路、它们的真实入口在哪里、以及该怎么走对。**