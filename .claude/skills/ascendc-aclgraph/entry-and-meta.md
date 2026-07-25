# 入图方式与 meta 接口

源：wiki WIKI2026040910715297 §1.1/§2/§2.1/§2.2。

## 两种入 ACLGraph 方式

### 1. 裸 ACLGraph
理论上单算子功能跑通即可入裸 ACLGraph，**不需要额外交付件**。

```python
g = torch_npu.npu.NPUGraph()
with torch.npu.graph(g):
    res = model(input, ...)
g.replay()
```

### 2. torch.compile 后入 ACLGraph（npugraph_ex 后端）
算子先经 torch.compile，须满足 compile 交付件：**支持 device="meta" 的 tensor 输出推导接口**（compile 时推导算子输出）。

涉及交付件：**meta 接口注册与实现**。

## meta 接口（torch.compile 路径必需）

### 算子原型定义
```cpp
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_ai_infra_xxx(Tensor input, Tensor indices, Tensor update) -> Tensor");
    m.def("npu_ai_infra_xxx_(Tensor(a!) input, Tensor indices, Tensor update) -> ()");
}
```

### META 实现（仅 shape 推断，不执行计算）
```cpp
// 非原地
at::Tensor npu_ai_infra_xxx_meta(const at::Tensor &input, const at::Tensor &indices,
                                  const at::Tensor &update) {
    std::vector<c10::SymInt> output_size{input.size(0), input.size(1), input.size(2)};
    at::Tensor out = at::empty_symint(output_size, input.options());  // ← 必须 empty_symint
    return out;
}
// 原地
void npu_ai_infra_xxx__meta(at::Tensor &input, const at::Tensor &indices,
                             const at::Tensor &update) {}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_ai_infra_xxx", &custom::npu_ai_infra_xxx_meta);
    m.impl("npu_ai_infra_xxx_", &custom::npu_ai_infra_xxx__meta);
}
```

### 关键约束（wiki 红字）
- **推理算子 meta 申请输出内存，统一用 `at::empty_symint`**（不用 empty）
- **获取 size 统一用 `sym_size`** 接口（SymInt，支持动态 shape）
- 入 aclgraph **默认支持静态 kernel 编译**，ST 须含静态 kernel 验证，且性能不能比动态差
- 入 aclgraph **默认支持 SuperKernel**，ST 须含 SuperKernel 验证，性能不能劣化

## 入图场景自验用例（模型封装）

```python
class XxxModel(torch.nn.Module):
    def __init__(self): super().__init__()
    @torch.inference_mode()   # 训练算子加此装饰器强制转推理模式
    def forward(self, input, indices, update):
        torch.ops.custom.npu_ai_infra_xxx_(input, indices, update)
        return input
```

## 相关
- tiling 动态变化的算子（FIA 类）还需 [tiling-update-op.md](tiling-update-op.md) 的额外接口