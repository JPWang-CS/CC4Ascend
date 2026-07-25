# Foreach 批量融合范式

ops-nn foreach 类（68 算子，对一组 Tensor 做相同操作）。内容经真实代码 trace。源：`ops-nn/foreach/`。

## 基准算子：foreach_add_scalar（已 trace）

### Kernel 类（已 trace foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h:23）
```cpp
namespace ForeachAddScalar {
class ForeachAddScalarRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachAddScalarRegbase<...>> {  // CRTP
    using Base = ForeachRegbaseUnary<...>;
    void Init(inputs, scalar, outputs, workspace, ...);   // :28
    void Compute(LocalTensor<T> inLocal, outLocal, dataCount);   // :42
};
}
```
引用共享基类：`foreach_utils/arch35/foreach_regbase_unary.h`（:18）。

## CRTP 模板范式（foreach 核心，已 trace :23-25）
所有 foreach 一元算子继承 `ForeachRegbaseUnary<..., Self>`（CRTP，子类把自身传给基类）：
- 基类管批量调度（遍历 inputs 列表、核间分配）
- 子类只实现 `Compute(in, out, count)` 的单 Tensor 操作

## foreach 算子清单（已验证，部分）
`foreach_add_scalar`(+list+inplace) / `foreach_addcdiv`(+scalar+list) / `foreach_abs` / `foreach_acos` / `foreach_a_cos_inplace` ...

## base 设计理由
- **批量**：一组 Tensor 做相同 op（如优化器的多参数更新），逐个调 op 会有 kernel launch 开销
- **融合**：foreach 一个 kernel 处理整组，减 launch + 共享 tiling/sync
- **CRTP**：基类统一调度，子类只写计算，零虚函数开销

## A2A3 也广泛用（已验证 88 个 arch22）
foreach 在 A2A3 也有大量实现（88 个 arch22 目录），说明该范式跨代通用。

## 通用优化
- 核间按 Tensor 列表分配（非按元素），负载均衡
- inplace 变体（`_inplace`）省输出 GM
- list 变体（`_list`）支持 scalar 也分组