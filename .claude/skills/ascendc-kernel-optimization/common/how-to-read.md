# 怎么读 base 模板与优化偏离

## base 模板是什么

每个算子类的 kernel 都有一个**默认实现形态**（base 模板），它是该类算子在多数场景下的标准写法。优化都是在 base 上的**偏离**。

## 一条优化条目的标准写法

```
### 优化点名
- base 模板做法：xxx
- 本优化改成：xxx
- 适用条件：xxx（shape / 量化 / 芯片 / 场景）
- 收益：xxx
- 风险 / 负优化条件：xxx
- 真实代码路径：op_class/op/kernel/xxx.h → 函数名
```

## trace 路径要求

每条优化必须能沿真实调用链定位到：
```
op_def.cpp（注册/输入输出）
  → op_host/*_tiling.cpp（TilingFunc 算参）
  → op_kernel/*.cpp（kernel 入口，选 TilingKey 分支）
  → op_kernel/arch22|arch35/*.h（具体 Kernel 类与计算函数）
```

只有 trace 到最后一级（具体函数/变量），才能确认优化真的在这条链上。

## base 模板的设计理由

base 模板不是随意写的，它解决该类算子的基本问题：
- Attention base：三阶段流水（CopyIn→Compute→CopyOut）+ softmax 数值稳定
- GMM base：对角线分核规避同地址冲突 + group 边界对齐
- RoPE base：按 D 维度旋转 + cos/sin 预计算

理解 base 为什么这么写，才能判断什么时候值得偏离。