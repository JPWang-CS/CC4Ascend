# 高阶 API 编程范式

高阶 API（Matmul<> / SoftMax<> / Activation）是封装好的模板，组合在 Membase 三阶段流水里用，省去手写 Cube/Vector 指令。

## Matmul<> 模板（trace 自 attention arch22 s1s2_bn2gs1.h:98-119）

### 定义 MatmulType（A/B/C/Bias 各一个）
```cpp
using matmul::MatmulType;
using aType = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
using bType = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;  // true = B 转置
using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
```
- 第 1 参数：位置（GM/L1/L0A/L0B/L0C/UB）
- 第 2 参数：格式（ND/NZ/Zn 等）
- 第 3 参数：dtype
- 第 4 参数（B）：是否转置
- 第 7 参数（B）：enableL1Reuse

### 声明 Matmul 对象
```cpp
matmul::Matmul<aType, bType, cType, biasType, GetMmCfg(enableL1Reuse)> bmm1;
// NZ 输出变体
matmul::Matmul<aType, bType, cNzType, biasType, GetMmCfg(enableL1Reuse)> bmm1Nz;
```

### MatmulConfig（GetMmCfg）
```cpp
__aicore__ const constexpr MatmulConfig& GetMmCfg(bool enableL1Reuse) {
    // 返回 Norm/MDL/IBShare/BasicBlock 等 config + L1Reuse 开关
}
```

### 使用四步法（组合在三阶段里）
```cpp
bmm1.SetTensorA(inputGmX);          // 绑定 A
bmm1.SetTensorB(inputGmY);          // 绑定 B
// 可选: bmm1.SetBias(...);
bmm1.Iterate();                      // 算一次
bmm1.GetResult(outputLocal);         // 取结果（触发 Cube，结果进 LocalTensor/GM）
bmm1.End();                          // 释放
```

## SoftMax<> 模板（trace 自 attention，SoftMaxFlash/SoftMaxTiling）

```cpp
SoftMaxFlash<...> softmax;   // Flash Softmax（online，支持分块）
softmax.Init(maxLocal, sumLocal, expLocal, tiling);
softmax.Process(srcLocal, dstLocal, maskLocal);  // 算 softmax
```
- 内部做 max-shift-exp-sum 数值稳定
- Flash 变体支持分块流水（online softmax，见 kernel-optimization attention base）

## 为什么用高阶 API
- 省去手写 Cube 指令（LoadData/Mmad/Fixpipe 全包）
- 内置双缓冲、L1Reuse、分形处理
- 模板参数编译期选路，零运行时开销

## 何时用
- 矩阵乘 → Matmul<>（不要手写 Mmad）
- softmax 归一化 → SoftMax<>
- 激活函数 → Activation（GELU/SwiGLU 等）

## 与 Blaze 的区别
- 高阶 Matmul<>：AscendC 内置，通用，简单算子用
- Blaze（ops-tensor）：更细的 block/tile/epilogue 可组合，复杂 matmul（GMM/量化/MC2）用，见 [blaze.md](blaze.md)