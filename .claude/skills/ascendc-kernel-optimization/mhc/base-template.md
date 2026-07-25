# MHC 类 base 模板（深度版）

以 `mhc_sinkhorn` 为基准。trace 自 `mhc/mhc_sinkhorn/op_kernel/arch35/mhc_sinkhorn.h`。

## 数学本质（优化出发点）

Sinkhorn-Knopp 算法：**迭代将混合矩阵投影到双随机矩阵**（行和=1、列和=1）。每轮交替做行 softmax 和列 softmax，直到收敛。

核心矛盾：迭代次数不定（收敛才停）、行列交替归一化需要**反复读写整块矩阵**。base 优化是让单块的迭代全部在寄存器级完成（MicroAPI），避免每轮迭代回 GM。

## base 实现：MhcSinkhornSimd（已 trace mhc_sinkhorn.h:42-149）

### 三阶段（已 trace :124/:137/:149）
```
CopyIn(loopIdx, handleNum)    # 搬入一块 h_res 到 inputQue
CalcSoftmax(handleNum)         # 寄存器级 Sinkhorn 迭代
CopyOut(loopIdx, handleNum)    # 搬出 norm_out / sum_out
```

### CalcSoftmax 寄存器级迭代（已 trace CalcSoftmax 实现）
```cpp
LocalTensor<float> inputLocal = inputQue_.DeQue<float>();
LocalTensor<float> normOutLocal = normOutQue_.AllocTensor<float>();
__VEC_SCOPE__ {
    MicroAPI::RegTensor<float> inputReg, midReg, sumReg, sumoutReg;
    MicroAPI::RegTensor<int32_t> orderReg, tmpReg1, tmpReg2, tmpReg3, dupReg1;
    // 行/列 softmax 迭代，orderReg 维护置换顺序
}
```

### 为什么寄存器级（RegTensor）
Sinkhorn 每轮迭代 = 行 softmax + 列 softmax，需多次扫矩阵。若用 Membase（UB TBuf），每轮迭代读写 UB；矩阵大时 UB 放不下整块，要回 GM。MicroAPI RegTensor 让单 handle 的迭代全在寄存器完成，迭代轮次只动寄存器不访存。

### orderReg 的作用
Sinkhorn 收敛依赖行列交替归一化的顺序一致性。orderReg 维护元素在行/列扫描中的排列，保证迭代数值确定。

## handleNum 分块（已 trace CalcSoftmax 参数）
`handleNum` = 一次处理的行块数。矩阵按 handleNum 分块，每块独立做 Sinkhorn 迭代。blockCnt / loopSize 由 `GetVRegSize()` 决定（寄存器容量约束）。

## 输出
- `normOut`：归一化后的双随机矩阵
- `sumOut`：列和（反向传播用）

## MHC 系列
- `mhc_pre`：算 H_res/H_post 投影（Sinkhorn 的输入预处理）
- `mhc_post`：Post Mapping + 残差（Sinkhorn 输出后处理）
- `mhc_pre_sinkhorn`：Pre + Sinkhorn 融合（省中间回 GM）
- `mhc_post`/`mhc_post_backward`/`mhc_pre_sinkhorn_backward` 双架构（arch22+arch35），其余 arch35

## base 设计理由总结
Sinkhorn 是**迭代密集 + 访存敏感**。关键优化：
1. 寄存器级迭代（RegTensor）避免每轮回 GM
2. handleNum 分块让单块进寄存器
3. Pre/Sinkhorn/Post 融合省中间 GM 往返