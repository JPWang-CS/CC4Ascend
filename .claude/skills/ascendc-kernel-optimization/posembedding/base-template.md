# PosEmbedding 类 base 模板（深度版）

以 `apply_rotary_pos_emb` 为基准。trace 自 `posembedding/apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_ab.h`。

## 数学本质（优化出发点）

RoPE（旋转位置编码）= 对 Q/K 的每个 head，按位置 pos 和维度 d 做旋转：
```
旋转角 θ(pos, d) = pos / base^(d / dim)
out = [q·cos(θ) - rotate(q)·sin(θ)]
```
`rotate(q)` 按 d 维切分模式决定（HALF / INTERLEAVE / QUARTER）。

核心矛盾：RoPE 是**纯 elementwise**（每元素独立旋转），瓶颈在搬运（cos/sin 和 Q/K 的 GM 访问）而非算力。base 优化是按 d 维切分模式选最高效的旋转实现，并用 broadcast 后缀处理 cos/sin 对齐。

## base 实现：ApplyRotaryPosEmbAB（已 trace apply_rotary_pos_emb_ab.h:24）

### ProcessQKLoop 流程（已 trace ProcessQKLoop 实现）
```
1. AllocTensor(qkBuffer, outBuffer)
2. CopyIn:
   isPartialRope ? CopyInByRotaryMode(只搬 realDim 部分) : DataCopyPad(全搬)
3. 旋转（按 rotaryMode 分派）:
   HALF      → HalfAlignVF(sin, cos, qk, out, realDim, dAlign, currBSNum, count)
   QUARTER   → QuarterAlignVF(...)
   else      → InterleaveModeVF(...)
4. CopyOut:
   isPartialRope ? CopyOutByRotaryMode : DataCopyPad
```

### 为什么按 rotaryMode 分派三个 VF 函数
D 维切分模式决定 cos/sin 怎么和 Q 元素配对：
- **HALF（dSplitCoef=2）**：D 分两半，前半后半交叉配对
- **QUARTER（dSplitCoef=4）**：D 分四份，相邻两两配对
- **INTERLEAVE（dSplitCoef=1）**：偶奇维度配对

三种配对的数据重排逻辑不同（HalfAlign / QuarterAlign / Interleave），用独立 VF 函数各自高效实现，避免一个通用函数处理三种重排的低效。

### dSplitCoef 与 dSplitSize（已 trace :62,72）
```cpp
uint32_t dSplitCoef_{1};                      // :62 默认 INTERLEAVE
dSplitSize_ = realDim / dSplitCoef * sizeof(T);  // :72 每Split块的字节大小
DataCopyExtParams qkParams = {currBSNum * count * dSplitCoef, dSplitSize_, ...};  // :134
```
dSplitCoef 决定一次搬多少个 dSplit 块，直接影响 DataCopy 的参数。

## isPartialRope（部分旋转，已 trace ProcessQKLoop）
当 `realDim < headDim`（如只旋转前 D/4），isPartialRope 走特殊 CopyIn/CopyOut：只搬旋转部分，其余 passthrough。避免全量搬运的浪费。

## broadcast 后缀变体（文件名即证据）
`_ab` / `_aba` / `_ba` / `_bab` 对应 cos/sin 的 B 维 broadcast 模式。不同 layout（BNSD/BSND/SBND/TND）cos/sin 与 Q 的 B 轴对齐方式不同，用独立类处理，避免通用广播逻辑的开销。

## A2A3 vs A5
- **A2A3（Membase）**：LocalTensor + TQue + DataCopyPad，UB 中间缓冲
- **A5（Regbase，`_apt.cpp`）**：MicroAPI RegTensor + LoadDist/StoreDist（`DIST_UNPACK_B16` BF16 解包 / `DINTLV_B32` De-Interleave / `DIST_PACK4_B32` int8 打包），寄存器级旋转无 UB 中间缓冲

### 为什么 A5 用 LoadDist
INTERLEAVE 模式需把偶奇维度拆开配对。A5 `DINTLV_B32`（De-Interleave）在加载时即拆分，省一次 UB 重排。BF16 输入用 `DIST_UNPACK_B16` 解包升 FP32 算。量化输出 int8 用 `DIST_PACK4_B32` 打包。

## base 设计理由总结
1. elementwise 瓶颈在搬运 → 优化 DataCopy（dSplitCoef 调块大小、isPartialRope 只搬必要部分）
2. 三种 d 切分模式独立 VF（各自最高效重排）
3. broadcast 后缀独立类（layout 对齐高效）
4. A5 Regbase + LoadDist/StoreDist（加载即拆分，省 UB 重排）