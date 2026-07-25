# MC2 类 base 模板（深度版）

以 `matmul_all_reduce` A5（arch35）为基准。trace 自 `mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_910_general.h`。

## 数学本质（优化出发点）

MC2 = MatMul + 集合通信（AllReduce / AllToAll / ReduceScatter）融合。例如 `matmul_all_reduce` = `C = MatMul(A,B)`，然后 `C = AllReduce(C)`。

核心矛盾：若串行（先算完整 MatMul 写 GM，再 AllReduce 读回），**MatMul 完成前通信单元空等，AllReduce 时 Cube 空等**，且中间结果 C 要完整回 GM。base 优化的核心是**按 tile 交错计算与通信**，让 Cube 和通信引擎重叠。

## base 实现：MatmulAllReduce910General（已 trace :32-81）

### Process（已 trace :44-54）
```
InnerProcess(false, tileCnt, tileInfo_)       # 主 tile 批
if (tailFlag_) InnerProcess(true, tailCnt, tailInfo_)  # 尾 tile 批
if (basedA2aRsAg) ReduceSumAndAllGather()     # A2A+RS+AG 融合的收尾
HcclFinalize()                                 # 通信资源释放
```

### InnerProcess（已 trace :57-77）
```
for (i < turnCnt) {
    if (block < usedCoreNum) {
        tPipe->Reset();
        mmOp.Init(aGM, bGM, cGM, biasGM, workspaceGM, tiling, tPipe);
        mmOp.Process();        # Cube: 算一个 MatMul tile
        mmOp.End();
    }
    PostProcEachTurn(hcclHandleId, aAddrOffset, cAddrOffset, index);  # 通信后处理
}
if (basedA2aRsAg) WaitAlltoAllEachTurn(tailFlag, turnCnt);
```

### 为什么按 turnCnt 交错
串行方式：算完 N 个 tile 的 MatMul（Cube 忙、通信闲）→ AllReduce 整个 C（通信忙、Cube 闲）。交错方式：每算完一个 tile 立即 PostProcEachTurn 触发该 tile 的通信，**Cube 算第 i+1 tile 时通信在传第 i tile**。两者重叠，总耗时接近 max(算力, 通信) 而非 sum。

### basedA2aRsAg 模板参数（已 trace :50,74）
`basedA2aRsAg` 决定是否走 AllToAll + ReduceScatter + AllGather 融合路径。为 true 时：
- 每 turn 后 `WaitAlltoAllEachTurn`（等 AllToAll）
- 末尾 `ReduceSumAndAllGather`（ReduceScatter 后的求和与再 AllGather）

为 false 时是纯 AllReduce 路径（无 A2A）。

### tailFlag（已 trace :47-49,71）
MatMul tile 数非整数倍时，主批 `tileCnt` + 尾批 `tailCnt` 分开处理。尾批的 tiling 可能不同（tile 更小）。`index = tailFlag ? i + tileCnt : i` 保证地址连续。

## MatMul 类型（已 trace :85-87）
```cpp
using AType = MatmulType<GM, ND, DTYPE_X1, false>;
using BType = MatmulType<GM, ND, DTYPE_X2, bTransFlag>;  // bTransFlag 支持转置
using CType = MatmulType<GM, ND, DTYPE_Y>;
```
GM + ND 格式，输入输出都在 Global Memory。

## CCU 适配（A5，见 ascendc-development §CCU）
A5 通信引擎从 AICPU 改为 CCU1.0：
- Eager：`NnopbaseSetHcclServerType(executor, CCU)`
- Graph：`CreateCcuTask` + `ccu server`/`ccu_stream`

base 的 HcclFinalize / PostProcEachTurn 在 A5 走 CCU 路径，A2A3 走 AICPU 路径。

## 三档 tiling（已验证文件名）
`quant_matmul_all_reduce_tiling_data.h` / `unquant_...` / `weight_quant_...` —— 量化形态决定 tiling 结构（quant 多 scale 处理，weight_quant 多反量化）。

## base 设计理由总结
1. 按 tile 交错计算与通信（Cube 与通信引擎重叠）
2. basedA2aRsAg 支持 A2A+RS+AG 融合（比纯 AllReduce 更优的通信拓扑）
3. tail 分批处理非整数倍 tile
4. A5 走 CCU（低时延），A2A3 走 AICPU
5. 量化三档独立 tiling

## 相关
- ascendc-development §CCU（CCU Eager/Graph 适配代码）
- ascendc-development §核间同步（通信同步）