# aclnn Checker 报错码

aclnn 两段式接口的 GetWorkspaceSize 阶段 checker 拒绝时的报错。真实 checker 在 `op_api/*.cpp` 的 `CheckXXX` 函数。

## EZ0012 "at least ND" 尾数 = 编译常量指纹（已验证）

### 现象
```
EZ0012: ... at least ND ... < trailing number>
```

### 根因机制（已验证）
报错信息尾部的数字是**编译进 .so 的常量指纹**。同一份代码不同编译会产生不同尾数。若尾数与你预期不符，说明加载的还是旧 op-api .so（没重装/没选到新包）。

### 诊断
对比报错尾数与重新编译后的 .so 指纹。不一致 = stale .so（见 stale-deployment.md）。

### 修法
重新 `build.sh --pkg` + 重装算子包，确认 `ASCEND_CUSTOM_OPP_PATH` / vendors 指向新包。

## EZ0013 shape/dim 不匹配（已验证 QBMM）

### 现象
checker 校验 shape/dim 约束失败。如 QBMM pertoken_scale 非 G-B/B-B 模式须 1D[M] 不能 [M,1]。

### 根因
checker 对特定模式有 shape 约束（如 scale 必须特定 rank/shape）。输入构造不符合该模式约束。

### 修法
看报错指向的约束，调整输入 shape 或换 TilingKey 模式。

## EZ0020 dtype 不符（已验证 QBMM）

### 现象
如 QBMM V5 x1Scale(pertoken) 须 DT_FLOAT。

### 修法
按 checker 要求的 dtype 构造输入。pertoken scale 用 fp32，weight scale 可 bf16/[N]。

## EZ0027（已验证 QBMM）

QBMM 系列 checker 约束报错，按报错文本定位具体约束（scale 转置一致、groupSize 等）。

## 通用诊断法

1. 报错码 EZ 后 4 位 + 文本 → 定位是哪类约束（shape/dtype/format/量化）
2. 看 `op_api/*_common.h` / checker 函数里对应常量
3. 对比输入构造 vs checker 要求
4. 注意：checker 报错的尾数/常量是指纹，可检测 stale .so