# AI Core 算子开发指南（通用）

## 概述

AI Core 算子使用 **Ascend C** 语言开发，运行在 AI Core 硬件单元。本指南覆盖 A2A3 和 A5 通用开发流程。

## 开发全流程

```
工程创建 → 算子定义 → Tiling实现 → Kernel实现 → aclnn适配 → 编译部署 → 算子验证
```

## 1. 工程创建

```bash
bash build.sh --genop=${op_class}/${op_name}
```

例：`bash build.sh --genop=attention/flash_attention_score`

目录结构：

```
${op_name}/
├── examples/                  # 调用示例
│   ├── test_aclnn_${op_name}.cpp
│   └── test_geir_${op_name}.cpp
├── op_host/                   # Host侧
│   ├── ${op_name}_def.cpp     # 算子信息库
│   ├── ${op_name}_infershape.cpp  # Shape推导
│   ├── ${op_name}_tiling.cpp  # Tiling切分策略
│   └── CMakeLists.txt
├── op_kernel/                 # Device侧Kernel
│   ├── ${op_name}_tiling_key.h    # TilingKey模板参数
│   ├── ${op_name}_tiling_data.h   # TilingData结构体
│   ├── ${op_name}.cpp         # Kernel入口(核函数)
│   ├── ${op_name}.h           # Kernel实现(算子类)
│   ├── arch22/                # (可选) A2A3 架构代码
│   └── arch35/                # (可选) A5 架构代码
└── CMakeLists.txt
```

新增算子分类需在 `cmake/custom_build.cmake` 中添加 `add_subdirectory(${op_class})`。

## 2. 算子定义

### 交付件：`${op_name}_def.cpp`（算子信息库）

```
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore()
            .AddConfig("ascend910b")    // A2A3
            .AddConfig("ascend950");    // A5
    }
};
OP_ADD(AddCustom);
}
```

## 3. Tiling 实现

Tiling = 将大张量切分为多个小块（Tile），逐块计算。

### 交付件

| 文件 | 位置 | 说明 |
|------|------|------|
| `${op_name}_tiling.cpp` | op_host/ | Tiling 主逻辑 + 注册 |
| `${op_name}_tiling_key.h` | op_kernel/ | 模板参数枚举，区分不同算法分支 |
| `${op_name}_tiling_data.h` | op_kernel/ | TilingData 结构体 |

### Tiling 主函数模板

```cpp
// 获取平台信息
uint64_t ubSize;
int64_t coreNum;
GetPlatformInfo(context, ubSize, coreNum);

// 获取输入信息
auto inputX = context->GetInputShape(0);
auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
auto dataType = context->GetInputDesc(0)->GetDataType();

// 计算 Tiling 参数
// ...

// 设置 TilingData
${op_name}TilingData* tiling = context->GetTilingData<${op_name}TilingData>();
tiling->totalLength = totalIdx;
tiling->tileNum = TILE_NUM;

// 注册
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
```

### TilingKey 模板参数

```cpp
ASCENDC_TPL_ARGS_DECL(
    ${op_name},
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, MODE_0, MODE_1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, MODE_0, MODE_1)));
```

### TilingData 结构体

```cpp
struct ${op_name}TilingData {
    int64_t totalLength;
    int64_t tileNum;
    // ... 自定义字段
};
```

## 4. Kernel 实现

### 核函数入口 (`${op_name}.cpp`)

```cpp
template <uint32_t schMode>
__global__ __aicore__ void add_example(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);

    if constexpr (schMode == TILING_KEY_FLOAT) {
        NsAddExample::AddExample<float> op;
        op.Init(x, y, z, &tilingData);
        op.Process();
    }
}
```

### Kernel 类 (`${op_name}.h`)

```cpp
template <typename T>
class AddExample {
public:
    __aicore__ inline AddExample(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 const AddExampleTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(const int32_t dataLength);

    TPipe pipe_;                    // 流水管道
    TQue<VECIN, BUFFER_NUM> inQ_;   // 输入队列(double buffer)
    TQue<VECOUT, BUFFER_NUM> outQ_; // 输出队列
    GlobalTensor<T> inputGM_, outputGM_;
};
```

### Process 主流程

```cpp
void Process() {
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);      // 数据搬入 (GM → UB)
        Compute(i);     // 计算 (UB上)
        CopyOut(i);     // 数据搬出 (UB → GM)
    }
}
```

## 5. aclnn 适配

算子编译后自动生成 aclnn 接口，无需手动配置。通过 `${op_name}_def.cpp` 自动生成算子二进制包。

## 6. 编译部署

```bash
# 编译指定算子
bash build.sh --pkg --soc=${soc} --vendor_name=${vendor} --ops=${op_list}

# SoC版本:
# ascend910b   → Atlas A2
# ascend910_93 → Atlas A3
# ascend950    → Ascend 950
```

## 7. 算子验证

### UT 验证（无需 NPU）

| UT 类型 | 验证目标 |
|---------|----------|
| Infershape UT | Host 侧 Shape 推导逻辑 |
| Tiling UT | Host 侧 Tiling 切分逻辑 |
| Kernel UT | Device 侧 Kernel 计算逻辑 |

### Kernel UT 模板

```cpp
TEST_F(${OpName}KernelTest, test_case_basic) {
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(...);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TilingData));
    auto* tilingData = reinterpret_cast<TilingData*>(tiling);
    tilingData->totalLength = ...;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(${op_name}, blockDim, x, y, z, workspace, tiling);
    EXPECT_EQ(...);
    AscendC::GmFree(x);
}
```

### aclnn 调用验证

```bash
bash build.sh --run_example ${op} eager cust [--soc=ascend950]
```

## 来源
- `ops-transformer_AI/docs/zh/develop/aicore_develop_guide.md`
