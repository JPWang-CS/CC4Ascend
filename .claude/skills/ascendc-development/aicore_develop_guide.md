# AI Core 算子开发指南

真实源：`ops-transformer_AI/docs/zh/develop/aicore_develop_guide.md`（1068 行）。本文件是结构化速查，详细代码模板见真实文档。

## 开发全流程

```
工程创建 → 算子定义 → Tiling 实现 → Kernel 实现 → aclnn 适配 → 编译部署 → 算子验证
```

## 1. 工程创建

```bash
# AI Core 算子
bash build.sh --genop=${op_class}/${op_name}
# 例：bash build.sh --genop=examples/add_example

# AI CPU 算子（非矩阵、分支密集型）
bash build.sh --genop_aicpu=${op_class}/${op_name}
```

- `${op_class}` = 算子类型，如 attention / ffn / gmm / examples
- `${op_name}` = 小写下划线形式，不可与已有算子重名
- 新增算子分类需在 `cmake/custom_build.cmake` 添加 `add_subdirectory`

目录结构见 `ascendc-install/dir_structure.md`。

## 2. 算子定义 — `${op_name}_def.cpp`

```cpp
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
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

交付件：
- `${op_name}_tiling.cpp` — Tiling 主逻辑 + 注册
- `${op_name}_tiling.h` — 头文件
- `${op_name}_tiling_${sub_case}.cpp` — 可选，子场景 Tiling
- `op_kernel/${op_name}_tiling_key.h` — TilingKey 模板参数
- `op_kernel/${op_name}_tiling_data.h` — TilingData 结构体

```cpp
// TilingFunc 骨架
uint64_t ubSize; int64_t coreNum;
GetPlatformInfo(context, ubSize, coreNum);
auto shape = context->GetInputShape(0)->GetStorageShape();
auto dtype = context->GetInputDesc(0)->GetDataType();
// ... 计算 tile 参数 ...
auto* tiling = context->GetTilingData<${op_name}TilingData>();
tiling->totalLength = totalIdx;
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
```

## 4. Kernel 实现

- `${op_name}.cpp` — 核函数入口（`__global__ __aicore__`）
- `${op_name}.h` — Kernel 类（Init / Process / CopyIn / Compute / CopyOut）
- `arch22/` 或 `arch35/` — 可选架构子目录

Process 三阶段流水：`CopyIn(GM→UB) → Compute(UB) → CopyOut(UB→GM)`，配合 TQue 双缓冲。

## 5. aclnn 适配

编译后自动生成 aclnn 接口；若需手动逻辑，在 `op_api/aclnn_${op_name}.cpp` 实现。图模式另需 `op_graph/` 交付件（见 `graph_develop_guide.md`）。

## 6. 编译部署

```bash
bash build.sh --pkg --soc=${soc} --vendor_name=${vendor} --ops=${op_list}
```
SoC 与包形态详见 `ascendc-install`。

## 7. 算子验证

### UT（无需 NPU）
```bash
bash build.sh -u --[opapi|ophost|opkernel] --ops=${op}
```

### aclnn 样例验证
```bash
bash build.sh --run_example ${op} eager cust [--soc=ascend950]
```

### 算子工程迁移 / 跨平台迁移
真实文档附录有「算子工程迁移」和「算子跨平台迁移」章节，迁移方法论见本 skill [cross_platform_migration_guide.md](cross_platform_migration_guide.md)。

---

## 来源
- `ops-transformer_AI/docs/zh/develop/aicore_develop_guide.md`（1068 行，全量代码模板）
- `ops-transformer_AI/docs/zh/develop/aicpu_develop_guide.md`（AI CPU 开发）