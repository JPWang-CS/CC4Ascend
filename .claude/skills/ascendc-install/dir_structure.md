# 项目目录结构

真实源：`ops-transformer_AI/docs/zh/install/dir_structure.md`

## 顶层目录

```
ops-transformer_AI/
├── cmake/                # 编译配置（含 aclnn_ops_transformer.h.in 汇总头模板）
├── common/               # 公共头文件(inc) + 公共代码(src)
├── experimental/         # 贡献/实验性算子（按算子类分子目录）
├── ${op_class}/          # 算子分类：attention / ffn / gmm / mc2 / moe / mhc / posembedding
├── docs/                 # 项目文档
├── examples/             # 端到端算子开发和调用示例（如 add_example）
├── scripts/              # 辅助脚本（含 third_lib_download.py）
├── tests/                # 项目级测试
├── torch_extension/      # torch 扩展
│   └── cann_ops_transformer/
│       ├── common/       # 扩展公共能力
│       ├── docs/         # 扩展 api 文档（含 torch_extension_guidelines.md）
│       ├── op_builder/   # OpBuilder 基类，管理 JIT 编译与 schema/meta 注册
│       └── ops/
│           ├── csrc/     # 算子 api 的 C++ 实现
│           ├── graph_convert/  # TorchAir 图模式 Converter
│           ├── ${api_name}.py  # Python 前端
│           └── __init__.py
├── build.sh              # 编译脚本
├── install_deps.sh       # 依赖安装脚本
├── requirements.txt      # Python 三方依赖
├── classify_rule.yaml    # 组件划分信息
└── version.info          # 版本信息
```

---

## 单个算子目录结构

> 部分目录可选，以实际交付件为准。缺少某目录有明确含义（见下）。

```
${op_class}/${op_name}/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── aclnn${OpName}.md                 # aclnn 接口文档（${OpName} 大驼峰）
├── examples/
│   ├── test_aclnn_${op_name}.cpp         # aclnn 调用示例
│   └── test_geir_${op_name}.cpp          # geir 调用示例（图模式）
├── op_graph/                             # 图融合实现
│   ├── CMakeLists.txt
│   ├── ${op_name}_graph_infer.cpp        # InferDataType
│   ├── ${op_name}_proto.h                # 算子原型定义（融合识别）
│   └── fusion_pass/                      # 融合规则
├── op_host/                              # Host 侧
│   ├── config/                           # 可选，二进制配置
│   │   └── ${soc_version}/
│   │       ├── ${op_name}_binary.json
│   │       └── ${op_name}_simplified_key.ini
│   ├── ${op_name}_def.cpp               # 算子信息库
│   ├── ${op_name}_infershape.cpp        # 可选，InferShape
│   ├── ${op_name}_tiling.cpp            # 可选，Tiling
│   ├── ${op_name}_tiling.h
│   ├── ${op_name}_tiling_${sub_case}.cpp # 可选，子场景 Tiling
│   ├── ${op_name}_tiling_${sub_case}.h
│   └── CMakeLists.txt
├── op_api/                               # 可选，aclnn 实现
│   ├── aclnn_${op_name}.cpp
│   ├── aclnn_${op_name}.h
│   ├── ${op_name}.cpp                    # l0 接口
│   ├── ${op_name}.h
│   └── CMakeLists.txt
├── op_kernel/                            # Device 侧 Kernel
│   ├── ${sub_case}/                      # 可选，子场景目录
│   ├── ${op_name}_tiling_key.h           # 可选，TilingKey
│   ├── ${op_name}_tiling_data.h          # 可选，TilingData
│   ├── ${op_name}.cpp                    # Kernel 入口
│   └── ${op_name}.h                      # Kernel 实现
└── tests/
    └── ut/                               # 可选，UT 用例
```

---

## 缺目录的含义（真实，易踩坑）

| 缺少目录 | 含义 |
|----------|------|
| 缺 `op_host/` | 调用了其他算子的 op_host；或 Kernel 暂无 Ascend C 实现 |
| 缺 `op_kernel/` | 调用了其他算子的 op_kernel；或暂无 Ascend C 实现 |
| 缺 `op_api/` | **暂不支持 aclnn 调用** |
| 缺 `op_graph/` | **暂不支持图模式调用** |

---

## A2A3 vs A5 代码组织

- **A2A3 代码**：通常平铺在 `op_kernel/` 下，或放 `arch22/` 子目录
- **A5 代码**：放 `arch35/` 子目录
- `arch35` = A5；`arch22` 一般是 A2A3，但需看代码内容确认
- 子场景也可用 `${sub_case}/` 目录（不限 arch）

---

## 来源
- `ops-transformer_AI/docs/zh/install/dir_structure.md`