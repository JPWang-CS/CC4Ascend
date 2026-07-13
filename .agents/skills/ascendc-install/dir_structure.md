# 项目目录结构

## ops-transformer_AI 顶层目录

```
ops-transformer_AI/
├── attention/        # Attention 类算子 (FlashAttention, SparseAttention 等)
├── ffn/              # FFN 类算子 (MoE FFN, SwinTransformer 等)
├── gmm/              # 分组矩阵乘 (GroupedMatMul 等)
├── mc2/              # 通信+计算融合 (MatMulAllReduce, MoEDispatch 等)
├── moe/              # MoE 路由/排列 (MoeInitRouting, TokenPermute 等)
├── mhc/              # mHC 架构算子 (Sinkhorn, Pre/Post 等)
├── posembedding/     # 位置编码 (RoPE, ApplyRotaryPosEmb 等)
├── common/           # 公共基础设施
├── examples/         # 示例算子 (AddExample 等)
├── experimental/     # 贡献/实验性算子
├── 3rdparty/         # 第三方依赖
├── cmake/            # CMake 构建配置
├── docs/zh/          # 中文文档
├── scripts/          # 辅助脚本
├── tests/            # 测试框架
└── torch_extension/  # PyTorch 扩展
```

## 单个算子目录结构

```
${op_name}/
├── README.md                     # 算子说明
├── CMakeLists.txt
├── examples/                     # 调用示例
│   ├── test_aclnn_${op_name}.cpp
│   └── test_geir_${op_name}.cpp
├── op_host/                      # Host侧
│   ├── ${op_name}_def.cpp        # 算子信息库
│   ├── ${op_name}_infershape.cpp # Shape推导
│   ├── ${op_name}_tiling.cpp     # Tiling切分
│   └── CMakeLists.txt
├── op_kernel/                    # Device侧
│   ├── ${op_name}_tiling_key.h   # TilingKey
│   ├── ${op_name}_tiling_data.h  # TilingData
│   ├── ${op_name}.cpp            # 核函数入口
│   ├── ${op_name}.h              # Kernel类定义
│   ├── arch22/                   # (可选) A2A3 架构代码
│   └── arch35/                   # (可选) A5 架构代码
├── op_api/                       # aclnn API
├── op_graph/                     # 图模式 IR
├── tests/ut/                     # 单元测试
│   ├── op_host/
│   └── op_kernel/
└── docs/                         # 算子文档
```

## A2A3 vs A5 代码组织规律

- **A2A3 代码**：通常直接放在 `op_kernel/` 下（平铺），或放在 `arch22/` 子目录
- **A5 代码**：放在 `arch35/` 子目录下，不一定是所有 A5 算子
- **arch35 一定是 A5**，但 arch22 不一定是 A2A3（需看代码内容）

## 来源
- `ops-transformer_AI/docs/zh/install/dir_structure.md`
