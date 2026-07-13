# 源码构建指南

## 环境准备

### 硬件要求

| 项目 | A2A3 | A5 |
|------|------|-----|
| NPU | Atlas 300T/9000 A2 | Ascend 950 |
| SoC | ascend910b, ascend910_93 | ascend950 |

### 软件依赖

1. CANN toolkit (社区版/商业版)
2. Python 3.8+
3. CMake 3.14+
4. GCC 7.3+
5. 第三方依赖（自动下载或离线准备）：gtest, protobuf, json, yaml-cpp 等

## 联网编译

```bash
bash build.sh --pkg --soc=ascend910b --ops=add_example
```

## 离线编译

联网场景自动下载第三方依赖；离线场景需预先准备依赖包，详见[离线编译指导](../invocation/quick_op_invocation.md#未联网编译)。

## 生成产物

编译成功后输出：
```
Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
```
产物位于 `build_out/` 目录。

## 来源
- `ops-transformer_AI/docs/zh/install/compile.md`
