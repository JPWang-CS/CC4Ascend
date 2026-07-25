# 静态 kernel 编译

源：wiki WIKI2026040910715297 §2.2.4。入 ACLGraph 默认要求支持静态 kernel 编译。

## 机制

ACLGraph Capture 时把 kernel 编译固化为静态二进制（static_kernel_*），Replay 时直接用，不再动态编译。相同 shape 下静态 kernel 性能不能比动态差。

## 启用（npugraph_ex + force_eager）

```python
model_compiled = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True,
    options={"force_eager": True, "static_kernel_compile": True})
res = model_compiled(input, ...)  # 必须预执行一次完成静态编译，否则直接 capture 报错

g = torch_npu.npu.NPUGraph()
with torch.npu.graph(g):
    res = model_compiled(input, ...)
g.replay()
```

> 多 DIE 场景需配 `export LOCAL_WORLD_SIZE=${local_world_size}`，每节点一致。

## 编译前清缓存
```bash
rm -rf ./log ./sk_meta ./static_kernel_compile_outputs
rm -rf ${ASCEND_HOME_PATH}/opp/static_kernel/
```

## 验证静态 kernel 生效

### 方式一：profiling
```bash
msprof python3 test.py
```
看 `task_time_xxx.csv` 的 kernel_name 是否含 `static_kernel_`。
例：`static_kernel_AiInfraFusedInferAttentionSink_a7c7..._3119786_d0`
不含则未生效，用方式二定位。

### 方式二：plog
```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```
plog 含以下字段即生效：
- 自动框架算子搜：`Launch static kernel ${算子类型名}`
- 手动框架算子搜：`Available static bin for op ${算子类型名}`
搜不到 = 未生效。

## 编译报错定位
查脚本同级：
```
static_kernel_compile_outputs/xxx/xxx/summary.txt
static_kernel_compile_outputs/xxx/xxx/xxxxx_compile_error.log
```

## 相关
- 静态 kernel 是 SuperKernel 融合的前提（不支持静态 kernel 的算子不支持 SK 融合）