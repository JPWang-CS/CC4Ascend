# 改了但没生效

代码改了、重编译了，但行为没变。最常见且最易误判的失效类。

## 根因总览

"改了没生效"几乎都是**运行的还是旧产物**，而非代码没改对。按可能性排序：

## 1. so 未替换 / 未重装（最常见）

### 现象
改了 op_api/op_kernel，重编 `build.sh --pkg`，但 aclnn 行为没变。

### 根因
- 没跑 `./build_out/cann-*.run` 安装
- 或 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH` 指向旧 vendor 目录
- vendors/config.ini 的 load_priority 指向旧包

### 诊断
```bash
# 确认实际加载的 .so 路径
ldd <executable> | grep opmaster
ls -la ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_api/lib/
```

### 修法
重装 + 确认环境变量指向新包。

## 2. builtin 包优先级低于 custom / 或反之

custom 包优先级高于 builtin（挂载方式）。若同时存在，行为取决于 load_priority。
- 想用 custom：确认 vendors/config.ini 有该 vendor 且 load_priority 靠前
- 想用 builtin：清 `ASCEND_CUSTOM_OPP_PATH`

## 3. 常量指纹检测（EZ0012 尾数，见 checker-errors.md）

报错尾数 ≠ 重新编译的指纹 = 加载的还是旧 .so。这是检测 stale .so 的可靠手段。

## 4. example 可执行文件没重编

`test_aclnn_*.cpp` 改了但没重跑 `build.sh --run_example`，运行的还是旧可执行文件。

## 5. UT 二进制没重编

`build.sh -u` 改了源码但没 `--make_clean`，旧 .o 残留。修：`--make_clean` 后重编。

## 诊断流程（推荐顺序）

1. 确认重装了（`./build_out/cann-*.run` 跑过）
2. `ldd` 确认可执行文件链到的新 .so 时间戳
3. 确认 `ASCEND_CUSTOM_OPP_PATH` / vendors 指向新包
4. 触发 EZ0012 报错，对比尾数指纹
5. 确认 example/UT 二进制重编

> 经验：90% 的"改了没生效"是步骤 1-2（没重装或链到旧 so），先查这个再怀疑代码。