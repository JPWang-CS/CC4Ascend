# Utils API 清单（官网捕获）

> 源：`atlasascendc_api_07_11095.html`（2026-07-25 WebFetch）。URL 以官网为最终真值。
> 子页编号 = `atlasascendc_api_07_<编号>.html`。本文件只给"哪个 API 在哪页"，签名/约束按需 WebFetch 对应页，并回 ops-tensor 头文件核对。

## 1. C++ 标准库 API

| API | 用途 | 页 |
|---|---|---|
| max / min | 两同类型数的最大/最小值 | 10053 / 10054 |
| abs / sqrt | 绝对值 / 平方根 | 10184 / 10183 |
| integer_sequence | 生成整数序列 | 10106 |
| tuple / get / make_tuple | 多类型元素容器 / 取元素 / 创建 | 10108 / 10109 / 10110 |
| is_convertible / is_base_of / is_same | 编译时类型关系判断 | 10114 / 10115 / 10116 |
| is_void / is_integral / is_floating_point / is_array / is_pointer / is_reference / is_const | 编译时类型检测 | 10430-10436 |
| remove_const / remove_volatile / remove_cv / remove_reference / remove_pointer | 移除限定符 | 10437-10441 |
| add_const / add_volatile / add_cv / add_pointer / add_lvalue_reference / add_rvalue_reference | 添加限定符 | 10442-10447 |
| enable_if / conditional / integral_constant | 编译时条件启用/选择/常量 | 10117 / 10118 / 10198 |

## 2. 平台信息获取 API（Tiling 必查）

| API | 用途 | 页 |
|---|---|---|
| PlatformAscendC | 获取硬件平台信息（核数等）支撑 Tiling 计算 | 1026 |
| PlatformAscendCManager | Kernel Launch 场景获取平台信息 | 1039 |

## 3. 原型注册与管理 API（host 注册面）

| API | 用途 | 页 |
|---|---|---|
| OP_ADD | 注册算子原型定义 | 0945 |
| OpDef | 算子原型定义 | 0946 |
| OpParamDef | 算子参数定义 | 0957 |
| OpAttrDef | 算子属性定义 | 0976 |
| OpAICoreDef | AI 处理器实现信息 | 0978 |
| OpAICoreConfig | AI Core 配置信息 | 0988 |
| OpMC2Def | 通算融合算子通信域名配置 | 0999 |

## 4. Tiling 数据结构注册 API

| API | 用途 | 页 |
|---|---|---|
| TilingData 结构定义 | 定义 TilingData 类保存所需参数 | 1005 |
| TilingData 结构注册 | 注册 TilingData 结构体并与算子绑定 | 1006 |

## 5. Tiling 调测 API

| API | 用途 | 页 |
|---|---|---|
| OpTilingRegistry | 加载 Tiling 动态库，获取 Tiling 函数指针 | 00071 |
| ContextBuilder | 手动构造 TilingContext 验证 Tiling 函数 | 1007 |

## 6. Tiling 模板编程 API

| API | 用途 | 页 |
|---|---|---|
| 模板参数定义 | 定义模板参数与模板参数组合 | 00011 |
| GET_TPL_TILING_KEY | 自动生成 TilingKey | 00008 |
| ASCENDC_TPL_SEL_PARAM | 自动生成并配置 TilingKey | 00057 |

## 7. Tiling 下沉 API

| API | 用途 | 页 |
|---|---|---|
| DEVICE_IMPL_OP_OPTILING | Tiling 下沉场景生成注册类 | 00060 |

## 8. RTC（运行时编译）API

| API | 用途 | 页 |
|---|---|---|
| aclrtcCompileProg / aclrtcCreateProg / aclrtcDestroyProg | 编译/创建/销毁程序实例 | 00154 / 00155 / 00156 |
| aclrtcGetBinData / aclrtcGetBinDataSize | 获取编译二进制数据/大小 | 00157 / 00158 |
| aclrtcGetCompileLogSize / aclrtcGetCompileLog | 获取编译日志大小/内容 | 00159 / 00160 |

## 9. Log API

| API | 用途 | 页 |
|---|---|---|
| ASC_CPU_LOG | Host 侧打印 Log（TilingFunc 代码用） | 00152 |

## 10. 调测接口

| API | 用途 | 页 |
|---|---|---|
| printf | Kernel 侧输出日志 | 10426 |
| assert | SIMT VF 调试 assert 断言 | 10428 |
| __trap | SIMT VF 中断算子运行 | 10429 |
| clock | 记录启动到调用的时钟周期数 | 10448 |

## 选用映射

- **host 注册面**（OP_ADD/OpDef/OpParamDef/OpAttrDef/OpAICoreDef/OpMC2Def）→ ascendc-host-engineer
- **Tiling 全家**（结构注册/调测/模板/下沉 + PlatformAscendC）→ ascendc-tiling-expert
- **RTC / 调测**（aclrtc* / printf / assert / __trap / clock）→ ascendc-debug / kernel-expert
