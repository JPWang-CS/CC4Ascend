# Agent Memory Index -- ascendc-architect

AscendC 算子架构专家记忆。扁平文件模型：分类靠文件名前缀 + 本索引分区，无物理子目录。通用知识在前，项目与环境在后。

## 通用知识：MX / 量化 Matmul（Ascend 950）
- [MX scale 语义](mx-quant-scale-semantics.md) — e8m0 1字节不打包、scale shape (M,ceil(K/64),2) +batch 前缀、groupK=32、batch stride、cube 内 per-group apply
- [fp4/e2m1 打包](fp4-e2m1-packing.md) — float4_e2m1fn_x2 真 2:1 打包(最内层存储维)、NPU ×2 解释、OCP E2M1 16码、int-passthrough 失败、ND 下 N=1 合法、两结构盲区(nibble序/符号翻转)
- [scale 转置 stride 判定](scale-transpose-stride-semantics.md) — 约束A：NPU 靠 stride(非 shape)判转置；scale 用 .transpose(-3,-2) 跳过末维"2"；checker @ QuantMatmulKernelNpuOpApi.cpp
- [Blaze/CMCT 编译期选路](cann-blaze-cmct-gating.md) — IS_BLAZE=(MAJOR>=9&&MINOR>0) 编译期定路径(9.0=CMCT/9.1=Blaze)；dav_c220/c310 芯片映射不可自选；9.0.x fp4 需 fp4x2 白名单；MX 仅 950
- [device PRINTF 验证边界](device-print-verification.md) — 只能证实"执行到了"、不能证伪"没执行"；真实证据靠数值(multi-batch PASS)
- [可证伪 golden 测试设计](golden-falsifiable-testing.md) — 全码覆盖+阴性对照；机制见 skill `ascendc-golden-testing` §falsifiable-design
- [cos+相对L2 判据](golden-cos-rell2-criterion.md) — 现行 gating 已改为 flat isclose+err_ratio（见 skill `ascendc-golden-testing` §criteria）；本条留 cos/rel_l2 诊断价值与历史判据演变
- [抵消场景判据修法](golden-cancellation-criterion.md) — (已被cos判据取代,存病态诊断价值)极端K+对称抵消误杀零误差核；噪声地板检测+peak缩放；证伪电池 A-F
- [fp4+bias 近零 err_small 判别](golden-fp4-bias-nearzero.md) — 1b_fp4_bias 偶发FAIL根因;fp4 matmul是fp32精确(dyadic)非换序噪声;近零靠bias抵消;正确核也因L0C bias-first累加序~1.7%擦门(|err|~1e-5) vs bias降精bug(|err|~1e-3);max_diff=1.99是大值无害;判据本身不误杀正确核
- [仓库 mxfp4 测试不可参照](repo-mxfp4-tests-broken.md) — 仓库无真实 e2m1 解码；op-plugin 4 个 mxfp4 测试 NameError 从未跑、数据未打包，不可作 fp4 调用参照

## 通用知识：硬件 / 调优
- [硬件规格速查](hardware-specs.md) — 已降级为指针：原始数字真值在 `AscendC_platform/*.ini` + skill `ascendc-hardware`；本条只留跨代 Tiling 影响归纳（旧表核数曾漂移 36/32/28）
- [核间同步](expert-cross-core-sync.md) — A5 CrossCoreSetFlag/WaitFlag 死锁排查三步法、mode=3
- [性能调优](expert-performance-tuning.md) — A5 性能不升反降排查清单(错位分核负优化/CCU/Tiling 沿用/ND DMA)+通用流程
- [Cube-Vector 融合](expert-cv-fusion.md) — L0C2UB/UB2L1 直连消除 GM 往返、切 K 累加、后处理融合管线、同步时序
- [Tiling 策略](expert-tiling-patterns.md) — TilingKey 新旧系统、双/三缓冲判定、分核策略、量化 scale 的 UB 预算、A5 MX 量化
- [常见陷阱](expert-common-pitfalls.md) — 类型推导/A5 删接口/arch 命名/aclnn 两段式；编译/安装陷阱机制见 skill `ascendc-build-errors`，语义陷阱见 `ascendc-data-context`

## 通用知识：A2A3 (910B/910C)
- [分核策略](a2a3-core-split.md) — GMM 对角线分核、Attention S1/S2/B 多维分核、QuantGMM 2D Split
- [流水同步](a2a3-pipeline-sync.md) — 三缓冲流水线、ProcessVec1 事件链、Events 全谱系、GM Workspace 双缓冲
- [双缓冲](a2a3-double-buffer.md) — BigDoubleBuffer、TQue ping-pong、UB 分配、Triple Buffer、Pre-deferred MMCompute
- [Tiling 优化](a2a3-tiling-optimization.md) — TilingKey 旧系统位域、TilingData 层级、min-MTE2 分核、GEMV 阈值、workspace-split、静态 Tiling

## 通用知识：A5 (950)
- [分核策略](a5-core-split.md) — 无需错位、三种分核模式、GMM ASWT 对角线分组、容量反推 Tiling
- [流水同步](a5-pipeline-sync.md) — 四阶流水线、CrossCoreSetFlag/WaitFlag(mode 2/4)、cgmct 框架、TSCM 双缓冲、死锁排查
- [Regbase 优化](a5-regbase-optimization.md) — VF 寄存器计算替代 UB、LoadDist/StoreDist、IFA/RoPE/Norm 融合、Regbase vs Membase
- [SIMT 优化](a5-simt-optimization.md) — SIMT vs SIMD、线程独立分支适用场景、与 Regbase 配合

## 通用知识：迁移 (A2A3 → A5)
- [Attention 迁移](migration-attention.md) — FlashAttention TilingKey 简化、分核模式、Buffer 容量变化、CV 通路、IFA Regbase 化
- [GMM 迁移](migration-gmm.md) — cgmct 框架、AICPU→CCU 通信、SwiGLU V1→V2、模板参数扩展
- [PosEmbedding 迁移](migration-posembedding.md) — Membase→Regbase、TilingKey 20000+ 体系、融合算子流水线、迁移 Checklist

## 通用知识：CANN 构建
- [custom 构建陷阱](cann-custom-build-gotchas.md) — 自建 custom 包 tiling-parse 失败=legacy.so dlopen(非缺注册)；es whl 离线编 build-isolation 卡死与绕过
- [binary opc_cmd 静默失败](binary-opc-silent-return-swallow.md) — build_binary_opc.sh 裸 return 吞 gen_task 退出码→空 opc_cmd.sh→OPC_NUM=0→1 variant；omni 已修 return+build.sh guard，nn 未修；真因是 op 未进 autogen ini/csv

## 工作流偏好
- [AscendC 工作流偏好](ascendc-workflow-prefs.md) — 改码前先讨论达成一致、跑服务器前先取得同意、小步迭代、先改先编 ops-nn 再 ops-tensor
- [设计文档保持纯净](feedback-design-doc-pristine.md) — 实施方案.md 当纯设计文档；施工进度另起单独文档

## 项目与环境
- [omni AiInfraMatmul 迁移](omni-aiinframatmul-migration.md) — =ops-nn mat_mul_v3 改名;fp32+NZ 全错根因(GetMatMulOp 丢 NdNzNd ReFormat)、IS_ND_NZ_FP32 dispatch、isNzB 误导、残留风险
- [QBMM-Batch 项目档案](qbmm-batch-status.md) — quantBatchMatmulV3 MX scale 加 batch 维(A5/950)；已上板验证完成(Blaze+CMCT)；指向其产出的通用知识
- [QBMM MX batchA合轴守卫](qbmm-mx-batchA-mfusion-guard.md) — scale-batch提交误删CheckFusionBatchA的IsMicroScaling禁合轴守卫→A_batch-B_no_batch合M走WITHOUT_BATCH kernel→TilingKey 8196应变10500;含arch35 TilingKey位域解码
- [工作区布局](workspace-layout.md) — ops-nn/ops-tensor/op-plugin/ops-transformer 兄弟仓角色、skills/、agent-memory、算子分类、芯片支持
- [设计规范来源](spec-sources.md) — skills/、ops-transformer docs/zh、在线 AscendC API 指针
- [MM-fp32nz-aclgraph 项目范围](mm-fp32nz-aclgraph-project-scope.md) — 阶段1只验 nn 仓 torch.matmul fp32+NZ 在 ACLGraph 真实行为(不动 omni fix);无NPU无nn源码,板上出真值
