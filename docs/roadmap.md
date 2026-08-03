# 学习路线：推理系统的 12 节核心课

## 结论

核心课程固定为 **12 节**，预计需要 **约 32～44 个专注小时**。六个心智模型是分析问题的视角，不是六节课；具体论文、框架和优化方法是案例，也不单独占用核心课编号。

这条路线面向已经维护推理服务、希望补齐理论主干的工程师。目标不是记住术语，而是形成一套稳定的判断过程：

```text
先确定模型必须做什么
→ 再计算需要多少工作和数据
→ 再判断硬件上限与实际执行
→ 再分析并发、调度和通信
→ 最后用测量验证优化是否成立
```

正确性、数值误差、业务效果和 SLO 是贯穿全程的约束。

## 为什么是 12 节

课程按知识依赖拆分，而不是按流行术语分类：

1. 不认识完整计算图，就不知道该统计哪些工作；
2. 不会计算工作量、数据量和状态量，就无法判断硬件瓶颈；
3. 不理解 GPU 与 Runtime，就无法解释理论值和实测值的差距；
4. 不会正确测量，就无法证明瓶颈和优化收益；
5. 不理解单请求的自回归和 KV，就无法推导 LLM 调度；
6. 不理解单卡关键路径，就无法计算多卡通信是否值得。

因此，第一课之后不能直接跳到 Roofline、KV Cache 或调度。第二课先补齐“完整计算图与算子族”，第三课才开始算三本账。

## 六个分析视角

| 视角 | 必须回答的问题 | 主要对应课程 |
| --- | --- | --- |
| 计算语义与依赖 | 输入输出是什么？哪些步骤并行，哪些步骤串行？ | 1、2、7 |
| 计算、搬运与状态 | 做多少计算？搬多少数据？保留多少容量？ | 3、8、10 |
| 硬件资源上界 | 受计算、带宽、延迟、容量还是并行度限制？ | 4 |
| 图到 Kernel 的执行 | 图怎样被专化、融合、选 Kernel 并真正运行？ | 5、6 |
| 并发、排队与调度 | Batch 为什么同时改变吞吐、等待和尾延迟？ | 9 |
| 分片、通信与同步 | 扩卡减少了什么，又引入了哪些通信和等待？ | 11 |

第 12 课把六个视角合并成一次完整判断。

## 课程总览

| 阶段 | 课次 | 主题 | 学完后的能力 | 预计时间 | 状态 |
| --- | ---: | --- | --- | ---: | --- |
| 一、看懂模型 | 1 | 张量、算子、依赖与 Linear | 能标注 shape、依赖和并行边界 | 2～3 小时 | 已完成 |
| 一、看懂模型 | 2 | 完整计算图与核心算子族 | 能从请求画到模型输出，并识别主要算子族 | 2～3 小时 | 已完成 |
| 二、建立性能模型 | 3 | 计算、搬运与状态三本账 | 能完成数量级成本估算 | 3～4 小时 | 下一课 |
| 二、建立性能模型 | 4 | GPU 执行模型与性能上界 | 能区分计算、带宽、延迟和并行度限制 | 3～4 小时 | 待学习 |
| 二、建立性能模型 | 5 | 计算图如何变成 GPU 工作 | 能解释专化、融合、Kernel 选择和动态 Shape | 2～3 小时 | 待学习 |
| 二、建立性能模型 | 6 | 测量、Timeline 与证据闭环 | 能选择指标和工具验证瓶颈 | 2～3 小时 | 待学习 |
| 三、理解 LLM 链路 | 7 | Decoder-only LLM 的单请求链路 | 能解释一次 Prefill 和逐步 Decode | 2～3 小时 | 待学习 |
| 三、理解 LLM 链路 | 8 | Attention 与 KV Cache 的成本 | 能估算 KV，并判断相关优化的作用条件 | 3～4 小时 | 待学习 |
| 三、理解 LLM 链路 | 9 | 在线服务、Batching 与调度 | 能从 workload 和 SLO 判断调度方案 | 3～4 小时 | 待学习 |
| 四、做优化判断 | 10 | 数值精度与量化 | 能判断精度、容量、带宽和计算的交换 | 2～3 小时 | 待学习 |
| 四、做优化判断 | 11 | 多 GPU、通信与集群扩展 | 能画出分片、集合通信和关键路径 | 3～4 小时 | 待学习 |
| 四、做优化判断 | 12 | 综合案例与优化决策闭环 | 能独立提出、验证和复盘优化方案 | 4～6 小时 | 待学习 |

## 第一阶段：看懂模型必须做什么

### 第 1 课：张量、算子、依赖与 Linear 的成本直觉

**核心问题**：一次推理的最小组成是什么？`T`、shape 和数据依赖分别意味着什么？

**学习内容**：

- 张量的 shape、dtype、layout 和存储大小；
- 算子、数据依赖、串行关键路径和批内并行；
- Linear 的 `FLOPs`、权重容量和 `T` 的关系；
- `FLOPs` 与 `FLOP/s` 的量纲区别；
- Prefill 和 Decode 的第一层性能直觉。

**本课边界**：只用 Linear 建立数量级直觉，不展开完整 Transformer、Roofline 或 KV 公式。

**通过标准**：给定一个 Linear 和输入 shape，能写出输出 shape、计算量、权重大小，并判断哪些 token 能并行。

**正文**：[第一课：张量、算子、依赖与 Linear 的成本直觉](lessons/01-tensors-operators-and-dependencies.md)

### 第 2 课：完整计算图与核心算子族

**核心问题**：一个请求从输入到输出到底经过哪些节点？不同算子在图中承担什么角色？

**学习内容**：

- 计算图的节点、边、参数、常量、中间值和拓扑顺序；
- shape 传播、广播、reshape/transpose 和 layout 变化；
- GEMM/GEMV、Conv、Attention、Reduction、Elementwise、Gather/Scatter；
- Tokenizer、Resize、Decode、NMS、Sampling 等前后处理；
- 用一个 Decoder Block 和一个 OCR/CV 图识别算子族、依赖与活跃张量。

**本课边界**：只回答“算什么、按什么顺序算”，暂不计算完整 FLOPs/Bytes，也不讨论具体 Kernel 快慢。

**通过标准**：能从模型图中圈出主干、分支、可融合邻接段和关键路径，并说明每个主要张量为何存在。

**正文**：[第二课：完整计算图与核心算子族](lessons/02-computation-graphs-and-operator-families.md)

**主要资料**：

- [ONNX Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 第二阶段：从理论成本走到真实执行

### 第 3 课：计算、搬运与状态三本账

**核心问题**：一次推理需要做多少计算、搬多少数据、保留多少容量？

**学习内容**：

- 计算账：GEMM/GEMV、Conv、Attention、Reduction 的数量级；
- 搬运账：权重、输入输出和中间值的理想最小流量与实际流量；
- 状态账：权重、激活、KV、workspace、通信 buffer 和碎片；
- 算术强度 `FLOPs/Byte` 及其假设；
- 缓存、重计算、量化和 Fusion 分别改变哪本账。

**本课边界**：先建立算法级成本模型，不把理论字节数误称为实测 HBM 流量。

**通过标准**：能为一个 Decoder Block 和一个 OCR 子图列出三本账，写清单位、假设和数量级。

**主要资料**：

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

### 第 4 课：GPU 执行模型与性能上界

**核心问题**：为什么同样的 FLOPs 在不同 shape、batch 和 GPU 上耗时不同？

**学习内容**：

- Host、Device、Kernel、Grid、Block、Warp、SM 和 SIMT；
- Register、Shared Memory、Cache、HBM 与主机—设备链路；
- 峰值计算速率、带宽、延迟、并行度和占用率；
- Roofline、ridge point 和分层 Roofline；
- Kernel launch、同步、尾块、分支和小工作量造成的效率损失。

**本课边界**：目标是读懂性能现象，不要求编写 CUDA 或学习 PTX/ISA。

**通过标准**：给定三本账和硬件规格，能估计理论下界，并列出不能达到下界的主要原因。

**主要资料**：

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Nsight Compute：Roofline](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#rooflines)

### 第 5 课：计算图如何变成 GPU 工作

**核心问题**：模型图经过哪些步骤，才变成 Timeline 上的 Kernel 和 Memcpy？

**学习内容**：

- 模型导出、图优化、常量折叠和子图分区；
- shape、dtype、layout 专化与 tactic/Kernel 选择；
- 算子 Fusion、Plugin、自定义 Kernel 和回退路径；
- 内存规划、workspace、stream 与 CUDA Graph；
- 动态 shape、optimization profile 和首次运行成本。

**本课边界**：用 TensorRT/ONNX 说明通用机制，不学习一套框架的全部参数。

**通过标准**：能把“模型节点少了”“Kernel 变了”“动态 shape 变慢”分别定位到正确层次。

**主要资料**：

- [TensorRT Inference Library](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/)
- [TensorRT Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)

### 第 6 课：测量、Timeline 与证据闭环

**核心问题**：怎样证明瓶颈在哪里，而不是根据单个利用率指标猜测？

**学习内容**：

- workload 的 batch、shape、输入长度、输出长度和到达分布；
- 冷启动、warmup、稳态、同步方式和测量边界；
- 端到端、Server、Model、GPU、Kernel 五层时间；
- latency、throughput、p50/p99、TTFT、TPOT/ITL 和 goodput；
- Nsight Systems 看系统 Timeline，Nsight Compute 看单 Kernel；
- 假设、观测、反证、修改和复测的最小闭环。

**本课边界**：工具是取证手段，不把某个 profiler 指标当作通用结论。

**通过标准**：能为一次性能回归设计可复现的基线，并说明每条证据支持或排除了什么。

**主要资料**：

- [TensorRT Performance Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)

## 第三阶段：理解自回归 LLM 的特殊链路

### 第 7 课：Decoder-only LLM 的单请求链路

**核心问题**：Prompt 怎样变成第一个 token，后续 token 为什么必须逐步生成？

**学习内容**：

- Tokenizer、Embedding、Decoder Blocks、LM Head、Logits 和 Sampling；
- Norm、Q/K/V Projection、Attention、MLP、Residual 的依赖；
- Causal Mask 与自回归条件概率；
- Prefill、首次 Decode 和后续 Decode 的输入输出；
- 请求内串行、单步内部并行和请求间并行的区别。

**本课边界**：先建立正确的执行语义，暂不展开 KV 内存管理和在线调度。

**通过标准**：能画出一个请求从 Prompt 到多个输出 token 的时序图，并指出每一步可复用的结果。

**主要资料**：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

### 第 8 课：Attention 与 KV Cache 的成本

**核心问题**：上下文增长时，Attention 的计算、KV 容量和读取流量怎样变化？

**学习内容**：

- Q、K、V、Attention Score、Softmax 和输出投影的 shape；
- Prefill Attention 与 Decode Attention 的不同成本；
- KV Cache 的逐层公式、生命周期、容量和每步读取；
- MHA、GQA、MQA 的核心差别；
- FlashAttention、PagedAttention 和 Prefix Cache 分别解决哪本账。

**本课边界**：第一轮不系统展开所有 Attention 变体、MLA、稀疏 Attention 或长上下文论文。

**通过标准**：给定模型配置、并发和上下文长度，能估算 KV 容量，并判断容量瓶颈和带宽瓶颈是否混淆。

**主要资料**：

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

### 第 9 课：在线服务、Batching 与调度

**核心问题**：为什么提高吞吐常常会增加等待和尾延迟，调度器到底在分配什么？

**学习内容**：

- 到达率、服务时间、队列等待、利用率和 Little's Law；
- request throughput、token throughput、TTFT、TPOT/ITL 和 SLO goodput；
- 静态 Batch、Continuous Batching 和 iteration-level scheduling；
- token budget、KV budget、preemption、chunked prefill 和优先级；
- Prefill/Decode 干扰与分离的收益、KV 传输和资源代价。

**本课边界**：先学习可推导的调度约束，不比较某一版本框架的所有调度参数。

**通过标准**：给定请求长度分布、到达率、KV 容量和 SLO，能解释调度策略改善了哪个指标、牺牲了什么。

**主要资料**：

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [DistServe](https://arxiv.org/abs/2401.09670)

## 第四阶段：把原理用于优化与扩展

### 第 10 课：数值精度与量化

**核心问题**：低精度为什么可能更快、更省，又为什么有时没有收益或损害效果？

**学习内容**：

- 表示范围、有效精度、舍入、溢出和累加；
- FP32、TF32、FP16、BF16、FP8、INT8、INT4 的核心差异；
- Weight-only、Weight-Activation 和 KV 量化；
- scale、granularity、对称/非对称、静态/动态和 calibration；
- Q/DQ、转换开销、硬件支持、Kernel 可用性与精度验证。

**本课边界**：掌握统一原理，不背完所有厂商格式或每种量化算法。

**通过标准**：能说明一个量化方案改变了哪些字节、哪些运算和哪些误差，并设计性能与效果的双重验收。

**主要资料**：

- [TensorRT Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [AWQ](https://arxiv.org/abs/2306.00978)

### 第 11 课：多 GPU、通信与集群扩展

**核心问题**：增加 GPU 后，每张卡少算了什么，又新增了哪些通信、同步和空泡？

**学习内容**：

- Data、Tensor、Pipeline、Expert 和 Context Parallel 的分片对象；
- AllReduce、AllGather、ReduceScatter、AllToAll 和点对点通信；
- 消息大小、带宽、单次延迟、频率、拓扑和关键路径；
- Pipeline bubble、负载不均、跨节点链路和副本路由；
- 单实例模型并行、跨实例数据并行与 Prefill/Decode 分离的边界。

**本课边界**：先会计算通信与同步，不深入 NCCL 算法、协议和环境变量调优。

**通过标准**：能为一种并行方案画出每卡权重、激活、KV 和通信路径，并判断扩卡收益何时被通信抵消。

**主要资料**：

- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

### 第 12 课：综合案例与优化决策闭环

**核心问题**：面对一个真实性能问题，怎样从现象走到可验证的优化结论？

**学习内容**：

- 固定 workload、硬件、软件版本、SLO 和正确性基线；
- 画计算图与关键路径，完成三本账和理论下界；
- 读取端到端 Timeline，定位主要差距所在层次；
- 提出一个优化假设，明确收益、代价和适用条件；
- 用对照实验验证性能、容量、尾延迟、稳定性和效果；
- 把结论写成可复查的工程判断，而不是只报告一个加速比。

**贯穿案例**：

1. 一个常规 Decoder-only LLM，用于自回归、KV、调度和多卡；
2. 一个常规 OCR/CV TensorRT 模型，用于完整计算图、动态 shape、前后处理、Plugin 和 Fusion。

**通过标准**：能独立完成“问题定义 → 理论估算 → 证据定位 → 方案选择 → 实验验证 → 适用边界”的报告。

## 第一轮明确不学什么

以下内容有价值，但不是掌握主线的前置条件：

- 训练、反向传播和优化器；
- CUDA ISA/PTX 与手写高性能 Kernel；
- 每一种量化格式和每一种 Attention 变体；
- MoE、MLA、稀疏 Attention 的全部实现细节；
- 每个推理框架的参数清单；
- NCCL 内部算法、协议和网卡调优；
- 每篇最新调度论文的具体策略。

完成 12 节后，再按工作需要选择 Speculative Decoding、MoE/MLA、长上下文与 Offload、Prefix Cache、Triton/CUDA Kernel、超低比特量化或多模态推理等专题。它们都应从六个分析视角推导，不改变核心课编号。

## 每节课的固定写法

后续正文统一采用以下结构：

```text
核心问题
→ 必要概念与量纲
→ 一个最小定量例子
→ LLM 与 OCR/CV 两类工程映射
→ 常见误判
→ 理解检查
→ 学习过程中的疑问与解答
→ 原始资料
```

课程变更遵循两条规则：

1. 开始新课前，先检查本路线中的前置知识、边界和通过标准；
2. 若必须调整核心顺序，先更新本文件和 README，再编写正文，避免在对话中临时改变课次。

## 资料选择原则

- 数学与算法结论优先引用原始论文；
- 图格式、Runtime、CUDA 和通信语义优先引用官方规范与文档；
- 版本相关实现只作为案例，写作时标明版本和测量条件；
- 论文中的加速比只证明论文实验范围内的结果，不直接外推到其他 workload。
