# 学习路线：推理系统的核心主线

## 目标

这条路线不以“记住多少术语”为目标，而是训练一种判断能力：拿到一个模型、服务或优化方案后，能够沿着计算依赖、数据与状态、硬件上界、实际执行、并发调度和分布式通信逐层分析。

推理系统的主线可以概括为：

```text
模型语义与依赖
  → 计算量、数据量与状态
  → 硬件资源上界
  → 计算图到 Kernel 的实际执行
  → 并发、排队与调度
  → 分片、通信与同步
```

正确性、数值误差和业务效果是贯穿所有层次的约束。

## 六个核心心智模型

### 1. 计算语义与依赖

需要回答：

- 输入和输出张量是什么形状？
- 哪些计算可以并行，哪些必须串行？
- 哪些结果可以缓存、复用或重算？
- 请求的串行关键路径在哪里？

第一阶段只需掌握少数算子族：

- GEMM / GEMV；
- Convolution；
- Attention；
- Reduction，例如 Softmax、Norm、TopK；
- Elementwise，例如激活、残差和位置编码；
- Gather、Scatter 和 Routing；
- Tokenizer、Sampling、Resize、NMS 等前后处理。

LLM 的特殊之处是自回归依赖；MoE 增加了条件路由；扩散模型增加了重复去噪步骤。它们仍然可以用张量和依赖图分析。

### 2. 计算量、数据量与状态

每次分析都要算三本账：

1. **计算账**：需要多少 `FLOPs`，计算量怎样随 batch、序列长度和分辨率变化？
2. **搬运账**：权重、激活和状态在 CPU、HBM、Cache、片上存储和设备之间搬运多少字节？
3. **状态账**：权重、KV Cache、激活、workspace、通信 buffer 和内存碎片分别占多少容量？

很多优化只是这三本账之间的交换。例如，缓存用容量换计算，量化用数值精度换容量、带宽或计算效率，重计算则用计算换状态容量。

### 3. 硬件资源上界

先用下面的近似式判断主要约束：

$$
T_{\text{lower bound}}
\gtrsim
\max\left(
\frac{F}{P},
\frac{B_{\text{HBM}}}{BW_{\text{HBM}}},
\frac{B_{\text{link}}}{BW_{\text{link}}}+N_{\text{msg}}\times L_{\text{link}}
\right)
$$

其中：

- `F` 是计算量，单位为 `FLOPs`；
- `P` 是有效计算速率，单位为 `FLOP/s`；
- `B` 是数据量，单位为 Bytes；
- `BW` 是带宽，单位为 Bytes/s；
- `L` 是单次通信延迟，单位为秒。

实际时间还会包含 Kernel launch、CPU 调度、同步、pipeline bubble、低并行度和实现效率损失。

算术强度定义为：

$$
AI=\frac{FLOPs}{Bytes}
$$

它是判断 workload 更可能受计算还是数据搬运限制的起点，但不是对真实性能的完整预测。

### 4. 计算图到 Kernel 的实际执行

模型图不会直接在 GPU 上运行。典型过程是：

```text
模型代码或模型文件
→ 计算图
→ 图优化与分区
→ shape、dtype 和 layout 专化
→ tactic / kernel 选择
→ 内存规划
→ stream 上的 kernel 与 memcpy
→ GPU 执行
```

需要理解：

- shape、dtype、layout 为什么会改变 Kernel 选择；
- Fusion 为什么可能减少 HBM 往返和 launch；
- 动态 shape 为什么会影响专化和图捕获；
- 大 GEMM、GEMV、小 Reduction 的硬件行为为何不同；
- 理论下界和实测之间的差距来自哪里。

### 5. 并发、排队与调度

单请求 Kernel 很快，不等于在线服务就快。需要区分：

- 服务时间和队列等待时间；
- TTFT、TPOT/ITL 和端到端延迟；
- token throughput 和 request throughput；
- 平均值和 p95/p99；
- 最大吞吐和满足 SLO 的 goodput。

两个基础关系是：

$$
\rho=\frac{\lambda}{\mu}
$$

当到达率 `λ` 接近服务能力 `μ` 时，排队和尾延迟通常会快速恶化。

$$
L=\lambda W
$$

平均在途请求数 `L` 等于到达率 `λ` 乘以平均停留时间 `W`。

Batching 的本质是用等待时间和状态容量换权重及计算复用。Continuous Batching、Chunked Prefill 和优先级调度都是在异长请求、动态到达和有限状态预算下的具体策略。

### 6. 分片、通信与同步

学习并行方案时不先背名称，而是统一回答：

1. 权重和请求状态放在哪里？
2. 每张设备计算什么？
3. 设备之间传输什么数据？
4. 通信量和通信频率分别是多少？
5. 通信是否位于关键路径？
6. 是否引入同步、负载不均或 pipeline bubble？

TP、PP、DP、EP、CP 和 Prefill/Decode 分离，都是容量、计算时间、通信和同步之间的不同交换。第一阶段只需掌握 AllReduce、AllGather、ReduceScatter 和 AllToAll 的语义。

## 为什么具体技术不是主线本身

具体技术应该作为六个心智模型的练习题：

| 方法 | 它改变了什么 | 主要成立条件 |
| --- | --- | --- |
| FlashAttention | 减少 Attention 中间结果的 HBM 读写 | Attention 的 IO 成本足够显著 |
| PagedAttention | 减少 KV Cache 的预留和碎片浪费 | KV 容量限制有效并发 |
| 权重量化 | 减少权重容量和读取字节，或使用更快低精度单元 | 权重带宽或计算是瓶颈，精度可接受 |
| Kernel Fusion | 减少中间张量读写和 Kernel launch | 小算子或内存往返占比较高 |
| Speculative Decoding | 用额外计算和验证减少大模型串行步数 | Decode 昂贵且候选接受率足够高 |
| Tensor Parallel | 用集合通信换单卡容量和计算时间 | 分片收益超过通信与同步成本 |
| Prefill/Decode 分离 | 用 KV 传输和额外调度换阶段隔离 | 两阶段干扰或资源需求差异明显 |

遇到任何新优化，都应回答：

1. 它改变的是计算量、数据量、状态容量、串行步数、调度还是通信？
2. 它作用于哪个阶段、哪些 shape 和哪类 workload？
3. 它成立的瓶颈前提是什么？
4. 它减少了什么资源消耗？
5. 它增加了什么开销？
6. 它改善的是延迟、吞吐还是 SLO goodput？
7. 它是否改变数值结果、概率分布或业务效果？

## 学习顺序

建立能够用于技术判断的基础主线，预计需要约 25～35 个专注小时。

| 阶段 | 核心问题 | 完成标准 |
| --- | --- | --- |
| 1. 张量、算子与依赖 | 模型必须计算什么，哪里可以并行？ | 能标注关键张量 shape 和依赖 |
| 2. 三本账 | 一次推理要做多少计算、搬多少数据、保留多少状态？ | 能完成一个模型的数量级估算 |
| 3. GPU 与 Roofline | 为什么没有达到理论算力？ | 能区分计算、带宽、延迟和并行度限制 |
| 4. Runtime 与 Kernel | 图怎样变成真正的 GPU 工作？ | 能读懂一条基本 timeline |
| 5. LLM 专题 | Prefill、Decode、KV 和 Attention 有何不同？ | 能估算 KV，并解释两阶段性能差异 |
| 6. 在线服务 | Batch 和调度为什么同时改变吞吐与尾延迟？ | 能从 workload 和 SLO 分析调度方案 |
| 7. 多卡与集群 | 扩卡减少了什么，又增加了什么？ | 能列出计算、状态和通信路径 |
| 8. 综合判断 | 某项优化为何对这个 workload 有效？ | 能形成假设、证据、方案和验证闭环 |

第一轮不需要系统学习训练与反向传播、CUDA ISA/PTX、所有量化格式、所有 Attention 变体、每个框架参数或 NCCL 内部算法。

## 建议的贯穿案例

- 一个常规 Decoder-only LLM：用于学习自回归、KV、调度和多卡；
- 一个常规 OCR/CV 模型：用于学习通用计算图、动态 shape、前后处理、插件和 Fusion。

使用两个案例可以避免把推理理论误学成某个 LLM Serving 框架的使用手册。

## 主要参考资料

- [ONNX Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [TensorRT Inference Library](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/index.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [DistServe](https://arxiv.org/abs/2401.09670)
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
