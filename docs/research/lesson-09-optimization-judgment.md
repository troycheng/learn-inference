# 第 9 课研究笔记：怎样判断一种推理优化是否有用

第 9 课不应变成优化名词清单。每种方法都要放回同一条链路，回答五个问题：

```text
它改了哪个对象？
少算、少存或少搬了什么？
新增了什么成本？
什么 workload 才能兑现收益？
应该看哪些指标验证？
```

## 总览

| 优化 | 直接改变的对象 | 最可能改善 | 主要代价或失效条件 |
| --- | --- | --- | --- |
| 权重量化 | 权重表示、读取字节、GEMM Kernel | 模型容量、Decode 带宽、单机可部署性 | 精度、反量化、Kernel/硬件不匹配 |
| KV/状态量化 | 每请求状态的表示和读取字节 | 长上下文容量、并发、Attention Decode | 精度、scale 开销，只覆盖被量化的状态 |
| FlashAttention | Full Attention 的分块和 HBM 读写 | 长序列 Prefill、Attention 临时显存 | 不优化 FFN/Gated DeltaNet，不消除自回归 |
| Prefix Cache | 跨请求复用相同前缀状态 | 重复长前缀的 TTFT 和 Prefill 吞吐 | 命中率、块粒度、驱逐、混合状态支持 |
| Batching | 一轮一起处理的 token 数和请求组成 | GPU 利用率、总体吞吐 | 排队、尾延迟、Padding/调度开销 |
| TP | 同一层矩阵和计算在多卡分片 | 模型可部署性、单请求算力 | 每层集合通信、小 Batch 效率、拓扑 |
| EP | 专家权重和 token 在设备间分布 | MoE 容量、专家带宽、吞吐 | All-to-All、负载倾斜、小专家 GEMM |
| 推测解码/MTP | 一次目标模型前向验证多个候选 | 低并发 TPOT、串行步数 | 草稿成本、拒绝浪费、额外状态和 Batch 占用 |

下面逐项说明。

## 1. 权重量化：少搬权重不等于自动更快

### 改变了什么

权重量化把 BF16/FP16 权重编码为 INT8、INT4、FP8、FP4 等更低位格式，并附带 scale 等量化参数。理想 payload 为：

```text
BF16：2 Byte / 参数
INT8：1 Byte / 参数
INT4：0.5 Byte / 参数
```

Qwen3.5-35B-A3B 的 BF16 tensor payload 约 66.97 GiB，纯 INT4 编码的理想值约 16.74 GiB。实际检查点会更大，因为 scale、分组元数据、对齐和部分未量化权重仍需空间。

Weight-only 量化通常让激活保持 BF16/FP16。Kernel 读取压缩权重后，在寄存器或片上存储中反量化并完成矩阵运算。AWQ 原论文就是 activation-aware 的低比特 weight-only 方法，并同时强调：理论容量下降只有配合权重打包、融合反量化和硬件优化 Kernel，才能转成实测加速。[AWQ v6](https://arxiv.org/abs/2306.00978v6) [vLLM 量化支持，revision `653ebb5`](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/docs/features/quantization/README.md)

### 什么时候容易收益

- 模型原来放不进目标设备，量化使单机或更小并行度部署成为可能。
- 小 Batch Decode 需要反复从 HBM 读取大部分活跃权重，权重带宽是主要瓶颈。
- 硬件和推理框架有成熟的对应低比特 GEMM/GEMV Kernel。
- 省下的显存可以换成更多 KV Cache 或更高并发。

### 什么时候不一定收益

- 大 Batch/长 Prefill 的矩阵乘已经偏计算受限，低比特路径的算力、反量化或数据布局未必更快。
- 目标 GPU 不原生支持该格式，Kernel 需要昂贵的格式转换。
- 某些层回退到通用实现，频繁 Kernel 切换抵消收益。
- MoE 的低 Batch 请求把 token 分散到多个专家，每个专家 GEMM 太小，量化带宽收益被调度和 Kernel 开销掩盖。

### 验证

不能只看模型文件大小。至少同时比较：

```text
质量：目标任务指标、困惑度或固定回归集
容量：真实 GPU 权重占用、剩余 Cache 容量
延迟：同输入输出长度和并发下的 TTFT、TPOT
吞吐：达到同一 SLO 时的 request/s 或 token/s
执行：量化 Linear 是否都落到预期 Kernel，有无 fallback
```

## 2. KV 量化：省的是请求状态，不是模型权重

Full Attention KV Cache 的逻辑公式是：

$$
2L_{full}N_{kv}TDs
$$

把 KV 从 BF16 的 2 Byte 改为 FP8 的 1 Byte，理论有效载荷约减半。它可以提高长上下文并发，并减少 Decode 读取历史 K/V 的字节数。但它不会：

- 缩小模型权重；
- 减少 QK/AV 的数学运算数量；
- 自动量化 Qwen3.5 Gated DeltaNet 的 `conv_state` 与 `recurrent_state`；
- 保证质量完全不变。

Qwen3.5-9B 只有 8 个 Full Attention 层，35B-A3B 只有 10 个。评估 KV 量化时应使用这部分状态，而不是把全部 Decoder Layer 都当成 KV 层。

需要检查 scale 是静态还是动态、按 tensor/头/块怎样分组，Kernel 是否原生读取量化 Cache，以及长上下文质量是否退化。若并发主要受固定 shape 的 Gated DeltaNet 状态限制，只量化 KV 的收益会小于对纯 Transformer 的直觉。

## 3. FlashAttention：改变数据搬运，不改变 Attention 含义

标准实现容易把 `QK^T` 分数和 Softmax 概率矩阵写回 HBM。FlashAttention 把 Q/K/V 分块装入片上 SRAM，在块内维护 Softmax 统计量并直接累计输出，避免在 HBM 完整物化巨大的 `T x T` 中间矩阵。它是精确 Attention，不是近似 Attention。[FlashAttention 原论文 v2](https://arxiv.org/abs/2205.14135v2)

### 它真正减少的东西

```text
减少：Attention 中间矩阵的 HBM 读写和临时显存
没有减少：模型层数、Q/K/V 投影、Attention 语义、自回归依赖
```

它通常对长序列 Prefill 更重要，因为中间矩阵随序列长度快速增长。单 token Decode 没有 `T x T` 的当前 Query 矩阵，但仍要读取历史 K/V，此时使用的通常是专门的 Paged/Decode Attention 路径，不能把所有 Attention Kernel 都笼统称为 FlashAttention。

对 Qwen3.5 还要增加两条边界：

1. 文本模型只有四分之一层是 Full Attention，FlashAttention 不会直接加速其余 Gated DeltaNet 层。
2. Dense/MoE FFN 常占大量投影计算，Attention 子图快一倍不表示端到端也快一倍。

验证时应同时看 Attention Kernel 时间和端到端 TTFT/TPOT，并按 Prompt 长度分桶。

## 4. Prefix Cache：命中的是完整前缀状态

两个请求若拥有完全相同的 token 前缀，第二个请求可以复用第一个请求已经计算出的前缀状态，跳过对应 Prefill。vLLM 以包含父前缀哈希和当前 token block 的键识别可复用块。[vLLM Prefix Cache 设计，revision `653ebb5`](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/docs/design/prefix_caching.md) [PagedAttention v1](https://arxiv.org/abs/2309.06180v1)

### 它不等于什么

- 不等于 KV Cache。KV Cache 主要供同一请求的后续 Decode 使用；Prefix Cache 让不同请求复用相同前缀状态。
- 不等于语义缓存。文字含义接近但 token 不同，不能直接复用模型内部状态。
- 不会加速没有缓存命中的后缀 Prefill 和后续 Decode。
- 不只需要保存 Full Attention K/V。混合模型还要在正确边界恢复 Gated DeltaNet 状态。

### 收益条件

可用一个粗略乘积判断：

$$
收益\propto命中率\times可复用token数\times每token\ Prefill成本
$$

系统 Prompt、固定工具说明、多轮会话和共享文档前缀往往适合。请求前缀差异很早、缓存容易被驱逐或 hash/block 粒度不合适时，命中率会很低。

Qwen3.5 是 Full Attention 与 Gated DeltaNet 混合模型。vLLM 官方 Qwen3.5 配方截至固定 revision 仍把 Mamba/GDN `align` 模式的 Prefix Cache 标为 experimental。课程可以讲通用原理，但不能承诺某个 runtime 版本对 Qwen3.5 已有与纯 Attention 模型完全相同的命中行为。[vLLM Qwen3.5 配方，revision `689d6b9`](https://github.com/vllm-project/recipes/blob/689d6b98c05ec4e92523a231afe9dce97e5d83dc/Qwen/Qwen3.5.md)

验证需记录真实缓存命中 token 数、TTFT、首次与重复请求差异、缓存占用和驱逐率，而不是只确认开关已打开。

## 5. Batching：用等待换利用率

Linear 输入从单 token 的 `[1,H]` 变为多个 token 的 `[M,H]` 后，同一份权重可以服务更多输入，通常能提高矩阵运算规模和权重读取复用。Batching 因此主要改变每次执行包含多少工作，不改变模型数学。

### Static Batching 的问题

一批请求如果必须一起开始、一起结束，短请求完成后可能仍要等待长请求，设备槽位不能及时补入新工作。

### Continuous Batching 的改变

Orca 的 iteration-level scheduling 在生成迭代边界重新组成 Batch，让完成的请求退出，等待请求进入。这样减少不同输出长度带来的空转。[Orca 原论文与系统页面](https://www.usenix.org/conference/osdi22/presentation/yu)

但“Continuous Batching”本身只说明 Batch 成员可以动态变化，不自动保证 Prefill chunk 与 Decode token 能放进同一次模型执行。后者还需要调度器、模型执行器和 Kernel 支持。Sarathi-Serve 使用 Chunked Prefill 和 stall-free batching 来改善 Prefill/Decode 干扰，正说明混合两类工作不是一个开关就自然完成。[Sarathi-Serve v3](https://arxiv.org/abs/2403.02310v3)

### 典型交换

```text
更大 Batch：吞吐通常更高，但请求可能排队更久
更小 Batch：更快开始执行，但权重复用和 GPU 利用率可能更低
优先 Decode：改善正在生成请求的 TPOT，可能推迟新请求 TTFT
优先 Prefill：新请求更快进入，可能造成 Decode 抖动
```

优化目标应写成“在某个 TTFT/TPOT/P99 SLO 下最大化吞吐”，而不是只追求离线 token/s。

## 6. Tensor Parallel：少算一部分，层层做通信

TP 把同一层的权重矩阵按列或行分到多张 GPU。以 FFN 为例，可以把第一组投影按输出列切分，各卡独立完成激活，再把 `down_proj` 按输入维切分并聚合结果。Attention 的 Q/K/V 头也可以分片。

Megatron-LM 的经典做法把同一 Transformer Layer 内的列并行和行并行组合，使中间激活不必在每个 Linear 后都完整聚合，但每个层块仍需少量 All-Reduce。[Megatron-LM v4](https://arxiv.org/abs/1909.08053v4) [vLLM 并行部署说明，revision `653ebb5`](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/docs/serving/parallelism_scaling.md)

### 收益

- 单卡放不下的层权重可以分片。
- 单请求或单 Batch 的大矩阵计算由多卡分担。
- Attention 头和部分 KV 状态可以按头分布到设备。

### 成本

- 每层或每个子层的集合通信进入关键路径。
- TP 越大，每卡 GEMM 可能越小，Kernel 效率下降。
- 单 token、小 Batch Decode 的计算量很小，通信延迟更容易占主导。
- PCIe、NVLink、节点间 IB 的带宽和延迟不同，不能只看 GPU 型号和理论 FLOPS。

判断 TP=2 是否比单卡快，至少要比较：

$$
单卡节省的计算和权重读取时间\quad vs\quad新增的集合通信和同步时间
$$

TP 首先是“让模型放得下并缩短单层计算”的手段，不是卡数翻倍、吞吐必然翻倍。

## 7. Expert Parallel：只把专家分开

EP 将不同 Routed Expert 放到不同设备。Router 为 token 选出专家后，系统把 token 特征发送到专家所在设备，计算后再把结果发回原序列位置。跨设备实现通常包含 dispatch 和 combine 两段 All-to-All。[DeepSpeed-MoE v2](https://arxiv.org/abs/2201.05596v2) [vLLM EP 部署说明，revision `653ebb5`](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/docs/serving/expert_parallel_deployment.md)

Qwen3.5-35B-A3B 每层有 256 个 Routed Expert，每 token 选 8 个，还会经过 Shared Expert。EP 主要分片 Routed Expert，不会让 Attention、Gated DeltaNet、Router 和 Shared Expert 自动消失。

### 适合的情况

- 全部专家权重无法在单卡或较小 TP 组中容纳。
- Batch 中 token 足够多，可以形成较大的 grouped GEMM。
- 专家负载较均衡，或者系统有可靠的负载监控和重映射机制。
- 卡间互联能承受 token dispatch/combine。

### 常见失效点

- 低并发下每个专家只收到少量 token，小 GEMM 和 All-to-All 延迟占主导。
- Router 热点造成某些 GPU 排队，整层等待最慢专家。
- Top-K 较大时，每个 token 需要发送到更多专家，通信量上升。
- 跨节点 EP 的网络成本可能高于专家计算节省。

不要把“每 token 只激活少量专家”直接推导成“只需读取这几个专家且无通信”。一个 Batch 中不同 token 可能覆盖很多专家，实际权重流量和负载分布取决于路由结果。

## 8. 推测解码：减少目标模型的串行调用次数

推测解码先用更便宜的 Drafter 提出多个候选 token，再让 Target Model 在一次较大的前向中并行验证这些位置。目标模型从前往后接受候选，遇到第一个拒绝后修正并重新开始。

严格的 speculative sampling 使用接受概率和修正分布，可以保持目标模型原本的输出分布。简单地“草稿与目标贪心 token 相同就接受”只覆盖贪心场景，不能代表一般采样下的无损证明。[Speculative Decoding 原论文 v2](https://arxiv.org/abs/2211.17192v2)

### 粗略收益条件

设每轮提出 `k` 个候选，平均接受 `a` 个，草稿成本为 `C_draft`，目标验证成本为 `C_verify(k)`。只有当：

```text
获得 a+1 个 token 的总成本
小于目标模型逐个生成 a+1 次的成本
```

才有实际加速。它要求候选接受率高、Drafter 足够便宜，而且 Target 一次验证多个位置明显便宜于多次单 token Decode。

### 新增成本

- Drafter 权重、计算和状态。
- 为候选 token 预留的 lookahead Cache。
- 被拒绝候选的无效计算。
- Batch 扩张、回滚和调度复杂度。
- 高并发时占用本可服务其他请求的 token budget。

因此，低并发下 TPOT 变好，不代表高并发总体吞吐也会变好。

## 9. MTP：内置 Drafter，但仍要验证

Multi-Token Prediction 在训练中加入辅助模块，让一个位置除下一个 token 外，还学习预测更远的 token。DeepSeek-V3 报告说明，MTP 模块可以在普通推理时丢弃，也可以作为推测解码 Drafter 使用。[DeepSeek-V3 MTP v2](https://arxiv.org/abs/2412.19437v2)

Qwen3.5 两个配置都有：

```text
mtp_num_hidden_layers = 1
```

对应检查点也保存了一套 MTP 辅助层。启用 MTP 推测解码时，它提出候选，主语言模型仍负责验证；MTP 不是让主 Decoder 违反因果关系一次直接确定未来答案。

Qwen3.5-35B-A3B 官方模型卡给出了 vLLM `method=mtp` 的启动方式。vLLM 官方 Qwen3.5 配方同时提醒：MTP-1 适合低并发、延迟敏感场景，可能降低高并发文本吞吐，因为候选 token 会占用 KV 容量并减小有效 Batch。[Qwen3.5-35B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md) [vLLM Qwen3.5 配方](https://github.com/vllm-project/recipes/blob/689d6b98c05ec4e92523a231afe9dce97e5d83dc/Qwen/Qwen3.5.md) [vLLM 推测解码文档，revision `653ebb5`](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/docs/features/speculative_decoding/README.md)

`mtp_num_hidden_layers=1` 表示检查点含一层 MTP 模块，不保证候选接受率，也不等同于固定加速比例。runtime 还可能重复使用同一 MTP 层提出多个 speculative step，必须按实际配置测试。

## 10. 一套可执行的判断顺序

面对一个新优化方案，可以按下面顺序提问。

### 第一步：定位对象

```text
权重？
Full Attention 中间张量？
KV/GDN 请求状态？
Batch 组成？
单层矩阵分片？
MoE 专家放置？
自回归串行步数？
```

如果说不清直接改变哪个对象，就还不能判断收益。

### 第二步：写出理论上少掉的量

例子：

```text
INT4：每参数理想从 2 Byte 降到 0.5 Byte
KV FP8：Full Attention KV 每元素从 2 Byte 降到 1 Byte
Prefix Cache：命中的前缀 token 不再做 Prefill
TP=2：主要矩阵每卡约保存和计算一半
推测解码：每次 Target forward 平均确认不止 1 个 token
```

### 第三步：列出新成本

反量化、scale、通信、hash、缓存占用、调度、草稿模型、拒绝浪费、质量误差都必须写出来。

### 第四步：找出收益出现的 workload

至少固定或记录：

```text
Prompt/Output 长度分布
并发或到达率
重复前缀比例
Batch token 数
采样参数
GPU 和互联拓扑
SLO
模型 dtype 与 runtime 版本
```

### 第五步：用端到端指标裁决

算子时间只能证明局部变化。最终还要看：

```text
TTFT / TPOT / ITL / P50 / P99
在同一 SLO 下的 request/s 与 token/s
真实 GPU 权重、Cache、临时显存
跨卡通信时间和链路带宽
目标任务质量
```

## 11. 最容易讲错的边界

1. **量化容量下降不保证同比例加速。** 加速依赖 Kernel、硬件和 workload 是否受相应带宽限制。
2. **FlashAttention 不是稀疏或近似 Attention。** 它主要减少 HBM IO 和中间物化，输出语义仍是精确 Attention。
3. **Prefix Cache 不是语义缓存。** 必须命中兼容的 token 前缀和完整模型状态。
4. **Batching 不只带来收益。** 它通常用等待时间换取设备利用率。
5. **Continuous Batching 不自动等于 Prefill/Decode 混批。** 混批需要额外执行支持。
6. **TP 不会消除通信。** 同一层的分片结果需要在关键位置聚合。
7. **EP 只针对专家。** 非专家层仍需复制、TP 或其他并行方式。
8. **MoE 激活参数少不等于所有请求只触碰同一小组专家。** Batch 路由分布决定聚合权重流量。
9. **推测解码不改变目标模型。** 候选仍必须验证，接受率低时可能更慢。
10. **MTP 是可选 Drafter，不是一次确定多个未来 token 的魔法。** 它仍受自回归验证和运行时调度约束。

## 资料来源

- [AWQ: Activation-aware Weight Quantization v6](https://arxiv.org/abs/2306.00978v6)
- [FlashAttention v2](https://arxiv.org/abs/2205.14135v2)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention v1](https://arxiv.org/abs/2309.06180v1)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve v3](https://arxiv.org/abs/2403.02310v3)
- [Megatron-LM v4](https://arxiv.org/abs/1909.08053v4)
- [DeepSpeed-MoE v2](https://arxiv.org/abs/2201.05596v2)
- [Fast Inference from Transformers via Speculative Decoding v2](https://arxiv.org/abs/2211.17192v2)
- [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)
- [vLLM 固定 revision `653ebb5` 文档与源码](https://github.com/vllm-project/vllm/tree/653ebb52dffd8b4653b430302473c771117529f1)
- [vLLM Qwen3.5 配方，revision `689d6b9`](https://github.com/vllm-project/recipes/blob/689d6b98c05ec4e92523a231afe9dce97e5d83dc/Qwen/Qwen3.5.md)
- [Qwen3.5-9B 官方配置，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
- [Qwen3.5-35B-A3B 官方配置与模型卡，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754)
