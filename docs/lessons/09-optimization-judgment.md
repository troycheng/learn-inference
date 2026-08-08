# 第 9 课：一种优化到底有没有用

推理优化的名词很多，但判断方法可以固定下来。遇到任何方案，先回答五个问题：

```text
1. 它直接改变了链路中的哪个对象？
2. 理论上少算、少存或少搬了什么？
3. 新增了什么成本？
4. 哪种 workload 才能兑现收益？
5. 用什么端到端指标裁决？
```

如果第一问都说不清，后面的“提速百分比”通常没有可靠基础。

![常见优化分别改了推理链路中的什么](../assets/09-optimization-map.svg)

这一课不会给每种优化下一个永远成立的结论，而是把量化、FlashAttention、Prefix Cache、Batching、TP、EP 和推测解码放回前八课建立的成本模型中。

## 1. 先建立统一判断表

| 优化 | 直接改变的对象 | 首先可能改善 | 最常见的代价 |
| --- | --- | --- | --- |
| 权重量化 | 权重编码与 Linear Kernel | 权重容量、Decode 权重读取 | 质量、反量化、Kernel 不匹配 |
| KV 量化 | Full Attention 请求状态 | 长上下文容量、KV 读取 | 质量、Scale、只覆盖 KV |
| FlashAttention | Full Attention 中间数据流 | 长序列 Prefill、临时显存 | 只优化 Attention 子图 |
| Prefix Cache | 跨请求的前缀状态复用 | 重复前缀 TTFT | 命中率、驱逐、状态兼容 |
| Batching | 一次执行包含的 token | GPU 利用率、吞吐 | 排队和尾延迟 |
| TP | 同一层矩阵的设备分片 | 单卡容量、单请求计算 | 层间集合通信 |
| EP | Routed Expert 的设备分布 | MoE 容量与吞吐 | Dispatch、负载倾斜 |
| 推测解码 / MTP | 每次目标模型验证的位置数 | 低并发 TPOT | 草稿成本、拒绝浪费、Cache 占用 |

先定位对象，是为了避免把局部收益误认为端到端收益。例如 FlashAttention 只改变 Full Attention 的执行方式，不会直接加速 FFN 和 Gated DeltaNet。

## 2. 权重量化：先问省下的字节能不能被 Kernel 用起来

### 改了什么

BF16 每个参数占 2 Byte。INT8 理想上是 1 Byte，INT4 是 0.5 Byte：

```text
BF16：2 Byte / 参数
INT8：1 Byte / 参数
INT4：0.5 Byte / 参数
```

Qwen3.5-35B-A3B 的 BF16 权重有效载荷约 66.97 GiB。若所有参数都用纯 INT4 编码，理想下限约 16.74 GiB。

Weight-only 量化通常仍让激活保持 BF16 或 FP16。低比特 Kernel 读取压缩权重，再在寄存器或片上存储中完成反量化和矩阵运算。

### 为什么可能更快

小 Batch Decode 经常需要为很少的 token 读取大量活跃权重。若时间主要花在 HBM 搬权重，读取字节减少，延迟就有下降空间。省下的显存还可以换成更多 KV Cache 或更高并发。

### 为什么可能不快

```text
低比特格式没有硬件原生支持
反量化与格式转换代价很大
部分层回退到通用 Kernel
大 Batch GEMM 已偏计算受限
MoE 每个 Expert 的 GEMM 太小
```

权重从 2 Byte 变成 0.5 Byte，只证明编码数据理论上缩小四倍，不证明端到端时间也缩短四倍。

### 怎样验证

至少固定同一模型输入、输出长度、采样参数、并发和 SLO，比较：

```text
质量：固定回归集或目标任务指标
容量：进程真实权重显存与剩余 Cache 容量
延迟：TTFT、TPOT、P99
吞吐：同一 SLO 下的 request/s、token/s
执行：量化层是否使用预期 Kernel，有无 fallback
```

只看模型文件大小，不能判断量化是否成功兑现性能收益。

## 3. KV 量化：省的是历史状态，不是模型权重

第 8 课的 KV 公式是：

$$
KV\ Bytes=2B L_{full}N_{kv}TDs
$$

若把 KV 从 BF16 的 2 Byte 改为 FP8 的 1 Byte，逻辑有效载荷约减半。它可能带来两类收益：

1. 相同显存能容纳更多长上下文请求。
2. Decode 读取历史 K/V 的字节数减少。

但它不会：

- 缩小模型权重；
- 减少 QK 和 AV 的数学运算次数；
- 自动量化 Gated DeltaNet 的 `conv_state` 和 `recurrent_state`；
- 保证长上下文质量不变。

对 Qwen3.5，只有 8 或 10 个 Full Attention 层使用随长度增长的 KV。若服务容量主要受每请求几十 MiB 的 Gated DeltaNet 固定状态限制，只量化 KV 的收益可能小于纯 Transformer 模型。

验证时还要确认 Scale 的粒度、静态或动态计算方式，以及 Attention Kernel 是否能直接读取量化 Cache。若先把 KV 还原成 BF16 再执行，节省的存储字节不一定能完全转成读取加速。

## 4. FlashAttention：少写中间矩阵，计算含义不变

普通 Attention 可以按下面的直观步骤实现：

```text
QKᵀ → 写出完整分数矩阵
读取分数 → Softmax → 写出概率矩阵
读取概率和 V → 得到输出
```

长序列时，`T×T` 分数和概率矩阵很大，反复写入和读取 HBM 的成本很高。

FlashAttention 把 Q/K/V 分块搬到片上 SRAM，在块内维护 Softmax 所需统计量，并直接累计最终输出。它避免把完整 `T×T` 中间矩阵物化到 HBM。

![普通 Attention 与 FlashAttention 的数据流](../assets/09-flashattention.svg)

### 它减少什么

```text
减少：Attention 中间矩阵的 HBM 读写和临时显存
保留：Q/K/V 投影、QK/AV 数学、Softmax、因果关系
```

FlashAttention 是精确 Attention 算法，不是稀疏 Attention，也不是近似检索。

### Qwen3.5 的边界

Qwen3.5 只有四分之一语言层是 Full Attention。其余层是 Gated DeltaNet，FlashAttention 不会直接加速它们；Dense 或 MoE FFN 也不在这个子图内。

长 Prompt Prefill 更容易受益，因为标准实现的 Attention 中间数据随长度快速增长。单 token Decode 不会产生当前 Query 的 `T×T` 矩阵，但仍要读取历史 K/V，通常会走 Paged/Decode Attention Kernel。

所以验证时要同时看：

```text
Attention Kernel 时间
按 Prompt 长度分桶的端到端 TTFT
Decode TPOT
临时显存峰值
```

“Attention Kernel 快两倍”不等于完整模型快两倍。

## 5. Prefix Cache：第二个请求复用第一个请求的前缀计算

假设许多请求都以同一个系统 Prompt 开头：

```text
请求 A：[共同系统 Prompt][用户问题 A]
请求 B：[共同系统 Prompt][用户问题 B]
```

请求 A 已经算过共同前缀并建立了模型状态。Prefix Cache 命中后，请求 B 可以直接恢复兼容的前缀状态，只计算自己的后缀。

粗略收益可以写成：

$$
收益\propto命中率\times可复用token数\times每token\ Prefill成本
$$

### 它不是什么

- 不是同一请求 Decode 使用的普通 KV Cache。
- 不是语义缓存。两句话意思接近但 Token ID 不同，不能直接复用内部状态。
- 不会减少未命中后缀的 Prefill。
- 不会加速之后每个新 token 的正常 Decode。

Qwen3.5 还是混合模型。可复用前缀不仅包含 Full Attention K/V，还要在正确边界恢复 Gated DeltaNet 的卷积和递归状态。vLLM 固定版本的 Qwen3.5 配方仍把相关 `align` 模式标为 experimental，因此不能只看“Prefix Cache 开关已打开”就假定行为与纯 Attention 模型完全一致。

### 怎样验证

```text
命中的前缀 token 数
首次请求与重复请求 TTFT
真实命中率和驱逐率
Cache 占用
不同前缀长度下的收益
混合状态恢复是否正确
```

如果请求在很靠前的位置就不同，或者缓存频繁被驱逐，Prefix Cache 可能几乎没有收益。

## 6. Batching：把同一份权重服务给更多 token

单 token Linear 输入是 `[1,H]`。若一次处理 `M` 个 token，输入变成 `[M,H]`。同一张权重矩阵在一轮中服务更多输入，矩阵规模和权重读取复用通常更好。

Batching 不改变模型公式，改变的是每次执行装入多少工作。

### Static Batching

一批请求一起开始，通常也要一起等待。短请求先完成后，空出来的位置不能及时补入新请求，直到整批结束。

### Continuous Batching

系统在生成迭代边界重新组织 Batch：完成的请求退出，等待请求进入。不同输出长度造成的空转因此减少。

![Static Batching 与 Continuous Batching](../assets/09-batching.svg)

Continuous Batching 只说明 Batch 成员能动态变化，不自动表示 Prefill chunk 和 Decode token 一定能进入同一次模型执行。混批还需要调度器、执行器和 Kernel 支持。

### 它在交换什么

```text
更大 Batch：权重复用和吞吐往往更好，但请求可能排队更久
更小 Batch：更快开始，但矩阵太小、GPU 利用率可能更低
优先 Decode：在途请求 TPOT 更稳，新请求 TTFT 可能变差
优先 Prefill：新请求更快进入，Decode 可能产生抖动
```

因此，服务优化目标不应只写“最大化离线 token/s”，而应写成：

> 在给定 TTFT、TPOT 和 P99 SLO 下，最大化可持续吞吐。

## 7. TP：切同一层的矩阵，用通信合并部分结果

Tensor Parallelism 把一张大权重矩阵分到多张 GPU。以 FFN 为例：

```text
gate/up：按输出特征切分，各卡产生一部分中间特征
down：   按输入特征切分，各卡产生一部分输出
合并：   用集合通信得到完整层输出
```

### 可能收益

- 单卡放不下的层权重能够分片部署。
- 大矩阵计算由多卡分担。
- Attention 头和部分 KV 状态可以按头分布。

### 新增成本

- 集合通信进入每层关键路径。
- TP 越大，每卡 GEMM 越小，Kernel 效率可能下降。
- 小 Batch Decode 的本地计算很少，通信延迟更容易占主导。
- PCIe、NVLink 和跨节点网络的延迟与带宽差异很大。

判断 TP 是否值得，比较的是：

```text
每卡少掉的权重读取和计算时间
vs
新增的集合通信、同步与小矩阵效率损失
```

更多 GPU 首先解决模型容量和单请求计算分担，不保证吞吐按卡数线性增长。

## 8. EP：切不同 Expert，用 Dispatch 和 Combine 交换 token

Expert Parallelism 把 Routed Expert 分布到不同设备。Router 选完 Top-K 后，token 特征被送到持有相应 Expert 的设备；Expert 算完，再把结果送回原 token 位置。

![TP 与 EP 的通信位置](../assets/09-tp-ep.svg)

### 适合什么情况

```text
全部 Expert 权重无法在较小设备组中容纳
Batch token 足够多，能形成较大的 Grouped GEMM
专家负载相对均衡
卡间互联能承受 Dispatch 和 Combine
```

### 常见失效点

- 低并发时每个 Expert 只收到少量 token，小 GEMM 和通信延迟占主导。
- 热点 Expert 集中在少数 Rank，整层等待最慢设备。
- Top-K 越大，每个 token 的 Routed Assignment 和通信通常越多。
- 跨节点网络成本可能超过专家计算节省。

Qwen3.5-35B-A3B 的 EP 主要分布 256 个 Routed Experts。Attention、Gated DeltaNet、Router 与 Shared Expert 仍要由其他并行或复制策略处理。

第 6 课已经强调：EP 的稳定边界是 Dispatch 和 Combine，具体使用 All-to-All、All-Reduce 或其他 Dispatcher 取决于 runtime。课程不能把 EP 与某一个 Collective 永久画等号。

## 9. 推测解码：用一次目标模型前向验证多个候选

正常 Decode 每次目标模型前向只确认一个新 token。推测解码先用更便宜的 Drafter 提出多个候选，再让 Target Model 一次验证多个位置。

![普通 Decode 与推测解码](../assets/09-speculative-decoding.svg)

验证会从前往后接受候选。遇到第一个拒绝位置后，系统按目标模型结果修正，再开始下一轮。

### 什么时候可能更快

设每轮草稿提出 `k` 个候选，平均接受 `a` 个。只有当：

```text
草稿成本 + 一次 k 位置验证成本
小于
目标模型逐个生成 a+1 次的成本
```

才有实际收益。通常需要：

```text
Drafter 足够便宜
候选接受率足够高
一次验证多个位置比多次单 token Decode 更高效
低并发下仍有空余计算能力
```

### 新增了什么

- Drafter 权重、计算和请求状态。
- 候选位置占用的 Lookahead Cache。
- 被拒绝候选产生的无效计算。
- Batch 扩张、回滚和调度复杂度。

严格的 Speculative Sampling 可以保持目标模型原有采样分布。简单的“草稿 token 与目标贪心 token 一样就接受”只覆盖贪心生成，不能代表一般采样场景。

低并发 TPOT 变好，也不保证高并发吞吐变好。候选 token 会占用本来可服务其他请求的 Token Budget 和 Cache。

## 10. MTP：模型自带 Drafter，仍要由主模型验证

Multi-Token Prediction 在训练时增加辅助模块，让当前位置学习预测更远的 token。它在普通生成中可以不启用，也可以作为推测解码的 Drafter。

Qwen3.5 配置中：

```text
mtp_num_hidden_layers = 1
```

这表示检查点保存了一层 MTP 辅助模块，不表示主 Decoder 一次可以不经验证地确定多个未来 token。

启用时，MTP 提出候选，主语言模型仍负责验证。候选接受率、一次提出几个、是否重复使用同一 MTP 层，都由实际模型行为和 runtime 配置决定。

Qwen3.5 的 vLLM 配方把 MTP-1 更偏向低并发、延迟敏感场景，并提醒它可能降低高并发文本吞吐。正确做法是分别测试：

```text
低并发 TPOT
高并发同 SLO 吞吐
平均接受长度
候选拒绝率
Lookahead Cache 占用
MTP 开关前后的质量一致性
```

## 11. 把判断过程做成一张检查表

### 第一步：指出被改变的对象

```text
权重编码？
Full Attention 中间张量？
KV 或 Gated DeltaNet 状态？
Batch 组成？
单层矩阵分片？
Expert 放置？
自回归目标模型调用次数？
```

### 第二步：写出理论上少掉的量

```text
INT4：每参数理想从 2 Byte 变成 0.5 Byte
KV FP8：Full Attention KV 每元素从 2 Byte 变成 1 Byte
Prefix Cache：命中的前缀 token 不再做 Prefill
TP=2：主要矩阵每卡约保存和计算一部分
推测解码：每次 Target forward 平均确认不止 1 个 token
```

### 第三步：列出新增成本

反量化、Scale、通信、Hash、Cache 占用、排队、草稿模型、拒绝浪费和质量误差都必须进入判断。

### 第四步：固定 workload

至少记录：

```text
Prompt 与 Output 长度分布
并发或到达率
重复前缀比例
Batch token 数
采样参数
GPU 与互联拓扑
SLO
模型 dtype、runtime 版本和关键开关
```

### 第五步：用端到端指标裁决

```text
TTFT、TPOT、ITL、P50、P99
同一 SLO 下 request/s 与 token/s
权重、请求状态与临时显存
跨卡通信时间和真实链路带宽
目标任务质量
```

Kernel 时间用来解释原因，端到端指标决定方案是否值得上线。

## 12. 三个完整判断例子

### 例 1：把 Qwen3.5-35B-A3B 从 BF16 改为 INT4

```text
改变对象：全部或部分 Linear 权重编码
理论收益：权重有效载荷显著下降，Decode 权重读取减少
新增成本：Scale、反量化、未量化层、低比特 Kernel
收益 workload：小 Batch Decode、显存紧张、硬件有成熟 INT4 Kernel
验证：质量、真实显存、Kernel 覆盖、TTFT/TPOT、同 SLO 吞吐
```

若服务的主要瓶颈是跨节点 EP 通信，单看 INT4 权重缩小无法证明端到端有同等收益。

### 例 2：为共享 20K 系统 Prompt 开启 Prefix Cache

```text
改变对象：跨请求复用完整前缀状态
理论收益：命中请求少做 20K token Prefill
新增成本：Cache 空间、Hash、块粒度、驱逐和混合状态恢复
收益 workload：大量请求拥有完全相同的长前缀
验证：命中 token、首次/重复 TTFT、驱逐率、状态正确性
```

若每个请求在前几十个 token 就不同，20K 文档只是语义相似而不是 Token ID 相同，缓存不会命中。

### 例 3：把 TP 从 4 提高到 8

```text
改变对象：每层矩阵切得更细
理论收益：每卡权重和本地计算减少，单卡容量压力下降
新增成本：更频繁或更大范围的集合通信，每卡 GEMM 变小
收益 workload：模型放不下，或大 Batch 计算足以覆盖通信
验证：逐层 GEMM、Collective 时间、链路、TTFT/TPOT、同 SLO 吞吐
```

如果跨节点 TP 使用较慢网络，小 Batch Decode 可能因为通信占比上升而变慢。

## 13. 容易犯的判断错误

1. 量化文件缩小四倍，就宣称服务必然快四倍。
2. FlashAttention Kernel 更快，就把完整模型加速按同样比例计算。
3. Prefix Cache 开关成功，就当作所有重复问题都会命中。
4. Batch 越大越好，不再检查排队和 P99。
5. GPU 数翻倍，就预期 TP 吞吐线性翻倍。
6. MoE 每 token 只选少数 Expert，就忽略 Batch 路由分布与通信。
7. 推测解码平均接受长度变大，就不再观察 Drafter 成本和高并发吞吐。
8. 用算子 Microbenchmark 代替端到端服务测量。
9. 改变输入分布、并发和 SLO 后，仍把两组数字当成公平对比。
10. 只报平均值，不看 P99、显存峰值、回退路径与质量变化。

## 14. 练习

1. 判断一个优化的第一问是什么？
2. INT4 权重理想容量缩小四倍，为什么延迟不一定缩小四倍？
3. KV 量化会缩小 Gated DeltaNet recurrent state 吗？
4. FlashAttention 减少的主要是什么？它是否改变 Attention 输出含义？
5. Prefix Cache 为什么不能按“语义相似”命中？
6. Continuous Batching 是否自动表示 Prefill 和 Decode 混批？
7. 为什么更大 Batch 可能伤害 TTFT？
8. TP 的本地计算收益需要与什么成本比较？
9. EP 为什么容易受到专家负载倾斜影响？
10. 推测解码为什么仍然满足自回归约束？
11. MTP 一层是否表示一次必然多生成一个 token？
12. 为什么只看 Kernel 时间不能决定优化是否上线？
13. 比较两个方案时，至少应固定哪些 workload 条件？
14. 一个方案 TPOT 下降，但 P99 TTFT 和同 SLO 吞吐都变差，能否直接说整体更优？

## 15. 参考答案

1. 它直接改变了推理链路中的哪个对象。
2. 还受反量化、Kernel、硬件、Batch 和原始瓶颈影响。
3. 不会。它只改变 Full Attention KV，除非 runtime 另有状态量化方案。
4. Attention 中间矩阵的 HBM 读写和临时显存；它仍是精确 Attention。
5. 内部状态由确切 Token ID、位置和模型计算决定，意思接近不能保证状态相同。
6. 不是。混批还需要调度器、执行器和 Kernel 支持。
7. 请求可能在队列中等待凑批，或被更大 Prefill 工作阻塞。
8. 集合通信、同步和每卡矩阵变小带来的效率损失。
9. 最慢 Expert 所在设备可能成为整层瓶颈，且通信量也会失衡。
10. 候选只是草稿，Target Model 仍按顺序验证并在拒绝点修正。
11. 不是。它只说明有一层辅助模块，接受率和候选数取决于实际运行。
12. 局部变快可能被其他模块、排队、通信或质量代价抵消。
13. 输入输出长度、并发或到达率、采样参数、Batch、硬件拓扑、SLO、dtype 和 runtime 版本。
14. 不能。要根据业务 SLO 和整体目标权衡，至少不能仅凭 TPOT 下结论。

## 16. 学完以后，怎样用这套方法

现在可以从用户输入一路解释到优化判断：

```text
Chat Template 与 Tokenizer
→ Embedding 或视觉编码
→ 多层 Decoder
   → RMSNorm、Token Mixer、残差、Dense/MoE FFN
   → Full Attention KV 或 Gated DeltaNet 状态
→ LM Head、采样与下一个 token
→ Prefill 和逐 token Decode
→ 权重、计算、请求状态、通信与调度成本
→ 根据 workload 和 SLO 选择优化
```

这套课最终要留下的是推理链路中的对象、shape、依赖和成本。遇到一种新方案，先指出它改了哪里、少了什么、多了什么，再判断是否值得测试。

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
- [vLLM 文档与源码，revision 653ebb5](https://github.com/vllm-project/vllm/tree/653ebb52dffd8b4653b430302473c771117529f1)
- [vLLM Qwen3.5 配方，revision 689d6b9](https://github.com/vllm-project/recipes/blob/689d6b98c05ec4e92523a231afe9dce97e5d83dc/Qwen/Qwen3.5.md)
- [Qwen3.5-35B-A3B 模型卡，revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)
