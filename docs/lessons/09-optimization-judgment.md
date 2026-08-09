# 第 9 课：推理优化的分析与评估

量化、FlashAttention、Prefix Cache、Batching 和并行策略改的不是同一部分。判断一种方案有没有用，先回答五个问题：

1. 它改了哪段计算、哪份数据或哪条调度规则？
2. 因此少做了多少计算，少占了多少显存，或少传了多少数据？
3. 它又增加了哪些计算、通信、缓存或排队时间？
4. 哪种输入长度、输出长度、并发和硬件配置下可能更快？
5. 最后用哪些端到端指标比较？

如果第一步说不清，后面的“提速百分比”就无法解释，也很难复现。

![常见优化分别改了哪些计算、数据和调度](../assets/09-optimization-map.svg?rev=20260809-1)

## 1. 推理优化的分类与作用范围

| 优化 | 直接改动 | 可能先改善什么 | 需要同时计入什么 |
| --- | --- | --- | --- |
| 权重量化 | 用低比特格式保存 Linear 权重，并换用相应 Kernel | 权重显存、Decode 读取权重的字节数 | 质量、Scale、反量化、Kernel 回退 |
| KV 量化 | 用低比特格式保存 Full Attention 的 K/V | 长上下文容量、读取历史 K/V 的字节数 | 质量、Scale、量化与反量化开销 |
| FlashAttention | 分块计算 Attention，不把完整中间矩阵写入 HBM | 长序列 Prefill、临时显存 | 只影响 Full Attention |
| Prefix Cache | 让后续请求复用完全相同的前缀状态 | 重复前缀请求的 TTFT | 命中率、驱逐、状态兼容性 |
| Batching | 改变每轮放入多少 token，以及哪些请求一起执行 | GPU 利用率、吞吐 | 排队、尾延迟、调度开销 |
| DP | 复制完整模型，把独立请求分给不同副本 | 集群吞吐、并发容量 | 每份权重占用、负载均衡、Cache 分散 |
| TP | 把同一层的矩阵切到多张 GPU 上，再合并部分结果 | 单卡容量、单请求计算时间 | 集合通信、同步、小矩阵效率 |
| PP | 把连续 Decoder Layer 分给不同设备阶段 | 单副本模型容量 | 阶段传输、流水线空泡、阶段不均衡 |
| EP | 按 Expert ID 分布权重，并在设备间分发 token | MoE 容量、Expert 计算吞吐 | Dispatch、Combine、负载倾斜 |
| 推测解码 / MTP | 先提出候选，再让目标模型一次验证多个位置 | 低并发 TPOT | Drafter、候选状态、被拒绝候选的计算 |

一项改动通常只影响部分耗时。例如 FlashAttention 减少 Full Attention 的中间读写，但 FFN 和 Gated DeltaNet 仍照常计算。因此，算子时间下降多少，不能直接当成整个请求的加速比例。

### 1.1 `token/s` 的统计口径

同一个 `token/s` 可能统计不同内容。比较前要把分子和时间窗口写清楚。本课使用以下口径：

| 指标 | 分子 | 分母 |
| --- | --- | --- |
| `input token/s` | 窗口内完成请求的 Prompt token 总数 | 测量窗口秒数 |
| `output token/s` | 窗口内完成请求的输出 token 总数 | 测量窗口秒数 |
| `goodput` | 窗口内满足预先指定 SLO 的完成请求数 | 测量窗口秒数 |
| `engine token/s` | runtime 实际安排进入模型执行的 token 位置数 | 测量窗口秒数 |

`input token/s` 和 `output token/s` 是用户可见工作量。报告 `goodput` 时，要同时写出逐请求判定所用的 TTFT、TPOT 等 SLO 阈值，P99 等分位数另行报告。如果工具把 goodput 定义成 token/s 而不是 request/s，也要明确标注。

`engine token/s` 统计引擎内部工作量，不等于用户收到的 token 数。使用推测解码、Padding 或重算时，还要说明 runtime 的计数器是否包含候选位置、填充位置和重复执行的位置。

## 2. 权重量化

### 2.1 权重量化改变了哪些数据

BF16 每个参数占 2 Byte。INT8 的理想下限是 1 Byte，INT4 是 0.5 Byte。第 8 课已经说明，这只是保存 dtype 的有效载荷；计算 dtype、累加 dtype、Scale 和对齐要另算。

Qwen3.5-35B-A3B 的 BF16 权重有效载荷约 66.97 GiB。若所有参数都用纯 INT4 编码，理想下限约 16.74 GiB。

Weight-only 量化通常仍让激活保持 BF16 或 FP16。低比特 Kernel 读取压缩权重，再在寄存器或片上存储中完成反量化和矩阵运算。

### 2.2 理论容量收益

小 Batch Decode 经常需要为很少的 token 读取大量活跃权重。若时间主要花在 HBM 搬权重，读取字节减少，延迟就有下降空间。省下的显存还可以换成更多 KV Cache 或更高并发。

### 2.3 延迟收益受哪些条件限制

低比特格式缺少合适的硬件路径、反量化代价过大或部分层回退到通用 Kernel 时，省下的读取时间会被新增工作抵消。大 Batch GEMM 已经偏计算受限，或者 MoE 单个 Expert 收到的 token 太少时，低比特 Kernel 也未必高效。

权重从 2 Byte 变成 0.5 Byte，只证明编码数据理论上缩小四倍，不证明端到端时间也缩短四倍。

### 2.4 怎样验证权重量化

至少固定同一模型输入、输出长度、采样参数、并发和 SLO，再比较固定回归集质量、进程真实显存、TTFT、TPOT、P99 以及同 SLO 吞吐。Profile 还要确认量化层是否命中预期 Kernel，有没有回退路径。

只看模型文件大小，不能判断量化是否真的降低了服务延迟或提高了同 SLO 吞吐。

## 3. KV Cache 量化

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

## 4. FlashAttention

普通实现会先把 `QKᵀ` 的完整分数矩阵写到 HBM，Softmax 读取分数后再写出概率矩阵，最后重新读取概率和 V。长序列下，两张 `T×T` 中间矩阵会带来大量 HBM 读写。

FlashAttention 把 Q/K/V 分块搬到片上 SRAM，在块内维护 Softmax 所需统计量，并直接累计最终输出。它避免把完整 `T×T` 中间矩阵物化到 HBM。

![普通 Attention 与 FlashAttention 的数据流](../assets/09-flashattention.svg)

### 4.1 FlashAttention 怎样减少 HBM 读写

FlashAttention 减少的是 Attention 中间矩阵的 HBM 读写和临时显存。Q/K/V 投影、QK 与 AV 的数学运算、Softmax 和因果关系仍然存在。

FlashAttention 是精确 Attention 算法，不是稀疏 Attention，也不是近似检索。

### 4.2 FlashAttention 只改变 Full Attention

Qwen3.5 只有四分之一语言层是 Full Attention。其余层是 Gated DeltaNet，FlashAttention 不会直接加速它们；Dense 或 MoE FFN 也不在这个子图内。

长 Prompt Prefill 更容易受益，因为标准实现的 Attention 中间数据随长度快速增长。单 token Decode 不会产生当前 Query 的 `T×T` 矩阵，但仍要读取历史 K/V，通常会走 Paged/Decode Attention Kernel。

验证时应同时看 Attention Kernel 时间、按 Prompt 长度分桶的端到端 TTFT、Decode TPOT 和临时显存峰值。

“Attention Kernel 快两倍”不等于完整模型快两倍。

## 5. Prefix Cache

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

### 5.1 哪些前缀可以复用

- 不是同一请求 Decode 使用的普通 KV Cache。
- 不是语义缓存。两句话意思接近但 Token ID 不同，不能直接复用内部状态。
- 不会减少未命中后缀的 Prefill。
- 不会加速之后每个新 token 的正常 Decode。

Qwen3.5 还是混合模型。可复用前缀不仅包含 Full Attention K/V，还要在正确的 token 位置恢复 Gated DeltaNet 的卷积和递归状态。vLLM 固定版本的 Qwen3.5 配方仍把相关 `align` 模式标为 experimental，因此不能只看“Prefix Cache 开关已打开”就假定行为与纯 Attention 模型完全一致。

### 5.2 怎样验证 Prefix Cache

需要记录命中的前缀 token 数、首次与重复请求 TTFT、真实命中率、驱逐率和 Cache 占用。对混合模型还要验证 Gated DeltaNet 状态能否正确恢复。

如果请求在很靠前的位置就不同，或者缓存频繁被驱逐，Prefix Cache 可能几乎没有收益。

## 6. Batching 与调度

单 token Linear 输入是 `[1,H]`。若一次处理 `M` 个 token，输入变成 `[M,H]`。同一张权重矩阵在一轮中服务更多输入，矩阵规模和权重读取复用通常更好。

Batching 不改变模型公式，改变的是每次执行装入多少工作。

### 6.1 Static Batching

一批请求一起开始，通常也要一起等待。短请求先完成后，空出来的位置不能及时补入新请求，直到整批结束。

### 6.2 Continuous Batching

系统在每轮生成结束时重新组织 Batch：完成的请求退出，等待请求进入。不同输出长度造成的空转因此减少。

![Static Batching 与 Continuous Batching](../assets/09-batching.svg)

Continuous Batching 只说明 Batch 成员能动态变化，不自动表示 Prefill chunk 和 Decode token 一定能进入同一次模型执行。混批还需要调度器、执行器和 Kernel 支持。

### 6.3 Batch 大小怎样影响吞吐与排队时间

| 调度选择 | 可能得到什么 | 可能付出什么 |
| --- | --- | --- |
| 更大 Batch | 权重复用和吞吐通常更好 | 凑批与排队更久 |
| 更小 Batch | 请求更快开始 | 矩阵过小，GPU 利用率较低 |
| 优先 Decode | 在途请求的 TPOT 更稳定 | 新请求 TTFT 可能变差 |
| 优先 Prefill | 新请求更快进入模型 | Decode 可能出现抖动 |

因此，服务优化目标不应只写“最大化离线 token/s”，而应写成：

> 先满足给定的 TTFT、TPOT 和 P99 SLO，再比较可持续的 request/s、input token/s 和 output token/s。

## 7. DP、TP、PP 与 EP 的切分方式

四种并行策略都使用多张 GPU，但切分对象不同：

| 策略 | 切分或复制什么 | 一个请求怎样执行 | 主要解决什么 |
| --- | --- | --- | --- |
| 数据并行（DP） | 复制完整模型，分配不同请求 | 通常只进入一个模型副本 | 集群吞吐与并发容量 |
| 张量并行（TP） | 切分同一层中的矩阵或 Attention 头 | 每层都由多个 Rank 共同计算 | 单卡容量与单请求计算分担 |
| 流水线并行（PP） | 按深度切分连续 Decoder Layer | 依次经过多个阶段 | 单副本模型容量 |
| 专家并行（EP） | 把不同 Routed Experts 放到不同设备 | 每个 MoE 层按路由结果跨设备分发 | MoE 专家容量与计算组织 |

### 7.1 数据并行（DP）

DP 的每个副本都能独立完成一次前向。在线服务把不同请求或不同 Batch 分给不同副本，因此它主要提高集群总吞吐和可承载并发，不会直接缩短单个请求经过模型的计算路径。

代价是每个副本都保存一份完整权重和独立请求状态。Prefix Cache 也通常分散在各副本中，负载均衡如果只看请求数而忽略队列与 Cache 命中，可能让部分副本排队、另一部分空闲。MoE runtime 还可能组合 DP 与 EP，此时各 DP Rank 未必完全独立，必须按具体实现判断通信边界。

### 7.2 张量并行（TP）

TP 把一张大权重矩阵分到多张 GPU。以 FFN 为例，`gate_proj` 和 `up_proj` 可以按输出特征切分，`down_proj` 再按输入特征切分。各卡得到部分结果后，通过集合通信恢复下一步需要的层输出。

TP 能降低每卡权重和计算量，但集合通信进入每层关键路径。TP 越大，每卡 GEMM 越小，Kernel 效率也可能下降。小 Batch Decode 的本地计算很少，通信延迟尤其容易占主导。

第 8 课已经说明固定版本 vLLM 在 `Nkv<TP` 时会复制 K/V 头。计算每卡 KV Cache 时，应使用 runtime 实际分配的本地 K/V 头数，不能把全局数量直接除以 TP。

### 7.3 流水线并行（PP）

PP 按模型深度切分层。例如 32 层模型分成 4 个阶段，每个阶段保存连续 8 层。一个 token 的 Hidden State 先经过阶段 0，再传给阶段 1，直到最后一个阶段输出 Logits。

PP 能让单个模型跨越多张卡或多个节点，但不会让同一个 token 跳过任何阶段。只有多个请求或 microbatch 同时处在不同阶段时，设备才能形成流水。Batch 太小、阶段耗时不均或 Decode 每步工作过少时，部分阶段会等待，形成流水线空泡。跨阶段还要传输激活。

### 7.4 专家并行（EP）

EP 把 Routed Experts 分布到不同设备。Router 选完 Top-K 后，token 特征被送到持有相应 Expert 的设备；Expert 算完，再把结果送回原 token 位置。

![TP 与 EP 的通信位置](../assets/09-tp-ep.svg)

低并发时，每个 Expert 可能只收到少量 token，小 GEMM 和通信延迟占主导。热点 Expert 若集中在少数 Rank，整层还要等待最慢设备。Top-K 越大，每个 token 的路由分配和通信通常也越多。

Qwen3.5-35B-A3B 的 EP 主要分布 256 个 Routed Experts。Attention、Gated DeltaNet、Router 与 Shared Expert 仍要由其他并行或复制策略处理。具体使用哪种 Collective 取决于 runtime，不能把 EP 固定等同于 All-to-All。

## 8. 推测解码（Speculative Decoding）

正常 Decode 每次目标模型前向只确认一个新 token。推测解码先用更便宜的 Drafter 提出多个候选，再让 Target Model 一次验证多个位置。

![普通 Decode 与推测解码](../assets/09-speculative-decoding.svg)

验证会从前往后接受候选。遇到第一个拒绝位置后，系统按目标模型结果修正，再开始下一轮。

### 8.1 推测解码在什么条件下有收益

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

### 8.2 推测解码增加了哪些成本

- Drafter 权重、计算和请求状态。
- 候选位置占用的 Lookahead Cache。
- 被拒绝候选产生的无效计算。
- Batch 扩张、回滚和调度复杂度。

严格的 Speculative Sampling 可以保持目标模型原有采样分布。简单的“草稿 token 与目标贪心 token 一样就接受”只覆盖贪心生成，不能代表一般采样场景。

低并发 TPOT 变好，也不保证高并发吞吐变好。候选 token 会占用本来可服务其他请求的 Token Budget 和 Cache。

## 9. Multi-Token Prediction（MTP）

Multi-Token Prediction 在训练时增加辅助模块，让当前位置学习预测更远的 token。它在普通生成中可以不启用，也可以作为推测解码的 Drafter。

Qwen3.5 配置中：

```text
mtp_num_hidden_layers = 1
```

这表示检查点保存了一层 MTP 辅助模块，不表示主 Decoder 一次可以不经验证地确定多个未来 token。

启用时，MTP 提出候选，主语言模型仍负责验证。候选接受率、一次提出几个、是否重复使用同一 MTP 层，都由实际模型行为和 runtime 配置决定。

Qwen3.5 的 vLLM 配方建议优先在低并发、延迟敏感场景尝试 MTP-1，并提醒它可能降低高并发文本吞吐。因此要分别测试：

```text
低并发 TPOT
高并发同 SLO 吞吐
平均接受长度
候选拒绝率
Lookahead Cache 占用
MTP 开关前后的质量一致性
```

## 10. 局部加速的端到端上限

一种优化往往只覆盖整条链路的一部分。设这部分原来占总时间的比例为 `f`，优化后快了 `s` 倍，其余部分不变。端到端加速比的理论上限是：

$$
Speedup=\frac{1}{(1-f)+\frac{f}{s}}
$$

例如，Profile 显示 Full Attention 占端到端时间的 20%，新的 Kernel 把这部分加速 2 倍：

$$
Speedup=\frac{1}{0.8+0.2/2}=\frac{1}{0.9}\approx1.11
$$

局部快了 2 倍，整条链路的理论结果只有约 1.11 倍。即使把这 20% 完全消除，整体也最多达到：

$$
\frac{1}{1-0.2}=1.25
$$

这就是 Amdahl 定律在推理优化中的用法。`f` 必须来自目标工作负载的实测分解，不能用另一个 Prompt 长度、Batch 或并发下的比例。公式还假设没有新增开销；量化的反量化、TP 的通信或 Prefix Cache 的查找都要加入新路径后重新计算。

Amdahl 定律估算的是服务时间组成，不直接描述排队系统。一次优化改变 Batch、资源容量或到达率后，排队时间可能发生非线性变化，最终仍要用端到端压测确认。

## 11. 推理优化的实验评估方法

一项优化应当按同一条证据链评估：

| 步骤 | 要回答的问题 | 需要留下的证据 |
| --- | --- | --- |
| 确定范围 | 改的是权重、Attention、请求状态、Batch、模型切分还是目标模型调用次数？ | 修改前后的数据流和配置 |
| 估算上限 | 少了多少计算、字节或重复执行？原路径占总时间多少？ | 容量公式、FLOPs、Profile 和 Amdahl 上限 |
| 计入代价 | 新增了反量化、Hash、通信、Cache、排队还是 Drafter？ | 新增 Kernel、通信量和显存 |
| 固定条件 | 哪些请求和硬件条件保持不变？ | Prompt/Output 分布、到达率、采样、GPU、互联、dtype、runtime 和 SLO |
| 比较结果 | 业务指标、容量和质量是否同时达标？ | TTFT、TPOT、P99、同 SLO 吞吐、显存、质量和 Profile |

吞吐测试还要固定统计窗口，并明确报告 request/s、input token/s、output token/s、engine token/s 还是 goodput。不同分子得到的数字不能直接横向比较。

先用端到端指标决定方案是否有效，再用 Kernel 和通信时间解释原因。Microbenchmark 可以证明局部实现更快，不能替代完整服务结果。

## 12. 用对照数据做优化决策

下表使用教学用的假设压测数据，只用于练习分析方法，不代表 Qwen3.5 的公开 benchmark。每组实验都保持模型版本、硬件、runtime 和请求数据不变，只修改表中说明的配置。读表时应根据目标和证据作出取舍，无需记住数字。

### 12.1 INT4 节省了显存，为什么还不能直接上线

团队准备把 Qwen3.5-35B-A3B 的权重从 BF16 改为 INT4。目标有两个：降低模型常驻显存，并改善低并发 Decode 延迟。服务还必须通过现有质量回归集，高并发吞吐不能下降超过 2%。

| 观测项 | BF16 | INT4 |
| --- | ---: | ---: |
| 4 卡模型组的进程显存 | 91 GiB | 43 GiB |
| 并发 1 的 TPOT P50 | 23.8 ms | 16.7 ms |
| 并发 64 的 output token/s | 6400 | 6150 |
| Linear 命中预期 INT4 Kernel 的比例 | 不适用 | 71% |
| 质量回归集 | 全部达标 | 2 项超过退化阈值 |

这次实验已经证明两件事：INT4 显著降低了实际进程显存，低并发 TPOT 也下降了约 30%。但它没有通过完整上线条件。高并发吞吐下降约 3.9%，质量回归还有两项超出阈值。

Profile 还显示，29% 的 Linear 没有走预期 INT4 Kernel。高并发下，大 Batch GEMM 的计算占比上升，反量化和回退路径抵消了部分权重读取收益。这里不能用“INT4 理论上少读四分之三的权重字节”解释全部端到端结果。

因此，当前版本已经达到容量目标，也改善了低并发延迟，但还不能替换所有线上实例。下一轮应先处理质量退化和 Kernel 覆盖率，再分别评估低并发延迟实例与高并发吞吐实例。一个配置不必同时适合两种工作负载。

### 12.2 Prefix Cache 命中率很高，为什么 P99 TTFT 仍未达标

某服务中约 80% 的请求共享一段 8192-token 的精确前缀，其余请求从开头就不同。目标是把总体 TTFT P99 降到 1.5 秒以内。开启 Prefix Cache 后得到：

| 观测项 | 关闭 Cache | 开启 Cache |
| --- | ---: | ---: |
| 请求命中率 | 0 | 81.7% |
| 命中请求的 TTFT P50 | 1.82 s | 0.38 s |
| 未命中请求的 TTFT P99 | 2.74 s | 2.79 s |
| 全部请求的 TTFT P50 | 1.79 s | 0.44 s |
| 全部请求的 TTFT P99 | 2.82 s | 2.85 s |

Cache 确实生效了。命中请求跳过大段重复 Prefill，P50 明显下降。总体 P99 仍由最慢的 1% 请求决定，而未命中请求占 18.3%，足以覆盖整个 P99 尾部。它们仍要执行完整 Prefill，所以总体 P99 几乎没有变化。

这组数据说明 Prefix Cache 已经减少了命中请求的重复计算。81.7% 的请求命中率不等于 TTFT 下降 81.7%，总体 P99 仍取决于未命中的慢请求。接下来应把两个请求群体分开报告，并继续定位未命中路径中的 Prefill、排队和调度时间。

### 12.3 TP=8 的单请求更快，为什么集群吞吐反而下降

一台机器有 8 张 GPU，模型能以 TP=4 运行。团队比较两种占用相同硬件的部署：

```text
方案 A：2 个模型副本，每个副本 TP=4
方案 B：1 个模型副本，TP=8
```

请求分布固定为 2048-token Prompt、最多生成 256 个 token，到达率为每秒 12 个请求。SLO 要求 TTFT P99 不超过 1 秒，TPOT P99 不超过 25 ms。

| 观测项 | 2×TP4 | 1×TP8 |
| --- | ---: | ---: |
| 并发 1 的 TPOT P50 | 21.8 ms | 15.9 ms |
| 单步 Decode 的 Collective 时间 | 3.1 ms | 6.0 ms |
| 在线 TTFT P99 | 0.82 s | 1.74 s |
| 在线 TPOT P99 | 24.0 ms | 20.1 ms |
| 满足两项 SLO 的 `goodput` | 11.5 request/s | 8.7 request/s |

TP=8 分担了更多本地计算，所以单请求 TPOT 更低；更大的通信组也让 Collective 时间上升。更重要的是，方案 B 把两个可独立接收请求的模型副本合并成了一个。在线到达率不变时，请求更容易在唯一副本前排队，TTFT P99 因此超标，`goodput` 也下降。

如果目标是低到达率下的单请求延迟，TP=8 可能更合适。对当前工作负载，模型已经能以 TP=4 部署，目标又是同一台机器的在线承载能力，因此应保留两个 TP=4 副本。比较并行策略时，必须固定总 GPU 数，并把副本数、通信和排队一起纳入实验。

## 13. 常见评估误区

1. 量化文件缩小四倍，就宣称服务必然快四倍。
2. FlashAttention Kernel 更快，就把完整模型加速按同样比例计算。
3. Prefix Cache 开关成功，就当作所有重复问题都会命中。
4. Batch 越大越好，不再检查排队和 P99。
5. GPU 数翻倍，就预期 TP 吞吐线性翻倍，或认为 DP 会直接降低单请求延迟。
6. MoE 每 token 只选少数 Expert，就忽略 Batch 路由分布与通信。
7. 推测解码平均接受长度变大，就不再观察 Drafter 成本和高并发吞吐。
8. 用算子 Microbenchmark 代替端到端服务测量。
9. 改变输入分布、并发和 SLO 后，仍把两组数字当成公平对比。
10. 只报平均值，不看 P99、显存峰值、回退路径与质量变化。

## 14. 练习

1. 测试一个优化时，为什么要先说明它改了哪段计算或数据？
2. INT4 权重理想容量缩小四倍，为什么延迟不一定缩小四倍？
3. FlashAttention 把占端到端时间 20% 的部分加速 2 倍，按 Amdahl 定律，整体理论加速约是多少？
4. Prefix Cache 为什么不能按“语义相似”命中？对 Qwen3.5 还要恢复哪些状态？
5. Continuous Batching 是否自动表示 Prefill 和 Decode 混批？更大 Batch 为什么可能伤害 TTFT？
6. DP、TP、PP 和 EP 分别切分或复制什么？哪一种通常不缩短单请求模型路径？
7. TP 的本地计算收益需要与什么成本比较？PP 在低并发 Decode 中为什么容易出现空泡？
8. EP 为什么容易受到专家负载倾斜影响？
9. 推测解码为什么仍满足自回归约束？MTP 一层是否表示一次必然多生成一个 token？
10. 一个方案 TPOT 下降，但 P99 TTFT、质量和同 SLO 吞吐都变差，能否直接说整体更优？

<details>
<summary>查看参考答案</summary>


1. 只有确定作用范围，才能计算原路径占比、理论上限和新增成本，并选择正确的验证指标。
2. 还受反量化、Kernel 覆盖、硬件、Batch、通信和原始瓶颈影响。
3. `1/(0.8+0.2/2)≈1.11` 倍。
4. 内部状态由确切 Token IDs、位置和模型计算决定。Qwen3.5 还要恢复 Full Attention KV、卷积状态和 recurrent state。
5. 不是，混批还需要执行器和 Kernel 支持。更大 Batch 可能增加凑批、排队和长 Prefill 阻塞时间。
6. DP 复制完整模型并分请求；TP 切同一层矩阵；PP 按层深度切阶段；EP 按 Expert ID 分专家。DP 通常不缩短单请求模型路径。
7. TP 要计入集合通信、同步和小矩阵效率。PP 需要多个请求或 microbatch 同时占据不同阶段，低并发 Decode 很难填满流水。
8. 热点 Expert 会让少数 Rank 同时承担更多计算和通信，整层等待最慢设备。
9. 候选只是草稿，Target Model 仍按顺序验证并在拒绝点修正。MTP 一层只表示存在辅助模块，不保证候选被接受。
10. 不能。上线判断要同时满足业务 SLO、质量、容量和吞吐目标，不能只凭一个平均指标。

</details>

## 15. 实践：审查一份上线结论

某次评审给出下面的结论：

> FP8 KV Cache 比 BF16 少占一半空间，Attention Microbenchmark 又快了 12%，因此建议在全部 Qwen3.5 服务上直接开启。预计并发翻倍，TPOT 降低 12%，输出质量不受影响。

请指出这段结论中哪些内容已有证据，哪些只是推断，并列出作出上线决策前必须补充的实验。

<details>
<summary>查看审查要点</summary>


已有证据只能支持两个局部结论：FP8 的 KV 有效载荷理论上约为 BF16 的一半；目标 Microbenchmark 中的 Attention 实现快了 12%。下面这些结论尚未得到证明：

- 实际进程显存是否减半。权重、Gated DeltaNet 固定状态和运行时预留没有随 KV dtype 一起缩小。
- 并发是否翻倍。还要看原来是不是 KV 容量受限，以及分页、复制和每请求固定状态占比。
- 端到端 TPOT 是否降低 12%。Attention 只是完整 Decode 路径的一部分，量化与反量化也可能增加工作。
- 全部服务是否都受益。Prompt、Output、并发、Batch 和硬件不同，路径占比也不同。
- 输出质量是否不变。需要固定回归集和长上下文测试。

补充实验应固定模型、runtime、硬件、请求分布和采样设置，比较每 Rank 显存、可承载并发、TTFT、TPOT P50/P99、`goodput`、Kernel 路径和质量。还要按上下文长度分桶，确认 Attention Kernel 直接读取 FP8 Cache，没有先完整还原为 BF16 的回退路径。

</details>

## 参考资料

- [AWQ: Activation-aware Weight Quantization v6](https://arxiv.org/abs/2306.00978v6)
- [FlashAttention v2](https://arxiv.org/abs/2205.14135v2)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention v1](https://arxiv.org/abs/2309.06180v1)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve v3](https://arxiv.org/abs/2403.02310v3)
- [Megatron-LM v4](https://arxiv.org/abs/1909.08053v4)
- [DeepSpeed-MoE v2](https://arxiv.org/abs/2201.05596v2)
- [Fast Inference from Transformers via Speculative Decoding v2](https://arxiv.org/abs/2211.17192v2)
- [DeepSeek-V3 Technical Report v2](https://arxiv.org/abs/2412.19437v2)
- [Amdahl：Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities](https://dl.acm.org/doi/10.1145/1465482.1465560)
- [vLLM：Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [vLLM：Tensor Parallel 与 Pipeline Parallel 部署](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [NVIDIA Megatron Core：并行策略对比](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [vLLM 文档与源码，revision 653ebb5](https://github.com/vllm-project/vllm/tree/653ebb52dffd8b4653b430302473c771117529f1)
- [vLLM Qwen3.5 配方，revision 689d6b9](https://github.com/vllm-project/recipes/blob/689d6b98c05ec4e92523a231afe9dce97e5d83dc/Qwen/Qwen3.5.md)
- [Qwen3.5-35B-A3B 模型卡，revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)

---

[上一课：模型配置与资源估算](08-config-and-sizing.md) · [返回课程路线](../roadmap.md)

完成本课后，继续做[结业案例：从模型配置到优化判断](../capstone.md)。案例不会再按课程顺序提示公式，需要独立还原一次完整分析。仓库中的[请求容量复算程序](../../examples/request_budget_walkthrough.py)可以核对案例使用的容量数字。
