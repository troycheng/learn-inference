# 第 4 课研究笔记：Prefill、Decode 与推理状态

这份笔记为第 4 课核实模型行为和系统行为。资料只取自 Qwen3.5-9B-Base 官方配置、Hugging Face Transformers 官方实现与文档、vLLM 官方源码，以及 Orca、PagedAttention、Sarathi-Serve 原论文。

## 核心结论

第 4 课应当沿一条请求时间线展开，而不是分别解释几个术语：

```text
Prompt p1 p2 p3 p4
        │
        └─ Prefill：处理 4 个已知 token，得到预测 y1 的 Logits
                         │
                         └─ 采样或贪心选择 y1
                                      │
                                      └─ Decode 1：把 y1 作为输入，预测 y2
                                                           │
                                                           └─ Decode 2：把 y2 作为输入，预测 y3
```

最容易讲错的是缓存与生成 token 的对应关系：

- Prefill 结束时，Full Attention 层的 KV Cache 已包含 `p1...p4`，但还不包含 `y1`。
- Prefill 输出的最后一个位置经过 LM Head 得到 Logits，再从中选出 `y1`。
- 下一次模型前向以 `y1` 为新输入。各层在这次前向中计算并缓存 `y1` 的 K/V，最后得到预测 `y2` 的 Logits。

Transformers 的生成循环就是这个顺序：先调用 `_prefill`，取最后一个位置的 Logits，选出下一个 token 并追加到 `input_ids`；后续循环在启用缓存时只取一个新 token 送入模型。[Transformers 生成循环，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/generation/utils.py#L2856-L2923)

## 1. Prefill 到底做了什么

Prefill 是生成开始前，对当前已知输入进行的模型前向计算。对于没有前缀缓存命中的普通文本请求，已知输入就是完整 Prompt。

以 Prompt 长度 `T=4` 为例，模型接收的 Hidden States 是 `[B,4,H]`。在 Full Attention 层中，四个位置各自产生 Q、K、V，并受因果遮罩约束：

```text
p1 只能读 p1
p2 可以读 p1、p2
p3 可以读 p1、p2、p3
p4 可以读 p1、p2、p3、p4
```

四个位置可以放进同一层的张量计算，并不表示它们可以偷看未来。能否读取某个位置由因果遮罩决定，张量能否一起送入算子是另一件事。

模型对所有位置产生 Hidden States 和 Logits，但普通生成只需要最后一个 Prompt 位置的 Logits 来选第一个输出 token。Qwen3.5 的 Causal LM 先运行文本模型，再将所需 Hidden States 送入 LM Head。[Qwen3.5 Causal LM forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1571-L1647)

### “Prompt 可以并行”只是简写

这句话适合描述标准 Full Attention 在一个层内同时处理多个已知位置，但不应扩大成“模型中所有 Prompt 计算彼此独立”。

- Decoder Layer 之间仍按层顺序执行。
- Full Attention 同时计算多个 Query，但因果遮罩仍限制每个 Query 可读取的 Key。
- Qwen3.5 的 Gated DeltaNet 层存在递归状态依赖。官方实现对多 token 输入调用分块 Gated Delta Rule，对单 token 缓存解码调用递归版本。[Qwen3.5 Gated DeltaNet forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L432-L540)

因此，第 4 课应写成“Prompt 的 token 已全部给出，runtime 可以把多个已知位置组织进一次或若干次较大的前向计算”，不要写成“Prompt token 没有依赖，所以完全并行”。

## 2. 一步 Decode 的准确数据流

假设 Prompt 后已经选出 `y1`，且历史 Full Attention 缓存包含 `p1...p4`。预测 `y2` 的这一步是：

```text
输入 token：y1
    ↓ Embedding
当前位置 Hidden State
    ↓ 每个 Decoder Layer
Full Attention 层：
    只为 y1 计算新的 Q、K、V
    新 Q 读取历史 K/V 与 y1 自己的 K/V
    把 y1 的 K/V 追加到本层缓存
Gated DeltaNet 层：
    用 y1 更新卷积状态和递归状态
    ↓ 最终 RMSNorm 与 LM Head
预测 y2 的 Logits
    ↓ 采样或贪心
得到 y2
```

在 Qwen3.5 的 Full Attention 实现中，当前输入先产生 Q、K、V，Q/K 再应用 RoPE，然后当前 K/V 交给 Cache 的 `update`。该方法返回历史加当前的完整 K/V，供本次 Attention 使用。[Qwen3.5 Full Attention forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L656-L695)

对普通单 token Decode，关键 shape 可以写成：

```text
当前 Q：[B,Nq,1,D]
当前 K：[B,Nkv,1,D]
当前 V：[B,Nkv,1,D]

追加后的 K Cache：[B,Nkv,T+1,D]
追加后的 V Cache：[B,Nkv,T+1,D]
```

其中 `T` 是进入本步前已经缓存的 token 数。新 Query 要与 `T+1` 个 Key 比较，包括历史位置和当前位置。

同一请求的 `y2` 不能提前计算，因为 `y2` 的输入依赖 `y1` 实际选中了什么。KV Cache 省掉历史 K/V 的重复投影，却没有改变这条自回归依赖。

## 3. KV Cache 保存什么

KV Cache 保存的是每个 Full Attention 层中，已经处理过的位置所产生的 K 和 V：

```text
不保存：模型权重的副本
不保存：未来 token
不保存：每一步最终答案
不保存：所有层的完整 Hidden States
保存：每个 Full Attention 层的历史 K/V
```

因果模型处理过去位置时看不到未来，因此未来 token 到来以后，过去位置在该层已经算出的 K/V 不需要重新计算。后续步骤只要计算当前 K/V，再与历史缓存拼接。Transformers 官方缓存说明也按这一顺序描述，并明确缓存按层保存。[Transformers 缓存说明，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/docs/source/en/cache_explanation.md#L34-L90)

### 3.1 逻辑 shape

一个普通动态缓存层的 K 和 V 都是：

```text
[B,Nkv,T,D]
```

| 符号 | 含义 |
|---|---|
| `B` | 该张量中的序列数量 |
| `Nkv` | K/V 头数量 |
| `T` | 已经缓存的序列长度 |
| `D` | 每个 K/V 头的维度 |

Transformers 的 `DynamicLayer` 也把缓存 shape 定义为 `[batch_size,num_heads,seq_len,head_dim]`，并沿序列轴拼接新的 K/V。[Transformers `DynamicLayer`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L113-L159)

这里使用 `Nkv` 而不是 `Nq`。GQA 中多个 Query 头共享 K/V 头，缓存没有必要按 Query 头数量复制。某些朴素 Attention 实现会在计算时逻辑展开 K/V，优化 Kernel 通常直接处理分组关系；这不改变缓存本身的宽度。

### 3.2 元素数和字节数

设 Full Attention 层数为 `L_full`，一个元素占 `s` 字节，则等长 Batch 的逻辑 KV 数据量为：

$$
\text{KV bytes}
=2\times B\times L_{full}\times N_{kv}\times T\times D\times s
$$

最前面的 `2` 表示 K 和 V 两份张量。

如果各请求长度不同，更准确的服务端写法是：

$$
\text{KV bytes}
=2\times L_{full}\times N_{kv}\times D\times s\times\sum_i T_i
$$

实际显存还会受到物理布局影响：静态缓存可能预留最大长度，分页缓存按块分配会有尾块空余，Tensor Parallel 可能把头分散到多个设备，还会有块表和对齐开销。因此，上式首先表示逻辑有效载荷，不应直接当成 runtime 的显存峰值。

### 3.3 代入 Qwen3.5-9B-Base

官方配置给出：

```text
总 Decoder Layer：32
Full Attention 层：8
Nkv：4
D：256
配置 dtype：bfloat16
```

32 层按 `3 × linear_attention + 1 × full_attention` 重复，所以只有 8 层使用随长度增长的 KV Cache。[Qwen3.5-9B-Base 配置，revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)

若 runtime 按 BF16 保存 KV，每个元素是 2 字节。此时每个 token、每个请求的 Full Attention KV 有效载荷为：

$$
2\times 8\times 4\times 256\times 2
=32768\ \text{bytes}
=32\ \text{KiB}
$$

因此：

```text
1,000 token：约 31.25 MiB
131,072 token：4 GiB
262,144 token：8 GiB
```

这些数字只包括 8 个 Full Attention 层的 BF16 K/V，不包括 Gated DeltaNet 状态、视觉输入、分配块空余、临时张量、模型权重和 runtime 开销。runtime 也可以另行配置 KV Cache dtype；如果不是 BF16，应替换公式中的元素字节数，不能直接沿用上述结果。

### 3.4 PagedAttention 改变的是物理管理方式

PagedAttention 不会把 K/V 换成另一种模型语义。它把每条序列的 KV Cache 分成固定大小的块，允许逻辑连续的缓存放在不连续的物理块中，以减少碎片和重复副本，并方便不同请求共享缓存块。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)

课程中应当分清：

```text
KV Cache：模型层面保存并复用什么
PagedAttention：服务系统怎样分配、定位和共享这些缓存
Prefix Cache：不同请求何时可以复用相同前缀的状态
```

## 4. Qwen3.5 还有另一套推理状态

不能把 Qwen3.5 的全部 32 层都画成 KV Cache。

Transformers 根据 `layer_types` 为不同层建立不同的缓存对象：`full_attention` 对应会增长的 `DynamicLayer`，`linear_attention` 对应 `LinearAttentionLayer`。[Transformers Cache 类型映射](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L1210-L1230)

Qwen3.5-9B-Base 的 24 个 Gated DeltaNet 层保存两类固定形状状态：

1. 深度卷积所需的最近窗口 `conv_state`。
2. Gated Delta Rule 的 `recurrent_state`。

官方实现的线性 Attention 缓存不会沿 token 维无限增长。卷积状态通常只保留最近 `conv_kernel_size` 个位置，递归状态则原地更新。[Transformers `LinearAttentionLayer`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L998-L1086)

按当前官方 Python 回退实现推导，Qwen3.5-9B 的单个 Gated DeltaNet 层状态 shape 是：

```text
conv_state：[B,8192,4]
recurrent_state：[B,32,128,128]
```

推导依据是 `16` 个 Key 头、`32` 个 Value 头、Key/Value 头维均为 `128`，卷积宽度为 `2 × key_dim + value_dim = 8192`；递归计算会把 Q/K 头扩展到 `32`，最终状态 shape 为 `[B,num_heads,k_head_dim,v_head_dim]`。[Qwen3.5 Gated DeltaNet 定义](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L387-L425) [Gated Delta Rule 状态创建](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L330-L380)

第 4 课只需讲清边界：Full Attention 使用随序列长度增长的 K/V，Gated DeltaNet 使用固定 shape 的状态。状态内部更新过程留到第 5 课。

## 5. TTFT、ITL 与 TPOT 的边界

### 5.1 从用户视角定义

用户感知的 TTFT 是从请求发出到收到第一个输出 token 的时间。它可能包含：

```text
网络与协议处理
→ 排队
→ Tokenize 或多模态预处理
→ Prefill
→ LM Head 与采样
→ 首个 token 序列化并返回
```

因此，“TTFT 就是 Prefill Kernel 时间”不成立。

ITL 是相邻两个输出 token 之间的时间。TPOT 通常是首 token 之后所有输出间隔的平均值：

$$
\text{TPOT}
=\frac{t_{last}-t_{first}}{N_{output}-1}
$$

首 token 不计入分母。只有一个输出 token 时，这个值没有可解释的间隔，系统可能返回 `0` 或 `null`。

### 5.2 指标名字相同，不代表计时边界相同

当前 vLLM 源码中的 Prometheus TTFT 使用请求 `arrival_time` 到首 token 的间隔，因此包含排队时间；其请求阶段统计则把排队、Prefill 和 Decode 分开：

```text
queue_time   = first_scheduled - queued
prefill_time = first_token - first_scheduled
decode_time  = last_token - first_token
TPOT         = decode_time / (output_tokens - 1)
```

[vLLM 时间统计源码，revision `643c125`](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/metrics/stats.py#L381-L500)

同一 revision 的 vLLM 每请求指标文档却把 `time_to_first_token_ms` 定义为“从被调度到产生首 token”，并单独返回 `queue_time_ms`。[vLLM 每请求指标文档](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/docs/features/per_request_metrics.md#L33-L55)

课程应明确告诉读者：比较 TTFT 或 TPOT 前，必须查看采集点和计算公式，不能只看指标名字。

## 6. Continuous Batching 为什么成立

同一请求的未来 token 必须串行确定，但不同请求之间没有这种依赖。请求 A 等待下一步 Decode 时，runtime 可以在下一轮重新组成 Batch：

```text
第 1 轮：A 的 Decode，B 的 Decode，C 的 Prefill
第 2 轮：A 的 Decode，C 的 Decode，D 的 Prefill
```

Orca 将这种思想表述为 Iteration-level Scheduling：调度粒度从整个请求变成一次模型迭代，完成的请求可以退出，新请求也能进入下一轮 Batch。[Orca 原论文与 USENIX 页面](https://www.usenix.org/conference/osdi22/presentation/yu)

现代 runtime 的实现不必把调度对象硬分成两类。当前 vLLM V1 记录每个请求“已经算到多少 token”和“目前有多少 token 已知”，再从全局 token 预算中分配本轮计算量。源码明确说明 Scheduler 中没有严格的 Prefill phase 和 Decode phase，这套表示同时覆盖 Chunked Prefill、Prefix Cache 和 Speculative Decoding。[vLLM V1 Scheduler，revision `643c125`](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/core/sched/scheduler.py#L440-L528)

这里仍要区分两个概念：

- Continuous Batching 表示 Batch 可以在模型迭代之间改变。
- Prefill 和 Decode 能否放进同一次模型执行，还取决于 runtime 是否支持混合 Batch、对应 Attention Kernel 和序列元数据。

## 7. Chunked Prefill 为什么能与 Decode 同批

长 Prompt 的 token 已由用户给出。runtime 可以把它切成多段：

```text
Prompt：p1 ... p3000

Chunk 1：p1    ... p512
Chunk 2：p513  ... p1024
Chunk 3：p1025 ... p1536
...
```

处理后一个 Chunk 时，前面 Chunk 的 K/V 或递归状态已经存在。Chunk 边界改变了调度单位，不改变“后面位置只能读取前面位置”的因果关系。

一个 Decode token 也属于已经确定的输入：它是上一轮采样得到的 token。因此，Prefill Chunk 与其他请求当前的 Decode token 都可以成为本轮已知输入。runtime 还必须携带请求边界、位置编号、因果遮罩、缓存块表和 Slot 映射，确保它们不会互相读取缓存。

Sarathi-Serve 把长 Prefill 切块，再让 Decode token 与 Prefill Chunk 组成 Decode-maximal Batch。论文给出的动机是用计算量较大的 Prefill Chunk 承载 Decode，从而改善 GPU 利用率，并控制长 Prefill 对 Decode 间隔造成的停顿。[Sarathi 原论文](https://arxiv.org/abs/2308.16369) [Sarathi-Serve 原论文](https://arxiv.org/abs/2403.02310)

Chunked Prefill 没有让同一请求的未来生成 token 变成已知：

```text
可以切：用户已经提供的 p1...p3000
不能切：尚未采样出来的 y1...y100
```

### 同一个名字可能指不同能力

Transformers 的 `prefill_chunk_size` 会把一个输入 Batch 的 Prompt 依次切块，并在块之间传递 Cache；这段生成代码本身没有把 Chunk 与在线服务中的其他 Decode 请求混批。[Transformers `_prefill`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/generation/utils.py#L3893-L3976)

vLLM 或 Sarathi-Serve 所说的 Chunked Prefill 通常还包含服务调度：把不同请求的 Prefill Chunk 和 Decode 工作放进同一调度轮次。课程不能仅凭“都叫 Chunked Prefill”就认为两种实现能力相同。

## 8. 性能判断中不要过度概括

### 8.1 Prefill 不一定永远是计算受限

Prefill 一次处理较多位置，Linear 往往形成更大的矩阵乘法，Full Attention 还要处理随 Prompt 长度增长的 Q/K 组合。Sarathi 系列论文观察到其测试模型和硬件上的 Prefill 更容易充分利用计算资源。[Sarathi](https://arxiv.org/abs/2308.16369)

但实际瓶颈还受 Prompt 长度、Batch、模型结构、量化、Kernel、Tensor Parallel 和硬件影响。短 Prompt、小模型或低效实现可能仍受权重读取、Kernel Launch 或调度影响。Qwen3.5 还包含 24 个 Gated DeltaNet 层，不能把纯 Full Attention 模型的复杂度结论原样套过来。

### 8.2 Decode 不一定永远是带宽受限

单请求或小 Batch Decode 每轮只有少量新 token，Linear 的权重复用不足，且 Full Attention 要读取逐渐增长的 KV Cache，所以经常暴露显存带宽和固定开销。Batch 增大后，同一份权重可服务更多 token，算术强度会上升；超长上下文下，KV 读取和 Attention 计算也会改变瓶颈。

因此，更稳妥的表述是“较小 Decode Batch 常常更偏带宽或固定开销受限”，不是“Decode 天生只能受显存带宽限制”。

### 8.3 混批不是无条件提高所有指标

Prefill Chunk 和 Decode 混批有机会提高设备利用率，也会增加单轮工作量。Chunk 太大时，活跃 Decode 请求可能等待更久，ITL 和尾延迟变差；Chunk 太小时，Prefill 要经过更多轮次，TTFT 和调度开销可能上升。Sarathi-Serve 的目标正是调整这组吞吐与延迟的交换关系，不是证明一个 Chunk 大小适用于所有 workload。[Sarathi-Serve](https://arxiv.org/abs/2403.02310)

### 8.4 KV Cache 只避免一部分重复计算

KV Cache 避免历史位置在各 Full Attention 层重复生成 K/V，也让 Decode 只更新当前位置的 Hidden State。它没有消除：

- 当前 token 通过全部 Decoder Layer 的计算；
- 当前 Query 对历史 Key 的 Attention；
- 历史 K/V 的读取；
- LM Head、采样和服务调度；
- 自回归步骤之间的依赖。

“有 KV Cache 后 Decode 是常数成本”是错误说法。对 Full Attention 而言，单步需要读取的历史 K/V 随上下文长度增长。

## 9. 推荐的正文讲解顺序

课程正文适合按下面的顺序写：

1. 用 4 个 Prompt token 和 3 个输出 token 画完整时间线。
2. 明确 Prefill 产生的是第一个输出 token 的 Logits，不是一次生成所有答案。
3. 放大第一步 Decode，标出 `y1` 何时作为输入、何时写入缓存。
4. 回到单个 Full Attention 层，解释为什么只保存 K/V，不保存 Q。
5. 推导 `[B,Nkv,T,D]` 和字节公式，再代入 Qwen3.5-9B。
6. 把 Qwen3.5 的 8 个 KV Cache 层与 24 个固定状态层分开画。
7. 由多请求时间线引出 Continuous Batching。
8. 把长 Prompt 切块，解释 Chunked Prefill 为什么可与 Decode 混批。
9. 最后定义 TTFT、ITL、TPOT，并提醒指标采集边界。

建议至少配三张图：

- 一张请求时间线，突出“Logits 预测下一个 token”的错位关系。
- 一张 Full Attention 缓存增长图，显示每步只追加一列 K/V。
- 一张多请求调度图，区分请求内串行与请求间混批。

## 10. 常见错误清单

| 容易写错的说法 | 更准确的说法 |
|---|---|
| Prefill 一次生成第一个 token | Prefill 产生预测第一个 token 的 Logits，采样或贪心才选出 token |
| Prefill 后 KV Cache 已经包含第一个生成 token | 此时只包含已处理的 Prompt；第一个生成 token 在下一次前向中写入缓存 |
| Decode 每步把全文重新输入模型 | API 可能保留完整 `input_ids`，启用 Cache 的模型前向通常只接收尚未处理的新 token |
| KV Cache 保存每个 token 的 Hidden State | Full Attention KV Cache 保存各层历史 K/V |
| 所有 32 层各有一份 KV Cache | Qwen3.5-9B 只有 8 个 Full Attention 层使用增长的 K/V，24 个 Gated DeltaNet 层维护固定状态 |
| KV Cache 使用 `Nq` 计算容量 | GQA 缓存宽度使用 `Nkv` |
| PagedAttention 就是 KV Cache | KV Cache 是模型状态；PagedAttention 是其块式内存管理和访问方法 |
| Prompt token 没有依赖，所以完全并行 | Prompt 已全部已知，可组织为较大的前向计算；因果遮罩和递归层依赖仍存在 |
| Continuous Batching 就是把 Prefill 与 Decode 放在同一 Batch | Continuous Batching 允许迭代间重组 Batch；混合 Prefill/Decode 还需 runtime 和 Kernel 支持 |
| Chunked Prefill 能把未来 100 个输出 token 分块算完 | 它只能切分已经知道的 Prompt，不能创造尚未采样的未来 token |
| TTFT 等于 Prefill Kernel 时间 | TTFT 的采集边界可能还包括排队、预处理、采样和返回，必须查看指标定义 |
| Decode 永远是显存带宽受限 | 小 Batch Decode 经常如此，但瓶颈会随 Batch、上下文、模型、Kernel 和硬件变化 |

## 来源

- Qwen, [Qwen3.5-9B-Base `config.json`, revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)。
- Hugging Face Transformers, [Qwen3.5 模型实现, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
- Hugging Face Transformers, [Cache 实现, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py)。
- Hugging Face Transformers, [生成循环与 Chunked Prefill, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/generation/utils.py)。
- Hugging Face Transformers, [Caching 官方说明, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/docs/source/en/cache_explanation.md)。
- vLLM, [V1 Scheduler, revision `643c125`](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/core/sched/scheduler.py)。
- vLLM, [时间指标实现, revision `643c125`](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/metrics/stats.py)。
- Gyeong-In Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022。
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。
- Amey Agrawal et al., [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)。
- Amey Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024。
