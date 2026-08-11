# 第 4 课：Prefill、Decode 与 KV Cache

假设 Prompt 已经被分成 4 个 token：

```text
p1  p2  p3  p4
```

模型接下来生成 `y1 y2 y3`。整个过程不是把 7 个 token 一次送进模型，而是执行 3 次前向：

| 模型前向 | 本轮输入 | 前向结束后的请求状态 | 根据本轮 Logits 选出 |
| --- | --- | --- | --- |
| Prefill | `p1 p2 p3 p4` | Prompt 的状态 | `y1` |
| Decode 第 1 轮 | `y1` | Prompt 加 `y1` 的状态 | `y2` |
| Decode 第 2 轮 | `y2` | Prompt 加 `y1 y2` 的状态 | `y3` |

这张表有一个很容易看错的地方：Prefill 选出了 `y1`，但 Prefill 结束时的 Cache 还不包含 `y1`。`y1` 要在下一次前向中经过各层，它的 K/V 才会写入 Cache。

![一次请求中的 Prefill 和 Decode](../assets/04-generation-timeline.svg)

## 1. 一次请求怎样向前生成

### 1.1 Prefill 处理完整 Prompt

模型开始执行前，`p1` 到 `p4` 的 Token ID 已经全部确定。Prefill 把这些已知位置送入模型：

```text
Token IDs                    [1,4]
→ Embedding                  [1,4,H]
→ 多层 Decoder               [1,4,H]
→ 取最后一个位置 p4          [1,H]
→ LM Head                    [1,V]
→ 贪心或采样
→ y1
```

`V` 是词表大小。LM Head 得到的是候选 token 的分数，贪心或采样才从中选出 `y1`。所以 Prefill 的模型前向产生 Logits，不会直接产生文字。

每个 Full Attention 层还会保存 `p1` 到 `p4` 的 K/V。以 Qwen3.5-9B 为例，单层缓存的逻辑 shape 是：

```text
K Cache：[B,Nkv,T,D] = [1,4,4,256]
V Cache：[B,Nkv,T,D] = [1,4,4,256]
```

第一个 `4` 是 K/V 头数 `Nkv`，第二个 `4` 是 Prompt 长度 `T`。它们只是碰巧相同。

### 1.2 Decode 每轮接收一个已经选出的 token

Prefill 选出 `y1` 后，下一次前向只接收 `y1`：

```text
输入 y1
→ 读取 Prompt 留下的状态
→ 用 y1 更新各层状态
→ 得到预测 y2 的 Logits
→ 选出 y2
```

随后用 `y2` 预测 `y3`。启用 Cache 时，模型不必在每一轮重新处理 `p1 p2 p3 p4`。

有些框架把产生 `y1` 也叫作第一步 Decode，有些框架把输入 `y1`、产生 `y2` 的前向叫作 Decode 第 1 轮。名称可能不同，数据依赖没有变化。读代码或指标时，先确认阶段边界。

### 1.3 Prompt 可以批量计算，未来输出不行

Prompt 的 4 个位置在前向开始前已经知道。以一个 Full Attention 层为例，它们的 Q、K、V 可以组织成同一组张量计算：

```text
Q：[B,Nq,4,D]
K：[B,Nkv,4,D]
V：[B,Nkv,4,D]
```

因果遮罩仍然限制每个位置可以读取的范围：

```text
p1 读取 p1
p2 读取 p1、p2
p3 读取 p1、p2、p3
p4 读取 p1、p2、p3、p4
```

遮罩控制“能读哪些位置”，不要求 GPU 先算完 `p1` 才开始算 `p2`。因此，Prompt 的多个已知位置可以放进较大的张量运算。

未来输出不同。它们的联合概率是：

$$
P(y_1,y_2,y_3\mid p)
=P(y_1\mid p)P(y_2\mid p,y_1)P(y_3\mid p,y_1,y_2)
$$

预测 `y2` 时必须知道实际选中了哪个 `y1`。如果 `y1` 还没确定，模型就不知道下一轮的输入。普通自回归生成因此不能把未来 100 个 token 当成已知的 `T=100` 一次算完。

“Prompt 可以并行”是一种简写。Decoder Layer 仍要逐层执行，Qwen3.5 的 Gated DeltaNet 层也保留递归状态依赖。准确的说法是：Prompt 已全部给出，runtime 可以把多个已知位置组织成一次或若干次较大的前向计算。

## 2. KV Cache 省掉了哪些重复计算

没有 Cache 时，为了生成 `y2`，模型要重新处理：

```text
p1 p2 p3 p4 y1
```

生成 `y3` 时又要处理：

```text
p1 p2 p3 p4 y1 y2
```

历史前缀反复经过 Embedding、所有 Decoder Layer 和 LM Head，结果虽然正确，计算却重复了。

Full Attention 的历史 K/V 已经由过去位置算好。由于因果遮罩，后来的 token 不会改变过去位置当时能够读取的内容，所以这些 K/V 可以直接复用。KV Cache 保存的正是每个 Full Attention 层的历史 K/V。

![没有 KV Cache 与使用 KV Cache 的计算对比](../assets/04-without-vs-with-kv-cache.svg)

### 2.1 放大一次 Decode

假设 Prefill 后已经缓存 4 个位置：

```text
K_past：[B,Nkv,4,D]
V_past：[B,Nkv,4,D]
```

现在输入 `y1`。当前层只为这个新位置计算：

```text
q_new：[B,Nq,1,D]
k_new：[B,Nkv,1,D]
v_new：[B,Nkv,1,D]
```

从模型语义看，新 K/V 追加到历史缓存：

```text
K_all = concat(K_past, k_new)  → [B,Nkv,5,D]
V_all = concat(V_past, v_new)  → [B,Nkv,5,D]
```

新 Query 读取 5 个位置：

```text
q_new 与 K_all 打分      → [B,Nq,1,5]
Softmax 后汇总 V_all     → [B,Nq,1,D]
```

![一步 Decode 怎样读写 KV Cache](../assets/04-kv-cache-growth.svg)

这次前向结束后，Cache 才从 4 个位置增长到 5 个位置。刚刚选出的 `y2` 还没有经过模型，所以此时 Cache 不包含 `y2`。

PagedAttention 一类实现不需要每轮申请一段更长的连续内存，再执行一次物理 `concat`。上面的写法只表示 Cache 的逻辑长度增加了一个位置。

### 2.2 为什么保存 K/V，不保存 Q

Q 只负责当前读取。`y1` 的 Q 在处理 `y1` 时使用一次，后续 token 会产生自己的 Q，不会再次使用 `y1` 的 Q。

K/V 代表历史位置以后怎样接受查询、怎样贡献信息。每个新 Query 都可能读取它们，因此必须保留。

Qwen3.5 的 Full Attention 先对 Q/K 应用 RoPE，再把 K 写入 Cache。缓存中的 K 已经带有原位置的信息。前缀复用或缓存拼接除了 shape 正确，还要保证位置编号和 RoPE 规则一致。

## 3. KV Cache 的容量

下面先计算 K/V 数值本身占用的空间，不计分配器块、对齐、元数据和碎片。

$$
\text{KV Cache Bytes}
=2\times L_{full}\times B\times T\times N_{kv}\times D\times s
$$

| 符号 | 含义 |
| --- | --- |
| `2` | K 和 V 两份缓存 |
| `L_full` | Full Attention 层数 |
| `B` | 同时保存状态的序列数 |
| `T` | 已缓存的位置数 |
| `Nkv` | 每层 K/V 头数 |
| `D` | 每个 K/V 头的维度 |
| `s` | 每个元素占用的字节数 |

Qwen3.5-9B 有 32 个 Decoder Layer，其中 8 层使用 Full Attention。按 BF16 KV Cache 计算：

```text
L_full = 8
Nkv    = 4
D      = 256
s      = 2 Byte
```

一个请求每增加一个缓存位置，8 层共增加：

$$
2\times8\times4\times256\times2
=32768\ \text{Byte}
=32\ \text{KiB}
$$

| 已缓存长度 `T` | Full Attention KV Cache |
| ---: | ---: |
| 1 token | 32 KiB |
| 4096 token | 128 MiB |
| 32768 token | 1 GiB |

`T` 包括 Prompt 和已经经过模型的输出 token。刚被采样出来、尚未进入下一次前向的 token 不计入 Cache。

这些数字是逻辑有效载荷，不等于 runtime 实际申请的显存。实际值还会受到缓存 dtype、块大小、尾块空余、Prefix Cache 共享、TP 下的头分布和预留策略影响。`partial_rotary_factor=0.25` 也不会把 KV Cache 缩小到四分之一，RoPE 只旋转部分 K 维度，Cache 仍保存完整 K 和 V。

### 3.1 PagedAttention 管理的是物理内存

如果每个请求都申请一段连续显存，系统很难提前知道它最终会生成多长。PagedAttention 把 KV Cache 分成固定大小的块：

```text
逻辑顺序：token 1 → token 2 → token 3 → ...
物理块：  block 7 → block 21 → block 9 → ...
```

块表记录逻辑位置与物理块的对应关系。请求增长时可以继续分配块，不必预留最大长度，也不必把整段 Cache 搬到新的连续地址。

三个名称回答三个不同问题：

| 名称 | 回答的问题 |
| --- | --- |
| KV Cache | Full Attention 为历史 token 保存什么 |
| PagedAttention | runtime 怎样分配和定位这些 K/V |
| Prefix Cache | 不同请求何时可以复用相同前缀的状态 |

## 4. Qwen3.5 同时保留两类状态

Qwen3.5-9B 每 4 层中有 3 个 Gated DeltaNet 层和 1 个 Full Attention 层。因此一个请求同时保留：

| 层类型 | 层数 | 跨前向保存的状态 | 是否随上下文长度增长 |
| --- | ---: | --- | --- |
| Full Attention | 8 | 历史 K/V | 是 |
| Gated DeltaNet | 24 | 卷积状态和递归状态 | shape 固定 |

本课的 KV Cache 公式只适用于 8 个 Full Attention 层。24 个 Gated DeltaNet 层没有逐 token K/V，却不等于没有状态。第 5 课会解释它怎样更新固定 shape 的状态。

## 5. 多个请求怎样组成 Batch

同一个请求的未来 token 需要依次确定，不同请求却没有这种依赖。假设调度器现在有三份工作：

```text
请求 A：输入 a5，预测 a6       1 个 Decode token
请求 B：输入 b2，预测 b3       1 个 Decode token
请求 C：处理 c1 到 c4          4 个 Prefill token
```

如果 runtime 支持混合执行，这 6 个已知 token 可以组成同一轮模型输入。Linear 和 FFN 可以批量处理它们；Token Mixer 仍要根据请求归属、位置编号、因果边界和 Cache 地址把三条序列隔开。

![不同请求怎样组成同一轮执行](../assets/04-mixed-batch.svg)

这里有两条不同的并行关系：

```text
同一请求的未来 token：值尚未确定，必须逐轮生成
不同请求的已知 token：彼此独立，可以批量执行
```

### 5.1 Continuous Batching

传统静态 Batch 常让一组请求一起运行到结束。短请求先完成，也要等长请求；空出的计算位置不能及时交给新请求。

Continuous Batching 会在相邻模型迭代之间重组 Batch：

```text
本轮结束
→ 移除已完成的请求
→ 加入可以运行的新请求
→ 为未完成请求安排下一步
→ 开始下一轮
```

它改变的是调度粒度，不会取消单个请求的自回归顺序。

Continuous Batching 也不自动等于 Prefill 和 Decode 混批。混批还要求 runtime 能描述不同长度、不同阶段的序列，并由相应 Attention Kernel 正确处理。日志中的 Batch Size 也可能指请求数、序列数或本轮 token 数，分析前要先确认统计口径。

## 6. 长 Prompt 怎样分块

一个长 Prompt 全部放进同一轮，可能占满 token 预算，让正在 Decode 的请求等待。Chunked Prefill 把已经知道的 Prompt 切成几段：

```text
8-token Prompt
→ Chunk 1：p1 p2 p3 p4
→ Chunk 2：p5 p6 p7 p8
```

Chunk 2 必须接着 Chunk 1 的状态继续计算。Full Attention 读取前一段 K/V，Gated DeltaNet 读取前一段的卷积状态和递归状态。Chunk 边界不会清空上下文。

调度器可以先放入活跃请求的 Decode token，再用剩余 token 预算放入 Prefill Chunk。这样可以减少长 Prefill 对 Decode 请求的阻塞，并让空余计算容量得到利用。

Chunk 大小需要结合负载选择。较大的 Chunk 能更快完成 Prefill，却可能拉长同批 Decode 的间隔；较小的 Chunk 让 Decode 更容易插入，但一个 Prompt 需要经过更多轮次，调度开销和 TTFT 可能上升。

还要区分两个同名能力。Transformers 的 `prefill_chunk_size` 会把一个输入的 Prompt 顺序切开，并在 Chunk 之间传递 Cache。vLLM 或 Sarathi-Serve 所说的 Chunked Prefill 通常还包含在线调度，可以把某个请求的 Prefill Chunk 与其他请求的 Decode token 放进同一轮。

## 7. Prefill 和 Decode 的计算差异

两阶段使用同一套权重，区别在于本轮处理多少新位置，以及读取多少历史状态。

| 对比项 | Prefill | Decode |
| --- | --- | --- |
| 每个请求本轮输入 | 多个 Prompt token | 通常 1 个新 token |
| Linear/FFN | 同一权重处理多行输入 | 同一权重处理少量新位置 |
| Full Attention 查询数 | 多个 Query | 每个请求通常 1 个新 Query |
| KV Cache | 本轮建立 | 每轮读取并追加 |
| 单请求可用并行度 | 较大 | 较小 |

Prefill 的 Linear 和 FFN 通常能形成较大的矩阵乘法。Decode 的小 Batch 每轮工作量较少，权重复用不足，Kernel Launch 和 CPU 调度等固定开销更容易占据较高比例；Full Attention 还要读取不断增长的历史 K/V。

“Prefill 偏计算，Decode 偏访存”是常见现象，不是模型定律。Prompt 长度、Batch、量化、Kernel、模型结构和硬件都可能改变瓶颈。性能判断要回到实际矩阵 shape、Cache 长度和执行时间线。

## 8. 延迟指标：TTFT、ITL 和 TPOT

![TTFT、TPOT 与 ITL](../assets/04-latency-metrics.svg)

首 token 延迟（Time to First Token，TTFT）从请求发出开始，到客户端收到第一个输出 token 结束。端到端 TTFT 可能包括：

```text
网络和请求解析
→ 排队与调度
→ Chat Template、Tokenize 或多模态预处理
→ Prefill
→ LM Head 与采样
→ 首 token 返回
```

TTFT 因此不等于 GPU 上的 Prefill Kernel 时间。

Token 间延迟（Inter-token Latency，ITL）是相邻两个输出 token 的到达间隔。一个请求有多段 ITL，它们会随 Batch、上下文长度和调度情况变化。

每输出 token 时间（Time per Output Token，TPOT）通常排除首 token，计算后续间隔的平均值。vLLM serving benchmark 使用：

$$
TPOT=\frac{E2E-TTFT}{N_{out}-1},\qquad N_{out}>1
$$

平均 TPOT 相同的两个请求，ITL 分布可能不同，其中一个仍可能出现明显卡顿。

指标同名不代表采集边界相同。本课引用的 vLLM revision 中，Prometheus TTFT 从请求到达开始，包含排队；每请求指标文档把排队时间单列，TTFT 从首次被调度开始。比较不同数据前，应先查看采集点和计算公式。

## 9. 练习：跟踪一次生成

Prompt 是 `p1 p2 p3 p4`，模型最终生成 `y1 y2 y3`。补全下表：

| 模型前向 | 本轮输入 | 前向结束后的 Cache | 本轮选出的 token |
| --- | --- | --- | --- |
| Prefill | `p1 p2 p3 p4` |  |  |
| Decode 第 1 轮 |  |  |  |
| Decode 第 2 轮 |  |  |  |

然后回答：

1. Decode 第 2 轮结束后，Cache 中有几个位置？`y3` 是否已经在 Cache 中？
2. 给定 `L_full=2`、`B=1`、`Nkv=2`、`D=4`，KV Cache 使用 BF16，这时一共需要多少字节？
3. Continuous Batching 能否让同一个请求在一轮中直接确定 `y1 y2 y3`？
4. 为什么 TTFT 不能直接当作 Prefill Kernel 时间？

<details>
<summary>查看答案</summary>

| 模型前向 | 本轮输入 | 前向结束后的 Cache | 本轮选出的 token |
| --- | --- | --- | --- |
| Prefill | `p1 p2 p3 p4` | `p1 p2 p3 p4` | `y1` |
| Decode 第 1 轮 | `y1` | `p1 p2 p3 p4 y1` | `y2` |
| Decode 第 2 轮 | `y2` | `p1 p2 p3 p4 y1 y2` | `y3` |

Decode 第 2 轮结束后，Cache 中有 6 个位置。`y3` 刚从 Logits 中选出，还没有进入下一次模型前向，因此不在 Cache 中。

```text
2 × 2 × 1 × 6 × 2 × 4 × 2 = 384 Byte
```

这些因子依次表示 K/V 两份、2 个 Full Attention 层、1 条序列、6 个缓存位置、2 个 K/V 头、每头 4 维和 BF16 每元素 2 Byte。

Continuous Batching 只能重组不同请求在每一轮的工作，不能消除同一请求未来 token 的依赖。端到端 TTFT 还可能包含网络、输入处理、排队、调度、采样和返回时间。

</details>

## 参考资料

以下 Qwen3.5 配置、Transformers 和 vLLM 实现于 2026-08-08 按固定 revision 复核：

- [Qwen3.5-9B-Base `config.json`，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)
- [Transformers：Qwen3.5 模型实现，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Transformers：Cache 实现，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py)
- [Transformers：生成循环与 Prompt 分块，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/generation/utils.py)
- [Transformers：KV Cache 说明，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/docs/source/en/cache_explanation.md)
- [vLLM：V1 Scheduler，revision 643c125](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/core/sched/scheduler.py)
- [vLLM：时间指标实现，revision 643c125](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/v1/metrics/stats.py)
- [vLLM：每请求指标说明，revision 643c125](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec314734722f/docs/features/per_request_metrics.md)
- [vLLM：Serving Benchmark 的 TTFT、TPOT 与 ITL 计算，revision 643c125](https://github.com/vllm-project/vllm/blob/643c125fab66d5ed5ec3143b7e764a77e7ae8ac7/vllm/benchmarks/serve.py#L582-L613)

服务调度和缓存管理参考以下原始论文：

- [Orca：Iteration-level Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [SARATHI：Chunked Prefill 与 Decode 混批](https://arxiv.org/abs/2308.16369)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

---

[上一课：Attention 的计算原理](03-attention.md) · [返回课程路线](../roadmap.md) · [下一课：Gated DeltaNet 的状态读写](05-gated-deltanet.md)
