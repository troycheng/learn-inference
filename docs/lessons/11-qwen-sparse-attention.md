# 第 11 课：Qwen Sparse Attention 的计算过程

第 3 课中的 Full Attention 让当前位置与所有可见位置计算 QK 分数，再用 Softmax 权重汇总 V。它保留了完整历史，但上下文越长，每个 Query 需要读取和比较的 Key 越多。

Qwen Sparse Attention，简称 QSA，没有修改 Attention 最终的 QK、Softmax 和 V 加权求和。它在正式 Attention 前增加一个轻量索引器，先选出优先读取的位置，正式 Attention 只计算这些位置。

理解 QSA，必须始终区分两套计算：

```text
索引器：决定读取哪些 token
核心 Attention：决定怎样汇总这些 token 的 V
```

## 1. Full Attention 的成本来自哪里

设序列长度为 `T`。一次 Prefill 中，每个位置都要与此前可见位置打分，QK 分数矩阵接近 `T×T`。一次 Decode 只产生一个新 Query，但它仍要读取 `T` 个历史 Key 和 Value。

```text
Prefill：T 个 Query × 最多 T 个历史位置
Decode：1 个 Query × T 个历史位置
```

长上下文中，模型通常不会平均使用所有历史 token。QSA 的目标是用便宜的计算筛出候选位置，再把正式 Attention 的读取范围限制在固定预算内。

## 2. QSA 的两级计算

QSA 的完整数据流如下：

![QSA 先选择位置，再执行正式 Attention](../assets/11-qsa-two-stage-flow.svg?rev=20260828-1)

第一阶段使用独立的索引器 Q/K：

```text
隐藏状态
→ 轻量 Query 投影
→ 轻量 Key 投影
→ 把连续 Key 压缩成微块
→ Query 为每个微块打分
→ Top-K 选择微块
→ 展开为原始 token 位置
```

第二阶段仍使用原来的 Attention Q/K/V：

```text
隐藏状态
→ 正式 Q/K/V 投影
→ 只取被选中位置的 K/V
→ QK 点积、缩放、因果遮罩、Softmax
→ 权重汇总 V
→ 输出投影
```

索引器压缩后的 Key 只用于选择位置，不能代替正式 Attention 的 Key 和 Value。

## 3. 第一步：把历史 Key 压缩成微块

真实 QSA 每 4 个连续 token 组成一个微块。为了手算，先把压缩比例缩小为 `r=2`。假设当前位置能看到 8 个 token：

```text
位置：  1  2 | 3  4 | 5  6 | 7  8
微块：   B0  |  B1  |  B2  |  B3
```

索引器先为每个 token 计算一条轻量 Key，再对同一微块中的 Key 求平均：

$$
\overline k_b=\mathrm{RMSNorm}\left(\frac{1}{r}\sum_{j=0}^{r-1}k_{br+j}\right)
$$

例如：

```text
k3 = [1, 3]
k4 = [3, 1]
```

忽略 RMSNorm，微块 `B1` 的压缩 Key 是：

$$
\overline k_1=\frac{[1,3]+[3,1]}{2}=[2,2]
$$

四个微块最终只保留四条压缩 Key，而不是八条逐 token Key。真实配置 `r=4`，所以索引器需要比较的 Key 数量约为原来的四分之一。

### 3.1 为什么先平均，再加入 RoPE

RoPE 会让不同位置的同一组特征旋转不同角度。如果先给每个 token 加 RoPE，再把四个不同角度的向量平均，内容和位置信息会混在一起。

QSA 采用下面的顺序：

```text
原始索引器 Key
→ 微块内 AvgPool
→ RMSNorm
→ 给整个微块设置一个位置
→ Partial RoPE
```

微块使用起始 token 的位置。Query 仍使用当前 token 自己的位置。

## 4. 第二步：Query 为每个微块打分

索引器有多个 Query 头和一个共享 Key 头。每个 Query 头分别与微块 Key 点积，负数经过 ReLU 变成 0，再把各头分数相加：

$$
I_{i,b}=\frac{1}{\sqrt{D_{idx}}}\sum_h\mathrm{ReLU}\left(q_i^{(h)}\cdot\overline k_b\right)
$$

`D_idx` 是索引器头维度，真实配置为 128。`I[i,b]` 表示位置 `i` 对微块 `b` 的重要性估计。它不是正式 Attention Score，也不进入最终的 Softmax。

继续使用 8 个位置的例子。假设当前位置的索引器为四个微块得到：

```text
微块          B0    B1    B2    B3
索引器分数    0.2   1.7   0.6   1.1
```

如果 token 预算为 4、每块 2 个 token，那么最多选择两个完整微块。Top-2 结果为：

```text
B1、B3
```

展开成原始 token 位置：

```text
B1 → 位置 3、4
B3 → 位置 7、8

正式 Attention 的候选位置：3、4、7、8
```

微块只减少了索引器的比较次数。选中后仍会回到原始 token 位置。

## 5. 因果关系怎样作用于微块

位置 `i` 只能给已经完整出现的微块打分。若一个微块包含位置 9～12，而当前只处理到位置 10，这个微块还不能作为完整微块参与 Top-K。

最后一个不完整微块中的可见 token 会直接加入候选集合：

```text
完整微块：参与索引器打分和 Top-K
末尾未满 4 个 token：直接保留
未来 token：因果遮罩禁止读取
```

这样既不会泄露未来位置，也不会因为等待凑满四个 token 而丢掉最近上下文。

Prefill 时，每一行 Query 能看到的完整微块数不同；Decode 时，当前 Query 可以给此前所有完整微块打分，再把末尾不足四个的历史 token 加进候选集合。

## 6. 第三步：对选中的原始 K/V 做正式 Attention

索引器已经选出位置 3、4、7、8。接下来使用正式 Attention 的 Q/K/V，为这四个位置重新计算分数。

假设缩放和因果遮罩后的正式 Attention Score 为：

```text
位置          3     4      7     8
Score        0.4   1.2   -0.3   0.7
```

Softmax 权重约为：

```text
位置          3      4      7      8
权重        0.197  0.439  0.098  0.266
```

四个位置的 Value 为：

```text
V3 = [1,0]
V4 = [0,2]
V7 = [1,1]
V8 = [2,0]
```

输出为：

$$
\begin{aligned}
o={}&0.197[1,0]+0.439[0,2]\\
&+0.098[1,1]+0.266[2,0]\\
\approx{}&[0.827,0.976]
\end{aligned}
$$

这一步与第 3 课完全相同。变化只在于 Softmax 的输入从八个历史位置缩小到四个候选位置。

## 7. 为什么不能直接使用索引器分数

索引器追求低成本，所以它的头数、宽度和 Key 数量都比正式 Attention 小。它只要大致找对相关位置，不需要生成最终上下文向量。

两套分数承担不同职责：

| 分数 | 比较对象 | 作用 | 是否进入 V 加权求和 |
| --- | --- | --- | --- |
| 索引器分数 | Query 与压缩微块 Key | 选择候选位置 | 否 |
| Attention Score | 正式 Query 与原始 token Key | 计算 Softmax 权重 | 是 |

如果直接用索引器分数汇总 V，微块中的多个 token 只能共享一个粗粒度分数，正式 Q/K 投影也失去作用。QSA 因而是一套“粗选加精算”的两级结构。

## 8. 回到 Qwen3.8-Flash-Next 的真实配置

QSA 层的核心 Attention 仍是带输出门控的 GQA：

```text
Hidden Size                 2560
Query Heads                 24
KV Heads                    2
Head Dimension              256
```

轻量索引器使用：

```text
Indexer Query Heads         4
Indexer Shared Key Heads    1
Indexer Head Dimension      128
Compression Ratio r         4 token / 微块
Token Budget K              2048
Block Budget                2048 / 4 = 512 个完整微块
```

正式 Attention 和索引器都使用 Partial RoPE。正式头的 256 维中有 64 维参与 RoPE；索引器头的 128 维中同样有 64 维参与 RoPE。

模型共有 12 个 QSA 层，每隔三个 Gated DeltaNet 层出现一次。QSA 负责从原始长上下文中重新取回具体 token 信息，Gated DeltaNet 则用固定大小状态连续汇总历史。

## 9. QSA 保存哪些请求状态

按当前实现，QSA 层需要保存两类随上下文增长的数据：

```text
正式 Attention：每个历史位置的 K/V
轻量索引器：每个历史位置的原始 indexer Key
```

微块平均在执行索引时计算。QSA 没有把所有历史 K/V 永久压缩为 512 个块，也没有把未选中的 K/V 从 Cache 删除。不同 Query 可能选择不同微块，因此完整历史仍要保留。

这意味着 QSA 主要改变：

- 一次 Attention 实际读取多少历史 K/V；
- QK、Softmax 和 V 汇总处理多少候选位置；
- 长上下文下 Attention Kernel 的计算和访存量。

它不会按 `2048/T` 的比例缩小 KV Cache 容量。相比普通 GQA，QSA 还多了一份索引器 Key 状态。

## 10. Prefill 与 Decode 中的成本

设上下文长度为 `T`、压缩比例为 `r`、正式 Attention 的 token 预算为 `K`。

Prefill 中，索引器仍要让大量 Query 与约 `T/r` 个微块比较，复杂度约为：

$$
O\left(\frac{T^2}{r}\right)
$$

核心稀疏 Attention 每个 Query 最多处理 `K` 个候选位置，复杂度约为：

$$
O(TK)
$$

Decode 中只有一个新 Query：

```text
索引器：与约 T/r 个微块比较
核心 Attention：读取最多 K 个候选位置的 K/V
```

因此 QSA 的收益会随上下文增长而扩大。技术报告中的 1M 上下文实验显示，QSA 相比稠密 GQA 的 Attention 模块内核，Prefill 加速 7.6 倍，Decode 加速 4.9 倍。该结果包含索引器和稀疏核心 Attention，但不包含 MoE、Gated DeltaNet、调度、通信和其他端到端时间。

稀疏结构本身不保证加速。Kernel 必须根据候选位置只读取并计算选中的 K/V；如果只是给完整 `T×T` 分数矩阵加一层遮罩，矩阵乘法仍然是稠密的。Transformers 参考实现主要用于表达模型语义，报告中的性能来自专用的融合 QSA Kernel。

## 11. QSA 与 MTP 的关系

模型的 MTP 模块也使用 QSA。推测解码的多个预测步骤处理相近的上下文，技术报告中的推理方案让后续 MTP 步骤复用第一次选择的 Top-K 位置，省去重复索引。

这里复用的是“候选位置编号”，不是复用正式 Attention 输出。每个预测步骤仍要用自己的 Query 对选中的 K/V 执行核心 Attention。

## 12. 理解检查

1. QSA 的索引器和核心 Attention 分别使用哪套 Q/K？
2. 微块平均后的 Key 能否直接替代正式 Attention 的 Key？
3. `r=4`、`K=2048` 时，最多选择多少个完整微块？
4. 当前 Query 只能看到最后一个微块中的两个 token，这两个 token 会怎样处理？
5. QSA 为什么不能把 KV Cache 直接缩小到 2048 个位置？
6. 报告中的 Attention Kernel 加速能否直接当作端到端服务加速？

<details>
<summary>参考答案</summary>

1. 索引器使用独立的轻量 Q/K；核心 Attention 使用正式 Q/K/V。
2. 不能。压缩 Key 只用于估计微块重要性；选中后仍要读取原始 token 的正式 K/V。
3. 最多 512 个完整微块，末尾未满四个 token 的部分另行保留。
4. 它们属于末尾不完整微块，不参加完整微块 Top-K，但会直接加入候选集合。
5. 每个 Query 可能选择不同位置，完整历史 K/V 仍需保留；2048 是一次计算的读取预算，不是 Cache 的保存上限。
6. 不能。端到端时间还包含其他模型层、通信、调度、采样和服务开销。

</details>

## 13. 资料来源

- [Qwen3.8-Flash-Next 技术报告，revision 6988587](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)
- [Qwen3.8-Flash-Next 配置，revision de4b8e4](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [Transformers 参考实现，revision 281dd53](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

---

[上一课：Gated Residual 与 Decoder Layer 数据流](10-gated-residual.md) · [返回课程路线](../roadmap.md) · [下一课：N-gram Embedding 与推理状态](12-ngram-embedding.md)
