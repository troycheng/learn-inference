# 第 12 课：N-gram Embedding 与推理状态

第 1 课介绍的 Token Embedding 只使用当前 Token ID 查表。同一个 token 无论出现在什么局部词组中，最初取出的都是同一行向量；上下文差异要靠后续 Token Mixer 和 FFN 逐层建立。

Qwen3.8-Flash-Next 增加了一块 N-gram Embedding：模型把当前位置和前面一两个 Token ID 组合成查表地址，直接取出与局部词组有关的向量，再把它写入第 2 个 Decoder Layer 的四条残差支路。

这块参数表有约 51B 参数，却几乎不增加大矩阵乘法。代价从 GPU 计算转向了大容量参数存储、主机内存读取和预取。

## 1. Unigram 与 N-gram 的区别

普通 Token Embedding 可以称为 Unigram Embedding。它的查表键只有一个 Token ID：

```text
当前 Token ID 17
→ Embedding[17]
```

N-gram 表示连续的 `n` 个 token。模型当前看到：

```text
大 / 模型 / 推理
```

处理“推理”这个位置时，可以构造：

```text
2-gram：模型 / 推理
3-gram：大 / 模型 / 推理
```

因此，同一个“推理”token 出现在“大模型推理”和“逻辑推理”中，会形成不同的 N-gram 查表键。

N-gram Embedding 只补充短距离词组信息。它不能代替 Attention 或 Gated DeltaNet，因为它看不到任意远的位置，也不会根据当前语义动态搜索整段上下文。

## 2. Token ID 怎样变成查表地址

如果直接把三个 Token ID 拼成一个整数，可能的组合数量会大到无法建立参数表。模型使用哈希把组合映射到固定大小的表中。

为了手算，假设最近三个 Token ID 为：

```text
前两个位置    2、5
当前位置      7
```

再假设一张只有 11 行的小表，三个位置使用的乘数分别是 7、5、3。2-gram 地址可以写成：

$$
h_2=((7\times3)\oplus(5\times5))\bmod 11
$$

其中 `⊕` 表示按位异或。逐步计算：

```text
7 × 3 = 21 = 二进制 10101
5 × 5 = 25 = 二进制 11001
按位异或       二进制 01100 = 12
12 mod 11 = 1
```

所以 2-gram 查询第 1 行。

3-gram 再加入更早的 Token ID：

$$
h_3=((7\times3)\oplus(5\times5)\oplus(2\times7))\bmod 11
$$

前两项异或得到 12，`2×7=14`，因此：

```text
12 XOR 14 = 2
2 mod 11 = 2
```

3-gram 查询第 2 行。

真实实现使用 64 位整数、训练配置生成的奇数乘数，以及接近 2000 万的不同质数表长。公式规模更大，计算顺序仍是：

```text
各位置 Token ID 乘不同乘数
→ 按位异或
→ 对当前哈希表长度取余
→ 得到一行 Embedding 的编号
```

哈希碰撞不可避免：不同 N-gram 可能落到同一行。模型使用多张不同表和不同表长，让同一对 N-gram 在所有表中同时碰撞的概率降低。

## 3. 为什么一次要查询 16 张表

当前配置使用：

```text
N-gram Size       = 3
每种 N-gram 头数 = 8
```

`N-gram Size=3` 表示同时使用 2-gram 和 3-gram，不包含普通 Token Embedding 已经处理的 1-gram。因此总共有：

```text
2-gram：8 个哈希头
3-gram：8 个哈希头
合计：16 个哈希头
```

每个头从自己的参数表中读取一条 160 维向量：

```text
16 个头 × 每头 160 维 = 2560 维
```

拼接结果正好与模型的 `H=2560` 相同。

每个头约有 2000 万行，因此参数量约为：

$$
16\times20{,}000{,}000\times160
=51.2\ \text{B parameters}
$$

如果使用 BF16 保存，有效载荷约为 102.4 GB，也就是约 95.4 GiB。一次处理一个 token 只读取 16 行，共 2560 个参数；模型不会为这个 token 遍历 51B 参数。

![N-gram 哈希、查表与向量拼接](../assets/12-ngram-lookup.svg?rev=20260828-1)

## 4. 查表结果怎样写进模型

N-gram 查表得到一条 `e:[H]` 向量，但模型没有把它简单加到四条残差支路上。第 10 课已经说明，四条支路可能保存不同的跨层信息，因此每条支路应独立判断这条局部词组信息是否有用。

当前实现使用 PLE（Per-Layer Embedding）完成注入。这里的 PLE 是把额外 Embedding 写进指定 Decoder Layer 的模块，可以按四步理解。

### 4.1 N-gram 向量生成 Key 和 Value

查表结果经过两个 Linear：

```text
e [H]
├→ key_proj   → [4,H]，每条残差支路一条 Key
└→ value_proj → [H]，四条支路共享一条 Value
```

Key 用于判断匹配程度，Value 是准备写入残差支路的内容。

### 4.2 四条残差支路充当 Query

当前四条残差支路经过 RMSNorm 后，各自形成一条 Query：

```text
Residual Queries [4,H]
N-gram Keys       [4,H]
```

第 `i` 条支路的 Query 与第 `i` 条 N-gram Key 点积，得到一个门控分数：

$$
a_i=\frac{q_i\cdot k_i}{\sqrt H}
$$

实现会先对分数做带符号的平方根变换，再执行 Sigmoid：

$$
u_i=\mathrm{sign}(a_i)\sqrt{\max(|a_i|,10^{-6})}
$$

$$
g_i=\mathrm{Sigmoid}(u_i)
$$

平方根会压缩绝对值较大的分数，同时保留正负号。`g_i` 才是 0 到 1 之间的写入比例。

### 4.3 同一条 Value 按不同强度写入四条支路

第 `i` 条支路接收：

$$
z_i=g_i v
$$

四条支路使用同一条 Value，但门控系数不同。某条支路认为这个局部词组与当前内容匹配，就多写一些；匹配较弱，就少写一些。

### 4.4 因果卷积补充附近位置的信息

PLE 还对门控后的结果执行一次带间隔的逐通道因果卷积，再与直接门控结果相加。当前配置的卷积核大小为 4、间隔为 3，所以位置 `t` 的卷积会读取 `t`、`t-3`、`t-6` 和 `t-9`，不会读取未来 token。

完整过程为：

![N-gram Embedding 通过 PLE 写入四条残差支路](../assets/12-ple-injection.svg?rev=20260828-1)

```text
N-gram 查表向量
→ 生成每支路 Key 和共享 Value
→ 与四条残差 Query 分别打分
→ 每条支路独立门控 Value
→ 加入局部因果卷积结果
→ 写入四条残差支路
```

这一步发生在第 2 个 Decoder Layer 的 Token Mixer 之前。之后该层仍按正常顺序执行 GR Read、Gated DeltaNet、GR Write、GR Read、MoE 和 GR Write。

## 5. 为什么只放在第 2 层

技术报告比较了把 N-gram Embedding 放在浅层、中间层、深层以及同时放在多层的效果。多层注入没有稳定优于单层，单独放在较浅位置已经足够。

最终选择第 2 层还有一个推理原因：

```text
开始处理请求
├→ GPU 执行第 1 层
└→ CPU/主机内存同时预取 N-gram 行

进入第 2 层
→ 使用已经取回的 N-gram 向量
```

把查表放在第 1 层会让模型更早等待主机内存；放在第 2 层给预取留下了一层计算时间。

这不表示主机内存读取一定免费。能否隐藏延迟取决于批次中的 token 数、查表地址分布、主机内存带宽、CPU 与 GPU 互连、缓存命中以及预取实现。Transformers 参考实现允许把大表留在 CPU，但真正把读取隐藏在第 1 层计算后面，还需要推理 Runtime 提前、异步地发起查表和传输。

这笔数据量可以直接估算。每个 token 查询 16 行，每行 160 个元素；若表使用 BF16，命中行的逻辑有效载荷为：

$$
16\times160\times2\ \text{Byte}=5120\ \text{Byte}=5\ \text{KiB}
$$

一轮处理 `M` 个 token 时，理想有效载荷是 `5M KiB`。它既可以表示从主机表中读出的命中数据，也可以表示这些 BF16 向量交给 GPU 时的最低数据量。

实际总流量通常与这个数不同。缓存行放大、索引、DMA、预取缓冲、批内重复地址、传输 dtype 和表是否已驻留 GPU，都会改变实测结果。

## 6. Prefill 与 Decode 怎样生成 N-gram

Prefill 已经拿到完整 Prompt，可以一次为所有位置构造 2-gram 和 3-gram 地址。每个位置只使用自己和左侧 token，不读取右侧未来位置。

Decode 每轮只有一个新 token。为了给下一轮构造 2-gram 和 3-gram，请求只需额外保存最近两个 Token ID：

```text
上一轮保存：[token t-1, token t]
新 token：   token t+1

2-gram：[token t, token t+1]
3-gram：[token t-1, token t, token t+1]
```

最近两个 Token ID 的状态大小固定，不会随上下文长度增长。遇到 EOS 或新的独立序列时，实现会重置局部 N-gram 上下文，避免跨样本拼接词组。

## 7. 模型参数与请求状态的完整清单

学完 GR、QSA 和 N-gram 后，可以把 Qwen3.8-Flash-Next 的主要数据分成三类。

### 7.1 模型参数

| 参数 | 是否为所有请求共享 | 每个 token 使用多少 |
| --- | --- | --- |
| Token Embedding | 是 | 当前 Token ID 对应的一行 |
| QSA、GDN、GR 的 Linear 权重 | 是 | 当前层需要的完整权重 |
| 512 个路由专家 | 是 | Router 选中的 10 个，加共享专家 |
| N-gram Embedding | 是 | 16 张表各一行 |

主模型有约 125B 参数，N-gram 表另有约 51B 参数。51B 表可以放在主机内存，不应直接加进 GPU 常驻权重估算。

### 7.2 随上下文增长的请求状态

| 状态 | 所在层 | 是否随 `T` 增长 |
| --- | --- | --- |
| 正式 K/V Cache | 12 个 QSA 层 | 是 |
| 索引器原始 Key | 12 个 QSA 层 | 是 |

QSA 每次只读取最多约 2048 个候选位置，但仍保存完整历史 K/V 和索引器 Key。

### 7.3 固定大小或临时状态

| 状态 | 生命周期 | 是否随 `T` 增长 |
| --- | --- | --- |
| GDN 递归状态和卷积状态 | 跨 Decode 轮保存 | 否 |
| 最近两个 Token ID | 跨 Decode 轮保存 | 否 |
| 四条 GR 残差支路 | 本轮输入位置从浅层传到深层 | 否 |
| MoE 路由结果 | 当前层计算期间 | 否 |

这张表决定了资源估算方法：QSA 状态按上下文长度计算，GDN 和 N-gram 局部历史按请求数计算，GR 激活按当前执行批次和层内数据流计算。

## 8. 从推理系统角度判断三项新结构

三项结构优化的对象不同：

| 结构 | 直接改变什么 | 主要新增成本 | 主要收益方向 |
| --- | --- | --- | --- |
| QSA | 正式 Attention 每次读取的历史位置 | 索引器计算和索引器 Key | 长上下文 Attention 计算与访存 |
| GR | 不同层之间怎样保留和读取信息 | 四支路激活流量、门控 Linear | 模型能力和训练稳定性 |
| N-gram Embedding | 局部词组怎样直接查询参数 | 大容量主机参数表、随机读取 | 低 FLOPs 扩大模型容量 |

因此，面对性能问题时不能笼统地说“新架构更省”：

- 长上下文 Attention 慢，要检查 QSA 索引和稀疏核心 Attention；
- Decode 显存不足，要分别计算 QSA KV、索引器 Key 和 GDN 固定状态；
- 第 2 层出现等待，要检查 N-gram 预取、主机内存和互连；
- 层间 Kernel 变碎或访存增加，要检查 GR Read/Write 是否融合、残差支路使用什么 dtype；
- MoE 吞吐不足，仍要回到专家路由、EP 通信和每专家 token 数。

## 9. 理解检查

1. N-gram Embedding 与普通 Token Embedding 的查表键有什么区别？
2. `ngram_size=3`、每种 N-gram 使用 8 个头时，一次共查询多少行？
3. 51B N-gram 参数是否表示每个 token 要执行 51B 参数的矩阵乘法？
4. PLE 为什么要让四条残差支路分别计算门控？
5. 为什么 N-gram Embedding 放在第 2 层有利于主机内存预取？
6. GR 残差支路、GDN 状态和 QSA KV Cache 的生命周期分别是什么？

<details>
<summary>参考答案</summary>

1. 普通 Token Embedding 只使用当前 Token ID；N-gram Embedding 使用当前和此前一两个 Token ID 的哈希组合。
2. 2-gram 查询 8 行，3-gram 查询 8 行，共 16 行。
3. 不是。每个 token 只从 16 张表各读取一行；大量参数提供的是可查询容量，不是逐 token 稠密计算量。
4. 四条支路保存的跨层信息不同，对同一条局部词组信息的需求也可能不同。
5. GPU 执行第 1 层时可以并行取回第 2 层需要的行，减少进入 PLE 时的等待。
6. GR 支路只在当前前向中沿层深传递；GDN 状态跨 Decode 轮保存但大小固定；QSA KV Cache 跨 Decode 轮保存并随上下文增长。

</details>

## 10. 资料来源

- [Qwen3.8-Flash-Next 技术报告，revision 6988587](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)
- [Qwen3.8-Flash-Next 配置，revision de4b8e4](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [Transformers 参考实现，revision 281dd53](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

---

[上一课：Qwen Sparse Attention 的计算过程](11-qwen-sparse-attention.md) · [返回课程路线](../roadmap.md)

学完本课后，可以重新阅读[第 8 课：模型配置与资源估算](08-config-and-sizing.md)和[第 9 课：推理优化的分析与评估](09-optimization-judgment.md)，把其中的状态公式和优化分类应用到这套新结构。
