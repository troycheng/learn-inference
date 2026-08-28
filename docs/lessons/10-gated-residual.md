# 第 10 课：Gated Residual 与 Decoder Layer 数据流

前九课使用 Qwen3.5 建立了 Decoder 的基本结构：每层先用 Token Mixer 交换不同 token 的信息，再用 FFN 或 MoE 加工单个 token 的特征；两个子层各自通过残差连接把增量加回输入。

Qwen3.8-Flash-Next 保留这两个子层，但改变了它们共用的残差通道。模型不再让所有层读写同一条隐藏状态，而是并行维护四条残差支路，并由门控决定每个子层从四条支路中读取什么、又把输出写回多少。这套结构称为 Gated Residual，简称 GR。

读这一课时，先盯住四个问题：

1. 四条残差支路保存什么？
2. GR Read 怎样把四条支路合成子层输入？
3. GR Write 怎样把子层输出写回四条支路？
4. GR 改变了哪些推理成本，又没有改变哪些状态？

## 1. 先看完整的模型骨架

Qwen3.8-Flash-Next 的文本模型有 48 个 Decoder Layer，隐藏宽度为 `H=2560`。Token Mixer 仍按四层一组排列：

```text
Gated DeltaNet → Gated DeltaNet → Gated DeltaNet → QSA
```

这组结构重复 12 次，因此共有 36 个 Gated DeltaNet 层和 12 个 QSA 层。每层的第二个子层都是 MoE。

GR 位于每个 Token Mixer 和 MoE 的前后：

![Qwen3.8-Flash-Next Decoder 的完整数据流](../assets/10-qwen38-decoder-flow.svg?rev=20260828-1)

沿着一层的数据流，可以写成：

```text
四条残差支路 R
→ GR Read
→ 得到 Token Mixer 输入 x
→ Gated DeltaNet 或 QSA
→ 得到增量 y
→ GR Write，将 y 写回四条支路

→ GR Read
→ 得到 MoE 输入 x
→ MoE
→ 得到增量 y
→ GR Write，将 y 写回四条支路
```

一个 Decoder Layer 内有两套独立的 GR：一套服务于 Token Mixer，另一套服务于 MoE。两套 GR 的参数不同。

第 11 课会打开 QSA，第 12 课再说明第 2 层中的 N-gram Embedding。现在先把它们看作能产生一条 `H` 维增量的子层。

## 2. 普通残差连接只有一条通道

普通预归一化 Decoder 子层的计算是：

$$
y=F(\mathrm{RMSNorm}(x))
$$

$$
x'=x+y
$$

`x` 同时承担两种职责：

- 它是当前子层的输入；
- 它也是此前所有层输出的累计结果。

如果模型有很多层，较早层写入的信息会与后来各层的输出不断相加。后面的子层只能读取这一个总和，不能分别决定“多读一些早期信息、少读一些近期信息”。

GR 没有删除残差相加，而是把一条累计通道扩展成四条：

```text
普通残差状态：x  [H]

GR 残差状态：R1 [H]
             R2 [H]
             R3 [H]
             R4 [H]
```

模型输入的 Token Embedding 最初会复制四份，所以四条支路一开始相同。经过各层不同强度的写入后，它们逐渐保存不同的跨层信息。

## 3. GR Read：从四条支路读取子层输入

GR Read 分三步计算。

### 3.1 分别归一化四条支路

每条支路独立执行 RMSNorm：

$$
\widehat R_i=\mathrm{RMSNorm}(R_i),\qquad i=1,2,3,4
$$

如果输入包含 Batch 和序列轴，shape 为：

```text
R                    [B,T,4,H]
每条支路 RMSNorm 后  [B,T,4,H]
```

归一化只沿最后的 `H` 轴计算。四条支路不会混在一起求均方根。

### 3.2 为每条支路的每个特征计算门控系数

四条归一化支路先拼成 `4H` 维向量，再经过两次 Linear：

```text
[B,T,4H]
→ 降到 320 维
→ 除以 4
→ SiLU
→ 升回 4H 维
→ Sigmoid
→ [B,T,4,H]
```

最终得到门控张量 `G:[B,T,4,H]`。`G` 中每个数位于 0 和 1 之间：

```text
G[b,t,i,h]
= 第 b 个请求、第 t 个 token
  从第 i 条残差支路读取第 h 个特征的比例
```

这里的门控不是四个固定权重。不同请求、不同 token、不同特征都可以得到不同的值。

### 3.3 加权并平均四条支路

GR Read 的输出为：

$$
x=\frac{1}{4}\sum_{i=1}^{4}G_i\odot\widehat R_i
$$

逐元素乘法 `G_i\odot\widehat R_i` 决定从第 `i` 条支路的每个特征读取多少，四条结果相加后除以 4，得到当前子层的 `H` 维输入。

Shape 由四条支路回到普通子层接口：

```text
归一化支路  [B,T,4,H]
读取门控    [B,T,4,H]
GR Read     [B,T,H]
```

因此，GR 内部虽然保存四条支路，Gated DeltaNet、QSA 和 MoE 接收到的仍是 `[B,T,H]`。

## 4. 用两条支路手算一次 GR Read

真实模型有四条支路、每条 2560 维。为了看清逐元素门控，先缩小为两条支路、每条 2 维，并假设下面的数已经经过 RMSNorm：

```text
R̂1 = [2, 4]
R̂2 = [6, 2]

G1 = [0.8, 0.2]
G2 = [0.3, 0.9]
```

第一条支路的读取结果：

$$
G_1\odot \widehat R_1=[0.8\times2,\ 0.2\times4]=[1.6,0.8]
$$

第二条支路的读取结果：

$$
G_2\odot \widehat R_2=[0.3\times6,\ 0.9\times2]=[1.8,1.8]
$$

两条支路相加并平均：

$$
x=\frac{[1.6,0.8]+[1.8,1.8]}{2}=[1.7,1.3]
$$

从结果可以看出，第一个特征更多地读取 `R1`，第二个特征更多地读取 `R2`。如果每条支路只有一个标量权重，就不能对两个特征作出不同选择。

## 5. GR Write：把子层输出写回四条支路

子层读取 `x:[B,T,H]` 后产生增量 `y:[B,T,H]`。GR Write 再为每条支路计算一个写入系数：

$$
s=2\,\mathrm{Sigmoid}\left(\frac{1}{4}W\widehat R\right)
$$

真实模型得到四个系数：

```text
s [B,T,4]
```

每个系数位于 0 和 2 之间。第 `i` 条支路的更新为：

$$
R_i'=R_i+s_i y
$$

GR Read 的门控细到每个特征，GR Write 则只为每条支路产生一个标量。报告中的消融实验表明，细化读取比细化写入更有价值。

再单独看一次写入。这里取两条原始残差支路 `R1=[2,4]`、`R2=[6,2]`，并假设子层输出和写入系数为：

```text
y = [0.6, -1.0]
s1 = 1.5
s2 = 0.4
```

第一条支路更新为：

$$
R_1'=[2,4]+1.5[0.6,-1]=[2.9,2.5]
$$

第二条支路更新为：

$$
R_2'=[6,2]+0.4[0.6,-1]=[6.24,1.6]
$$

同一个子层输出写入了两条支路，但强度不同。后续子层再次执行 GR Read 时，可以按当前 token 和特征重新选择这些信息。

![GR Read 与 GR Write 的计算关系](../assets/10-gr-read-write.svg?rev=20260828-1)

## 6. 四条支路怎样贯穿 48 层

四条支路不是每层重新创建的临时变量。它们从 Token Embedding 开始，依次经过全部 48 层：

```text
Token Embedding
→ 复制成 R1、R2、R3、R4
→ Layer 0 Token Mixer GR Read / Write
→ Layer 0 MoE GR Read / Write
→ Layer 1 Token Mixer GR Read / Write
→ Layer 1 MoE GR Read / Write
→ ...
→ Layer 47 MoE GR Read / Write
→ 最终 GR Read
→ LM Head
```

最终 GR Read 只负责把四条支路混合回 `[B,T,H]`，不再执行写入。LM Head 仍然接收普通的 `H` 维隐藏状态。

研究报告对四条支路的分析显示，训练后通常有一条支路更偏向保存跨越许多层的信息，其余支路更偏向相邻层或短距离信息。这不是人为指定的固定职责，而是模型训练得到的读写模式。

## 7. 回到真实配置和实现 shape

Qwen3.8-Flash-Next 的关键配置为：

```text
Hidden Size H             = 2560
Residual Branches         = 4
Flattened Residual Width  = 4 × 2560 = 10240
GR Low-rank Width         = 320
Decoder Layers            = 48
```

当前 Transformers 实现为了方便 Linear 计算，把四条支路展平在最后一维：

```text
模型层之间保存       [B,T,10240]
按支路解释时         [B,T,4,2560]
GR Read 输出         [B,T,2560]
子层输出             [B,T,2560]
GR Write 后          [B,T,10240]
```

`10240` 是四条残差支路拼接后的宽度，不是 QSA、Gated DeltaNet 或 MoE 的输入宽度。把这两个概念混在一起，会错误地把所有子层计算量估大四倍。

## 8. GR 对推理系统的影响

GR 最直接增加的是层间激活的读写量。普通残差流每个 token 传递 `H` 个元素，GR 要传递 `4H` 个元素；每个 Token Mixer 和 MoE 还要额外执行门控所需的低秩 Linear。

报告采用两项措施降低这部分成本：

1. 用 FP8 保存四条残差支路，使有效载荷相对 BF16 减半；
2. 把每支路 RMSNorm、门控和混合融合为 GR Read Kernel，把写入融合为 GR Write Kernel。

GR 状态也容易与 KV Cache 或 Gated DeltaNet 状态混淆。它们的生命周期不同：

| 状态 | 沿什么方向传递 | 是否跨 Decode 轮保存 |
| --- | --- | --- |
| GR 四条残差支路 | 从浅层传到深层 | 否；本轮输入位置完成 48 层后即可释放 |
| QSA 层的 KV Cache | 从历史 token 传给未来 token | 是 |
| Gated DeltaNet 的递归状态 | 从历史 token 传给未来 token | 是 |

GR 解决跨层信息流，KV Cache 和递归状态解决跨 token 历史。虽然三者都可以叫“状态”，容量估算时不能放进同一个公式。

## 9. 理解检查

1. GR 为什么需要先把四条支路读成 `[B,T,H]`，再交给 Token Mixer 或 MoE？
2. GR Read 和 GR Write 的门控粒度有什么不同？
3. `H=2560`、四条残差支路时，层间张量最后一维是多少？MoE 接收到的最后一维又是多少？
4. GR 四条支路是否像 KV Cache 一样随上下文长度持续保存在请求状态中？
5. 如果某条支路的写入系数接近 0，这次子层输出会怎样影响它？

<details>
<summary>参考答案</summary>

1. 现有 Token Mixer 和 MoE 的接口仍是 `H` 维；GR Read 负责从四条跨层通道中选择并混合出本次子层输入。
2. GR Read 为每条支路的每个特征产生一个系数，shape 为 `[B,T,4,H]`；GR Write 为每条支路产生一个标量，shape 为 `[B,T,4]`。
3. 层间张量最后一维是 `4×2560=10240`；MoE 接收的最后一维仍是 2560。
4. 不是。GR 支路只沿模型深度传递当前计算的隐藏状态；KV Cache 和 Gated DeltaNet 状态才会跨 Decode 轮保存。
5. 该支路几乎不接收这次子层输出，但原来保存在支路中的内容仍然保留。

</details>

## 10. 资料来源

- [Qwen3.8-Flash-Next 技术报告，revision 6988587](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)
- [Qwen3.8-Flash-Next 配置，revision de4b8e4](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [Transformers 参考实现，revision 281dd53](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

---

[上一课：推理优化的分析与评估](09-optimization-judgment.md) · [返回课程路线](../roadmap.md) · [下一课：Qwen Sparse Attention 的计算过程](11-qwen-sparse-attention.md)
