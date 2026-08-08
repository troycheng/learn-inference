# RoPE 教学研究笔记

这份笔记只解决一个问题：怎样向数学基础不强的推理工程师解释 RoPE，同时不牺牲准确性。资料只取自 RoFormer 原论文、Transformer 原论文，以及 Qwen3.5-9B-Base 的官方配置和 Hugging Face Transformers 实现。

## 先说结论

RoPE 的重点不是“把向量转了一下”，而是利用旋转矩阵的组合规律，把两个 token 的绝对位置变成它们在 Attention 分数中的相对距离。

一句适合放进课程正文的话是：

> 每个 token 都按自己的位置转动 Q 和 K。Attention 比较两个 token 时，共同的转动会抵消，只留下二者相差了多少步。因此，模型既能比较内容，也能感知前后顺序和距离。

这里必须加一句边界：Attention 分数仍然取决于 Q、K 所表示的内容，只是其中的**位置影响**只依赖相对位置，并非整个分数只由距离决定。RoFormer 将目标明确写成 `g(x_m, x_n, m-n)`，输入同时包含两个 token 的内容和相对位置。[RoFormer §3.1，公式 11](https://arxiv.org/pdf/2104.09864)

## 1. RoPE 要解决什么问题

Self-Attention 会根据 Q、K 的点积判断两个 token 是否相关。只比较内容还不够：同一个词出现在前一句、上一词和几千个 token 之前，位置关系不同，作用也可能不同。

Transformer 没有循环结构和卷积结构，因此不会天然获得序列顺序。原始 Transformer 论文为此把位置编码加到输入 Embedding 上。[Attention Is All You Need §3.5](https://arxiv.org/pdf/1706.03762)

RoFormer 换了一个切入点：既然 token 之间的联系最终由 `q_m^T k_n` 决定，就直接让这个点积包含位置信息，而且希望位置部分只与 `m-n` 有关。[RoFormer §3.1，公式 11](https://arxiv.org/pdf/2104.09864)

课程里可以先用下面的问题引出 RoPE：

```text
“杯子”在“它”的前 2 个位置
“杯子”在“它”的前 2000 个位置
```

两个位置都可能有相似内容，但 Attention 还需要有机会区分“相隔 2 个 token”和“相隔 2000 个 token”。RoPE 提供的是这种位置关系信号，不是替模型预先规定“近的一定重要”。

## 2. 旋转到底有什么意义

### 2.1 先只看二维

把 Q 的两个数暂时看成平面上的一根箭头。位置为 `m` 的 token，把箭头转过 `mθ`；位置为 `n` 的 token，对 K 转过 `nθ`：

```text
q_m = R(mθ)q
k_n = R(nθ)k
```

旋转矩阵有两个对 RoPE 很关键的性质：

1. 旋转不会改变向量长度。
2. 连续旋转可以合并，反向旋转等于角度取负。

因此，旋转后的 Q、K 做点积时：

$$
q_m^T k_n
=q^T R(m\theta)^T R(n\theta)k
=q^T R((n-m)\theta)k
$$

第一项和最后一项中的 `q`、`k` 仍然携带内容；绝对位置 `m`、`n` 则合并成了相对位置 `n-m`。这就是旋转的用途。它不是为了改变向量的大小，而是把“相差几步”变成 Q、K 之间的相对角度。[RoFormer §3.2.1 至 §3.2.2，公式 12 至 16](https://arxiv.org/pdf/2104.09864)

### 2.2 为什么整体平移后关系不变

假设两个 token 都向后移动 `c` 个位置：

```text
原来：m, n       距离 n-m
移动：m+c, n+c   距离 (n+c)-(m+c)=n-m
```

代入旋转矩阵，新增的共同旋转会抵消。因此，同一对内容整体搬到序列中的另一个位置后，RoPE 在点积里表达的相对位置关系不变。

这句话比“RoPE 编码绝对位置，同时得到相对位置”更容易理解，也更接近公式真正表达的性质。

### 2.3 一个可以手算的小例子

为了只观察位置，临时令 `q=k=[1,0]`，每前进一个位置旋转 30°。

```text
token A 在位置 2：转到 60°
token B 在位置 4：转到 120°
二者夹角：60°
点积：cos(60°)=0.5
```

如果把两者一起移动到位置 12 和 14，角度分别变为 360° 和 420°，夹角仍是 60°，点积仍是 0.5。

这个例子只能说明“共同平移会抵消”。真实 Q、K 不相同，维度也远高于 2，不能据此说 Attention 分数等于某个距离的余弦值。

## 3. 为什么位置影响只依赖 `m-n`

可把推导压缩成三步：

1. 位置 `m` 对 Q 使用 `R_m`，位置 `n` 对 K 使用 `R_n`。
2. 点积里出现 `R_m^T R_n`。
3. 旋转矩阵满足 `R_m^T R_n = R_{n-m}`。

于是：

$$
(R_m q)^T(R_n k)=q^T R_{n-m}k
$$

不同资料可能写成 `m-n` 或 `n-m`，取决于旋转矩阵、行列向量及复数共轭的约定。核心不在正负号，而在于只剩两者之差。RoFormer 的矩阵形式写作 `R_{n-m}`，复数形式中出现 `e^{i(m-n)θ}`。[RoFormer §3.2.1 至 §3.2.2，公式 12、16](https://arxiv.org/pdf/2104.09864)

正文中不宜写“`q_m^T k_n` 只依赖 `m-n`”。更准确的写法是：

> `q_m^T k_n` 仍然依赖两个 token 的内容，但它对位置的依赖只通过相对位置 `m-n` 进入。

## 4. 为什么需要很多种旋转速度

真实的头维度不是 2。标准 RoPE 把维度两两分组，每一组都在自己的二维平面里旋转，并使用不同的 `θ_i`：[RoFormer §3.2.2，公式 14 至 15](https://arxiv.org/pdf/2104.09864)

```text
第 1 个二维组：转得快
第 2 个二维组：稍慢
……
最后一个二维组：转得很慢
```

相对距离为 `Δ` 时，第 `i` 组产生的相对角度是 `Δθ_i`。因此，同一个距离会在多组维度中形成一组快慢不同的相位变化。

“多只钟表”是比较合适的比喻：

- 快钟每走一步就变化明显，容易反映细小的位置差别。
- 慢钟走很多步才明显变化，可以保留更长尺度上的位置信号。
- 只看一只钟会周期性重合；一起看多种速度，得到的是更丰富的距离特征。

原始 Transformer 的正弦位置编码也使用不同频率，波长按几何级数排列；RoFormer 延续了这种频率设置，并把它用在 Q、K 的旋转中。[Attention Is All You Need §3.5](https://arxiv.org/pdf/1706.03762) [RoFormer §3.2.2、§3.3](https://arxiv.org/pdf/2104.09864)

必须保留三个边界：

1. 不能说某个维度“专门负责近距离”或“专门负责远距离”。每组特征的用途由训练共同形成。
2. 单个旋转频率具有周期性，不能单独唯一表示任意距离。
3. 不能说 RoPE 保证 Attention 分数随距离单调下降。RoFormer 证明并展示的是特定频率安排下的整体长期衰减性质，不代表任意一对 Q、K 的分数都会随距离增大而下降。[RoFormer §3.4.3，公式 35 至 37](https://arxiv.org/pdf/2104.09864)

## 5. Qwen3.5-9B-Base 的实际实现

以下数字来自官方 checkpoint 当前 `main` 的固定修订版 [`68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)：

| 配置 | 值 | 含义 |
|---|---:|---|
| `head_dim` | 256 | 每个 Full Attention 头的 Q/K/V 宽度 |
| `num_attention_heads` | 16 | Q 头数量 |
| `num_key_value_heads` | 4 | K/V 头数量，属于 GQA |
| `partial_rotary_factor` | 0.25 | 只旋转每头前 25% 的维度 |
| 实际旋转维度 | 64 | `256 × 0.25` |
| 不旋转维度 | 192 | 剩余部分原样通过 |
| `rope_theta` | 10,000,000 | 计算各组旋转频率使用的基数 |
| `max_position_embeddings` | 262,144 | checkpoint 配置的最大位置数 |
| `rope_type` | `default` | 使用默认 RoPE 频率计算 |
| `mrope_section` | `[11,11,10]` | 32 个旋转维度对在时间、高度、宽度轴上的分配 |

Transformers 代码按 `head_dim × partial_rotary_factor` 算出旋转宽度，因此这里是 64；再在这 64 维上生成 32 个频率。[Qwen3.5 `compute_default_rope_parameters`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L105-L125)

实现会把 Q、K 各自切成两段：前 64 维执行 RoPE，后 192 维直接拼回去。V 不参与 RoPE。[Qwen3.5 `apply_rotary_pos_emb`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L554-L589)

Qwen3.5 的代码布局使用 `rotate_half`，实际配对的是旋转区前半与后半的对应维度，不一定是肉眼看到的相邻两列。它与论文中的二维分组在数学上等价，只是张量布局不同。课程可以画相邻两维来解释原理，但不应声称官方代码按相邻列逐对处理。[Qwen3.5 `rotate_half` 与 `apply_rotary_pos_emb`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L546-L589)

在 Full Attention 层中，Q/K 先做每头 RMSNorm，再应用 RoPE，然后 K 才写入 KV Cache。因此缓存中的 K 已带有位置旋转，V 则没有旋转。[Qwen3.5 Full Attention forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L632-L680)

Qwen3.5-9B 使用 3 层 Gated DeltaNet 加 1 层 Full Attention 的重复结构。RoPE 实际用于 Full Attention 的 Q/K；Gated DeltaNet 分支不调用这段旋转代码。[官方配置中的 `layer_types`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json) [Qwen3.5 Decoder Layer 分支](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L782)

### MRoPE 与纯文本 RoPE 的边界

这个 checkpoint 使用多模态 RoPE。32 个频率槽按 `[11,11,10]` 分配给时间、高度、宽度，并在实现中交错排列。[Qwen3.5 MRoPE 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L127-L164)

对纯文本，三个轴使用相同的顺序位置，所以教学时可以先按一维 RoPE 理解；图片和视频才需要真正区分时间、高度、宽度位置。不要在基础 RoPE 小节提前展开多模态坐标，但应注明 Qwen3.5 的正式实现是 MRoPE。[Qwen3.5 文本与多模态位置编号](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1158-L1197)

## 6. 推荐的讲解顺序

建议把现有 RoPE 小节改成下面六步，而不是从旋转图直接起讲：

1. **先问缺少位置会怎样**：Attention 只看内容时，不能充分区分相同内容出现在何处。
2. **明确目标**：让 Q/K 的匹配分数知道两个 token 相差几步。
3. **只画一个二维特征对**：位置每增加 1，箭头多转 `θ`。
4. **比较两根箭头**：点积看到的是夹角；共同转动会抵消，只剩角度差。
5. **再推广到多频率**：很多快慢不同的“钟表”共同表示不同尺度的位置差。
6. **最后落到 Qwen3.5**：每头 256 维，只旋转前 64 维，只作用于 Full Attention 的 Q/K。

公式应当在读者理解“共同转动会抵消”之后出现。第一遍只保留：

$$
R_m^T R_n=R_{n-m}
$$

不需要一开始就给完整的块对角旋转矩阵。块对角矩阵适合放在“想继续看数学”折叠段或附录。

## 7. 哪些说法容易误导

| 不建议这样写 | 问题 | 更准确的写法 |
|---|---|---|
| RoPE 给每个 token 加上位置 | Qwen 实现不是对 Hidden States 做加法 | RoPE 按位置旋转 Full Attention 的 Q 和 K |
| 旋转后模型就知道 token 在第几个位置 | 过度拟人化，也忽略相对位置目标 | 旋转让 Q/K 点积中的位置影响只与两者距离有关 |
| 点积只由相对位置决定 | 忽略了 Q/K 内容 | 点积仍依赖内容，位置部分通过相对距离进入 |
| 越近的 token 权重一定越大 | RoPE 不提供这种保证 | 频率设计带来整体长期衰减倾向，具体权重仍由内容和训练共同决定 |
| 每两个相邻维度在代码里组成一对 | Qwen 的 `rotate_half` 是半区对应配对 | 数学上可按二维对子理解，代码可能采用等价的不同布局 |
| RoPE 作用于 Q、K、V | 官方实现只旋转 Q、K | V 携带要汇总的信息，不参与匹配位置的旋转 |
| 256 个维度全部旋转 | 与 Qwen3.5-9B 配置不符 | 每头 256 维中只有前 64 维旋转 |
| 每种频率分别负责一种固定距离 | 把训练形成的表示硬解释成人工职责 | 多种频率共同给相对距离提供多尺度相位特征 |

## 8. 建议配图

三张图足以把主线说明白：

1. **共同平移图**：位置 `(2,4)` 与 `(12,14)`，两对箭头的夹角相同。图下注明“绝对角度变了，相对角度没变”。
2. **多频率钟表图**：同一个位置差让快、中、慢三只表转过不同角度。图下注明“每只表会重复，多只表共同提供更丰富的距离特征”。
3. **Qwen3.5 维度条**：一个 256 维头，前 64 维标成“RoPE”，后 192 维标成“原样通过”；Q/K 有两条，V 整条不旋转。

不要只画“一根箭头从位置 1 转到位置 2”。那张图只能说明发生了旋转，解释不了为什么旋转对 Attention 有用。

## 来源

- Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §3.5。
- Jianlin Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), §3.1 至 §3.4。
- Qwen, [Qwen3.5-9B-Base config.json, revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)。
- Hugging Face Transformers, [Qwen3.5 modeling source, revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
