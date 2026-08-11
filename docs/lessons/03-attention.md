# 第 3 课：Attention 的计算原理

上一课把 Token Mixer 当成一个整体。本课只研究其中的 Full Attention，并回答一个具体问题：模型处理“它”时，怎样找到前文中的“杯子”，又怎样把“杯子”的信息写进“它”的隐藏状态？

后面的图和手算始终使用同一个简化例子：

```text
小明把杯子放在桌上，然后拿起它。

位置 1   位置 2   位置 3
小明     杯子     它
```

真实 Tokenizer 可能把这句话切成更多 token。这里省略其他位置，只保留三个标签，方便跟踪计算。所有 Q、K、V、分数和遮罩矩阵都沿用这组位置编号。

Full Attention 中的 `Full` 指读取范围。一个位置可以直接与所有允许读取的位置计算关系。Decoder 受因果约束，只能读取当前位置和左侧已经出现的 token。

## 1. Attention 如何读取上下文

进入 Attention 前，每个位置都有一条隐藏状态。位置 3 的隐藏状态记作 `x3`。模型要更新它，先让它与位置 1、2、3 分别比较，再决定从每个位置取回多少信息。

```text
比较三个位置
→ 得到三个权重
→ 按权重汇总三个位置携带的信息
→ 得到位置 3 的新向量
```

![位置 3 怎样读取三个可见位置](../assets/03-qkv-intuition.svg?rev=20260810-1)

Attention 不会硬选一个位置。三个位置通常都会参与输出，只是贡献大小不同。

### 1.1 Q、K、V 分别负责什么

Attention 把“用什么匹配”和“匹配后取回什么”拆成了两件事：

| 符号 | 中文名称 | 计算中的作用 | 检索系统中的近似类比 |
| --- | --- | --- | --- |
| `Q` | 查询向量（Query） | 当前处理的位置用什么特征发起匹配 | 查询条件 |
| `K` | 键向量（Key） | 候选位置用什么特征接受匹配 | 索引特征 |
| `V` | 值向量（Value） | 候选位置最终提供什么信息 | 返回内容 |

检索类比只能帮助记住分工。K 不是唯一主键，Attention 也不是精确查表。Q 与 K 的点积产生连续分数，多个位置的 V 通常会一起进入结果。

K 和 V 分开后，模型可以用一组特征判断两个位置是否相关，再从相关位置取回另一组特征。例如，匹配时可能用到词性、位置和上下文关系；写入输出的则可能是对象、属性或动作等信息。这些含义由训练形成，并没有固定在某个向量坐标上。

### 1.2 隐藏状态怎样变成 Q、K、V

设 RMSNorm 后的隐藏状态为 `X:[B,T,H]`。三组 Linear 分别计算：

$$
Q=XW_Q^T,\qquad K=XW_K^T,\qquad V=XW_V^T
$$

`W_Q`、`W_K`、`W_V` 是训练得到的模型参数。Q、K、V 是这一层为当前请求算出的中间数据。同一条隐藏状态经过三组不同权重重新组合，得到三种用途不同的表示。

这里的 Q、K、V 都来自同一组 `X`，因此叫自注意力（Self-Attention）。如果 Q 和 K/V 来自不同输入，才是交叉注意力。本课不展开交叉注意力。

## 2. 手算位置 3 的 Attention 输出

先只看一个 Attention 头，把每条向量缩小到两维，即 `D=2`。每个位置都有自己的 Q、K、V；这次只计算位置 3，所以表中只列出 `Q3`。

| 位置 | token | 本次使用的 Q | K | V |
| ---: | --- | --- | --- | --- |
| 1 | 小明 |  | `[1,0]` | `[2,0]` |
| 2 | 杯子 |  | `[1,1]` | `[0,2]` |
| 3 | 它 | `[1,1]` | `[0,1]` | `[1,1]` |

这些数字只为手算服务，不是三个词的固定编码，也不表示某一维永远对应某种语义。

### 2.1 Q/K 点积得到匹配分数

`Q3` 分别与三条 K 做点积：

$$
\begin{aligned}
Q_3\cdot K_1&=[1,1]\cdot[1,0]=1\\
Q_3\cdot K_2&=[1,1]\cdot[1,1]=2\\
Q_3\cdot K_3&=[1,1]\cdot[0,1]=1
\end{aligned}
$$

第二项最高。在这组数字中，位置 2 的“杯子”与位置 3 的“它”匹配最强。

![Q3 分别与三条 K 计算分数](../assets/03-one-query-scores.svg?rev=20260810-1)

点积会把 `D` 项乘积加起来。头维度增大后，点积的绝对值也容易增大。Attention 把分数除以 `√D`，避免 Softmax 过早集中到少数位置：

$$
\frac{[1,2,1]}{\sqrt{2}}\approx[0.707,1.414,0.707]
$$

Qwen3.5-9B 的 Full Attention 头维度是 `D=256`，缩放系数为 `1/√256=1/16`。缩放不会改变三项分数的大小顺序。

### 2.2 Softmax 将分数换成权重

位置 3 没有需要遮住的未来位置。三个缩放分数经过 Softmax 后得到：

![一行分数经过 Softmax 变成权重](../assets/03-score-to-weight.svg?rev=20260811-1)

未截断的权重是：

```text
位置 1“小明”：0.248255...
位置 2“杯子”：0.503490...
位置 3“它”：  0.248255...
```

Softmax 保留分数的大小关系，把每项变成非负权重，并让整行权重之和等于 1。图中只保留三位小数，所以三项显示值相加为 `0.999`；完整精度下仍然等于 1。

Attention Score 和 Attention Weight 不是同一组数：

```text
Score：Q/K 点积并缩放后的分数
Weight：分数经过遮罩和 Softmax 后得到的权重
```

这里的 Weight 只控制当前层、当前头怎样汇总 V，不是模型生成某个 token 的概率。

### 2.3 权重汇总 V

Q 和 K 的工作到权重为止。真正写入输出的是 V：

![三个权重怎样汇总三条 V](../assets/03-weighted-values.svg?rev=20260810-1)

用完整精度计算：

$$
0.248255[2,0]+0.503490[0,2]+0.248255[1,1]
\approx[0.745,1.255]
$$

`[0.745,1.255]` 是位置 3 在这个头中取回的信息。它仍是一条隐藏特征向量，不是 Token ID，也不是词表概率。

至此，一个位置的计算已经完整走通：

```text
Q/K 算出权重
权重汇总 V
得到这个头的输出向量
```

## 3. 从一个位置推广到整段序列

### 3.1 一次计算所有位置

前面只计算了 `Q3`。加入位置 1 和位置 2 的查询后，三组向量可以写成矩阵：

$$
Q=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},\quad
K=
\begin{bmatrix}
1&0\\
1&1\\
0&1
\end{bmatrix},\quad
V=
\begin{bmatrix}
2&0\\
0&2\\
1&1
\end{bmatrix}
$$

Q 的第 `i` 行与 K 的第 `j` 行点积，得到分数矩阵的第 `i` 行第 `j` 列：

$$
\frac{QK^T}{\sqrt{2}}=
\begin{bmatrix}
0.707&0.707&0\\
0&0.707&0.707\\
0.707&1.414&0.707
\end{bmatrix}
$$

行和列的含义不能弄反：

```text
第 i 行：位置 i 正在发起查询
第 j 列：位置 j 正在被读取
```

第 3 行就是上一节的三次点积。矩阵乘法没有改变算法，只是把多个查询位置放在一次张量计算中。

### 3.2 因果遮罩防止读取未来 token

位置 2 可以读取位置 1 和自己，不能读取位置 3。训练时，位置 2 的最终表示用来预测位置 3 的 token；如果允许它读取第 3 列，模型就提前看到了待预测内容。

![因果遮罩规定每一行可以读取哪些列](../assets/03-causal-mask.svg)

因果遮罩（Causal Mask）记作 `M`：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty\\
0 & 0 & -\infty\\
0 & 0 & 0
\end{bmatrix}
$$

遮罩不是 Q、K、V，也不是从 Q/K 计算出的分数。它在 Softmax 前与分数矩阵逐元素相加：

$$
S_{masked}=\frac{QK^T}{\sqrt{D}}+M
$$

未来位置的分数变成负无穷，Softmax 后的权重便是 0。这个例子得到：

$$
S_{masked}=
\begin{bmatrix}
0.707&-\infty&-\infty\\
0&0.707&-\infty\\
0.707&1.414&0.707
\end{bmatrix}
$$

每一行单独做 Softmax：

$$
A\approx
\begin{bmatrix}
1&0&0\\
0.330&0.670&0\\
0.248&0.503&0.248
\end{bmatrix}
$$

最后计算 `O=AV`：

$$
O\approx
\begin{bmatrix}
2.000&0\\
0.660&1.340\\
0.745&1.255
\end{bmatrix}
$$

第 3 行仍是上一节算出的结果。这里的 `A` 只显示三位小数，`O` 使用未截断权重计算。

因果遮罩限制每一行可以读取哪些列，不要求 Prompt 中的已知位置依次执行。Prefill 时，多个已知位置的 Q/K/V 可以放在较大的张量中计算，只是各行允许读取的范围不同。

### 3.3 两条公式概括主计算

一个 Attention 头可以写成两条公式：

$$
A=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{D}}+M\right),\qquad O=AV
$$

![缩放点积 Attention 的主计算](../assets/03-attention-flow.svg?rev=20260810-1)

| 公式部分 | 对应计算 |
| --- | --- |
| `QK^T` | 每条 Q 与每条 K 点积，得到分数矩阵 |
| 除以 `√D` | 控制分数尺度 |
| `+M` | 把未来位置的分数改成负无穷 |
| `Softmax` | 每一行分数变成权重 |
| `AV` | 按权重汇总每个位置的 V |

仓库中的 [Attention 手算程序](../../examples/attention_walkthrough.py) 使用同一组 Q、K、V，可以逐行核对缩放分数、遮罩、Softmax 权重和输出。

## 4. 多头 Attention

### 4.1 同一段序列做多组匹配

前面的手算只属于一个头。多头注意力（Multi-Head Attention，MHA）为每个头使用不同的投影权重，让同一段序列在多组表示空间中重复完成 Attention。

加入头以后，需要同时区分 token 位置和头编号：

| 下标 | 含义 | 例子 |
| --- | --- | --- |
| `i、j` | token 位置 | `j=2` 表示“杯子”所在的位置 |
| `h` | 查询头编号 | `h=1` 表示第 1 个查询头 |

用完整下标重写上一节：第 1 个头在位置 3 的查询是 `Q(h=1,i=3)`，它依次与同一头中的 `K(h=1,j=1...3)` 比较，最后得到：

$$
O(h=1,i=3)=[0.745,1.255]
$$

第 2 个头会使用另一组投影权重，再走一遍相同计算。各头读取同一段 token 序列，但 Q、K、V 数值不同，因此匹配分数和输出也可以不同。

![从前面的单头手算扩展到多个头](../assets/03-multihead-bridge.svg?rev=20260811-1)

模型没有为每个头预先指定“负责指代”或“负责语法”之类的固定职责。训练会让不同头形成不同的匹配方式，但并非每个头都能被清楚命名。

### 4.2 拼接各头结果

每个头都会为整段序列输出 `[B,T,D]`。`Nq` 个头放在一起得到 `[B,Nq,T,D]`，再调整轴顺序，把头编号和头内特征拼到同一条特征轴：

```text
[B,Nq,T,D]
→ [B,T,Nq,D]
→ Concat
→ [B,T,Nq×D]
```

Concat 只是把各头的结果首尾排列。随后执行的输出投影 `o_proj` 才会重新组合不同头的特征。

Qwen3.5-9B 中：

$$
Nq\times D=16\times256=4096=H
$$

因此，16 个头拼接后正好回到 Decoder Layer 约定的隐藏宽度 `H=4096`。

## 5. GQA 如何减少 K/V 头数

普通 MHA 中，每个查询头都有自己的 K/V 头。4 个查询头对应 4 组 K/V。分组查询注意力（Grouped-Query Attention，GQA）保留查询头数量，但让多个查询头共用一组 K/V。

一个 K/V 头仍然保存整段序列，不是只保存一个 token：

```text
K/V 头 g1：K(g1,j=1...T)，V(g1,j=1...T)
```

### 5.1 共享 K/V 后为什么还能得到不同结果

仍用前面的三位置手算，并让两个查询头共用同一组 K 和 V：

```text
查询头 h1：Q=[1,1] → 权重约 [0.248,0.503,0.248] → 输出 [0.745,1.255]
查询头 h2：Q=[0,1] → 权重约 [0.198,0.401,0.401] → 输出 [0.797,1.203]
```

两条查询头读取同一组 `K1...K3` 和 `V1...V3`，但它们的 Q 不同。Q/K 点积、Softmax 权重和最终输出自然也不同。GQA 共享的是被读取的 K/V，不是查询本身，更不是 Attention 结果。

![MHA 与 GQA 的 K/V 共享方式](../assets/03-gqa.svg?rev=20260811-1)

图中 `h` 表示查询头编号，`g` 表示 K/V 头编号，`j` 表示 token 位置。把头编号和位置编号分开，便不会把“第 1 个 K/V 头”误解为“位置 1 的 K/V”。

### 5.2 Qwen3.5-9B 的分组方式

Qwen3.5-9B 使用：

```text
Nq  = 16 个查询头
Nkv =  4 个 K/V 头
D   = 256
```

每 4 个查询头共用一个 K/V 头：

```text
h1  ～ h4   共用 g1
h5  ～ h8   共用 g2
h9  ～ h12  共用 g3
h13 ～ h16  共用 g4
```

三组投影的总宽度是：

```text
Q：16 × 256 = 4096
K： 4 × 256 = 1024
V： 4 × 256 = 1024
```

与 16 组 K/V 的普通 MHA 相比，K/V 投影宽度和 KV Cache 元素数降为 1/4。Q、输出投影和 FFN 并没有缩小，所以整模型显存和总计算量不会一起降为 1/4。

优化实现可以让四个查询头直接读取共享 K/V。朴素实现也可能在逻辑上把 K/V 展开到 16 头。两种实现的 Attention 结果相同，内存访问和临时数据量不同。

MLA 通常指 Multi-head Latent Attention。它会进一步把 K/V 压缩到潜在空间，结构和缓存方式与 GQA 不同。Qwen3.5-9B 的 Full Attention 使用 GQA，本课不把 MLA 混入这条计算链。

## 6. RoPE 如何把相对位置加入分数

到目前为止，Q/K 分数只来自向量内容。下面两句话包含相同的词，但词序改变后，谁批评谁也变了：

```text
小明批评小李
小李批评小明
```

同一个词出现在当前位置前 2 个 token 或前 2000 个 token，位置关系也不同。Attention 需要知道两个 token 谁在前、谁在后、相隔多远。

RoPE 在 Q/K 点积前旋转部分向量，让匹配分数同时受到内容和相对位置影响。旋转是计算手段，相对位置才是目的。

### 6.1 用二维箭头理解旋转

暂时从 Q 中取两个数，把它们看成平面上的一根箭头。假设 token 每向后移动一个位置，箭头就多转一个固定角度 `θ`：

```text
位置 0：转 0θ
位置 1：转 1θ
位置 2：转 2θ
...
位置 m：转 mθ
```

K 中对应的两个数也按自己的位置旋转。位置 3 的 Q 与位置 2 的 K 分别旋转 `3θ` 和 `2θ`，两根箭头相差 `θ`。把它们一起移到位置 13 和位置 12，两根箭头都多转了 `10θ`，夹角仍是 `θ`。

![RoPE 让点积感知相对位置](../assets/03-rope.svg?rev=20260811-2)

点积本来就与两根箭头的夹角有关。RoPE 让 Q/K 点积中的位置影响只取决于两者的相对距离。

取一个可以手算的特例。令原始 `q=k=[1,0]`，每前进一个位置旋转 30 度。两个 token 位于位置 2 和位置 3 时，箭头分别转到 60 度和 90 度，夹角为 30 度，点积是 `cos(30°)≈0.866`。把它们同时移到位置 12 和位置 13，点积仍然不变。

这个特例只说明共同移动不会改变相对位置。真实 Q、K 的内容不同，维度也更高。Attention 分数不是一张只按距离查询的固定表。

### 6.2 相对位置怎样进入点积

把位置 `m` 对应的旋转写成 `R_m`，位置 `n` 对应的旋转写成 `R_n`。这一小节把 `q`、`k` 写成列向量，便于表示旋转矩阵。旋转后的点积是：

$$
(R_m q)^T(R_n k)=q^T R_m^T R_n k=q^T R_{n-m}k
$$

公式最后只剩 `n-m`。两个绝对位置在点积中变成了相对位置。

这条公式不能读成“Attention 分数只由距离决定”。`q` 和 `k` 仍然来自 token 内容；RoPE 只是让位置影响通过 `n-m` 进入点积。内容和相对位置共同决定分数。

### 6.3 为什么使用多种旋转速度

真实 Q/K 不止二维。RoPE 把参与旋转的维度分成许多二维组，每组采用不同的旋转速度。可以把它们想成一排快慢不同的钟表：

- 转得快的组对较小的位置变化更敏感；
- 转得慢的组经过较长距离后仍在缓慢变化；
- 单独一组会周期性转回相同角度，多种速度组合后能提供更丰富的位置差特征。

![RoPE 使用多种旋转速度](../assets/03-rope-frequencies.svg)

这些维度没有被人工指定为“近距离通道”或“远距离通道”。模型会在训练中学习怎样使用不同频率。RoPE 也不保证附近 token 的权重一定更高，最终权重仍取决于 Q/K 内容、相对位置和训练结果。

### 6.4 RoPE 为什么只作用于 Q 和 K

RoPE 要改变的是位置之间的匹配分数，而分数来自 Q 与 K 的点积，所以旋转放在 Q/K 上。V 负责携带匹配后要汇总的信息，不参与打分。

旋转不会增加 token 数，也不会改变 Q/K 的 shape，只会改变其中的数值。

### 6.5 Qwen3.5-9B 只旋转部分维度

Qwen3.5-9B 的每个 Full Attention 头有 `256` 维。配置项 `text_config.rope_parameters.partial_rotary_factor=0.25` 表示只有前 `64` 维参与 RoPE，后 `192` 维原样通过：

```text
Q：前 64 维旋转，后 192 维原样通过
K：前 64 维旋转，后 192 维原样通过
V：256 维都不旋转
```

K 经过 RoPE 后才写入 KV Cache，所以缓存复用必须保持位置编号和 RoPE 规则一致。Qwen3.5 使用 MRoPE；纯文本可以先按一维位置理解，图片和视频的时间、高度、宽度坐标放到第 7 课。

## 7. Qwen3.5-9B 的完整 Attention 前向

前面的章节依次解释了单头计算、多头拼接、GQA 和 RoPE。把它们放回 Qwen3.5-9B 的一个 Full Attention 子层，完整顺序如下：

![Qwen3.5-9B 多头 Attention 的完整流程](../assets/03-qwen-attention-full-flow.svg?rev=20260811-3)

第 `h` 个查询头先找到对应的 K/V 头 `g`，再完成：

$$
A_h=\mathrm{softmax}\left(\frac{Q_hK_g^T}{\sqrt D}+M\right),\qquad O_h=A_hV_g
$$

16 个 `O_h` 先组成 `[B,16,T,256]`，再调整轴顺序并拼接：

```text
[B,16,T,256]
→ [B,T,16,256]
→ [B,T,4096]
```

Qwen3.5 在拼接结果进入 `o_proj` 前还会乘一组输出门控。`o_proj` 重新组合各头特征，最后与进入 Attention 前保存的 X 逐元素相加。

### 7.1 各阶段的实际 shape

| 阶段 | shape | 含义 |
| --- | --- | --- |
| 输入 X | `[B,T,4096]` | Attention 子层收到的隐藏状态 |
| 查询 Q | `[B,16,T,256]` | 16 个查询头 |
| 键 K | `[B,4,T,256]` | 4 个 K 头 |
| 值 V | `[B,4,T,256]` | 4 个 V 头 |
| 分数矩阵 | 逻辑上为 `[B,16,T,T]` | 每个查询位置给所有可见位置打分 |
| 各头输出 | `[B,16,T,256]` | 权重汇总 V 后的结果 |
| Concat | `[B,T,4096]` | `16×256=4096` |
| `o_proj` 输出 | `[B,T,4096]` | 与残差分支 shape 相同 |

“逻辑上为 `[B,16,T,T]`”描述模型语义。FlashAttention 一类实现会分块计算 Softmax 和加权结果，不必把完整分数矩阵长期写入显存。

### 7.2 Q/K 归一化、部分 RoPE 与输出门控

标准 Attention 公式之外，Qwen3.5-9B 还加入了三项处理：

1. Q 和 K 按每个头的 256 维做 RMSNorm，再进入 RoPE 和点积。
2. 每个头只有前 64 维参与 RoPE。
3. 各头拼接后乘 `Sigmoid(Gate)`，再进入 `o_proj`。

Transformers 实现中的 `q_proj` 一次产生 `[B,T,8192]`，随后拆成两半：

```text
前 4096 个数 → 16 个查询头的 Q
后 4096 个数 → Attention 输出门控
```

这组门控与 SwiGLU 的门控属于不同子层：

```text
SwiGLU 门控：调节 FFN 的中间特征
Attention 输出门控：调节各头拼接后的特征
```

## 8. 推理阶段的数据与开销

推理系统需要区分模型参数、当前计算的中间数据和跨 Decode 轮次保留的状态：

| 对象 | 属于什么 | 生命周期 |
| --- | --- | --- |
| `W_Q`、`W_K`、`W_V`、`W_O` | 模型参数 | 模型加载后由不同请求复用 |
| Q/K/V、分数和权重 | 当前请求的中间数据 | 本轮前向计算产生 |
| 历史 K/V | 当前请求的缓存 | 后续 Decode 轮次继续使用 |

Prefill 有 `T` 个查询位置，每个位置最多与 `T` 个键位置比较，所以分数关系在语义上是 `T×T`。FlashAttention 主要减少这部分中间结果的显存读写，不会改变 Q/K 打分、因果遮罩、Softmax 和汇总 V 的含义。

Qwen3.5-9B 的 KV Cache 按 `Nkv=4` 保存，不能误用 `Nq=16`。K 在写入缓存前已经做过 RoPE，复用缓存时必须保持位置编号和 RoPE 规则一致。

第 4 课会继续解释 Prefill、逐 token Decode、KV Cache 的建立过程和容量计算。

## 9. 练习

仍使用第 2 节的 K 和 V：

```text
K1 = [1,0]     V1 = [2,0]
K2 = [1,1]     V2 = [0,2]
K3 = [0,1]     V3 = [1,1]
```

现在令位置 3 在另一个查询头中的 `Q=[1,-1]`，头维度 `D=2`。

1. 计算 Q 与三条 K 的点积，再除以 `√2`。
2. Softmax 权重约是多少？保留三位小数。
3. 用权重汇总 V，得到这个头在位置 3 的输出。
4. 如果查询来自位置 2，哪一列必须被因果遮罩？
5. Qwen3.5-9B 一次 Decode 中，`B=1,Nq=16,Nkv=4,D=256`，加入新 token 后缓存长度为 101。写出新 Q、缓存 K/V 和逻辑 Attention Score 的 shape。
6. RoPE 在哪一步执行？它是否改变 Q/K shape，是否旋转 V？

<details>
<summary>查看计算结果</summary>


1. 点积为 `[1,0,-1]`，缩放后约为 `[0.707,0,-0.707]`。

2. Softmax 权重约为：

```text
[0.576,0.284,0.140]
```

3. 输出约为：

$$
0.576[2,0]+0.284[0,2]+0.140[1,1]
=[1.292,0.708]
$$

4. 位置 2 只能读取第 1、2 列，第 3 列属于未来，必须在 Softmax 前遮住。

5. Qwen3.5-9B 的 Decode shape 是：

```text
q_new:           [1,16,1,256]
K/V Cache:       [1,4,101,256]
Attention Score: [1,16,1,101]
```

6. RoPE 在 Q/K 点积前执行，只改变 Q 和 K 的数值，不改变 shape。V 不参与位置匹配，因此不做同样的旋转。

</details>

## 参考资料

以下 Qwen3.5 配置和实现于 2026-08-08 按固定 revision 复核：

- [Qwen3.5-9B-Base 模型卡，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)
- [Qwen3.5-9B-Base `config.json`，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)
- [Transformers：Qwen3.5 模型实现，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer：Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA：Grouped-Query Attention](https://arxiv.org/abs/2305.13245)

本课的讲解顺序和图示还参考了以下资料：

- [Transformer 模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)
- [3Blue1Brown：Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Dive into Deep Learning：Queries, Keys, and Values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html)

---

[上一课：Decoder Layer 的结构与计算](02-inside-a-decoder-layer.md) · [返回课程路线](../roadmap.md) · [下一课：Prefill、Decode 与 KV Cache](04-prefill-decode-kv-cache.md)
