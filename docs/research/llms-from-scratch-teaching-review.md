# LLMs-from-scratch 的图解方法：对本课程的改进建议

本文参考 Sebastian Raschka 的 [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)。它是《Build a Large Language Model (From Scratch)》的官方代码仓库，从零实现一个 GPT 风格模型，并配合文字、图和小例子逐步展开。[官方仓库 README](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/README.md)

本次研究以仓库快照 [`f77106d`](https://github.com/rasbt/LLMs-from-scratch/tree/f77106d3c66dc249e6b16e4b056534b4ca1820e6) 为准，只查看官方仓库、作者官网和书籍配套材料。

## 主要判断

本次重构开始前，第 0 至第 9 课已有 42 张 SVG，其中 Attention 一课有 8 张。问题不是图少。与 LLMs-from-scratch 相比，更值得改的是图和正文的衔接方式：

1. 一课尽量沿用同一个例子，不让读者每到一节都重新认识一批 token 和数字。
2. 先只跟踪一个 token、一个位置或一个请求，把一次计算讲清，再推广到整段序列和矩阵。
3. 连续几张图像连续镜头。上一张的输出就是下一张的输入，只新增一个动作。
4. 图里的数字、变量名和紧跟着的小计算完全一致。读者可以对着图逐项核对，而不是另外理解一套例子。
5. 公式放在直观过程之后，作为前面步骤的压缩写法。
6. 图只回答一个问题。全景图负责定位，局部图负责解释一个动作，不把两项任务塞进同一张图。

LLMs-from-scratch 的第 2 章主 notebook 有 20 个 `<img>`，扣除书籍封面后是 19 张章节图；第 3 章有 27 个 `<img>`，扣除封面后是 26 张章节图。这个数字只能说明作者愿意为一个难点画连续镜头，不应变成我们的配额。需要画几张，取决于读者需要在脑中模拟多少步。[第 2 章 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch02/01_main-chapter-code/ch02.ipynb) [第 3 章 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch03/01_main-chapter-code/ch03.ipynb)

| 材料 | 图示数量 | 这里能说明什么 |
| --- | ---: | --- |
| LLMs-from-scratch 第 2 章 | 19 张章节图 | Tokenization 和 Embedding 被拆成连续的小动作 |
| LLMs-from-scratch 第 3 章 | 26 张章节图 | Attention 先算一个位置，再逐步推广 |
| 本课程第 0 至第 9 课（重构前） | 42 张 SVG | 已经具备较完整的原创视觉素材 |
| 本课程第 3 课（重构前） | 8 张 SVG | Attention 的改进重点是顺序和连续性，不是单纯增加数量 |

## Tokenization 和 Embedding 是怎样讲的

第 2 章没有先用定义解释所有术语，而是从一个短句开始，一步只增加一个动作。

| 顺序 | 图承担的任务 | 图后面紧接什么 |
| --- | --- | --- |
| 1 | 把 `Hello, world. Is this-- a test?` 拆成小块 | 用相同字符串运行正则切分 |
| 2 | 把 token 去重并编号，得到词表和 Token ID | 用 `sorted(set(...))` 和 `enumerate` 建词表 |
| 3 | 画清 `encode` 和 `decode` 两条相反方向的路径 | 立即调用自己实现的 tokenizer |
| 4 | 用很小的词表和 3 维向量解释 Embedding 查表 | 设 `input_ids=[2,3,5,1]`，打印权重表，再逐行查出向量 |
| 5 | 把 Token Embedding、Position Embedding 和相加结果放在一起 | 才回到真实词表大小和三维 batch shape |
| 6 | 在章末把文本、token、ID、Embedding 和 Decoder 接回完整链路 | 让读者重新看到局部动作在全模型中的位置 |

对应的官方图可以直接查看：[文本切分](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/05.webp)、[token 到 ID](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/07.webp)、[Embedding 查表](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/16.webp)、[完整输入链路](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/19.webp)。图、数字和代码的具体位置见[第 2 章主 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch02/01_main-chapter-code/ch02.ipynb)。

这里最值得借鉴的是 Embedding 的入口。作者先让读者看到“ID 2 就是取第 2 行”，再谈高维表示。这样不会让“语义空间”这类抽象说法挡在计算过程前面。

## Attention 为什么更容易读

第 3 章把 Attention 走了两遍。

第一遍故意不用可训练的 Q、K、V。作者固定一句 `Your journey starts with one step`，每个 token 只有 3 个数，只计算第 2 个位置：

```text
第 2 个位置和所有位置点积
→ 把分数归一化
→ 每个输入向量乘对应权重
→ 相加得到第 2 个位置的上下文向量
```

四个动作分别用四张连续图解释：[整体预览](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/07.webp)、[点积得到分数](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/08.webp)、[分数变权重](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/09.webp)、[加权求和](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/10.webp)。每张图后面都用相同的 6×3 输入写几行代码，先算一个位置。等这条链路通了，才用双重循环计算所有位置，再改写成矩阵乘法。[第 3 章 3.3 节](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch03/01_main-chapter-code/ch03.ipynb)

第二遍保留同一句话，再给每个输入加上三组学习权重，得到真正的 Q、K、V。此时读者已经知道“比较”和“取回”是两件事，只需理解 Q、K 负责比较，V 负责提供被汇总的信息。[QKV 图](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/14.webp)

后面的因果遮罩也采用相同办法：先画哪些格子能看，接着画三角矩阵，最后展示在 Softmax 前把未来分数改为负无穷。[因果遮罩图](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/20.webp) [第 3 章 3.5 节](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch03/01_main-chapter-code/ch03.ipynb)

作者的[官方 Self-Attention 图文教程](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)也沿用这套方法：固定一句话，每次只高亮当前参与计算的向量，图后立即给出短代码。

这对我们的第 3 课最有价值，但不能原样照搬。现有正文一开始就给出完整流程和 Q、K、V，虽然准确，读者却还不知道为什么要分成三条向量。更合适的顺序是：

```text
先用概念图说明三个动作：比较位置、把分数变成权重、按权重取回信息
→ 引入 Q、K、V，用一个真实的 Q/K/V 小例子完成这三个动作
→ 推广到整张矩阵和因果遮罩
→ 压缩成完整公式，再讲多头、GQA 和真实 shape
```

概念图可以暂时不写公式，但第一次数值计算就应使用 Q、K、V。这样既保留“先理解动作，再看公式”的优点，也不会让读者先学一套 Qwen3.5 实际并不使用的简化算法。

## Transformer、生成和训练怎样衔接

第 4 章先分别解释归一化、FFN 和残差，再把这些模块装进一个 Transformer Block，最后把单层放回完整 GPT。每一块都先有独立小图和小数字，组装时只需要理解连接关系。[第 4 章主 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch04/01_main-chapter-code/ch04.ipynb)

具体顺序是：

- 用两行小输入计算 LayerNorm 的均值和方差，再画它在网络中的位置；
- 单独画 FFN 的扩展、激活和回收，并验证 `[2,3,768]` 输入输出 shape 不变；
- 单独解释残差路径；
- 把前面的模块组装成一层；
- 再把一层重复为完整 GPT。

对应的官方图包括：[单层 Transformer Block](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/13.webp)和[完整 GPT](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/15.webp)。

生成部分同样先画时间线，再给循环代码。一张图只画“一轮怎样从 logits 选出 Token ID”，下一张再画“新 token 怎样追加到输入并开始下一轮”。[一轮生成](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/17.webp) [逐轮追加 token](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/18.webp)

第 5 章讲训练时仍然使用小词表。它先让 target token 对准概率向量中的一个位置，再逐步接上对数、平均和负号。Top-K 也只用 9 个候选词，直接把未入选项改为负无穷，然后才给代码。[第 5 章主 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch05/01_main-chapter-code/ch05.ipynb) [Top-K 图](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/15.webp)

## 哪些方法适合我们

### 适合直接采用的方法

- **固定一个例子贯穿一课。** 读者把注意力放在新增计算上，不需要反复建立上下文。
- **先算一个位置，再推广到整个张量。** 这特别适合 Attention、KV Cache、Gated DeltaNet 和 Router。
- **图、小数字和计算共用变量。** 我们不必照搬代码优先的形式，可以改成“图 → 两三行手算或伪代码 → 通用 shape → Qwen3.5 配置”。
- **连续图只增加一个动作。** 比一张塞满十几个框的总图更容易跟。
- **讲完局部后回到全图。** 读者能确认当前模块位于 Decoder Layer、请求时间线或服务端链路的哪里。
- **玩具规模和真实规模分开。** 图里用 2 至 4 维，最后单独代入 `H=4096`、真实头数和层数。
- **颜色表达固定含义。** 同一课里，输入、Q、K、V、权重、缓存、模型参数和请求状态不随图改变颜色。

### 不适合照搬的部分

- 这套材料的目标是从零编码和训练 GPT-2，我们的读者主要维护推理系统。反向传播、优化器和完整训练循环不应挤占主线。
- 主线模型是较早的 GPT-2，很多实现与 Qwen3.5 的 RMSNorm、SwiGLU、GQA、RoPE、Gated DeltaNet 和 MoE 不同。可以学习讲法，不能直接把结构当成当前模型事实。
- 作者常在一张图里放较多英文注释。中文课程需要控制图中文字，让正文承担补充说明，手机和 GitHub 页面上也能读清。
- 图多不是目标。已经能从小数字和一张清楚的图理解的地方，不必再加一张同义图。
- 不要直接复制原图。官方仓库的 Apache 2.0 `LICENSE.txt` 在 `Source` 定义中明确排除了书籍内容和相关图片。可以借鉴教学结构，图必须使用自己的中文例子和视觉体系重新绘制。[官方 LICENSE](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/LICENSE.txt)

## 对第 0 至第 9 课的具体改法

### 第 0 课：张量与模型计算基础

现有三张图已经覆盖张量轴、归约和 Linear，但索引、广播、点积之间仍要靠读者在脑中换图。

可执行改法：

1. 固定一个 `X:[B=2,T=3,H=4]` 的小张量，后面的索引、沿 `H` 归约、`keepdim`、广播和 Linear 都用它。
2. 在张量图上直接高亮 `X[1,2,:]`，再显示结果 `[H]=[4]`。不要另换一组数字解释索引。
3. 广播图只回答一件事：`[B,T,1]` 怎样把每个 token 的一个缩放量应用到 `H` 个元素。把复制方向画出来。
4. 点积图固定两条 3 维向量，依次画“对应元素相乘”和“相加得到一个分数”。紧接同一组数字的手算。
5. Linear 图沿用这些向量，清楚标出“矩阵中的一行权重产生一个输出坐标”。

这里不需要增加完整线性代数章节。需要补的是读者无法仅靠文字在脑中完成的 shape 变化。

### 第 1 课：大模型生成下一个 Token 的过程

现有总览图把全链路列出来了，但后面的 Chat Template、Tokenizer、Embedding 和生成循环没有一直使用同一条输入。

可执行改法：

1. 选一条短的真实对话作为全课例子。第一张连续图展示：用户消息、Chat Template 补出的特殊 token、token 列表、Token ID 列表。
2. 下一张图保留相同的 Token ID，在旁边画 Embedding 表，逐个连到对应行。这样可以把“编号”和“向量”接起来。
3. LM Head 示例也继续使用这条输入，只把候选词表缩小到 4 至 6 项。用横向条形长度表示 logit，直接标出 Argmax 选中的 Token ID。
4. 再画三帧小图：第一次输入、选出一个 token、把它追加后开始下一轮。这样“为什么一次不能生成未来 100 个 token”会比一段文字更直观。
5. Chat Template 的示例应来自实际 tokenizer 输出或明确标注为示意，不能把用户输入的四个汉字直接当成最终模型输入。

### 第 2 课：Decoder Layer 的结构与计算

这一课的模块图目前使用 Mermaid，后面 RMSNorm、SwiGLU 和 SiLU 又各用不同例子。模块本身已经解释得比较完整，下一步应减少读者在这些图之间重新对应变量的成本。

可执行改法：

1. 把开头的 Mermaid 主骨架换成一张稳定 SVG，固定一个被高亮的 token 行，画出两段 `RMSNorm → 子层 → 残差相加`。
2. 第一遍只突出 Token Mixer 路径，其他模块变灰；第二遍突出 FFN 路径。两个残差加号分别从哪里接回要一眼可见。
3. FFN 从同一个 2 维输入开始，`gate_proj`、`up_proj`、SiLU、逐元素相乘、`down_proj` 全部沿用正文手算的数字。图和 8.1 至 8.4 节不要各自使用另一套记号。
4. 在完整层图每条主箭头上保留 `[B,T,H]`，只在 FFN 内部改成 `[B,T,I]`，让读者直接看到 token 数没有变化。
5. 章末回到同一张图，增加 Qwen3.5 中 Token Mixer 的两种实现标签，不再重新画另一套骨架。

### 第 3 课：Attention 的计算原理

这是最值得重排的一课。现在已经有 8 张图，短板不是数量，而是第一张就出现 Q、K、V、因果遮罩和完整流程，入口仍然偏陡。

可执行改法：

1. 把“Attention 的完整计算流程”移到直观过程之后，作为回顾图。
2. 新的第一张局部图固定一个三到四 token 句子，只用自然语言说明当前 token 要完成“比较、分配权重、取回信息”三个动作，暂时不展开公式。
3. 随后把同一个 Hidden State 通过三组 Linear 变成 Q、K、V。连续三张图继续使用完全相同的 token 和数字：Q/K 点积得到分数、Softmax 得到权重、权重乘 V 并相加。每张图只新增一步。
4. 第一次手算就使用真实 Q、K、V，不另外教授一套省略 Q/K/V 的数值算法。
5. 第三遍再把“只算当前 token”推广成整张 `T×T` 分数矩阵，并用同一张矩阵叠加因果遮罩。
6. RoPE 先画同一对 Q/K 在不同相对距离下点积分数怎样改变，再给旋转公式。需要回答的是“旋转为什么能把相对位置写进点积”，不只是“向量发生了旋转”。
7. 多头、GQA 和真实 shape 放在主计算已经完整走通以后，分别只回答“为什么分头”和“哪些头共享 K/V”。

### 第 4 课：Gated DeltaNet 的状态更新机制

这一课概念多，当前完整数据流图会让读者同时面对卷积、Q/K/V、状态矩阵和三个门。更适合先跟踪一次读和写。

可执行改法：

1. 先用一个 2×2 状态矩阵，只画“用 K 找到状态中的已有预测”“和 V 比较得到误差”“按误差修正状态”三步。
2. 下一张图沿用同一个状态，加入 `β`，只解释本次修正写入多少。
3. 再加入 `α`，只解释旧状态在写入前保留多少。输出门放到最后。
4. 因果卷积单独用一条三 token 滑动窗口解释，不要一开始与状态更新画在一起。
5. Full Attention 和 Gated DeltaNet 的比较图使用同一条长度逐步增加的序列：前者右侧不断增加 K/V 列，后者始终只有一个固定大小状态块。
6. 最后才把这些局部图装回现有完整数据流，并代入真实 shape。

作者的 Gated DeltaNet 配套材料也先画固定状态和门，再展开循环更新；它适合参考拆图顺序，但其中是为 Qwen3-Next 写的简化实现，不能直接替代我们已经核实过的 Qwen3.5 事实。[官方 Gated DeltaNet 材料](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch04/08_deltanet/README.md)

### 第 5 课：Dense FFN 与 MoE 的结构差异

现有图已经覆盖“替换 FFN”“单 token 路由”“完整 MoE”“EP 与 TP”。可以继续采用从单 token 推广到 batch 的顺序，重点补齐图之间的连续性。

可执行改法：

1. 第一张图只做一件事：把 Dense Layer 中的一套 FFN 替换成 Router 加多套 FFN，其他 Decoder Layer 模块保持灰色且不变。
2. 单 token 路由图固定一个 token 的 Router Logits、Top-K Expert ID 和归一化权重，正文手算使用图中同一组数字。
3. 下一张图加入第二、第三个 token，用相同专家颜色显示它们被分到不同 Expert。这样自然引出 Dispatch、Combine 和负载不均。
4. Shared Expert 始终用单独颜色并贯穿所有图，避免它看起来像 Top-K 里的另一个 Expert。
5. `35B Total / 3B Active` 用两层视觉表达：整机必须保存的全部权重，以及一个 token 本轮实际经过的彩色路径。不要只用数字框解释。

### 第 6 课：自回归推理的执行阶段与状态复用

现有生成时间线已经清楚，最需要补的是“缓存到底省掉了哪次重复计算”。Raschka 的官方 KV Cache 材料也是先用相邻两轮生成比较重复的 K/V，再进入缓存代码。[官方 KV Cache 说明](https://github.com/rasbt/LLMs-from-scratch/blob/f77106d3c66dc249e6b16e4b056534b4ca1820e6/ch04/03_kv-cache/README.md)

可执行改法：

1. 在“为什么过去的 K/V 可以复用”前加入一组并排图。左边是不使用缓存，第二轮重新计算旧 token 的 K/V；右边使用缓存，只计算新 token 的 K/V。
2. 两边使用同一句 Prompt 和同一组颜色，旧 K/V 用完全相同的数值，直接证明它们没有变化。
3. 把现有时间线拆成三帧：Prefill 结束、Decode 第 1 轮、Decode 第 2 轮。每帧分别标清“输入 token”“新写入的状态”“本轮预测出的 token”。
4. KV 容量计算继续使用正文公式，但在公式旁画一个 `[Layer,Nkv,T,D]` 的堆叠块，标出只有 `T` 随生成增长。
5. Mixed Batch 图保留，增加每个请求本轮贡献多少个 token 的明确标记，把“请求内串行”和“请求间可批处理”放在同一张时间切片里。

### 第 7 课：多模态输入与视觉编码

现有四张图分别解释双输入路径、Patch 与 Merger、占位替换和成本，主线已经比较顺。最直接的改进是让同一张图片贯穿所有步骤。

可执行改法：

1. 选一张简单的 512×512 示例图，从缩放、切 Patch、Patch Embedding、视觉编码器、Merger 一直沿用。
2. 在 Patch Embedding 处放一个 2×2 灰度小块，展开成 4 个数，再乘一个很小的权重矩阵得到向量。读者会看到“一个图块怎样变成一条向量”。
3. 用三种不同图形区分图像 Patch、视觉序列位置和词表 Token ID，不能只靠三个相近的术语。
4. 占位替换图保留相同的 Patch 颜色，让读者看出视觉向量写进了语言序列中的哪些位置。
5. 成本图沿用前面的网格，直接比较分辨率增加后 Patch 数、Merger 后视觉位置数和 Decoder 序列长度怎样变化。

### 第 8 课：模型配置与资源估算

这一课的主要难点是把字段名转成结构，再从结构算容量和计算量。现有第一张图按问题分类字段，但读者仍需在 JSON 和模型结构间来回对应。

可执行改法：

1. 放一段经过裁剪的真实 `config.json`，用连线把 `hidden_size`、层数、头数、KV 头数和 FFN 宽度接到同一张 Decoder 图中的对应位置。
2. Dense 和 MoE 分开做两张字段映射，不要把所有字段放进一个总图。
3. 参数量图按 Embedding、单层 Token Mixer、单层 FFN、层数乘积、LM Head 的顺序逐项累加，图中数字和正文公式一致。
4. 状态容量图继续使用第 6、4 课的颜色：KV Cache 随 `T` 增长，Gated DeltaNet 状态不随 `T` 增长。
5. 章末的配置检查表改成可直接复制的填写模板，读者拿到一个新模型后可以照表写出 shape、权重容量和请求状态。

### 第 9 课：推理优化的分析与评估

本课已经有多张前后对比图，但不同优化各自使用一套画法。读者更需要一套稳定的判断坐标。

可执行改法：

1. 所有优化都复用同一条推理链路底图，只把被改变的位置着色。
2. 每个案例固定四行：少搬了多少字节、少算了多少 FLOPs、新增了什么通信或调度、最终看哪个端到端指标。
3. 权重量化、FlashAttention、Prefix Cache、TP、EP 和推测解码分别只改动底图中的一个或两个位置，其他部分变灰。
4. 增加一张“理论收益被新增成本抵消”的图：左侧是省下的时间，右侧是反量化、通信或验证成本，最后比较总时间。它能直接解释为什么理论量下降不保证端到端变快。
5. 三个完整案例继续保留，但所有数字应落到同一张判断表中，避免案例写成三种不同格式。

## 建议的实施顺序

不需要同时重画 10 课。按读者理解成本排序：

1. 先重排第 3 课 Attention。现有素材最多，读者反馈也最集中，主要工作是固定例子、调整顺序和补齐连续镜头。
2. 接着改第 2 课的完整 Decoder Layer 图，再补第 1 课从 Chat Template 到生成循环的连续图。这两课是读懂 Attention 的直接前置。
3. 然后补第 6 课“无缓存与有缓存”的对照图，以及第 4 课状态矩阵的一次读写。
4. 第 0 课只补索引、广播和点积中确实需要读者在脑中模拟的步骤，不扩成完整线性代数图册。
5. 第 5 至第 9 课先保留现有结构，只修正图和手算不一致、同一对象颜色变化、总图承担过多任务的地方。

判断一张新图是否值得画，可以只问一个问题：如果没有这张图，读者是否必须在脑中同时保存两个以上的中间状态，才能继续理解正文？如果答案是肯定的，图通常有价值；如果只是把一段已经清楚的文字放进方框，便没有必要。
