# 课程路线：从模型怎样生成，到怎样判断推理优化

## 课程目标

这套课程写给已经参与推理系统研发、但模型理论基础还不完整的工程师。它不会覆盖所有论文和框架参数，而是集中讲清能解释大多数推理问题的那组核心概念。

学完后，可以沿着下面这条因果链分析问题：

```text
输入怎样表示
→ 一个 Decoder Layer 怎样处理表示
→ 不同 token 怎样交换信息
→ 为什么生成必须逐步进行并保存历史状态
→ Qwen3.5 怎样组合不同层和 Dense/MoE
→ 多模态输入怎样进入语言模型
→ 配置中的数字怎样变成参数、状态和计算
→ 一种优化到底改变了什么
```

## 三个贯穿问题

看一个语言模型结构时，可以从三个问题入手：

| 问题 | 要识别的对象 | 对工程判断的意义 |
| --- | --- | --- |
| 当前数据表示什么？ | Token ID、Embedding、Hidden State、Logit、概率 | 避免把不同阶段的张量混为一谈 |
| 信息怎样被混合？ | Token Mixer 跨 token；FFN 在单 token 内加工特征 | 找到模型能力和主要计算的来源 |
| 哪些历史结果需要保留？ | KV Cache、Gated DeltaNet recurrent state | 理解 Prefill、Decode、显存和并发限制 |

后面的算子、公式和优化方法，都会回到这三条主线中定位。

## 两种尺度的例子

基本计算先用小数字讲清，再回到真实模型，避免一开始就被 `4096×12288` 这样的规模遮住计算关系。

### 小数字教学模型

按概念需要选用类似下面的小尺寸：

```text
B=1，T=3，H=4，I=6，V=5，Attention Head=2
```

它不是任何真实模型，只用于手算、画矩阵和检查 shape。

### 真实模型

- Qwen3.5-9B-Base：讲 Dense FFN、混合层结构和视觉输入；
- Qwen3.5-35B-A3B：讲 MoE、总参数与激活参数；
- 官方 `config.json` 和 Transformers 实现：核对真实字段、shape 和数据流。

重要概念通常从小数字计算开始，再推广到通用 shape，最后用真实配置确认规模。

## 当前课程顺序

当前主线按 0～9 编号。编号是依赖顺序，不是对篇幅的限制；某个主题如果无法在一课内讲清，可以继续拆分。

| 课次 | 这一课回答什么 | 主要内容 | 学完后能做什么 |
| ---: | --- | --- | --- |
| 0 | 后面的张量和公式怎样读？ | 标量、向量、矩阵、shape、轴、索引、归约、广播、点积、矩阵乘法、Linear、Embedding | 能看懂算子对哪些数计算，并推导基本 shape |
| 1 | 一句话怎样变成下一个 token？ | Chat Template、Tokenizer、Embedding、Decoder 黑盒、LM Head、Logits、Softmax、采样 | 能区分文字、ID、向量、分数和概率，画出完整生成链路 |
| 2 | 一个 Decoder Layer 为什么这样设计？ | Hidden State、RMSNorm、Residual、Token Mixer/FFN 分工、Dense SwiGLU | 能解释一层中每个公共模块的目的、计算和 shape |
| 3 | 当前 token 怎样读取上下文？ | Q/K/V、因果 Mask、缩放点积 Attention、多头、GQA、RoPE | 能用小数字走完一次 Attention，解释位置和头的作用 |
| 4 | 为什么生成必须逐步进行？ | 条件概率、Prefill、Decode、KV Cache、请求内串行与批内并行 | 能解释缓存复用了什么、占用了什么，以及 TTFT/TPOT 的模型来源 |
| 5 | Qwen3.5 为什么混用两类 Token Mixer？ | Gated DeltaNet、因果卷积、门控更新、recurrent state、Full Attention 间隔层 | 能区分 KV Cache 与 recurrent state，读懂 3+1 混合排列 |
| 6 | Dense 和 MoE 到底差在哪里？ | Dense FFN、Router、Top-K、路由专家、共享专家、加权合并、token 分发 | 能解释总参数、激活参数、专家路由和通信来源 |
| 7 | 图片和视频怎样进入语言模型？ | 图像预处理、Patch、视觉编码器、特征压缩与投影、视觉 token、统一输入序列 | 能定位视觉编码与语言模型开销，解释分辨率为什么影响序列长度 |
| 8 | 怎样从配置还原模型和运行时数据？ | 层数、H/I/V、头数、专家数、dtype、权重、KV/状态容量、主要计算量 | 能读配置并完成带假设和单位的数量级估算 |
| 9 | 怎样用理论判断推理优化？ | 量化、FlashAttention、Prefix Cache、Batching、TP/EP、推测解码/MTP | 能说明优化改了什么、少了什么、新增什么、何时收益成立 |

## 第 0 课：只够后续使用的数学与张量

正文：[第 0 课：看懂推理链路所需的数学与张量](lessons/00-math-and-tensors.md)

这节课补齐阅读后文所需的数学，不展开完整线性代数。内容限于后面会直接用到的部分：

```text
数据怎样组织
→ shape 和索引怎样定位数据
→ 轴和归约怎样减少维度
→ 广播怎样应用逐元素计算
→ 点积怎样产生一个分数
→ 矩阵乘法怎样批量执行点积
→ Linear 和 Embedding 怎样使用这些动作
```

学到这里，应该能解释 `X:[B,T,H]` 和 `W:[I,H]` 中的 `B/T/H/I`，并推出 Linear 输出 `[B,T,I]`。

## 第 1 课：从一句话到下一个 token

正文：[第一课：从一句话到下一个 Token](lessons/01-text-to-next-token.md)

这一课建立完整的生成地图：

```text
对话消息
→ Chat Template
→ Tokenizer
→ Token IDs
→ Embedding
→ Decoder
→ LM Head
→ Logits
→ 贪心或采样
→ 下一个 Token ID
→ Tokenizer Decode
```

学完后，应该能解释模型为什么不直接接收或输出文字，并准确区分 Token ID、Embedding、Hidden State、Logit 和概率。

## 第 2 课：看懂一个 Decoder Layer

正文：[第二课：一个 Decoder Layer 怎样处理 Token](lessons/02-inside-a-decoder-layer.md)

一层反复使用同一骨架：

```text
保存输入 → RMSNorm → Token Mixer → 残差相加
保存输入 → RMSNorm → SwiGLU FFN → 残差相加
```

这里暂时不打开 Token Mixer 的内部计算，重点讲 Hidden State、RMSNorm、Residual 和 Dense SwiGLU。Attention 放到下一课单独展开。

学完后，应该能把 SwiGLU 展开为 `gate_proj → SiLU`、`up_proj`、逐元素乘法和 `down_proj`，并写出每一步的 shape。

## 第 3 课：Attention 怎样读取上下文

这一课回答：当前 token 怎样判断前文哪些位置与自己有关？

```text
Hidden State
→ 产生 Q、K、V
→ Q 与 K 点积打分
→ 加因果 Mask
→ Softmax 得到权重
→ 按权重汇总 V
```

在这个计算基础上，再解释多头、GQA 和 RoPE。讲解会区分结构事实与便于理解的直觉：不同头可以学习不同关系，但不表示每个头都有人工指定的固定职责。

## 第 4 课：自回归、Prefill、Decode 与 KV Cache

把静态模型结构变成时间过程：

```text
Prompt 已全部知道 → Prefill 可同时处理多个位置
第一个未来 token 尚未知 → 必须先生成
新 token 确定 → 才能继续生成下一个
```

KV Cache 保存已经计算出的历史 K/V，减少重复计算，但不能提前知道未来 token。这里再解释 Continuous Batching 和 Chunked Prefill 为什么能把不同请求、不同阶段的已知 token 组织到同一轮执行。

## 第 5 课：Qwen3.5 的混合 Token Mixer

Qwen3.5-9B 使用 8 组：

```text
3 × (Gated DeltaNet → Dense FFN)
+ 1 × (Full Attention → Dense FFN)
```

Full Attention 保留并读取较完整的历史 K/V；Gated DeltaNet 把历史持续更新到固定 shape 的 recurrent state。课程会解释两类状态的语义、shape 与生命周期，不展开训练推导和 Kernel 实现。

## 第 6 课：Dense 与 MoE

MoE 不替换整个 Decoder Layer，主要替换 FFN 子层：

```text
Dense：每个 token 使用同一套完整 FFN
MoE：Router 为每个 token 选择少量路由专家，并使用共享专家
```

重点区分：

- 总参数必须存储，不等于每个 token 都参与计算；
- 激活参数只描述本轮选中的参数，不直接等于显存、通信或延迟；
- 专家是训练形成的参数分工，不是人工指定的“代码专家”或“数学专家”。

## 第 7 课：视觉编码器与多模态输入

这节课只补充理解多模态推理链路所需的视觉知识：

```text
图片或视频
→ 缩放与归一化
→ 切成 Patch
→ 视觉编码器
→ 视觉特征压缩与投影
→ 与文本向量组成统一序列
→ 语言模型 Decoder
```

其中一个重要区别是，视觉特征不是由文本 Token ID 查 Embedding 得到的。图片分辨率、数量和视频帧数会改变视觉位置数，进而影响视觉编码耗时、语言模型 Prefill、缓存和显存。

## 第 8 课：从配置还原结构与数量级

前面先建立语义，这一课再集中估算：

```text
读配置字段
→ 翻译成模块和 shape
→ 用小数字检查公式
→ 代入真实配置
→ 标明单位、近似和忽略项
```

分别还原 Qwen3.5 Dense 与 MoE 模型的层排列、投影尺寸、头数、专家数、参数容量、KV Cache 和 recurrent state。

## 第 9 课：用理论判断优化方向

GPU 执行和服务调度不再各自成为孤立章节。每种优化都回到前面已经理解的模型对象：

| 优化 | 直接改变什么 | 首先检查什么 |
| --- | --- | --- |
| 权重量化 | 权重字节数、数值精度和计算路径 | 权重容量或带宽是否构成主要限制 |
| KV 量化 | Cache 容量和读取字节 | Full Attention 层、KV shape 和精度影响 |
| FlashAttention | Attention 中间值的读写与分块方式 | 它不改变 Attention 语义，也不消除自回归依赖 |
| Prefix Cache | 复用相同前缀的历史状态 | 命中率、状态容量和生命周期 |
| Batching | 一轮处理更多已知 token | 吞吐收益与排队、TTFT、TPOT 的交换 |
| TP | 分片同一层权重和计算 | 每层集合通信是否进入关键路径 |
| EP | 把专家放到不同设备 | token 分发、All-to-All 和负载均衡 |
| 推测解码/MTP | 提出并验证多个未来候选 | 接受率、验证成本和额外模型成本 |

可以用下面的问题检查一个优化方案：

```text
它改变链路中的哪个对象？
→ 少算、少存或少搬了什么？
→ 新增了什么计算、状态、通信或误差？
→ 哪种 workload 下收益才会出现？
→ 用什么指标和对照实验验证？
```

## 第一轮暂不展开

- 训练、反向传播、优化器和完整概率论；
- 视觉模型训练方法和完整计算机视觉课程；
- CUDA ISA、PTX、Kernel 编写和 profiler 操作教程；
- 每个框架的参数清单和每一种调度算法；
- 机械可解释性中的神经元、归纳头和知识编辑专题；
- 每一种 Attention、量化、并行和采样变体；
- OCR、语音和扩散模型等其他模型链路。

这些内容可以后续按工程需要作为专题添加，不打断第一轮主线。

## 总体验收标准

完成课程后，读者应当能够：

1. 画出文本或图片到下一个 token 的数据流；
2. 指着 Decoder Layer 解释 Norm、Token Mixer、FFN/MoE 和 Residual；
3. 用两三个 token 的小例子完成 Attention 计算；
4. 解释 Prefill、Decode、KV Cache 和 recurrent state；
5. 对比 Dense 与 MoE，并区分总参数和激活参数；
6. 从 Qwen3.5 配置恢复层数、层类型、头数和专家数；
7. 面对一种优化，先指出它改变哪个模型对象，再判断收益条件和代价。

## 原始资料

- [Qwen3.5-9B-Base 官方模型说明](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
- [Qwen3.5-9B-Base 官方配置](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/main/config.json)
- [Qwen3.5-35B-A3B 官方模型说明](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-35B-A3B 官方配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer：Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA：Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
