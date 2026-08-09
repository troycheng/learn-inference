# 《看懂大模型推理》课程路线

这套课面向已经参与推理系统研发、但没有系统学过模型原理的工程师。课程只保留会反复用于工程判断的模型主线，不追求收齐所有术语。

课程从用户输入出发，最后回到推理优化：

```text
文本输入与模型向量
→ Decoder Layer 的结构与计算
→ Attention 与上下文信息
→ Prefill、Decode 与请求状态
→ Dense、MoE 与 Gated DeltaNet
→ 多模态输入与视觉编码
→ 配置字段与资源估算
→ 推理优化的收益、成本与验证
```

## 课程依赖关系

推理系统里的很多概念互相依赖。还没分清 Token ID、Embedding 和 Hidden State，就很难理解 Q、K、V 从哪里来；没算过 Attention，也无法真正理解 KV Cache 保存了什么；不知道 Dense 和 MoE 的结构差异，就无法判断 TP、EP 或量化改变了哪部分成本。

因此前四课是一条连续主线：

```text
第 0 课  张量、shape 与基础计算
第 1 课  下一个 Token 的生成过程
第 2 课  Decoder Layer 的结构与计算
第 3 课  Full Attention 的计算原理
第 4 课  Prefill、Decode 与 KV Cache
```

第 5 至第 7 课加入 Qwen3.5 的混合结构、MoE 和图片输入。第 8、9 课不再堆新模块，而是练习估算和判断。

## 贯穿课程的三个分析维度

遇到任何新算子或模型结构，都可以先问三件事：

| 问题 | 需要看什么 | 为什么与推理有关 |
| --- | --- | --- |
| 当前数据表示什么？ | Token ID、Embedding、Hidden State、Logit、概率 | 避免把不同阶段的张量和指标混为一谈 |
| 信息怎样混合？ | Token Mixer 跨 token；FFN 在单个 token 内加工特征 | 找到模型计算和能力来自哪里 |
| 哪些历史结果要保留？ | KV Cache、卷积状态、recurrent state | 判断显存、并发和逐 token 延迟 |

## 课程目录与学习目标

| 课次 | 主题 | 学习目标 |
| ---: | --- | --- |
| [0](lessons/00-math-and-tensors.md) | 张量与模型计算基础 | 根据输入、运算和输出推导 shape |
| [1](lessons/01-text-to-next-token.md) | 下一个 Token 的生成过程 | 区分文本、Token ID、向量、Logit 和概率 |
| [2](lessons/02-inside-a-decoder-layer.md) | Decoder Layer 的结构与计算 | 解释 RMSNorm、Token Mixer、残差和 SwiGLU FFN |
| [3](lessons/03-attention.md) | Attention 的计算原理 | 手算 QK 点积、因果遮罩、Softmax 和 V 的加权求和 |
| [4](lessons/04-prefill-decode-kv-cache.md) | Prefill、Decode 与 KV Cache | 解释生成阶段、缓存复用和服务端批处理 |
| [5](lessons/05-gated-deltanet.md) | Gated DeltaNet 的状态更新 | 区分固定状态、因果卷积和 Full Attention KV |
| [6](lessons/06-dense-and-moe.md) | Dense FFN 与 MoE | 解释 Router、Top-K、共享专家、总参数和激活参数 |
| [7](lessons/07-multimodal-input.md) | 多模态输入与视觉编码 | 从像素、Patch 和视觉编码器推导到 `[B,T,H]` |
| [8](lessons/08-config-and-sizing.md) | 模型配置与资源估算 | 估算参数、权重容量、KV、固定状态和主要计算量 |
| [9](lessons/09-optimization-judgment.md) | 推理优化的分析与评估 | 说明优化的直接改动、收益条件、额外成本和验证指标 |

## 教学示例与真实模型

第一次解释计算时，课程会把向量缩到 2 至 4 维，把 token 或 Expert 缩到几个。缩小的是规模，不是算法。一个小例子会按下面的顺序推进：

```text
先算一个 token 或一个位置
→ 用同一组数字推广到矩阵和 batch
→ 写出通用 shape
→ 代入 Qwen3.5 的真实配置
→ 推出显存、计算、访存或通信影响
```

课程使用三个模型版本：

- Qwen3.5-9B-Base 用于核对 Dense 架构、层数和 shape；
- post-trained Qwen3.5-9B 用于 Chat Template、Tokenizer、多模态输入和生成行为；
- Qwen3.5-35B-A3B 用于 MoE、总参数和激活参数。

涉及配置、实现或真实数字时，正文会链接到固定 revision。教学用数值会明确说明，不会伪装成真实模型输出。

## 课程范围

- 训练、反向传播、优化器和完整概率论；
- CUDA ISA、PTX 和 Kernel 编写；
- 每一种 Attention、量化、采样和调度变体；
- 完整计算机视觉课程；
- 机械可解释性、OCR、语音和扩散模型。

这些主题可以以后单独增加，不挤占第一轮主线。

## 参考资料

- [Qwen3.5-9B-Base 模型说明，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)
- [Qwen3.5-9B-Base 配置，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)
- [Qwen3.5-9B 配置，revision c202236](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
- [Qwen3.5-35B-A3B 模型说明，revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)
- [Qwen3.5-35B-A3B 配置，revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
