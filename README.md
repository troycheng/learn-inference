# Learn Inference

这套课写给已经在做推理系统、但模型理论基础还不完整的工程师。它不要求读者先系统学完线性代数，也不打算罗列所有模型名词。课程沿着一次真实生成向里走：一段文字怎样变成 token，token 怎样经过 Decoder，模型怎样找到相关上下文，又为什么只能逐个生成后续 token。Prefill、Decode、KV Cache、Dense、MoE 和常见优化都会放回这条链路中解释。

## 这套课要解决什么问题

读完核心课程，应当能够：

- 从 Tokenizer 开始，完整解释一个新 token 是怎样产生的；
- 说明 Embedding、RMSNorm、Attention、FFN、Residual、LM Head 分别解决什么问题；
- 看懂 Qwen3.5 文本模型的一层结构和完整层排列；
- 解释 Dense 与 MoE 的共同骨架，以及 Router、Top-K、共享专家的作用；
- 解释 Prefill、Decode、KV Cache 和 Gated DeltaNet 状态之间的关系；
- 解释图片怎样经过视觉编码器变成语言模型可处理的视觉特征；
- 阅读模型 `config.json`，判断参数规模、状态规模和主要计算来自哪里；
- 判断量化、缓存、批处理和并行方案究竟改变了模型链路中的什么。

## 主线

```mermaid
flowchart LR
    A["文字"] --> B["Token IDs"]
    B --> C["文本 Embedding"]
    V["图片或视频"] --> VE["视觉编码器"]
    VE --> U["统一输入向量"]
    C --> U
    U --> D["多层 Decoder"]
    D --> E["Logits"]
    E --> F["概率与采样"]
    F --> G["下一个 token"]
    G --> D
```

前几课先讲文字生成，第 7 课再加入图片和视频。两类输入最终都会变成与语言模型 Hidden Size 对齐的向量，进入同一个 Decoder 主干。

课程一直用 Qwen3.5 作例子：

- 用 **Qwen3.5-9B** 讲 Dense 模型；
- 用 **Qwen3.5-35B-A3B** 讲 MoE 模型；
- 用两者共同的 Gated DeltaNet 与 Full Attention 混合结构讲现代模型；
- 用 Qwen3.5 的视觉输入路径讲视觉编码器与多模态序列；
- 第一轮不展开训练、CUDA 编程和框架参数清单。

## 从这里开始

1. [第 0 课：模型推理中的张量计算](docs/lessons/00-math-and-tensors.md)
2. [第 1 课：模型如何生成下一个 token](docs/lessons/01-text-to-next-token.md)
3. [第 2 课：Decoder Layer 内部的数据流](docs/lessons/02-inside-a-decoder-layer.md)
4. [第 3 课：图解 Attention 的完整计算过程](docs/lessons/03-attention.md)
5. [第 4 课：读完 Prompt 之后，模型怎样逐个生成 token](docs/lessons/04-prefill-decode-kv-cache.md)
6. [第 5 课：Gated DeltaNet 怎样用固定状态记录前文](docs/lessons/05-gated-deltanet.md)
7. [完整课程路线](docs/roadmap.md)
8. [课程术语与符号表](docs/glossary.md)
9. [课程讲解原则](docs/teaching-method.md)

正文按当前路线逐课编写。旧内容保存在 [`docs/archive`](docs/archive) 中，只用于记录学习过程，不再作为主课材料。

## 当前状态

- 核心路线：已按理论依赖重新组织；
- 课程讲解规范：已完成；
- 第 0～2 课正文：已完成第一轮学习和修改，仍会根据阅读反馈继续打磨；
- 第 3 课正文：已重写 Attention 和 RoPE 的讲解，等待学习反馈；
- 第 4 课正文：已完成本轮编写，等待学习反馈；
- 第 5 课正文：已完成本轮编写，等待学习反馈；
- 第 6～9 课正文：待逐课学习、问答和整理；
- 旧版第 1 课和第 2 课：已归档。

这个仓库目前首先用于个人学习和校验。每课经过提问、修正和复核后，会继续整理成其他工程师也能阅读的入门笔记。
