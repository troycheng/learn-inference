# 大模型推理原理与优化

面向推理系统工程师的模型原理入门课。

这套课写给正在维护推理系统、希望补齐模型原理的工程师。你可能已经处理过显存不足、吞吐下降或首 token 变慢，但碰到 QKV、RoPE、KV Cache 或 MoE 时，还很难从模型计算解释原因。

课程从一条输入开始，沿生成过程解释文字怎样变成 token、token 怎样经过 Decoder、模型怎样读取前文，以及回答为什么只能逐个 token 产生。学完后，你应该能把 Prefill、Decode、KV Cache、Dense、MoE 和常见优化放回具体的模型层、张量与请求状态中分析。

## 学习目标

这 10 课不要求你推导复杂公式，但要能做到：

- 从 Tokenizer 开始，完整解释一个新 token 是怎样产生的；
- 说明 Embedding、RMSNorm、Attention、FFN、Residual、LM Head 分别解决什么问题；
- 看懂 Qwen3.5 文本模型的一层结构和完整层排列；
- 解释 Dense 与 MoE 的共同骨架，以及 Router、Top-K、共享专家的作用；
- 解释 Prefill、Decode、KV Cache 和 Gated DeltaNet 状态之间的关系；
- 解释图片怎样经过视觉编码器变成语言模型可处理的视觉特征；
- 阅读模型 `config.json`，区分保存与计算 dtype，判断参数规模、状态规模和主要计算来自哪里；
- 判断量化、缓存、批处理和 DP、TP、PP、EP 究竟改变了模型链路中的什么。

## 课程主线

```text
用户输入 → 模型输入向量 → 多层 Decoder → Logits → 下一个 token → 下一轮 Decode
```

前几课先讲文字生成，第 7 课再加入图片和视频。两类输入最终都会变成与语言模型 Hidden Size 对齐的向量，进入同一个 Decoder 主干。

需要复习完整数据流、Decoder Layer 内部结构和两类请求状态时，可以打开带图的[大模型推理链路速查](docs/inference-map.md)。

课程一直用 Qwen3.5 作例子，但会区分模型版本的用途：

- 用 **Qwen3.5-9B-Base** 的配置讲 Dense 架构、层数和 shape；
- 讲 Chat Template、Tokenizer、多模态输入或生成行为时，使用对应的 **post-trained Qwen3.5-9B** 文档，不与 Base 的架构配置混用；
- 用 **Qwen3.5-35B-A3B** 讲 MoE 模型；
- 用两者共同的 Gated DeltaNet 与 Full Attention 混合结构讲现代模型；
- 用 Qwen3.5 的视觉输入路径讲视觉编码器与多模态序列；
- 第一轮只解释因果语言模型目标，不展开反向传播、优化器、CUDA 编程和框架参数清单。

## 课程目录

| 课次 | 内容 | 读完后能做什么 |
| ---: | --- | --- |
| 0 | [第 0 课：张量与模型计算基础](docs/lessons/00-math-and-tensors.md) | 根据轴和运算推导 shape |
| 1 | [第 1 课：大模型生成下一个 token 的过程](docs/lessons/01-text-to-next-token.md) | 区分文本、Token ID、向量、Logit 和概率，解释因果训练目标 |
| 2 | [第 2 课：Decoder Layer 的结构与计算](docs/lessons/02-inside-a-decoder-layer.md) | 解释 RMSNorm、残差连接和 SwiGLU FFN |
| 3 | [第 3 课：Attention 的计算原理](docs/lessons/03-attention.md) | 手算一次 Attention 并解释 RoPE、GQA |
| 4 | [第 4 课：Prefill、Decode 与 KV Cache](docs/lessons/04-prefill-decode-kv-cache.md) | 解释生成阶段、缓存复用和批处理 |
| 5 | [第 5 课：Gated DeltaNet 的状态更新机制](docs/lessons/05-gated-deltanet.md) | 区分固定状态与随长度增长的 KV Cache |
| 6 | [第 6 课：Dense FFN 与 MoE 的结构差异](docs/lessons/06-dense-and-moe.md) | 解释 Router、Top-K、共享专家和激活参数 |
| 7 | [第 7 课：多模态输入与视觉编码](docs/lessons/07-multimodal-input.md) | 从像素和 Patch 推导到 Decoder 输入 |
| 8 | [第 8 课：模型配置与资源估算](docs/lessons/08-config-and-sizing.md) | 估算权重、请求状态和主要计算量 |
| 9 | [第 9 课：推理优化的分析与评估](docs/lessons/09-optimization-judgment.md) | 判断优化改了什么、何时有效、怎样验证 |

配套资料：

- [完整课程路线](docs/roadmap.md)
- [大模型推理链路速查](docs/inference-map.md)
- [结业案例：从模型配置到优化判断](docs/capstone.md)
- [课程术语与符号表](docs/glossary.md)
- [模型分析工作表](docs/model-analysis-workbook.md)
- [课程编写与讲解规范](docs/teaching-method.md)

## 学习顺序

第 0 课补齐后文会用到的数学动作。第 1 至第 4 课是一条连续主线，先把文字生成和自回归过程读通；第 5 至第 7 课再加入 Gated DeltaNet、MoE 和图片输入；第 8、9 课把模型原理变成配置估算和优化判断。

建议读两遍。第一遍只追踪数据表示、shape 变化和请求需要保留的状态，不必记住 Qwen3.5 的每个配置数字。第二遍再带着真实模型和工作负载回来，用复算程序核对参数、状态和计算量，并完成每课末尾的综合练习。

读完第 9 课后，用[结业案例](docs/capstone.md)完成一次不按章节提示的综合分析。再拿一份自己正在部署的模型配置，填写[模型分析工作表](docs/model-analysis-workbook.md)。能把每个数字追溯到配置、源码或测量结果，才算把课程方法用到了真实问题中。

正文用小数字缩短手算过程，向量和矩阵的维度虽然变小，计算顺序与真实模型相同。每课末尾都有练习和一个综合自测，答案默认折叠。如果能在不看正文的情况下画出数据流并标对 shape，说明这一课已经掌握。

## 可运行示例

第 1、2、3、5、6、8、9 课提供了只依赖 Python 标准库的[复算程序](examples/README.md)。脚本与正文使用同一组数字，运行后会打印中间结果，并检查关键数值。它们适合在读完公式后自己算一遍，不要求安装 PyTorch 或下载模型权重。

提交前可以运行 `bash scripts/check-course.sh`，检查 Markdown 结构、本地链接、SVG 渲染和可运行示例。GitHub Actions 也会执行同一组检查。

早期草稿保存在[历史草稿目录](docs/archive/README.md)，只记录课程的迭代过程，不再作为主课材料。

## 当前状态

第 0～9 课已经形成完整主线。主课只保留经过资料核对、手算校验和图文检查的版本。

## 反馈与纠错

发现公式、配置数字、图示或引用有误时，请提交[内容纠错](https://github.com/troycheng/learn-inference/issues/new?template=content-error.yml)，注明课程位置、问题和可核对的来源。读到某一步无法继续，也可以提交[学习问题](https://github.com/troycheng/learn-inference/issues/new?template=learning-question.yml)。这类反馈会用来判断正文是否还缺少必要解释。

修改课程前请阅读[贡献说明](CONTRIBUTING.md)。

## 许可

课程文字与仓库中的原创图示采用 [CC BY 4.0](LICENSE)。转载或改编时，请署名、链接许可证并标明修改。引用的第三方资料仍归原作者，详见各课资料来源。
