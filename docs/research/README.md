# 研究底稿

这些文件记录课程写作前的资料核对、公式推导和教学方案比较。它们用于追溯正文依据，不属于学习路径；课程内容以 [`docs/lessons`](../lessons) 中的版本为准。

底稿可能保留论文中的另一套向量方向或符号。涉及 Qwen3.5 配置和实现的结论都链接到固定 revision，便于复核。

| 底稿 | 用途 |
| --- | --- |
| [Gated DeltaNet](lesson-04-gated-deltanet.md) | 核对状态更新公式、门控和请求状态 |
| [Dense FFN 与 MoE](lesson-05-dense-moe.md) | 核对 Router、Expert、参数口径和通信边界 |
| [Prefill、Decode 与推理状态](lesson-06-inference-phases-and-state-reuse.md) | 核对生成时间线、KV Cache 和服务指标边界 |
| [多模态输入](lesson-07-multimodal-input.md) | 核对 Patch、视觉编码器、Merger 和 MRoPE |
| [配置与资源估算](lesson-08-config-and-sizing.md) | 核对参数量、状态量和 FLOPs |
| [优化分析](lesson-09-optimization-judgment.md) | 核对常见优化的收益条件和失效边界 |
| [RoPE 讲解方案](rope-teaching-notes.md) | 比较 RoPE 的直观解释与数学边界 |
| [LLMs-from-scratch 教学方法](llms-from-scratch-teaching-review.md) | 分析图、正文、小数字和代码怎样配合 |
