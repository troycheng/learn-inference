# 可运行的手算示例

这些脚本把正文中的手算步骤写成代码，并沿用同一组数字。所有脚本只依赖 Python 标准库，便于逐项核对。

| 脚本 | 对应课程 | 复算内容 |
| --- | --- | --- |
| [`causal_lm_loss_walkthrough.py`](causal_lm_loss_walkthrough.py) | 第 1 课 | Softmax、正确 token 概率与交叉熵损失 |
| [`decoder_layer_walkthrough.py`](decoder_layer_walkthrough.py) | 第 2 课 | RMSNorm、SwiGLU 的三组投影与门控 |
| [`attention_walkthrough.py`](attention_walkthrough.py) | 第 3 课 | Attention 手算、Q/K 与 V 的分工、因果约束及 RoPE 位置平移 |
| [`gated_deltanet_walkthrough.py`](gated_deltanet_walkthrough.py) | 第 4 课 | Delta Rule、状态衰减、修正幅度与输出门控 |
| [`moe_routing_walkthrough.py`](moe_routing_walkthrough.py) | 第 5 课 | Router Softmax、Top-2、路由权重与 Shared Expert |
| [`full_forward_walkthrough.py`](full_forward_walkthrough.py) | 第 6 课配套实验 | 两层 Decoder、前文与未来 token 的影响、16 种分段的 Logits 与 K/V 等价性 |
| [`model_sizing_walkthrough.py`](model_sizing_walkthrough.py) | 第 8 课 | KV、TP 下每 Rank KV、固定状态与 Attention 长度项 |
| [`quantization_walkthrough.py`](quantization_walkthrough.py) | 数值精度与量化专题 | Scale、舍入、截断、分组及同一个 Linear 的输出误差 |
| [`roofline_walkthrough.py`](roofline_walkthrough.py) | 第 8、9 课 | Linear 算术强度、时间下界、饱和批次与长上下文 KV |
| [`optimization_decision_walkthrough.py`](optimization_decision_walkthrough.py) | 第 9 课 | 局部 Kernel 加速对同一条 TTFT trace 的实际影响 |
| [`tensor_parallel_walkthrough.py`](tensor_parallel_walkthrough.py) | 第 9 课 | Linear 两种切分及 SwiGLU 中间特征分片计算 |
| [`speculative_decoding_walkthrough.py`](speculative_decoding_walkthrough.py) | 第 9 课 | 候选接受、拒绝后的缓存处理、验证位置数与每输出成本 |
| [`gated_residual_walkthrough.py`](gated_residual_walkthrough.py) | 第 10 课 | 两支路 GR Read 的逐元素门控与 GR Write |
| [`qsa_walkthrough.py`](qsa_walkthrough.py) | 第 11 课 | 微块编号、Top-K、尾部候选上限、稀疏 Attention 与数据规模 |
| [`ngram_embedding_walkthrough.py`](ngram_embedding_walkthrough.py) | 第 12 课 | N-gram 哈希、参数表容量与局部卷积的历史复用 |
| [`request_budget_walkthrough.py`](request_budget_walkthrough.py) | 综合案例 | 长上下文的逻辑 KV、TP 下每 Rank KV 与双副本分流 |

在仓库根目录运行：

```bash
python3 examples/causal_lm_loss_walkthrough.py
python3 examples/decoder_layer_walkthrough.py
python3 examples/attention_walkthrough.py
python3 examples/gated_deltanet_walkthrough.py
python3 examples/moe_routing_walkthrough.py
python3 examples/full_forward_walkthrough.py
python3 examples/model_sizing_walkthrough.py
python3 examples/quantization_walkthrough.py
python3 examples/roofline_walkthrough.py
python3 examples/optimization_decision_walkthrough.py
python3 examples/tensor_parallel_walkthrough.py
python3 examples/speculative_decoding_walkthrough.py
python3 examples/gated_residual_walkthrough.py
python3 examples/qsa_walkthrough.py
python3 examples/ngram_embedding_walkthrough.py
python3 examples/request_budget_walkthrough.py
```

每个脚本都会打印中间结果，并用断言检查正文中的关键数值。它们用于复核公式和口径，不是模型 benchmark，也不能替代端到端压测。

完整前向实验默认打印生成结果和对照检查。需要逐层核对向量时，运行 `python3 examples/full_forward_walkthrough.py --trace`。
