# 可运行的手算示例

这些脚本把正文中的手算步骤写成代码，并沿用同一组数字。所有脚本只依赖 Python 标准库，便于逐项核对。

| 脚本 | 对应课程 | 复算内容 |
| --- | --- | --- |
| [`causal_lm_loss_walkthrough.py`](causal_lm_loss_walkthrough.py) | 第 1 课 | Softmax、正确 token 概率与交叉熵损失 |
| [`decoder_layer_walkthrough.py`](decoder_layer_walkthrough.py) | 第 2 课 | RMSNorm、SwiGLU 的三组投影与门控 |
| [`attention_walkthrough.py`](attention_walkthrough.py) | 第 3 课 | QK 点积、因果遮罩、Softmax 与 V 的加权求和 |
| [`gated_deltanet_walkthrough.py`](gated_deltanet_walkthrough.py) | 第 5 课 | Delta Rule、状态衰减、修正幅度与输出门控 |
| [`moe_routing_walkthrough.py`](moe_routing_walkthrough.py) | 第 6 课 | Router Softmax、Top-2、路由权重与 Shared Expert |
| [`model_sizing_walkthrough.py`](model_sizing_walkthrough.py) | 第 8 课 | KV、TP 下每 Rank KV、固定状态与 Attention 长度项 |
| [`optimization_decision_walkthrough.py`](optimization_decision_walkthrough.py) | 第 9 课 | 局部 Kernel 加速对同一条 TTFT trace 的实际影响 |
| [`request_budget_walkthrough.py`](request_budget_walkthrough.py) | 综合评审 | 逻辑状态、TP 下物理 KV、并发容量与可复用前缀 |

在仓库根目录运行：

```bash
python3 examples/causal_lm_loss_walkthrough.py
python3 examples/decoder_layer_walkthrough.py
python3 examples/attention_walkthrough.py
python3 examples/gated_deltanet_walkthrough.py
python3 examples/moe_routing_walkthrough.py
python3 examples/model_sizing_walkthrough.py
python3 examples/optimization_decision_walkthrough.py
python3 examples/request_budget_walkthrough.py
```

每个脚本都会打印中间结果，并用断言检查正文中的关键数值。它们用于复核公式和口径，不是模型 benchmark，也不能替代端到端压测。
