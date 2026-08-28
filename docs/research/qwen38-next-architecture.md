# Qwen3.8-Flash-Next 架构核对记录

这份底稿记录第 10 至第 12 课使用的配置、公式和实现边界。课程正文负责讲解，底稿只负责让数字和结论可追溯。

## 固定版本的资料

- [技术报告，revision `6988587`](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)
- [模型配置，revision `de4b8e4`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
- [Transformers 参考实现，revision `281dd53`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

## 正文使用的配置

| 项目 | 数值 | 配置字段或来源 |
| --- | ---: | --- |
| Decoder Layer | 48 | `num_hidden_layers` |
| Hidden Size | 2560 | `hidden_size` |
| GDN 与 QSA 排列 | 三层 GDN 后一层 QSA | `layer_types` |
| GR 残差支路 | 4 | `hc_count` |
| GR 低秩宽度 | 320 | `hc_lowrank` |
| QSA Query Heads | 24 | `num_attention_heads` |
| QSA KV Heads | 2 | `num_key_value_heads` |
| QSA Head Dimension | 256 | `head_dim` |
| Indexer Query Heads | 4 | `indexer_n_heads` |
| Indexer Shared Key Heads | 1 | `indexer_kv_heads` |
| Indexer Head Dimension | 128 | `indexer_head_dim` |
| Indexer Token Budget | 2048 | `indexer_budget` |
| 微块大小 | 4 token | `indexer_compress_ratio` |
| N-gram 最大阶数 | 3 | `ngram_size` |
| 每阶哈希头 | 8 | `heads_per_ngram` |
| PLE 插入层 | 第 2 层 | `ple_layer_ids=[2]`，实现按一开始计数 |

## 三个容易混淆的边界

1. GR 残差支路沿模型深度传递，但不跨 Decode 轮保存。QSA KV Cache、索引器 Key 和 GDN 状态才是请求历史状态。
2. QSA 的 `2048` 是一次核心 Attention 的读取预算。实现仍保存全部历史正式 K/V，并额外保存索引器原始 Key。
3. N-gram 表增加约 51B 参数，但每个 token 只查询 16 行。它增加的是查表容量和主机内存流量，不是 51B 参数规模的稠密矩阵乘。

## 实现核对

- `Qwen4ExpTextGatedResidual` 分别实现 GR Read 和 GR Write。Read 对四条支路做分组 RMSNorm，经 10240→320→10240 的低秩门控后逐元素混合；Write 为每条支路产生一个 0 到 2 之间的系数。
- `Qwen4ExpTextQSAIndexer` 缓存逐 token 原始 Key。执行时按四个 token 求平均，给压缩块加入 Partial RoPE，再选最多 512 个完整微块并补上末尾未满四个 token 的部分。
- `Qwen4ExpTextPLELayer` 让 N-gram 向量产生每支路 Key 和共享 Value。四条残差支路产生 Query，点积分数经过带符号平方根和 Sigmoid 后门控 Value，再叠加间隔为 3 的逐通道因果卷积。
