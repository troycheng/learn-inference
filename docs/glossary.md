# 课程术语与符号表

正文第一次引入概念时使用“中文（English）”，后续优先使用中文。代码、配置字段和类名保留原始英文，例如 `hidden_size`、`gate_proj`、`input_ids`。

## shape 符号

| 符号 | 中文 | 英文 | 含义 |
| --- | --- | --- | --- |
| `B` | 批大小 | Batch Size | 一批同时处理的序列数 |
| `T` | 序列长度 | Sequence Length | 每条序列当前包含的 token 位置数 |
| `H` | 隐藏维度 | Hidden Size | 每个 token 的公共表示宽度 |
| `I` | 中间维度 | Intermediate Size | FFN 内部的特征宽度 |
| `V` | 词表大小 | Vocabulary Size | 模型可输出的 token 候选数 |

## 数学与张量

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 标量 | Scalar | 一个数；标量 Tensor 的 shape 是 `[]` |
| 向量 | Vector | 一维数值序列 |
| 矩阵 | Matrix | 二维数值表 |
| 张量 | Tensor | 深度学习框架中多维数值数据的总称 |
| 形状 | Shape | 每个轴的长度；正文通常直接写 shape |
| 轴 | Axis / Dimension | shape 中某个维度的位置 |
| 索引 | Index | 定位某个元素或切片的位置编号 |
| 归约 | Reduction | 沿指定轴把多个数汇总为更少的数 |
| 广播 | Broadcasting | 让较小 shape 按规则参与较大 shape 的逐元素计算 |
| 逐元素计算 | Element-wise Operation | 对相同位置的元素分别计算 |
| 点积 | Dot Product | 两个等长向量对应相乘后求和 |
| 矩阵乘法 | Matrix Multiplication / `matmul` | 批量完成多次点积；Python 中常用 `@` |
| 线性层 | Linear Layer / `Linear` | 使用学习权重做线性变换，可选地加偏置 |
| 查表 | Lookup / `Gather` | 按索引从参数表中取指定行 |
| 模型参数 | Parameters | 训练得到、推理时被请求重复使用的数据 |
| 运行时数据 | Runtime Data | 当前请求在推理过程中产生的数据和状态 |

## 从文字到 Token

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 对话模板 | Chat Template | 把角色、消息边界和生成起点组织成模型格式 |
| 分词器 | Tokenizer | 在文字与 Token ID 之间转换 |
| token | Token | Tokenizer 使用的文本片段或特殊符号 |
| Token ID | Token ID / `input_ids` | token 在词表中的整数编号 |
| 嵌入 | Embedding | 根据 Token ID 取得的初始向量 |
| 隐藏状态 | Hidden State | 当前 token 位置经过模型层后的上下文表示 |
| 解码器层 | Decoder Layer | 反复更新 Hidden States 的模型层 |
| 语言模型输出层 | Language Model Head / LM Head | 把 Hidden State 变成全词表分数 |
| 未归一化分数 | Logit | LM Head 对一个候选 token 给出的原始分数 |
| 概率归一化 | Softmax | 把一组 Logits 转成总和为 1 的概率 |
| 最大值索引 | Argmax | 返回最大元素所在的位置 |
| 采样 | Sampling | 按候选概率随机选择 Token ID |
| 温度 | Temperature | 缩放 Logit 差距，改变概率分布的集中程度 |
| 最高 K 项 | Top-K | 只保留分数最高的 K 个候选 |
| 累计概率筛选 | Top-P | 保留累计概率达到阈值所需的最小候选集合 |

## Decoder Layer

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| Token Mixer | Token Mixer | 让不同 token 位置的信息发生联系的子层总称 |
| 前馈网络 | Feed-Forward Network / FFN / MLP | 独立加工每个 token 内部特征的子层 |
| 均方根归一化 | RMSNorm | 根据每个 token 向量的均方根调整尺度 |
| 平方根倒数 | Reciprocal Square Root / `rsqrt` | `1/sqrt(x)` |
| 极小稳定项 | Epsilon / `eps` | 防止除以 0，并降低极小分母的不稳定 |
| 残差连接 | Residual Connection | 把子层输出与原输入逐元素相加 |
| 预归一化 | Pre-Norm | 在子层变换前先做归一化 |
| 线性投影 | Linear Projection | 用 Linear 把一组特征重组为另一组特征 |
| 门控投影 | `gate_proj` | `H→I`，产生经过 SiLU 的调节系数 |
| 扩展投影 | `up_proj` | `H→I`，产生待调节的中间特征 |
| 回收投影 | `down_proj` | `I→H`，混合中间特征并回到层接口宽度 |
| 激活函数 | Activation Function | 给网络加入非线性变换 |
| SiLU | Sigmoid Linear Unit / `SiLU` | `z × sigmoid(z)` |
| SwiGLU | SwiGLU | `SiLU(gate_proj(x))` 与 `up_proj(x)` 逐元素相乘后做 `down_proj` |
| Dense FFN | Dense FFN | 每个 token 使用同一套完整 FFN 参数 |

## 书写约定

- `token` 在正文中通常小写；正式组件名 `Tokenizer`、`Token ID` 保留大写。
- `shape`、`dtype`、`runtime` 等工程语境中常见词保留英文，首次出现时解释中文含义。
- 数学公式使用 `XW^T`；Python/PyTorch 代码使用 `X @ W.T`。
- 逐元素乘法使用 `*`、`×` 或 `⊙`，并在上下文中明确说明；矩阵乘法不使用 `*`。
- 课程始终用 `B/T/H/I/V` 表示上述五个维度。需要把 `B×T` 合并为实现维度时，必须当场说明，不突然引入新符号。
