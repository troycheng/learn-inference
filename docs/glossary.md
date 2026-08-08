# 课程术语与符号表

正文第一次出现概念时写出中文和英文，后面优先使用中文。代码、配置字段和类名保留原名，例如 `hidden_size`、`gate_proj`、`input_ids`。

## shape 符号

| 符号 | 中文 | 英文 | 含义 |
| --- | --- | --- | --- |
| `B` | 批大小 | Batch Size | 一批同时处理的序列数 |
| `T` | 序列长度 | Sequence Length | 每条序列当前包含的 token 位置数 |
| `H` | 隐藏维度 | Hidden Size | 每个 token 的公共表示宽度 |
| `I` | 中间维度 | Intermediate Size | FFN 内部的特征宽度 |
| `V` | 词表大小 | Vocabulary Size | 模型可输出的 token 候选数 |
| `Nq` | 查询头数 | Number of Query Heads | Attention 中查询头的数量 |
| `Nkv` | 键值头数 | Number of Key/Value Heads | Attention 中键头和值头的数量 |
| `D` | 头维度 | Head Dimension | 每个 Attention 头包含的特征数 |

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
| 混合专家模型 | Mixture of Experts / MoE | 用 Router 为每个 token 选择少数 FFN 专家执行的稀疏结构 |
| 路由专家 | Routed Expert | 参加 Top-K 选择的一套独立 FFN 参数 |
| 共享专家 | Shared Expert | 不参加 Top-K、对所有 token 固定执行的 FFN |
| 路由器 | Router | 根据当前 Hidden State 为全部路由专家计算分数的 Linear |
| 路由分数 | Router Logit | Router 对每个路由专家给出的原始分数 |
| 路由权重 | Routing Weight | 选中专家的分数归一化后，用于合并专家输出的权重 |
| 最高 K 项路由 | Top-K Routing | 每层只选择路由分数最高的 K 个专家执行 |
| Token 分发 | Token Dispatch | 按 Expert ID 把 token 分组并送到持有相应专家权重的设备或计算组 |
| 分组矩阵乘 | Grouped GEMM | 把多组大小不同的 Expert 矩阵乘组织在一次计算中 |
| 总参数 | Total Parameters | 模型需要保存的全部参数，包括当前 token 未选中的专家 |
| 激活参数 | Active Parameters | 一个 token 本次前向实际使用的参数口径 |
| 专家并行 | Expert Parallelism / EP | 按 Expert ID 把专家权重和计算分到不同设备 |
| 张量并行 | Tensor Parallelism / TP | 切分一个 Linear 或 Expert 内部的权重矩阵并行计算 |

## Attention

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 自注意力 | Self-Attention | Q、K、V 来自同一组隐藏状态的 Attention |
| 查询向量 | Query / Q | 当前位置用于和其他位置比较的向量 |
| 键向量 | Key / K | 每个候选位置用于接受比较的向量 |
| 值向量 | Value / V | 每个位置被加权汇总的信息向量 |
| 相关性分数 | Attention Score | Q 与 K 点积并缩放后得到的原始分数 |
| 注意力权重 | Attention Weight | 分数经过遮罩和 Softmax 后得到的权重 |
| 因果遮罩 | Causal Mask | 让当前位置不能读取未来 token 的约束 |
| 头 | Attention Head | 独立执行一组 Q/K/V 和加权汇总的表示子空间 |
| 多头注意力 | Multi-Head Attention / MHA | 并行使用多个 Attention 头，再拼接结果 |
| 分组查询注意力 | Grouped-Query Attention / GQA | 多个查询头共享一组 K 和 V |
| 旋转位置编码 | Rotary Position Embedding / RoPE | 按各自位置旋转 Q/K，使点积中的位置影响通过两个 token 的相对距离进入 |
| 输出投影 | Output Projection / `o_proj` | 重新混合各头结果并回到 H 维的 Linear |

## 生成过程与请求状态

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 提示词处理 | Prefill | 处理已经给出的 Prompt，建立请求状态并产生预测首个输出 token 的 Logits |
| 逐 token 生成 | Decode | 把已经选出的最新 token 送入模型，更新请求状态并预测下一个 token |
| 键值缓存 | KV Cache | Full Attention 各层保存的历史 K/V，供后续 token 的新 Q 读取 |
| 递归状态 | Recurrent State | 把历史信息持续更新进固定 shape 状态的运行时数据 |
| 前缀缓存 | Prefix Cache | 让具有相同前缀的请求复用已经计算好的前缀状态 |
| 连续批处理 | Continuous Batching | 在模型执行轮次之间移除、保留或加入请求，重新组织 Batch |
| 分块提示词处理 | Chunked Prefill | 把已经给出的长 Prompt 分段计算，并在各段之间延续请求状态 |
| 首 token 延迟 | Time to First Token / TTFT | 从约定的请求起点到首个输出 token 的时间；使用前要确认计时边界 |
| Token 间延迟 | Inter-token Latency / ITL | 相邻两个输出 token 到达时间之差 |
| 每输出 token 时间 | Time per Output Token / TPOT | 通常指排除首 token 后，后续输出 token 的平均间隔 |

## Gated DeltaNet

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 因果卷积 | Causal Convolution | 沿 token 轴组合当前位置和左侧局部窗口，不读取未来位置 |
| 深度卷积 | Depthwise Convolution | 每个通道分别卷积，不在卷积内部混合不同通道 |
| 卷积状态 | Conv State | Decode 时为下一次因果卷积保留的最近局部窗口 |
| 递归状态 | Recurrent State | 每个 Gated DeltaNet 层跨 token 更新的固定 shape 状态矩阵 |
| 状态衰减 | Decay / `alpha` | 在写入当前 token 前缩放旧状态，控制过去保留多少 |
| 修正幅度 | Update Rate / `beta` | 控制当前 Value 与旧记录的误差写回多少 |
| 误差修正规则 | Delta Rule | 先读出状态当前记录，再沿 Key 方向写入它与目标 Value 的差值 |
| 输出门控 | Output Gate / `z` | 在状态读出并归一化后，逐元素调节本层输出 |
| 分块递归计算 | Chunk Gated Delta Rule | 把已知序列的递归更新改写为分块矩阵计算，并在块间传递状态 |

## 多模态输入

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 图像块 | Patch | 从图片或视频帧中切出的固定大小局部像素块 |
| 图像块嵌入 | Patch Embedding | 用可学习投影把局部像素块变成视觉特征向量 |
| 视觉编码器 | Vision Encoder | 在 Patch 位置之间交换信息并加工视觉特征的网络 |
| 视觉位置 | Visual Token / Visual Position | 视觉编码器和 Merger 产生、最终送入语言 Decoder 的向量位置 |
| 视觉特征合并器 | Patch Merger / Merger | 拼接相邻视觉特征，并投影到语言模型 Hidden Size |
| 图片占位符 | Image Placeholder / `image_pad` | 在统一输入序列中标记视觉向量应放入的位置 |
| 多模态旋转位置编码 | Multimodal RoPE / MRoPE | 为同一输入位置提供时间、高度和宽度坐标的 RoPE 变体 |
| 交错式多模态旋转位置编码 | Interleaved MRoPE | 把不同 RoPE 频率交错分配给时间、高度和宽度三个轴 |
| 时间图像块 | Temporal Patch | 由相邻若干视频帧组成的视觉时间单元 |

## 数量与容量

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 参数数量 | Parameter Count | 模型中可学习数字的总数，不包含请求运行状态 |
| 权重有效载荷 | Weight Payload | 参数按指定 dtype 编码后的理想数据字节数 |
| 浮点运算次数 | Floating-Point Operations / FLOPs | 一次计算执行的浮点加、乘等运算数量 |
| 每秒浮点运算次数 | FLOPs per Second / FLOPS | 硬件或程序每秒完成浮点运算的速率 |
| 临时激活 | Temporary Activations | 算子执行期间产生、随后可以释放或复用的中间张量 |
| 权重字节 | Weight Bytes | 参数量乘每参数编码字节数得到的理想权重容量 |
| 有效载荷 | Payload | 不含页表、对齐、块尾空余和运行时预留的逻辑数据量 |

## 优化判断

| 正文用词 | 英文或代码名 | 简短含义 |
| --- | --- | --- |
| 权重量化 | Weight Quantization | 用更低位格式保存和计算模型权重 |
| KV 量化 | KV Cache Quantization | 用更低位格式保存 Full Attention 的历史 K/V |
| 闪存注意力 | FlashAttention | 分块在片上计算精确 Attention，减少中间矩阵的 HBM 读写 |
| 语义缓存 | Semantic Cache | 根据输入语义相似性复用最终结果或上层结果，不等同于 Prefix Cache |
| 静态批处理 | Static Batching | 一组请求固定组成 Batch，通常整批结束后才补入新请求 |
| 集合通信 | Collective Communication | 多设备共同参与的 All-Reduce、All-to-All 等通信操作 |
| 推测解码 | Speculative Decoding | 用 Drafter 提出多个候选，再由 Target Model 一次验证多个位置 |
| 草稿模型 | Drafter | 在推测解码中以较低成本提出候选 token 的模型或辅助模块 |
| 目标模型 | Target Model | 决定候选是否接受并保持最终输出分布的主模型 |
| 多 Token 预测 | Multi-Token Prediction / MTP | 训练辅助模块预测更远 token，可作为推测解码 Drafter |
| 接受长度 | Acceptance Length | 一轮推测解码中被目标模型连续接受的候选 token 数 |
| 服务等级目标 | Service Level Objective / SLO | 对延迟、吞吐或可用性等指标设定的目标边界 |
| 回退路径 | Fallback | 预期优化 Kernel 不可用时转而执行的通用或较慢实现 |

## 书写约定

- `token` 在正文中通常小写；正式组件名 `Tokenizer`、`Token ID` 保留大写。
- `shape`、`dtype`、`runtime` 等工程语境中常见词保留英文，首次出现时解释中文含义。
- 数学公式使用 `XW^T`；Python/PyTorch 代码使用 `X @ W.T`。
- 逐元素乘法使用 `*`、`×` 或 `⊙`，并在上下文中明确说明；矩阵乘法不使用 `*`。
- 课程统一使用本表中的 shape 符号。某一课第一次使用新符号时，必须先说明含义。需要把多个轴合并为实现维度时，也要当场解释。
