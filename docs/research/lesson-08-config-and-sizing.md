# 第 8 课研究笔记：从配置还原 Qwen3.5 的结构和数量级

这份笔记给第 8 课准备一套可复核的计算口径。目标不是背参数，而是看到 `config.json` 后，能够回答四件事：模型由什么组成，权重为什么这么大，单个 token 实际经过多少参数，序列状态随长度怎样增长。

固定检查点：

- Dense：[Qwen3.5-9B，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
- MoE：[Qwen3.5-35B-A3B，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754)
- 实现：[Transformers，revision `9436284`](https://github.com/huggingface/transformers/tree/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models)

## 1. 先统一符号和口径

| 符号 | 配置字段 | 含义 |
| --- | --- | --- |
| `V` | `vocab_size` | 词表行数 |
| `H` | `hidden_size` | 每个语言模型位置的向量宽度 |
| `I` | `intermediate_size` | Dense FFN 的中间宽度 |
| `L` | `num_hidden_layers` | Decoder Layer 数量 |
| `Lfull` | 从 `layer_types` 计数 | Full Attention 层数 |
| `Nq` | `num_attention_heads` | Query 头数 |
| `Nkv` | `num_key_value_heads` | K/V 头数 |
| `D` | `head_dim` | 每个 Attention 头的宽度 |
| `E` | `num_experts` | Routed Expert 总数 |
| `K` | `num_experts_per_tok` | 每个 token 选中的 Routed Expert 数 |
| `Imoe` | `moe_intermediate_size` | 每个 Routed Expert 的中间宽度 |

本笔记区分下面几个容易混用的量：

```text
参数数量：模型中一共有多少个可学习数字
权重字节：这些数字按某种 dtype 保存后占多少空间
激活参数：本轮 token 实际经过的那部分参数
FLOPs：本轮进行了多少次浮点运算
推理状态：请求运行期间保存的 KV、卷积状态和递归状态
临时激活：算子执行中产生、随后可以释放的中间张量
```

“35B 参数”“3B 激活”“67 GiB BF16 权重”“一次 Decode 约 6 GFLOPs”不是同一个概念。

## 2. 两个检查点的主配置

| 字段 | Qwen3.5-9B | Qwen3.5-35B-A3B |
| --- | ---: | ---: |
| `V` | 248,320 | 248,320 |
| `H` | 4096 | 2048 |
| `L` | 32 | 40 |
| Gated DeltaNet 层 | 24 | 30 |
| Full Attention 层 `Lfull` | 8 | 10 |
| `Nq` | 16 | 16 |
| `Nkv` | 4 | 2 |
| `D` | 256 | 256 |
| Dense `I` | 12,288 | 不适用 |
| `E` | 不适用 | 256 |
| `K` | 不适用 | 8 |
| `Imoe` | 不适用 | 512 |
| Shared Expert 中间宽度 | 不适用 | 512 |
| MTP 层数 | 1 | 1 |
| 配置 dtype | BF16 | BF16 |

两者都按下面的 3:1 模式重复：

```text
Gated DeltaNet
Gated DeltaNet
Gated DeltaNet
Full Attention
```

因此 32 层得到 24 个 Gated DeltaNet 加 8 个 Full Attention；40 层得到 30 加 10。不要把 `full_attention_interval=4` 解释成“四层 Full Attention”，最终应直接检查 `layer_types` 列表。[9B 配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json) [35B-A3B 配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)

## 3. Dense 9B 的参数从哪里来

### 3.1 输入 Embedding 和 LM Head

输入 Embedding 表有：

$$
V\times H=248320\times4096=1,017,118,720
$$

Qwen3.5 配置中 `tie_word_embeddings=false`，所以 LM Head 还有一份同样大小、但参数独立的矩阵。两者合计约 2.034B 参数。

输入查 Embedding 时，只读取当前 Token ID 对应的行，不会对整张 1.017B 参数表做矩阵乘法。LM Head 则需要把 Hidden State 投影到完整词表，通常会使用整张输出矩阵。这是“参数数量不能直接等同于每 token FLOPs”的第一个例子。

### 3.2 Dense SwiGLU FFN

每层有 `gate_proj`、`up_proj` 和 `down_proj` 三个矩阵：

$$
P_{FFN}=H\times I+H\times I+I\times H=3HI
$$

代入 9B：

$$
3\times4096\times12288=150,994,944
$$

32 层合计：

$$
4,831,838,208\approx4.832B
$$

因此 Dense 模型里，FFN 通常是最大的 Decoder 参数来源。[Dense FFN 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L706-L717)

### 3.3 Full Attention 投影

Qwen3.5 的 Query 投影还同时产生一个输出门控，所以 `q_proj` 输出宽度是 `2 x Nq x D`。每个 Full Attention 层的主要参数为：

$$
P_{full}=H(2N_qD)+H(N_{kv}D)+H(N_{kv}D)+(N_qD)H
$$

再加 Q/K Norm 的少量参数。代入 9B 得每层 `58,720,768`，8 层约 `469.77M`。[Full Attention 定义](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L627-L700)

### 3.4 Gated DeltaNet 投影

9B 的线性 Attention 配置为：

```text
Key 头：16 x 128 = 2048
Value 头：32 x 128 = 4096
Q/K/V 卷积通道：2048 + 2048 + 4096 = 8192
```

每个 Gated DeltaNet 层包含 Q/K/V、门控 `z`、更新系数 `a/b`、输出投影和深度卷积等参数。按官方实现逐项相加，每层 `67,403,968`，24 层约 `1.618B`。[Gated DeltaNet 定义](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L387-L540)

### 3.5 与检查点总数对账

按配置和源码还原得到：

| 部分 | 参数数量 |
| --- | ---: |
| 文本模型、输入 Embedding、LM Head | 8,953,799,424 |
| 视觉编码器与 Merger | 456,010,480 |
| MTP 辅助层 | 243,294,464 |
| **检查点合计** | **9,653,104,368** |

官方 Safetensors API 统计为 `9,653,100,528` 个 BF16 参数和 `3,840` 个 FP32 参数，总计 `9,653,104,368` 个参数。对应 tensor payload 是 `19,306,216,416` 字节。不能直接用总字节除以 2，因为 FP32 参数每个占 4 Byte。[9B Safetensors 索引](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/model.safetensors.index.json) [9B 模型 API，revision `c202236`](https://huggingface.co/api/models/Qwen/Qwen3.5-9B/revision/c202236235762e1c871ad0ccb60c8ee5ba337b9a?expand%5B%5D=safetensors)

“Qwen3.5-9B”是产品级取整名称，不表示权重文件里恰好有 9,000,000,000 个数。

## 4. 35B-A3B 为什么总参数大、每 token 激活少

### 4.1 一层保存的全部专家

每个 Routed Expert 都是一个中间宽度为 512 的 SwiGLU FFN：

$$
P_{one\ expert}=3H I_{moe}
$$

35B-A3B 有 256 个 Routed Expert，因此一层仅 Routed Expert 权重就有：

$$
256\times3\times2048\times512=805,306,368
$$

再加 Shared Expert、Router 和 Shared Expert Gate，每层 MoE 子层共 `808,978,432` 个参数。40 层约 `32.359B`。这解释了 35B 总参数的主要来源。[MoE Expert、Router 与 Shared Expert 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L704-L796)

### 4.2 一个 token 只经过其中一部分

每个 token 选择 8 个 Routed Expert，并且始终经过 1 个 Shared Expert。该层实际使用的主要 FFN 权重约为：

$$
P_{active\ FFN}=3H I_{moe}(8+1)+H E+H
$$

代入配置：

$$
28,837,888\approx28.84M
$$

其中 `HE` 是 Router，最后的 `H` 是 Shared Expert Gate。40 层合计约 `1.154B` 激活 FFN 参数，远小于保存的 `32.359B` 专家参数。

“A3B”不表示每个 token 只读一个 3B 大专家，也不表示显存只需保存 3B 参数。它是全链路激活参数的取整口径：8 个 Routed Expert、Shared Expert、Token Mixer、LM Head 等共同构成单 token 的有效计算路径。

### 4.3 对账

| 部分 | 参数数量 |
| --- | ---: |
| 文本模型、全部专家、Embedding、LM Head | 34,660,605,888 |
| 视觉编码器与 Merger | 446,571,248 |
| MTP 辅助层 | 844,645,568 |
| **检查点合计** | **35,951,822,704** |

官方 Safetensors API 统计为 `35,951,817,904` 个 BF16 参数和 `4,800` 个 FP32 参数，总计 `35,951,822,704` 个参数。对应 payload 是 `71,903,655,008` 字节。这里同样不能把总字节直接除以 2。[35B-A3B Safetensors 索引](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/model.safetensors.index.json) [35B-A3B 模型 API，revision `59d61f3`](https://huggingface.co/api/models/Qwen/Qwen3.5-35B-A3B/revision/59d61f3ce65a6d9863b86d2e96597125219dc754?expand%5B%5D=safetensors)

官方模型卡将其写成“35B total and 3B activated”，属于便于交流的取整值。[35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)

## 5. 权重容量怎样估算

理想的权重 payload 公式很简单：

$$
Weight\ Bytes=P\times bytes\_per\_parameter
$$

| 模型 | 参数数 | BF16，2 Byte | INT8 理想值，1 Byte | INT4 理想值，0.5 Byte |
| --- | ---: | ---: | ---: | ---: |
| 9B 检查点 | 9.653B | 17.98 GiB | 8.99 GiB | 4.50 GiB |
| 35B-A3B 检查点 | 35.952B | 66.97 GiB | 33.48 GiB | 16.74 GiB |

INT8/INT4 两列只表示编码数据的理论下限。实际量化检查点还有 scale、zero point、分组元数据、对齐和未量化层，不能直接把 BF16 文件大小除以 2 或 4 当成最终显存。

正常文本推理也未必加载检查点所有组件：

- 文本专用模式可以不加载视觉编码器。
- MTP 权重只在启用相应推测解码路径时需要参与计算。
- MoE 的全部专家必须被某些设备保存，但每个 token 只经过选中的专家。

所以还要区分“仓库检查点总大小”“进程实际加载的权重”和“单 token 激活的权重”。

## 6. Full Attention 的 KV Cache

等长 Batch 的逻辑 KV payload 为：

$$
KV\ Bytes=2\times B\times L_{full}\times N_{kv}\times T\times D\times s
$$

最前面的 2 表示 K 和 V，`s` 是每个缓存元素字节数。

### 9B，BF16 KV

$$
2\times8\times4\times256\times2=32768\ Bytes=32\ KiB
$$

即每个请求每增加一个缓存位置，逻辑 KV 增加 32 KiB。

### 35B-A3B，BF16 KV

$$
2\times10\times2\times256\times2=20480\ Bytes=20\ KiB
$$

| 缓存长度 | 9B | 35B-A3B |
| ---: | ---: | ---: |
| 4,096 | 128 MiB | 80 MiB |
| 131,072 | 4 GiB | 2.5 GiB |

35B 模型更大，却因为 `Nkv=2`，每 token 的 KV 反而小于 9B。模型参数名不能替代 KV shape 计算。

这些数是单请求逻辑有效载荷，不含块尾空余、预分配、页表、对齐、跨设备布局和临时 workspace。视觉 token 进入统一 Decoder 序列，也会计入 Prefill 后的状态长度。

## 7. Gated DeltaNet 状态不是 KV Cache

9B 的每个 Gated DeltaNet 层按官方实现保存：

```text
conv_state：[B, 8192, 4]
recurrent_state：[B, 32, 128, 128]
```

35B-A3B 的线性 Attention 头配置相同，所以单层 shape 也相同。区别是 9B 有 24 层，35B-A3B 有 30 层。

Transformers 参考实现把递归计算转成 FP32，卷积状态随模型输入 dtype。按 BF16 卷积状态加 FP32 递归状态估算：

```text
单层 conv_state：8192 x 4 x 2 Byte = 64 KiB
单层 recurrent_state：32 x 128 x 128 x 4 Byte = 2 MiB
单层合计：约 2.0625 MiB

9B 的 24 层：约 49.5 MiB / 请求
35B-A3B 的 30 层：约 61.9 MiB / 请求
```

这部分不会随着序列长度线性增长，但会随着并发请求数增长。实际高性能 Kernel 可以使用不同 dtype 或布局，部署估算必须以 runtime 的真实分配为准。[Gated Delta Rule 参考实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L249-L380) [Linear Attention Cache](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L998-L1086)

## 8. 主要 FLOPs 怎样估算

### 8.1 Linear 的通用公式

输入 `[M,K]` 乘权重 `[K,N]`：

$$
FLOPs\approx2MKN
$$

乘和加各算一次。对单 token Decode，`M` 是这一轮一起处理的 token 数；单请求单步时可以先取 `M=1`。

### 8.2 Dense FFN

单层单 token：

$$
FLOPs_{FFN}\approx 2\times3HI=6HI
$$

9B 代入后每层约 `0.302 GFLOPs`，32 层 FFN 合计约 `9.66 GFLOPs/token`。

### 8.3 MoE FFN

35B-A3B 每 token 会计算 8 个 Routed Expert 和 1 个 Shared Expert：

$$
FLOPs_{MoE\ FFN}\approx2\times3H I_{moe}(8+1)
$$

Router 与门控还会增加少量计算。关键点是 FLOPs 跟激活的 9 条专家路径相关，不跟 256 个专家总数成正比；权重容量却跟全部 256 个专家相关。

### 8.4 一步文本 Decode 的线性投影数量级

将本步真正使用的 Linear/卷积权重按“每个权重约一次乘加”计算，并加入 LM Head：

```text
Qwen3.5-9B：约 15.9 GFLOPs / token
Qwen3.5-35B-A3B：约 5.9 GFLOPs / token
```

35B-A3B 的 `5.9G` 对应约 `2.946B` 个活跃线性权重，和官方“A3B”取整名称相符。这两个数没有包含 Full Attention 读取历史 K/V 的 QK 与 AV 计算、Gated DeltaNet 状态更新中的非 Linear 运算、采样、通信和 Kernel 开销。

### 8.5 Attention 的长度项

单个 Decode Query 读取长度为 `T` 的历史时，每个 Full Attention 层的 QK 和 AV 约为：

$$
FLOPs_{attn}\approx4N_qDT
$$

所有 Full Attention 层：

$$
4L_{full}N_qDT
$$

代入两个模型：

```text
9B：131,072 x T FLOPs
35B-A3B：163,840 x T FLOPs
```

| 历史长度 `T` | 9B Attention 长度项 | 35B-A3B Attention 长度项 |
| ---: | ---: | ---: |
| 4,096 | 0.54 GFLOPs | 0.67 GFLOPs |
| 131,072 | 17.18 GFLOPs | 21.47 GFLOPs |

长上下文下，Attention 的长度项可以超过其余单 token 线性投影。Qwen3.5 只有四分之一层是 Full Attention，所以上式必须使用 `Lfull`，不能直接用全部 Decoder Layer 数。

Prefill 中，Linear/FFN 主要随 token 数 `T` 线性增长；Full Attention 的 QK/AV 总运算随 `T^2` 增长。因果遮罩和具体 Kernel 会改变常数，但不会让标准 Full Attention 的长度依赖消失。

## 9. 为什么不能机械使用 `2 x 参数量`

“一次前向约等于两倍参数量 FLOPs”只适合做粗略口径，前提是大部分活跃 Linear 权重对每个 token 使用一次。Qwen3.5 中至少有六个例外：

1. 输入 Embedding 是查表，不是整张矩阵乘法。
2. LM Head 使用完整词表矩阵，而且与输入 Embedding 不共享。
3. MoE 保存 256 个专家，每 token 只计算其中 8 个加 Shared Expert。
4. Attention 的 QK/AV FLOPs 随上下文长度变化，不对应新的模型参数。
5. Gated DeltaNet 的递归状态计算不只由参数量决定。
6. 视觉编码器和 MTP 是否运行，取决于输入和服务配置。

工程判断时，先列模块和 shape，再分别估算权重、状态、FLOPs 和通信，比直接用模型名称乘一个常数可靠。

## 10. 最容易讲错的地方

1. **35B-A3B 的显存不是 3B 参数大小。** 全部专家仍要存储，只是每 token 激活其中一部分。
2. **8 个 Routed Expert 之外还有 Shared Expert。** 该模型不是简单 Top-8 后结束。
3. **Qwen3.5 的 q_proj 比普通 Q 投影宽一倍。** 另一半产生 Attention 输出门控，参数公式不能照搬普通 GQA。
4. **总层数不能代入 KV 公式。** 只有 8 或 10 个 Full Attention 层保存随长度增长的 KV。
5. **Gated DeltaNet 状态固定 shape，但不是零成本。** 它按并发请求数增长，且 FP32 递归状态可达到每请求几十 MiB。
6. **总参数包含视觉塔和 MTP 权重。** 纯文本基础解码未必使用它们。
7. **理想 INT4 容量不等于实际量化显存。** 还要加入 scale、元数据、未量化层和 workspace。
8. **FLOPs 不是延迟。** 权重搬运、Kernel 效率、通信、Batch 和硬件峰值共同决定时间。

## 资料来源

- [Qwen3.5-9B 官方配置，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
- [Qwen3.5-9B 官方模型卡，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)
- [Qwen3.5-9B Safetensors 索引，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/model.safetensors.index.json)
- [Qwen3.5-35B-A3B 官方配置，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
- [Qwen3.5-35B-A3B 官方模型卡，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)
- [Qwen3.5-35B-A3B Safetensors 索引，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/model.safetensors.index.json)
- [Transformers Qwen3.5 Dense 实现，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Transformers Qwen3.5 MoE 实现，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
- [Transformers Cache 实现，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py)
