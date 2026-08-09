# 第 7 课研究笔记：Qwen3.5 怎样把图片和视频送入 Decoder

这份笔记只核实课程需要讲清的推理链路，不展开视觉模型训练。结论来自 Qwen3.5 官方检查点、Transformers 固定 revision 的实现，以及 Qwen3-VL 原始技术报告。

## 核心结论

图片不会先被“识别成文字”再交给语言模型。Qwen3.5 的真实路径是：

```text
图片像素
  -> 调整尺寸、归一化
  -> 切成 16 x 16 的空间 Patch
  -> 27 层视觉编码器
  -> 每 2 x 2 个相邻视觉特征合并成 1 个
  -> 投影到语言模型 Hidden Size
  -> 替换输入序列中的 <|image_pad|> 占位位置
  -> 与文本向量一起进入同一个 Decoder
```

因此，文本 token 和视觉 token 的来源不同，但进入语言模型前会得到相同的最后一维：

```text
Qwen3.5-9B
文本 Embedding：[文本位置数, 4096]
视觉编码结果：[视觉位置数, 4096]

Qwen3.5-35B-A3B
文本 Embedding：[文本位置数, 2048]
视觉编码结果：[视觉位置数, 2048]
```

模型随后处理的是一条混合序列。视觉位置越多，视觉编码器的工作越多，语言模型的 Prefill 也越长。

## 1. 官方配置给出的视觉塔

Qwen3.5-9B 和 Qwen3.5-35B-A3B 的视觉配置相同，只有最后投影宽度随文本 Hidden Size 改变：

| 配置字段 | 9B | 35B-A3B | 含义 |
| --- | ---: | ---: | --- |
| `depth` | 27 | 27 | 视觉 Transformer Block 数量 |
| `hidden_size` | 1152 | 1152 | 视觉编码器内部宽度 |
| `intermediate_size` | 4304 | 4304 | 视觉 MLP 中间宽度 |
| `num_heads` | 16 | 16 | 视觉 Attention 头数 |
| `patch_size` | 16 | 16 | 每个空间 Patch 的边长 |
| `temporal_patch_size` | 2 | 2 | 一个时间 Patch 覆盖的帧数 |
| `spatial_merge_size` | 2 | 2 | 最后合并 2 x 2 个空间特征 |
| `out_hidden_size` | 4096 | 2048 | 输出到语言模型的宽度 |

来源：[Qwen3.5-9B 配置，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)；[Qwen3.5-35B-A3B 配置，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)

Qwen3.5 的配置把 `deepstack_visual_indexes` 设为空数组。这一点很重要：Qwen3-VL 论文介绍了把多个视觉层的特征注入语言模型的 DeepStack，但这两个 Qwen3.5 检查点没有启用该路径。课程不应照搬 Qwen3-VL 架构图，把 DeepStack 画进 Qwen3.5-9B 或 35B-A3B 的实际前向。

## 2. 图像预处理不是固定缩放到一个正方形

官方 `preprocessor_config.json` 给出：

```text
patch_size = 16
merge_size = 2
temporal_patch_size = 2
image_mean = [0.5, 0.5, 0.5]
image_std  = [0.5, 0.5, 0.5]
最小像素数 = 65,536
最大像素数 = 16,777,216
```

处理器会尽量保持原图宽高比，同时让调整后的高度和宽度都能被下面这个数整除：

$$
16 \times 2 = 32
$$

这里的 `16` 是 Patch 边长，`2` 是后面的空间合并倍数。图像因此可以保留不同宽高比和不同分辨率，不是所有图片都被压成同样数量的视觉位置。[Qwen3.5-9B 预处理配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/preprocessor_config.json) [Transformers 图像缩放与 Patch 化实现，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L62-L198)

### 一个能手算的例子

若图片调整后为 `512 x 512`：

```text
空间 Patch 网格：512/16 x 512/16 = 32 x 32
视觉编码器输入位置数：32 x 32 = 1024
2 x 2 合并后的视觉位置数：1024 / 4 = 256
```

这 256 个视觉位置会进入语言模型。再加上文本 token 和 `<|vision_start|>`、`<|vision_end|>` 等特殊 token，才是完整 Prefill 序列长度。

不要把“Patch 数”和“最终视觉 token 数”混为一谈。空间合并发生在视觉编码之后，`merge_size=2` 会把最终位置数降为原来的四分之一。

静态图片没有两个不同时间帧。官方图像处理实现会在 Patch 数据中复制静态图像的像素，以满足 `temporal_patch_size=2` 的卷积输入形状；它不会因此产生两倍的最终图片 token。[静态图像 Patch 化实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L170-L198)

## 3. Patch Embed 做了什么

视觉塔先用一个三维卷积完成 Patch Embedding：

```text
卷积核：[时间 2, 高 16, 宽 16]
输入通道：3
输出宽度：1152
```

对每个时间和空间小块，这个卷积把 `2 x 16 x 16 x 3` 个像素值重新组合成一条 1152 维向量。它和文本 Embedding 的共同点是“把离散输入变成向量”，区别是：

```text
文本：Token ID 查 Embedding 表
图片：Patch 像素经过可学习的卷积投影
```

随后，27 个视觉 Block 在这些 Patch 向量之间做视觉 Attention 和 MLP 计算。[Qwen3.5 视觉 Patch Embed 与视觉模型](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L839-L857) [视觉塔前向](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1002-L1112)

## 4. Merger 同时承担压缩和对齐

视觉编码器内部宽度是 1152，语言模型需要 4096 或 2048 维。`Qwen3_5VisionPatchMerger` 完成两件事：

1. 把同一 `2 x 2` 区域的四条 1152 维特征拼成一条 4608 维特征。
2. 经过两层 Linear 和 GELU，输出语言模型所需的 Hidden Size。

以 9B 为例：

```text
4 x 1152 = 4608
4608 -> 4608 -> 4096
```

35B-A3B 的最后一步则是 `4608 -> 2048`。所以课程可以把 Merger 解释为“视觉特征合并器兼投影器”，但不要画成简单平均池化，也不要说只是删掉四分之三的 Patch。[视觉 Patch Merger 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L859-L874)

## 5. 视觉特征怎样进入文本序列

聊天模板先为一张图片放入占位结构：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

Processor 得到实际图片网格后，把单个 `<|image_pad|>` 扩展成与视觉特征数量相同的占位 token。图片 token 数的公式是：

$$
T_{image}=\frac{T_{grid}\times H_{grid}\times W_{grid}}{merge\_size^2}
$$

静态图片的 `T_grid=1`。模型先为整条 `input_ids` 查文本 Embedding，再用视觉编码结果替换所有图片占位位置的向量。代码还会检查占位数量与视觉特征数量是否完全一致。[Qwen3VL Processor 占位扩展](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L40-L86) [Qwen3.5 占位检查与替换](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1393-L1542)

准确的说法是“视觉向量替换了统一输入序列中的占位向量”。它们不是追加在文本序列之外的另一条旁路，也不是普通词表 Token ID 查出的 Embedding。

## 6. 视频与图片共用视觉塔，但前处理不完全相同

`get_video_features` 直接复用 `get_image_features`，说明图片和视频进入同一套视觉编码器。差别主要发生在 Processor：

- 视频先选取或提供一组帧。
- 相邻 `temporal_patch_size=2` 帧组成时间 Patch。
- 每个时间 Patch 仍按空间 Patch 和 `2 x 2` Merger 处理。
- Qwen3.5/Qwen3-VL 在每组视频视觉 token 前插入可读的文本时间戳，例如 `<1.5 seconds>`。

如果视频元数据没有 FPS，Transformers 当前实现会警告并默认按 24 FPS 计算时间戳。因此，“帧下标”不能天然等同于“真实秒数”；正确 FPS 是时间定位准确的前提。[视频占位与时间戳实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L80-L111) [Qwen3.5 视频特征复用](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1380-L1406)

视频视觉位置数大致随“采样后时间 Patch 数 x 每帧空间位置数”增长。视频耗时不能只看文件时长，还要看采样帧数、每帧分辨率以及空间合并后的总视觉 token 数。

## 7. MRoPE 为什么需要三个位置坐标

纯文本只有一条先后顺序：第 0、1、2、3 个 token。图片还有上下左右，视频还涉及时间。Qwen3.5 因此为语言模型构造三组位置编号：

```text
position_ids shape = [3, B, T]
三行分别承载时间、高度、宽度坐标
```

文本 token 没有二维网格，三组坐标使用同一个文本位置编号。图片 token 的时间坐标不变，高度和宽度坐标按二维网格变化。后续文本位置从视觉区域占用的位置范围之后继续。

这三组数字不会让 token 数变成三倍。它们会作用在 RoPE 的不同旋转维度上，让 Attention 的 Q/K 同时带上顺序和空间位置信息。[Qwen3.5 三维位置构造](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1280-L1376)

### Interleaved MRoPE 的意义

早期 MRoPE 把一段连续的频率维度分给时间，再把另外两段分给高度和宽度。Qwen3-VL 报告指出，这会让三个轴获得不均衡的高低频范围。Interleaved MRoPE 改为交错分配时间、高度、宽度，使每个轴都能使用不同频率范围。Qwen3.5 配置用 `mrope_interleaved=true` 和 `mrope_section=[11,11,10]` 启用该实现。[Qwen3-VL 技术报告 v2](https://arxiv.org/abs/2511.21631v2) [Qwen3.5 MRoPE 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L84-L165)

不要把 MRoPE 讲成“模型因此精确知道真实世界三维坐标”。这里的三个轴是序列中视觉网格的位置编码，不是相机标定后的物理坐标。

### 视频时间的边界

Qwen3-VL 从把绝对时间直接编码进位置编号，改为在视频块前加入文本时间戳。Qwen3.5 的实现也会把视频网格按时间 Patch 拆开，并用文本时间戳分隔。因此，不应简单声称“MRoPE 的 T 轴数值就是视频秒数”。视觉块内仍有三维位置结构，但真实时间主要通过显式文本时间戳告诉模型。[Qwen3-VL 报告的视频时间戳设计 v2](https://arxiv.org/abs/2511.21631v2) [Qwen3.5 `get_rope_index` 注释与实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1280-L1376)

## 8. 对推理系统最有用的数量关系

对一张调整后尺寸为 `H_img x W_img` 的图片：

$$
T_{vision}=\frac{H_{img}}{16}\times\frac{W_{img}}{16}\div 4
$$

这里忽略两侧特殊 token。图片分辨率扩大一倍是指高和宽都扩大一倍，此时像素数和视觉 token 数都会约扩大四倍，而不是两倍。

视觉 token 会增加三部分成本：

1. 视觉塔计算和临时激活。
2. 语言模型 Prefill 的位置数。
3. Full Attention 层的 KV Cache，以及 Gated DeltaNet 层需要更新的状态。

生成阶段通常不会在每个 Decode step 重跑视觉塔。Transformers 的生成准备逻辑在有缓存且不是第一次迭代时把 `pixel_values` 和 `pixel_values_videos` 设为 `None`，后续依靠已经建立的语言模型状态继续解码。[Qwen3.5 generation 输入准备](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1734-L1775)

“视觉塔只跑一次”仍需加运行时边界：跨请求是否复用同一图片的视觉结果，取决于服务框架是否有多模态 Processor/Encoder Cache，不能由模型结构自动保证。

## 9. 最容易讲错的地方

1. **视觉 token 不是 Tokenizer 从词表里切出来的词。** 占位 token 有 Token ID，真正送入 Decoder 的视觉位置向量来自视觉编码器。
2. **Patch 数不等于最终视觉位置数。** Qwen3.5 还会做 `2 x 2` 空间合并。
3. **Merger 不只是降采样。** 它既合并相邻特征，也将 1152 维视觉空间投影到语言模型 Hidden Size。
4. **MRoPE 不会增加 token 数，也不表示物理三维坐标。** 它为同一批位置提供三组坐标，并分配到不同 RoPE 维度。
5. **视频的真实时间不应只靠帧序号解释。** Qwen3.5 通过文本时间戳表达秒数，准确性依赖 FPS 和采样元数据。
6. **Qwen3-VL 论文的所有模块不一定都在 Qwen3.5 检查点中启用。** 当前 9B 和 35B-A3B 配置没有启用 DeepStack。
7. **多模态成本不能只算视觉编码器。** 视觉位置还会进入语言模型 Prefill，并占用推理状态。

## 资料来源

- [Qwen3.5-9B 官方配置，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
- [Qwen3.5-9B 官方预处理配置，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/preprocessor_config.json)
- [Qwen3.5-9B 官方聊天模板，revision `c202236`](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer_config.json)
- [Qwen3.5-35B-A3B 官方配置，revision `59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
- [Transformers Qwen3.5 实现，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Transformers Qwen3VL Processor，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- [Transformers Qwen2VL 图像处理器，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- [Qwen3-VL Technical Report v2](https://arxiv.org/abs/2511.21631v2)
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution v2](https://arxiv.org/abs/2409.12191v2)
