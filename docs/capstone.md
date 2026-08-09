# 结业案例：从模型配置到优化判断

结业案例要求你独立完成一次模型分析。题目只给出模型配置、工作负载和当前症状，不再按课程顺序提示该用哪个公式。

建议先独立作答，再展开参考分析。文末给出了一页技术结论示例，可以用来检查自己的答案是否覆盖了模型结构、容量口径、实验设计和上线判断。

## 1. 已知条件

某在线服务使用 Qwen3.5-9B，当前只处理文字请求。模型和工作负载如下：

```text
语言模型层数：              32
Hidden Size：              4096
Full Attention 层数：       8
Gated DeltaNet 层数：      24
Query Heads：              16
Key/Value Heads：           4
Head Dimension：          256
KV Cache dtype：          BF16

Prompt：                  4096 token
最大输出：                  256 token
并发请求：                   32
相同系统前缀：              2048 token
前缀覆盖请求比例：             80%
当前现象：                  P99 TTFT 超标，TPOT 已达标
部署方式：                  TP=8
```

这里的 `4096 token` 指应用 Chat Template 并完成 Tokenizer 后的 Prompt 长度，不是用户输入的汉字数。

## 2. 需要完成的分析

1. 写出一条文字消息从 API 输入到首个输出 Token ID 的数据形式和主要 shape。
2. 画出一个 Decoder Layer 的两段残差路径，并说明这个模型的两类 Token Mixer 怎样排列。
3. 说明 Prefill 结束时已经产生了什么，缓存里又保存了什么。首个输出 token 的 K/V 什么时候写入 Cache？
4. 计算单请求、单缓存位置的逻辑 KV Cache；再计算为 `4096+256` 个位置预留时的容量。
5. 计算单请求的 Gated DeltaNet 固定状态，并与 KV Cache 相加。32 个并发请求的逻辑模型状态是多少？
6. TP=8 时，每个 Rank 实际保存几个 K/V 头？每 Rank、每位置的 KV 是多少？为什么不能把逻辑 KV 直接除以 8？
7. 根据当前症状，选择最先验证的优化，并说明为什么其他常见方案不应仅凭名称排在前面。
8. 写出一次可以复现的验证实验，至少包括工作负载、指标和正确性检查。

<details>
<summary>查看参考分析</summary>

## 3. 从 API 输入到首个输出 token

API 接收的是结构化消息，例如：

```json
{"role": "user", "content": "请解释这份日志"}
```

它不会直接送进 Decoder。主要数据变化是：

```text
消息对象
→ Chat Template 加入角色、边界和生成起点
→ Tokenizer 得到 Token IDs [1,4096]
→ Embedding 查表得到 [1,4096,4096]
→ 32 个 Decoder Layer，shape 保持 [1,4096,4096]
→ 最终 RMSNorm
→ LM Head 为最后一个 Prompt 位置计算词表 Logits [1,V]
→ 贪心或采样得到首个输出 Token ID
```

如果实现为所有 Prompt 位置保留完整 Logits，LM Head 输出可以写成 `[1,4096,V]`。生成首个 token 只需要最后一个位置的 `[1,V]`。课程使用哪种写法时，都要说明是否保留了全部位置。

## 4. Decoder Layer 与层排列

一个 Decoder Layer 有两段预归一化残差路径：

```text
x → RMSNorm → Token Mixer → 加回 x → y
y → RMSNorm → SwiGLU FFN  → 加回 y → z
```

Token Mixer 负责不同 token 位置之间的信息交换，FFN 分别加工每个 token 的内部特征。这个 9B 模型按下面的顺序重复 8 次：

```text
Gated DeltaNet
Gated DeltaNet
Gated DeltaNet
Full Attention
```

因此共有 24 个 Gated DeltaNet 层和 8 个 Full Attention 层。Dense SwiGLU FFN、RMSNorm 和残差连接仍存在于每个 Decoder Layer 中。

## 5. Prefill 结束时的输出与状态

Prefill 已经处理完整 Prompt，并为最后一个 Prompt 位置得到 Logits。选择策略用这组 Logits 选出首个输出 token，记为 `y1`。

此时各层请求状态只对应 Prompt：

```text
8 个 Full Attention 层：Prompt 的 K/V
24 个 Gated DeltaNet 层：处理完 Prompt 后的卷积状态和递归状态
```

`y1` 刚刚被选出来，还没有经过下一次模型前向。因此它的 K/V 不会在“选择出来的瞬间”自动出现在所有层的 Cache 中。下一轮 Decode 把 `y1` 送入模型，才会逐层计算并写入它的状态，同时预测 `y2`。

这个时间点很容易发生差一位错误。讨论“已输出 token 数”“已经执行过模型的 token 数”和“Cache 中的位置数”时，应分别说明口径。

## 6. 请求状态容量

Full Attention 的单请求逻辑 KV 为：

$$
KV\ Bytes=2L_{full}N_{kv}TDs
$$

最前面的 2 表示 K 和 V。代入 `L_full=8`、`Nkv=4`、`D=256`、BF16 `s=2 Byte`，每增加一个缓存位置：

$$
2\times8\times4\times256\times2=32768\ Byte=32\ KiB
$$

若按 `4096+256=4352` 个位置预留：

$$
4352\times32\ KiB=136\ MiB
$$

第 8 课已经按 Qwen3.5 参考实现核算出 Gated DeltaNet 固定状态：`conv_state` 按 BF16、`recurrent_state` 按 FP32 计算，约为：

```text
24 层 conv_state 与 recurrent_state：49.5 MiB / 请求
```

于是单请求两类模型状态合计：

$$
136+49.5=185.5\ MiB
$$

32 个并发请求为：

$$
185.5\times32=5936\ MiB\approx5.80\ GiB
$$

这是逻辑模型状态，不包括 PagedAttention 块尾空余、页表、临时激活、通信 Buffer、CUDA Graph、Kernel Workspace 和运行时内存池。

## 7. TP=8 时的 KV 头复制

固定版本的 vLLM 在 `Nkv<TP` 时，会让每个 Rank 至少保存一个 K/V 头：

$$
N_{kv,rank}=\max\left(1,\left\lfloor\frac{N_{kv}}{TP}\right\rfloor\right)
$$

本例 `Nkv=4`、`TP=8`，所以每个 Rank 保存 1 个 K/V 头。每 Rank、每请求、每新增位置的 KV 是：

$$
2\times8\times1\times256\times2=8192\ Byte=8\ KiB
$$

8 个 Rank 合计为 64 KiB，比模型逻辑 KV 的 32 KiB 更大。原因是 4 个逻辑 K/V 头在 8 个 Rank 上发生了复制。多卡容量不能默认按设备数平均分配，必须核对 runtime 的实际布局。

## 8. 优化方案的验证顺序

现有信息还不足以断定哪项优化能改善总体 P99 TTFT。必须先把最慢请求分成三类：命中共享前缀、未命中前缀，以及主要耗时来自排队的请求。80% 请求拥有相同前缀，说明 Prefix Cache 有明确的重复计算可消除，适合作为第一轮对照实验；它不是已经确定的 P99 解决方案。

在缓存已经建立、能够命中且没有被驱逐的理想条件下，平均每个请求可复用的 Prompt 位置数为：

$$
0.8\times2048=1638.4
$$

这个结果不能改写成“TTFT 降低 40%”。命中查找、剩余 Prefill、排队和其他计算仍然存在。

20% 未命中请求足以覆盖最慢的 1%。如果 P99 主要落在未命中路径上，Prefix Cache 即使显著改善命中请求，也可能只改善 P50，而不改变总体 P99。

Qwen3.5 的前缀状态还必须同时恢复 Full Attention KV、卷积状态和 recurrent state。

其他方案需要证据后再排序：

| 方案 | 现有证据为什么不足 |
| --- | --- |
| 权重量化 | 它可以减少权重容量和读取字节，但当前症状是长共享前缀带来的 TTFT；没有 Profile 不能断定权重读取是主要瓶颈。 |
| FlashAttention | 它可能改善长 Prompt 的 Full Attention，但不会跳过 Gated DeltaNet、FFN 和未命中的 Prefill。 |
| 增加 TP | 当前已经是 TP=8。继续增加设备会带来更多通信，还可能加重 K/V 头复制。 |
| 推测解码或 MTP | 它主要瞄准后续逐 token 生成，而本例 TPOT 已经达标。 |

## 9. 验证实验

至少固定下面这些条件：

```text
同一模型 revision 与权重 dtype
同一 runtime 版本、并行配置和硬件
同一 Prompt / Output 长度分布
同一并发到达方式和测试时长
同一采样设置与正确性输入集
```

对 Prefix Cache，测试必须区分冷请求和命中请求：

| 证据 | 需要记录什么 |
| --- | --- |
| 缓存行为 | 命中请求比例、命中 token 数、驱逐率、Cache 占用 |
| 用户延迟 | 命中、未命中、排队主导请求和总体的 TTFT P50/P99；同时保留 TPOT |
| 服务能力 | 固定 SLO 下的 `goodput`、排队时间、输出 token 吞吐 |
| 正确性 | 固定输入和采样条件，对比 Logits 或 Token IDs；确认三类前缀状态一起恢复 |
| 资源 | 每 Rank 显存峰值、KV 分配、通信与回退路径 |

如果 Prefix Cache 只改善命中请求的 P50，而总体 P99 仍由未命中请求主导，就不能说当前目标已经完成。反过来，如果它减少了整体 Prefill 负载和排队，使未命中请求的 P99 也下降，则应通过队列时间和分桶结果证明这条间接收益。Cache 驱逐导致尾延迟变差，或者 `goodput` 下降时，也不能只凭平均 TTFT 宣布方案有效。

</details>

## 10. 可选扩展：加入一张图片

假设请求再加入一张 `512×512` 图片。按本课程采用的 Qwen3.5 配置，图片先切成 `16×16` Patch，再由 Merger 把每 `2×2` 个视觉特征合并为一个，最终产生 256 个视觉位置。

如果原来的 `4096 token` 不包含这些视觉位置，那么 Decoder Prefill 要多处理 256 个位置。逻辑 KV 会额外增加：

$$
256\times32\ KiB=8\ MiB/请求
$$

Gated DeltaNet 的状态 shape 不会因此扩大，但 Prefill 计算、临时激活和视觉编码器本身的计算都会增加。只看图片文件大小，无法推出 Decoder 中增加了多少工作。

## 11. 一页评审结论示例

> **当前证据不足以批准 Prefix Cache 直接全量上线。** 80% 的请求共享 2048 个精确 token，说明系统存在可消除的重复 Prefill，值得优先做对照实验；但 20% 的未命中请求已经足以覆盖最慢的 1%，所以命中路径变快不代表 P99 TTFT 一定达标。
>
> Qwen3.5-9B 的语言模型有 32 层，其中 24 层使用 Gated DeltaNet，8 层使用 Full Attention。按 4352 个缓存位置计算，单请求 Full Attention KV 为 136 MiB，Gated DeltaNet 固定状态约 49.5 MiB，合计 185.5 MiB。32 个并发请求的逻辑状态约为 5.80 GiB。这个数字不包含块尾空余、运行时预留、临时激活和通信 Buffer，不能直接当作进程峰值显存。
>
> TP=8 时，固定版本 vLLM 会让每个 Rank 至少保存一个 K/V 头。8 个 Rank 的物理 KV 合计大于模型逻辑 KV，容量评估不能直接除以 8。Prefix Cache 还必须同时恢复 Full Attention KV、卷积状态和 recurrent state，不能只验证 K/V 命中。
>
> 第一轮实验应固定模型和 runtime revision、TP 配置、到达记录、Prompt/Output 长度分布与采样参数。结果按命中、未命中、驱逐和排队主导请求分桶，比较 TTFT P50/P99、TPOT、同 SLO `goodput`、每 Rank 显存和状态恢复正确性。只有总体 TTFT P99 达标，且质量、容量和未命中路径没有退化，才进入灰度上线。

这份评审区分了观测、估算和待验证结果。80% 是工作负载观测，185.5 MiB 是根据模型结构得到的逻辑容量，P99 能否达标仍要通过对照实验确认。逻辑状态与进程显存也使用不同口径。

## 资料来源

本案例沿用第 4、5、7、8、9 课已经核对的配置、实现和公式：

- [Qwen3.5-9B 配置，revision c202236](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)
- [Transformers Qwen3.5 实现，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [vLLM KV 头分布实现，revision 653ebb5](https://github.com/vllm-project/vllm/blob/653ebb52dffd8b4653b430302473c771117529f1/vllm/config/model.py#L1501-L1516)

---

[返回课程路线](roadmap.md)
