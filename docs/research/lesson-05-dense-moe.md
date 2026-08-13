# 第 5 课研究笔记：Dense FFN 与 MoE

这份笔记为第 5 课准备事实底稿。重点是 Qwen3.5 的语言 Decoder，不展开训练算法史，也不把某个推理框架的通信实现说成 MoE 唯一的执行方式。

## 课程主线

Dense 与 MoE 的共同点是：它们都在 Decoder Layer 的 FFN 位置逐 token 加工特征，输入输出仍是 `[B,T,H]`。

区别在于：

- Dense FFN：每个 token 都经过同一组 FFN 参数。
- MoE：每层有许多套 FFN 参数。Router 为每个 token 选出少数 routed experts，同时 Qwen3.5 还让所有 token 都经过一个 shared expert，最后把结果合并。

因此，MoE 扩大的是可用参数容量，并通过条件激活控制单个 token 的计算量。它没有替换 Attention、Gated DeltaNet、RMSNorm、残差连接，也不会让不同 token 在 FFN 内互相通信。

## 1. MoE 替换 Decoder Layer 的哪一部分

Qwen3.5 Dense Layer 的结构是：

```text
x
├─ RMSNorm → Token Mixer → 残差相加
└─ RMSNorm → Dense SwiGLU FFN → 残差相加
```

Dense 版本在第二条残差支路中实例化 `Qwen3_5MLP`。这个 MLP 由 `gate_proj`、`up_proj`、SiLU、逐元素乘法和 `down_proj` 组成。[Qwen3.5 Dense MLP 与 Decoder Layer](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L704-L793)

Qwen3.5 MoE Layer 保留同一骨架，只把 `self.mlp` 换成 `Qwen3_5MoeSparseMoeBlock`：[Qwen3.5 MoE Decoder Layer](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L821-L877)

```text
x
├─ RMSNorm → Token Mixer → 残差相加
└─ RMSNorm → Sparse MoE → 残差相加
```

准确边界如下：

1. 替换的是**语言 Decoder 每层的 FFN 子层**，不是整个 Decoder Layer。
2. Full Attention 与 Gated DeltaNet 的 3:1 排列不变。
3. 40 个语言 Decoder Layer 都使用 MoE，不是隔几层才放一个 MoE。
4. 视觉编码器仍使用自己的 Dense Vision MLP，不能说“Qwen3.5-MoE 全模型所有 FFN 都变成专家”。[Qwen3.5 MoE 语言层与视觉层实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L821-L877) [Qwen3.5 MoE Vision Block](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1062-L1090)

Mixtral 原论文也给出相同的结构边界：MoE 替换 Transformer Block 的 FFN 子块，每个 expert 本身是一个普通的 SwiGLU FFN。[Mixtral §2 至 §2.1](https://arxiv.org/pdf/2401.04088)

## 2. 先看 Dense FFN

对一个 token 的 `H` 维向量 `x`，Qwen3.5 Dense SwiGLU FFN 为：

$$
\mathrm{DenseFFN}(x)
=W_{down}\left(\mathrm{SiLU}(W_{gate}x)\odot W_{up}x\right)
$$

Shape 为：

```text
x                     [H]
gate_proj(x)          [I]
up_proj(x)            [I]
SiLU(gate) ⊙ up       [I]
down_proj(...)        [H]
```

批量写法只是把前面的 token 轴带上：

```text
[B,T,H] → [B,T,I] → [B,T,H]
```

`B` 和 `T` 不变，说明 FFN 分别处理每个 token。不同 token 之间的信息交换发生在 Token Mixer 中，不发生在 FFN 中。

Qwen3.5-9B 的官方配置为 `H=4096`、`I=12288`、32 层。三个无偏置矩阵一层共有：

$$
3HI=3\times4096\times12288=150,994,944
$$

这些 FFN 参数对每个 token 都会使用。[Qwen3.5-9B-Base 官方配置](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)

## 3. Qwen3.5 MoE 的完整数据流

以下用 Qwen3.5-35B-A3B 的配置：

```text
H = 2048
E = 256 个 routed experts
K = 8 个 selected experts / token
I = 512，每个 expert 的中间宽度
1 个 shared expert，I_shared = 512
```

官方模型卡给出的整体结构是 40 层，每 3 层 Gated DeltaNet 加 1 层 Full Attention，二者后面都接 MoE；每个 token 激活 8 个 routed experts 和 1 个 shared expert。[Qwen3.5-35B-A3B 官方模型卡，revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md) [官方配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)

令 `N=B×T`，把输入展平为 `[N,H]`。一层 MoE 的数据流如下：

```text
输入 X [B,T,H]
  ↓ 展平
X2 [N,H]
  ├─ Router Linear → logits [N,E]
  │    ↓ Softmax
  │  probabilities [N,E]
  │    ↓ 每个 token 取 Top-K，并重新归一化
  │  selected_experts [N,K]
  │  routing_weights [N,K]
  │    ↓ 按 expert 聚集 token
  │  8 个 routed expert 输出加权求和 → [N,H]
  │
  └─ shared expert 处理所有 token → [N,H]
       ↓ 每个 token 一个 Sigmoid 标量门控
     gated shared output [N,H]

routed sum + gated shared output
  ↓ reshape
输出 [B,T,H]
```

Transformers 参考实现就在 `Qwen3_5MoeTopKRouter` 和 `Qwen3_5MoeSparseMoeBlock` 中。[Qwen3.5 Router 与 Sparse MoE Block](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L760-L798)

### 3.1 Router logits、Top-k 与 routing weights

Router 是一个无偏置 Linear：

```text
Router 权重 W_r     [E,H] = [256,2048]
输入 X2             [N,H]
router_logits       [N,E] = [N,256]
```

对每个 token，代码先在 256 个 logits 上做 Softmax，然后选概率最大的 8 个，再把这 8 个数重新归一化，使它们的和为 1：

```text
router_logits       [N,256]
router_probs        [N,256]
selected_experts    [N,8]    整数 expert ID
routing_weights     [N,8]    浮点权重，每行之和为 1
```

这些概率只用于 expert 路由，不是词表上的下一个 token 概率。Router 也没有人工编写的“数学、代码、中文”等 expert 标签。[Qwen3.5 Top-k Router](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L760-L776)

### 3.2 每个 routed expert 仍是 SwiGLU FFN

每个 routed expert 都有自己独立的 gate、up、down 权重。Qwen3.5 为执行效率把 gate 与 up 存成一个融合张量：

```text
所有 experts 的 gate_up_proj  [E,2I,H] = [256,1024,2048]
所有 experts 的 down_proj     [E,H,I]  = [256,2048,512]
```

假设 expert `e` 在这一批收到 `n_e` 个 token：

```text
expert 输入                   [n_e,H]
gate_up Linear                [n_e,2I]
拆成 gate、up                 两个 [n_e,I]
SiLU(gate) ⊙ up               [n_e,I]
down Linear                   [n_e,H]
乘各 token 对该 expert 的权重 [n_e,H]
```

代码最后用 `index_add_` 把不同 expert 的贡献加回原 token 的 `[N,H]` 位置。同一 token 被复制到 8 个 expert 分支，8 个输出按 routing weights 加权相加。[Qwen3.5 Expert 权重与执行](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L720-L757)

用公式写就是：

$$
y_{routed}(x)=\sum_{i\in\mathrm{TopK}(x)}a_i(x)E_i(x),
\qquad \sum_i a_i(x)=1
$$

Mixtral 使用同一类加权合并方式，只是它有 8 个 experts、每个 token 选 2 个，也没有 Qwen3.5 的 shared expert。[Mixtral §2.1](https://arxiv.org/pdf/2401.04088)

### 3.3 Shared expert 不参加 Top-k

Shared expert 是另一套 SwiGLU FFN，所有 token 都会执行。它还有一个 `[1,H]` 的 Linear，为每个 token 生成一个标量，再经过 Sigmoid 调节 shared expert 输出：

```text
shared_expert(X2)            [N,H]
shared_expert_gate(X2)       [N,1]
sigmoid(gate) × shared       [N,H]
```

最终输出为：

$$
y(x)=y_{routed}(x)+\sigma(w_sx)E_s(x)
$$

Shared expert 不在 256 个 routed experts 之内，不参与 Top-8，也不与那 8 个 routing weights 一起归一化。[Qwen3.5 Sparse MoE Block](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L779-L798)

“Shared expert 承担通用知识”可以作为设计直觉，但不能当成已验证的 Qwen3.5 内部事实。Shared-expert 架构最初的一个明确设计动机确实是承接公共知识、减少 routed experts 的冗余；具体学到了什么仍由训练决定。[DeepSeekMoE §2.2](https://arxiv.org/pdf/2401.06066)

## 4. Shape 总表

对 Qwen3.5-35B-A3B，令 `N=B×T`：

| 阶段 | Shape | 说明 |
|---|---|---|
| MoE 输入 | `[B,T,2048]` | Layer 的 Hidden States |
| 展平输入 | `[N,2048]` | Router 与 experts 按 token 处理 |
| Router logits | `[N,256]` | 每个 token 对所有 routed experts 的原始分数 |
| Router probabilities | `[N,256]` | 对 256 维做 Softmax |
| Selected expert IDs | `[N,8]` | 每个 token 选 8 个 expert ID |
| Routing weights | `[N,8]` | 8 个选中 expert 的归一化权重 |
| Expert `e` 输入 | `[n_e,2048]` | `n_e` 每批都可能不同 |
| Expert 中间结果 | `[n_e,512]` | SwiGLU 中间宽度 |
| Expert `e` 输出 | `[n_e,2048]` | 乘路由权重后写回 |
| Routed 合并结果 | `[N,2048]` | 每 token 的 8 个输出求和 |
| Shared expert 输出 | `[N,2048]` | 所有 token 都执行 |
| Shared scalar gate | `[N,1]` | 每 token 一个 Sigmoid 系数 |
| MoE 输出 | `[B,T,2048]` | 与残差支路 Shape 相同 |

Shape 的关键不是背表，而是看出两点：

1. `N` 个 token 被临时按 expert 重新分组，处理后还要回到原顺序。
2. MoE 入口和出口都是 `H` 维，所以 Decoder Layer 外部接口不变。

## 5. 总参数与每 token 激活参数

### 5.1 两个口径分别回答什么问题

- **总参数**回答模型需要保存多少权重。即使某个 token 没选中某些 experts，那些 expert 权重仍属于模型。
- **每 token 激活参数**回答处理一个 token 时，哪些权重实际参与了这次前向计算。

因此，A3B 不表示模型只需存 3B 参数。官方模型卡给 Qwen3.5-35B-A3B 的口径是 35B 总参数、每 token 约 3B 激活参数。[Qwen3.5-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)

Mixtral 原论文专门提醒：active parameter count 与计算量相关，但服务内存仍按 sparse total parameters 计算，还要考虑路由、额外权重读取和硬件利用率；MoE 更适合能把 expert 工作聚成批的负载。[Mixtral §3 “Size and Efficiency”](https://arxiv.org/pdf/2401.04088)

### 5.2 Qwen3.5-35B-A3B 的一层 MoE 参数

忽略 bias，因为这些 Linear 都无 bias。`H=2048`、`I=512`、`E=256`、`K=8`。

一个 SwiGLU expert：

$$
3HI=3\times2048\times512=3,145,728
$$

一层的参数分解：

| 部分 | 参数数 |
|---|---:|
| 256 个 routed experts | `256 × 3,145,728 = 805,306,368` |
| Shared expert | `3,145,728` |
| Shared scalar gate | `2,048` |
| Router | `256 × 2,048 = 524,288` |
| 一层 MoE 合计 | `808,978,432` |

一个 token 在这一层实际用到：

| 部分 | 参数数 |
|---|---:|
| 8 个 selected experts | `8 × 3,145,728 = 25,165,824` |
| Shared expert 与 gate | `3,147,776` |
| Router | `524,288` |
| 一层激活合计 | `28,837,888` |

40 层 MoE 子层合计约 32.36B 总参数，而单 token 在这些 MoE 子层中使用约 1.15B。再加上 Embedding、Attention、Gated DeltaNet、Norm、输出层等共享模块，官方整模型口径约为 35B total / 3B active。上述分解由官方配置和权重 Shape 直接计算。[Qwen3.5-35B-A3B 配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json) [Expert 权重定义](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L720-L730)

还要区分“单 token 激活权重”和“一批请求实际读到的不同权重”。一批 token 可能合起来命中很多甚至全部 experts，所以 A3B 不能直接换算成整批的权重读取量，更不能直接推导吞吐倍数。

## 6. Token dispatch 与合并

### 6.1 单设备上的逻辑过程

Router 得到 `[N,K]` expert IDs 后，需要把 token 按 expert ID 重排：

```text
原顺序：token 0, token 1, token 2, ...
按 expert 分组：expert 7 收到若干 token，expert 81 收到若干 token，...
执行各 expert FFN
乘 routing weights
写回并恢复原 token 顺序
```

参考实现用 one-hot mask 找出每个 expert 收到的 token，用索引 gather 输入，再用 `index_add_` 合并输出。[Qwen3.5 Experts forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L733-L757)

生产推理框架通常会排序 token 并用 Grouped GEMM 或 fused MoE kernel 同时执行多个 expert。这个变化优化的是执行方式，不改变 Router、Top-k、expert FFN 和加权合并的数学语义。

### 6.2 为什么负载会不均

Router 是按内容逐 token 决策的，不保证每个 expert 恰好收到相同数量。`n_e` 的差异会带来：

- 某些 expert 或 GPU 工作很多，其他设备等待。
- 每个 expert 的 token 数太少，GEMM 变成许多小矩阵，GPU 利用率低。
- 分布式 dispatch 的发送量不均，最慢 rank 决定尾延迟。

Mixtral 原论文明确指出，expert 收到的 token 数可变，EP 必须处理负载均衡，否则会出现设备过载或计算瓶颈。[Mixtral §2.1](https://arxiv.org/pdf/2401.04088)

Qwen3.5-35B-A3B 每 token 产生 8 个 routed assignments。若 Decode 批次只有 8 个 token，一层也只有 64 个 assignments 分给 256 个 experts，很多 experts 没有 token，命中的 expert 也常只有很小的 micro-batch。这是“小 batch MoE 不一定快”的直接原因之一。

### 6.3 负载均衡损失做什么，不做什么

Transformers 实现提供 Switch Transformer 风格的辅助损失，统计每个 expert 的 token 占比和平均路由概率，并惩罚过度集中；Qwen3.5-35B-A3B 配置中的 `router_aux_loss_coef` 为 `0.001`。[Qwen3.5 load balancing loss](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1675-L1754) [官方配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)

它是训练时的正则项，不是推理时的调度器，也不能保证每个 batch 完全均匀。Switch Transformer 原论文给出了同类损失的定义和容量不均问题。[Switch Transformer §2.2，公式 4 至 6](https://arxiv.org/pdf/2101.03961)

Qwen 官方还提出过跨 micro-batch 统计的 global-batch load balancing，其目标也是让长期训练统计更均衡，而不是强制单个推理 batch 平均分配。[Qwen 官方 global-batch load balance 说明](https://qwenlm.github.io/blog/global-load-balance/)

Qwen3.5 的 Hugging Face 参考前向不会因为某个 expert 收到太多 token 而丢弃 token。Capacity factor、padding、token dropping 是部分训练或推理系统的实现选择，不应当写成 Qwen3.5 模型公式的一部分。[Qwen3.5 Experts forward](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L733-L757)

## 7. EP 与 TP 的通信边界

先区分两种并行：

| 并行方式 | 切分什么 | 每个 token 到哪里 | 主要通信边界 |
|---|---|---|---|
| Tensor Parallelism，TP | 一个 Linear 或一个 expert 内部的矩阵维度 | 同一 token 在多个权重分片上算部分结果 | Linear 内部或输出处交换、归并部分结果 |
| Expert Parallelism，EP | 不同 expert ID | token assignment 根据 Top-k 去持有对应 expert 的设备 | Router 之后 dispatch，expert 计算之后 combine |

### 7.1 Qwen3.5 官方 TP 计划

官方 Transformers 配置把 routed experts 的融合 `gate_up_proj` 设为列切分，把 `down_proj` 设为行切分；shared expert 的 gate/up/down 也分别采用列切分和行切分。也就是说，TP 会切一个 expert 内部的矩阵。[Qwen3.5 MoE `base_model_tp_plan`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L59-L77)

Transformers 官方文档将 TP 定义为切分单层，并指出每层都要交换部分结果，因此依赖快速的设备间互联。[Transformers Tensor Parallelism](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/docs/source/en/perf_infer_gpu_multi.md#L16-L20)

### 7.2 Qwen3.5 官方 EP 计划

EP 计划把 Router 标为 `ep_router`，把 routed expert 的 gate/up 和 down 交给 Grouped GEMM，并按 expert 切权重。Shared expert 没有出现在 EP 计划中，但出现在 TP 计划中。[Qwen3.5 MoE `base_model_ep_plan`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L83-L88)

Transformers 当前原生 EP 的做法是：每个设备只加载本地 experts，`ep_router` 处理路由，最后用 All-Reduce 合并各设备产生的 expert 输出。[Transformers Expert Parallelism](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/docs/source/en/expert_parallelism.md#L16-L48)

其他生产框架可能显式使用两次 All-to-All：一次把 token 发到 expert 所在 rank，一次把输出送回原 token 所在 rank。Mixtral 原论文将 EP 描述为 token 被送到对应 GPU，完成后返回原位置；Megatron Core 同时提供 All-to-All、All-Gather、Flex/DeepEP 等 dispatcher。因此，课程应讲清“dispatch 和 combine 是通信边界”，不能断言所有引擎都固定使用某一种 collective。[Mixtral §2.1](https://arxiv.org/pdf/2401.04088) [Megatron Core Token Dispatcher](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.token_dispatcher.html)

### 7.3 判断性能时该看什么

对推理工程师，MoE 并行至少要同时看：

1. 一步内有多少 token assignments，也就是大约 `N×K`。
2. 每个 expert 的 `n_e` 分布，不能只看平均值。
3. Experts 怎样映射到 GPU，热点 experts 是否集中到少数 rank。
4. EP dispatch/combine 的跨卡或跨机流量。
5. 每个 expert 是否还做 TP，TP collective 是否与 EP 通信叠加。
6. Shared expert 是复制、TP 切分还是与 routed expert overlap，取决于引擎实现。
7. Grouped GEMM 的 expert micro-batch 大小和 padding 情况。

## 8. Qwen3、Mixtral 与 Qwen3.5 不要混用

三者都能帮助理解 MoE，但结构不完全相同：

| 模型 | Routed experts | Top-k | Shared expert | Token Mixer |
|---|---:|---:|---|---|
| Mixtral 8x7B | 8 | 2 | 无 | Full Attention |
| Qwen3-30B-A3B / 235B-A22B | 128 | 8 | 无 | Full Attention |
| Qwen3.5-35B-A3B | 256 | 8 | 1 | 3:1 Gated DeltaNet / Full Attention |

Qwen3 技术报告明确写明 Qwen3 MoE 没有 shared expert。因此不能从 Qwen3 的结构直接推断 Qwen3.5；Qwen3.5 的 shared expert 以官方模型卡、配置和实现为准。[Qwen3 Technical Report §2](https://arxiv.org/pdf/2505.09388) [Qwen3.5-35B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)

## 9. 容易误导的说法

| 不建议这样写 | 问题 | 更准确的说法 |
|---|---|---|
| MoE 替换了整个 Decoder Layer | Attention、Norm、残差仍在 | MoE 替换语言 Decoder 的 Dense FFN 子层 |
| Dense 表示 token 与所有 token 全连接 | Dense 不是 token mixing | Dense 表示每个 token 都使用同一套完整 FFN 参数 |
| 一个 token 只进入一个 expert | 与 Qwen3.5 Top-8 不符 | 每层进入 8 个 routed experts，再进入 1 个 shared expert |
| Router 先给 experts 人工分类 | Router 权重由训练学习 | Router 用 Linear 为每个 token 计算 256 个路由分数 |
| Expert 3 是数学专家 | 没有这种人工标签或稳定保证 | Expert 是一套独立 FFN 权重，路由行为可分析但不能随意命名 |
| Shared expert 一定存通用知识 | 这是设计直觉，不是可直接验证的内容表 | Shared expert 对所有 token 执行，可为公共变换提供固定容量 |
| A3B 模型只占 3B 权重内存 | 混淆 active 与 total | 35B 权重需要保存，3B 是官方每 token 激活参数口径 |
| Active 参数少 10 倍，推理就快 10 倍 | 忽略通信、小 GEMM、路由和权重读取 | Active 参数主要反映计算稀疏度，实际性能还受 batch、dispatch 和硬件利用率影响 |
| EP 就是 All-to-All | 不同框架 collective 不同 | EP 的稳定语义是按 expert 分布计算，dispatch/combine 的具体 collective 由引擎决定 |
| Load balancing loss 会让推理流量均匀 | 它只是训练正则 | 推理时仍需观察每层、每批的 expert 负载分布 |

Mixtral 的路由分析也提醒：并没有观察到清晰、稳定的主题专家划分，只发现部分结构化和句法行为。课程不宜把 experts 画成“数学、代码、语言”三个固定部门。[Mixtral §5](https://arxiv.org/pdf/2401.04088)

## 10. 推荐的课程讲解顺序

1. 用一张 Decoder Layer 图标出 Dense FFN 的位置。
2. 手算一个 token 的 Dense SwiGLU，复习 `H → I → H`。
3. 把“一套 FFN”替换成“四套小 FFN”的玩具 MoE，Top-2 手算一次。
4. 画出 Router logits、Top-k IDs、weights，明确三者不是一回事。
5. 对两个 selected experts 的输出做加权求和。
6. 加入 shared expert，强调它不参加 Top-k。
7. 再换成 Qwen3.5 的真实 `256 / Top-8 / 1 shared` 和 Shape。
8. 最后解释 total / active 参数，以及为什么 active 少不等于线性加速。
9. 在优化判断段引入 token dispatch、负载不均、Grouped GEMM、EP 与 TP。

推荐至少画四张图：

1. Dense FFN 与 MoE 在 Decoder Layer 中的位置对照。
2. 一个 token 的 Router → Top-2 → 两个 experts → 加权合并玩具图。
3. Qwen3.5 的 8 routed + 1 shared 完整数据流和 Shape。
4. EP 图：Router 后按 expert 发往不同 GPU，计算后回到原 token 顺序；旁边单独画 TP 切分一个 expert 矩阵。

## 11. 学完后应能回答的问题

1. MoE 替换 Decoder Layer 的哪一部分？什么没有被替换？
2. Router logits `[N,256]` 与 selected expert IDs `[N,8]` 有什么区别？
3. 为什么 routing weights 每个 token 的 8 个数之和为 1？
4. 一个 routed expert 内部怎样完成 `H → I → H`？
5. Shared expert 为什么不在 Top-8 中？它的输出怎样加入结果？
6. 为什么 MoE 输入输出仍是 `[B,T,H]`？
7. 35B total / 3B active 分别说明什么？为什么 A3B 不代表只需保存 3B 权重？
8. Decode 小 batch 为什么可能让 MoE expert GEMM 很碎？
9. EP 与 TP 分别切什么，通信边界在哪里？
10. 为什么不能简单说“Expert 17 是代码专家”？

## 来源

- Qwen, [Qwen3.5-35B-A3B model card, revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md)。
- Qwen, [Qwen3.5-35B-A3B config, revision 59d61f3](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)。
- Hugging Face Transformers, [Qwen3.5 MoE modeling source, revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)。
- Hugging Face Transformers, [Qwen3.5 MoE configuration source, revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py)。
- Albert Q. Jiang et al., [Mixtral of Experts](https://arxiv.org/abs/2401.04088), §2.1、§3、§5。
- An Yang et al., [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), §2。
- William Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961), §2.2。
- Damai Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066), §2.2。
- Qwen Team, [Global-batch load balance](https://qwenlm.github.io/blog/global-load-balance/)。
- Hugging Face Transformers, [Expert parallelism](https://huggingface.co/docs/transformers/expert_parallelism) 与 [Tensor parallelism](https://huggingface.co/docs/transformers/perf_infer_gpu_multi)。
- NVIDIA Megatron Core, [MoE Token Dispatcher API](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.token_dispatcher.html)。
