# 第 5 课研究笔记：Qwen3.5 的 Gated DeltaNet

这份笔记为第 5 课核实 Gated DeltaNet 的原理、Qwen3.5-9B-Base 的具体结构和推理状态。资料只取自 Gated DeltaNet 与 DeltaNet 原论文、Qwen3.5 官方模型文件、Hugging Face Transformers 固定 revision 实现，以及论文作者维护的官方 Kernel 实现。

## 先说结论

Gated DeltaNet 不会像 Full Attention 那样保留每个历史 token 的 K/V，再让当前 Q 与全部历史 K 比较。它为每个头维护一张固定大小的状态矩阵。每读入一个 token，模型会：

```text
先按 α 衰减旧状态
→ 用当前 k 查询状态原来会返回什么
→ 比较“原来返回的值”和当前 v
→ 按 β 把差值写回状态
→ 用当前 q 从更新后的状态中读取输出
```

最小公式可以写成：

$$
\begin{aligned}
\bar S_t &= \alpha_t S_{t-1} \\
\hat v_t &= k_t^T\bar S_t \\
e_t &= \beta_t(v_t-\hat v_t) \\
S_t &= \bar S_t+k_t e_t^T \\
o_t &= \frac{q_t^T S_t}{\sqrt{D_k}}
\end{aligned}
$$

这里采用 Qwen 和 FLA Kernel 的状态布局，`S` 的 shape 是 `[Dk,Dv]`。最后的 `1/sqrt(Dk)` 是 Qwen 实现使用的 Query 缩放。Gated DeltaNet 论文把状态写成 `[Dv,Dk]`，所以论文公式左右相反，但两种写法只是互为转置。[Gated DeltaNet §3.1，公式 10](https://arxiv.org/pdf/2412.06464)

理解这五行以后，再看因果卷积、Q/K/V、多头和 Chunk Kernel。不要从论文的 WY 分解或 Triton Kernel 起讲。

## 1. 它在 Decoder Layer 中替换了什么

Qwen3.5 的一个 Decoder Layer 仍然保留预归一化、残差连接和 FFN。变化的是 Token Mixer：

```text
Full Attention Layer：
x → RMSNorm → Full Attention → Residual → RMSNorm → FFN → Residual

Gated DeltaNet Layer：
x → RMSNorm → Gated DeltaNet → Residual → RMSNorm → FFN → Residual
```

Transformers 根据 `layer_types` 在 `Qwen3_5Attention` 和 `Qwen3_5GatedDeltaNet` 之间选择，说明 Gated DeltaNet 替换的是 Attention 所在的 Token Mixer 子层，不是整个 Decoder Layer。[Qwen3.5 Decoder Layer 分支，revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L740-L790)

Qwen3.5-9B-Base 的 32 层按下面的顺序排列：

```text
3 × Gated DeltaNet Layer
1 × Full Attention Layer
```

这组结构重复 8 次，所以共有 24 个 Gated DeltaNet 层和 8 个 Full Attention 层。官方模型卡直接给出了 `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`。[Qwen3.5-9B-Base 模型卡，revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)

## 2. 为什么不保留全部历史 K/V

Full Attention 让当前 Q 直接与历史每个 K 比较：

```text
历史：k1/v1, k2/v2, ... , kt/vt
读取：q 与所有 k 打分，再按权重汇总所有 v
状态：随序列长度 T 增长
```

线性 Attention 可以把历史外积的和先整理成一张矩阵：

$$
S_t=S_{t-1}+k_tv_t^T,
\qquad
o_t=q_t^TS_t
$$

把递推式展开：

$$
q_t^TS_t
=\sum_{i=1}^{t}(q_t^Tk_i)v_i^T
$$

这表明，先积累 `k_i v_i^T`，再用 q 读取，与逐个计算线性点积后汇总 v 在代数上可以对应。状态矩阵的 shape 只由头数、`Dk` 和 `Dv` 决定，不随 `T` 增长。Gated DeltaNet 论文从这种矩阵状态形式引出 Mamba2、DeltaNet 和 Gated DeltaNet。[Gated DeltaNet §2.1 至 §2.2](https://arxiv.org/pdf/2412.06464)

但简单相加会有明显问题。新旧 token 的 key 相近时，它们的 value 会在同一方向上叠加，状态逐渐发生干扰。Delta Rule 的作用不是继续累加，而是先检查该 key 当前会读出什么，再只写入两者之间的差值。

状态矩阵不是一张可以逐行查看的“事实表”，也不是 KV Cache 的无损压缩。它是训练得到的连续数值关联，容量固定，key 不正交时仍会相互影响。

## 3. 用两个二维向量看懂 Delta Rule

先忽略遗忘，令 `α=1`、`β=1`。假设一个头的状态是 `2×2` 矩阵，初始全为 0：

$$
S_0=
\begin{bmatrix}
0&0\\
0&0
\end{bmatrix}
$$

第一个 token 要把 `v=[2,3]` 写到 `k=[1,0]` 对应的方向：

```text
状态当前返回：kᵀS₀ = [0,0]
需要修正：    v-kᵀS₀ = [2,3]
写回状态：    S₁=S₀+k[2,3]
```

结果是：

$$
S_1=
\begin{bmatrix}
2&3\\
0&0
\end{bmatrix}
$$

用 `q=[1,0]` 读取时，得到 `qᵀS₁=[2,3]`。

后来，相同的 key 要关联到新 value `v=[5,1]`。状态当前在这个 key 上返回 `[2,3]`，所以只写入差值：

```text
差值：[5,1]-[2,3]=[3,-2]
新状态第一行：[2,3]+[3,-2]=[5,1]
```

如果只是执行普通外积相加，第一行会变成 `[7,4]`，新旧 value 叠在一起。Delta Rule 得名于它写入的是误差 `delta=v-old_value`，而不是再次完整写入 v。

这个例子满足 key 已归一化、方向互不干扰、`β=1` 等简化条件。真实模型使用连续向量和软更新，不能把它理解成精确的 Python Dictionary。

## 4. α、β 和 z 分别控制什么

Gated DeltaNet 中有三组容易混淆的数：

| 名称 | Qwen 代码来源 | 范围 | 作用 |
|---|---|---|---|
| `α` | `α=exp(g)` | `(0,1)` | 更新前整体衰减一个头的旧状态 |
| `β` | `β=sigmoid(b)` | `(0,1)` | 控制当前 key 方向修正多少 |
| `z` | `SiLU(z)` | 不限于 0 到 1 | 在递归输出归一化后逐元素调节输出 |

### 4.1 α 是遗忘系数

每个 token、每个 Value 头都会产生一个 `α_t`。`α` 接近 1 时，旧状态大部分保留；接近 0 时，该头的旧状态会快速缩小。

Gated DeltaNet 论文强调，`α` 提供的是全局快速清理能力。纯 DeltaNet 只能沿当前 key 方向逐步修改，难以在上下文切换时迅速清空大量旧信息。[Gated DeltaNet §1、§3.1](https://arxiv.org/pdf/2412.06464)

Qwen 代码没有直接生成名为 `alpha` 的张量。它先计算：

$$
g=-\exp(A_{log})\times softplus(a+dt\_bias)
$$

因为右侧始终为负，所以递归 Kernel 中的 `α=exp(g)` 位于 `(0,1)`。代码变量 `a` 是输入相关的原始投影，`g` 是对数空间的衰减，二者都不应直接叫作 `α`。[Qwen3.5 门值计算](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L493-L502)

### 4.2 β 是修正幅度

`β` 决定当前误差写回多少：

```text
β 接近 0：几乎不改这个 key 对应的内容
β 接近 1：尽量把这个 key 的返回值推向当前 v
```

论文也把 `β` 解释为写入强度，并从在线回归角度把它看成自适应学习率。[Gated DeltaNet §2.2、§3.1](https://arxiv.org/pdf/2412.06464)

`α` 和 `β` 不可互换。`α` 先作用于整张头状态，`β` 再控制当前 key 方向的定向修正。

### 4.3 z 是输出门

递归状态读出的 `o` 还会执行：

$$
RMSNorm(o)\odot SiLU(z)
$$

然后展平所有 Value 头，再通过 `out_proj` 回到 Hidden Size。`z` 不参与状态遗忘，也不决定 Delta 写入强度；它只调节当前层送出的结果。Qwen 的 `Qwen3_5RMSNormGated` 明确采用“先归一化，再乘 `SiLU(gate)`”的顺序。[Qwen3.5 Gated RMSNorm](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L168-L187)

## 5. Q、K、V 在这里做什么

Gated DeltaNet 也使用 Q、K、V 这三个名字，但计算过程与 Softmax Attention 不同：

```text
K：决定从状态的哪个方向检查和修改
V：给出这个方向希望关联的新内容
Q：从更新后的状态中读取当前输出
```

代码在进入 Delta Rule 前对 Q/K 沿头维做 L2 归一化，并把 Q 乘以 `1/sqrt(Dk)`。论文的 Token Mixer 结构同样规定 Q/K 经过短卷积、SiLU 和 L2 归一化。[Gated DeltaNet §3.4、图 1](https://arxiv.org/pdf/2412.06464) [Qwen3.5 Q/K 归一化与递归实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L331-L380)

这里没有下面这条 Full Attention 主线：

```text
QKᵀ → 因果遮罩 → Softmax → 权重乘 V
```

递归形态直接通过状态更新保证因果顺序，也不会生成一张覆盖全部历史 token 的 Attention 权重矩阵。

## 6. 因果卷积为什么在 Delta Rule 前面

Qwen 先把 Hidden States 投影为混合的 Q/K/V 通道，再对每个通道沿 token 轴做长度为 4 的因果卷积和 SiLU：

```text
x
→ in_proj_qkv
→ 每个通道读取当前位置和前 3 个位置
→ SiLU
→ 拆成 Q、K、V
```

对某个通道，当前位置可以抽象成：

$$
c_t=SiLU(w_0u_{t-3}+w_1u_{t-2}+w_2u_{t-1}+w_3u_t)
$$

它只读取当前和过去，不读取未来，所以称为因果卷积。卷积的 `groups=conv_dim`，意味着 8192 个通道分别沿时间做自己的卷积，不在这一步相互混合；前面的 Linear 已经完成了特征组合。[Qwen3.5 因果卷积定义与回退实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L200-L240) [Qwen3.5 Gated DeltaNet 卷积层](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L387-L414)

因果卷积让当前 Q/K/V 在进入递归状态前，先混入很短的局部上下文。它不是图像卷积，也不是下采样，token 数量保持不变。

不要写成“因果卷积代替 RoPE”。Gated DeltaNet 层确实不调用 Full Attention 的 RoPE，但序列顺序同时体现在因果卷积和按时间更新的递归状态中，不能把位置能力归因于卷积一项。

### 为什么还需要 conv state

单 token Decode 时，要计算当前位置的长度 4 卷积，仍需最近几个投影值。recurrent state 只保存 Delta Rule 的矩阵状态，不保存这段精确局部窗口。因此每个 Gated DeltaNet 层还要维护独立的 `conv_state`。

Qwen 的缓存实现保留最后 4 个混合 Q/K/V 投影位置。新 token 到来时，把它追加进窗口，完成卷积，再原地更新这份状态。[Qwen3.5 单步因果卷积更新](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L200-L217) [Transformers Linear Attention Cache](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L998-L1086)

## 7. Qwen3.5-9B 的完整 shape

固定 revision 的官方配置给出：

```text
Hidden Size H            = 4096
Q/K 头数量               = 16
V 头数量                 = 32
Q/K 头维 Dk              = 128
V 头维 Dv                = 128
因果卷积宽度             = 4
Gated DeltaNet 层数      = 24
Full Attention 层数      = 8
```

[Qwen3.5-9B-Base 官方配置，revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)

输入为 `[B,T,4096]` 时，前半段数据流是：

```text
in_proj_qkv 输出      [B,T,8192]
├─ Q                  [B,T,16,128]
├─ K                  [B,T,16,128]
└─ V                  [B,T,32,128]

a 投影                [B,T,32]  → log-decay g → α
b 投影                [B,T,32]  → sigmoid      → β
z 投影                [B,T,4096] → [B,T,32,128]
```

`8192` 的来源是：

$$
16\times128+16\times128+32\times128=8192
$$

因为 V 有 32 个头，Q/K 只有 16 个头，Qwen 将每个 Q/K 头重复两次，与 32 个 Value 头对齐：

```text
Q：[B,T,16,128] → [B,T,32,128]
K：[B,T,16,128] → [B,T,32,128]
V：[B,T,32,128]
```

这不是 Full Attention 中的 GQA KV Cache。Gated DeltaNet 没有逐 token K/V Cache；这里的重复是为了让 32 个 Value 头各自执行状态更新。[Qwen3.5 Q/K 头扩展](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L482-L503)

每个 Gated DeltaNet 层的 recurrent state 是：

```text
[B,32,128,128]
```

可以读成：每个请求有 32 个头，每个头有一张 `128×128` 的状态矩阵。递归输出为 `[B,T,32,128]`，再经过 Gated RMSNorm、展平和 `out_proj`，回到 `[B,T,4096]`。[Qwen3.5 Gated DeltaNet 输出路径](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L504-L543)

## 8. 一步 Decode 的准确计算顺序

已有历史状态后，单个新 token 的输入是 `[B,1,4096]`：

```text
1. Linear 得到 mixed_qkv、a、b、z
2. mixed_qkv 与 conv state 组成长度 4 的局部窗口
3. 因果卷积与 SiLU 得到当前 Q、K、V
4. Q/K 从 16 个头扩展到 32 个头并做 L2Norm
5. 计算 α=exp(g) 和 β=sigmoid(b)
6. 旧 recurrent state 先乘 α
7. k 从衰减后的状态读取旧值
8. β×(v-旧值) 得到修正量
9. 沿 k 方向把修正量写回状态
10. q 从新状态读取输出
11. RMSNorm(output)×SiLU(z)
12. out_proj 回到 [B,1,4096]
```

Transformers 的逐 token 回退实现直接对应第 6 至 10 步：

```text
state = state * exp(g)
old_value = kᵀ state
delta = beta * (v - old_value)
state = state + k × deltaᵀ
output = qᵀ state
```

[Qwen3.5 recurrent Gated Delta Rule](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L331-L380)

FLA 的官方 Triton Kernel 使用同样顺序：先衰减 `b_h`，再计算 `beta×(v-retrieved)`，完成外积更新并读出 q。[FLA fused recurrent Kernel，revision `3c4c54a`](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/fused_recurrent.py#L114-L165)

输出是在当前 token 写入状态之后读取的，因此它可以使用当前位置的信息。这与因果 Full Attention 允许当前位置读取自己相符。

## 9. Prefill 为什么不逐 token 启动 Kernel

递推公式在语义上按 token 有先后顺序。如果 Prefill 真为每个 token 单独启动一次小 Kernel，GPU 利用率会很差。

Gated DeltaNet 论文利用 WY 表示把一段 token 的连续 Delta 更新改写为矩阵计算。实现会把序列分成 Chunk，在 Chunk 内使用较大的矩阵乘法，并把前一个 Chunk 的最终状态传给下一个 Chunk。[Gated DeltaNet §3.3](https://arxiv.org/pdf/2412.06464) [DeltaNet 硬件高效 Chunk 算法](https://arxiv.org/abs/2406.06484)

Qwen3.5 的 Transformers 实现据输入形态选择两条路径：

```text
已有状态且 T=1：recurrent_gated_delta_rule
其他情况：       chunk_gated_delta_rule
```

[Qwen3.5 Prefill 与 Decode 分支](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L432-L535)

FLA 的 Chunk Kernel 先构造 WY 表示，再计算 Chunk 间状态和 Chunk 内输出，最后返回与递推形式相同 shape 的最终状态。[FLA chunk Gated Delta Rule，revision `3c4c54a`](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/chunk.py#L33-L123)

“Chunk 并行”没有取消顺序语义，也不表示每 64 个 token 会清空一次记忆。Chunk 是等价计算的工程组织方式，状态会跨 Chunk 继续传递。

## 10. recurrent state 与 conv state 占多少空间

两类状态都不随上下文长度 `T` 增长，但会随请求数或 Beam 数增长。

### recurrent state

一个层、一个请求包含：

$$
32\times128\times128=524288
$$

个元素。Transformers 回退实现把递归计算转为 FP32，FLA Kernel 也将最终状态分配为 FP32。[Qwen3.5 recurrent fallback](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L331-L380) [FLA 最终状态 dtype](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/fused_recurrent.py#L184-L217)

按 FP32 计算：

```text
每层 recurrent state：2 MiB
24 层：               48 MiB
```

### conv state

一个层的混合 Q/K/V 宽度为 8192，窗口宽度为 4：

$$
8192\times4=32768
$$

个元素。若 conv state 跟随 BF16 模型输入 dtype：

```text
每层 conv state：64 KiB
24 层：          1.5 MiB
```

两者合计约 `49.5 MiB/请求`。这是当前固定 revision 和上述 dtype 假设下的逻辑状态数值容量，不包括分配器、对齐、Kernel 临时张量和 Full Attention KV Cache。其他 runtime 可能采用不同布局或 dtype，不能把这个数字视为所有 Qwen3.5 部署的固定显存值。

## 11. 与 Full Attention 放在一起比较

| 对比项 | Full Attention | Gated DeltaNet |
|---|---|---|
| 历史表示 | 每个位置的 K/V | 每头一张固定状态矩阵 |
| 当前读取 | Q 与所有历史 K 打分 | q 直接读取状态矩阵 |
| 历史长度轴 | 显式保留 | 不显式保留 |
| Decode 状态容量 | 随 `T` 线性增长 | 对 `T` 固定 |
| Decode 读取工作量 | 随历史长度增长 | 对历史长度固定 |
| 历史内容 | 可逐位置访问 | 已汇总进有限状态，可能干扰 |
| Softmax 权重 | 有 | 没有 |
| Qwen 位置处理 | Q/K 使用 RoPE | 因果递推加短卷积，不调用 RoPE |

这张表只比较算法形态，不能直接推出某个模型、Batch 和硬件上的实际延迟。Gated DeltaNet 每个头仍要读写 `128×128` 的状态矩阵，也依赖专用 Kernel；Full Attention 则已有非常成熟的 FlashAttention、PagedAttention 等实现。性能判断必须结合真实 Kernel 和 workload。

Qwen3.5 的混合结构从数值上同时拥有两类历史通道：24 层通过固定状态更新历史，8 层保留逐 token K/V 并执行 Full Attention。可以描述这种结构差异，不能未经官方消融就把某项能力简单归因于某一类层。

## 12. 适合放进正文的讲解顺序

面向数学基础较弱的工程师，建议按以下顺序写：

1. 接上第 4 课，先画出 Qwen3.5 请求同时拥有 KV Cache 和固定状态。
2. 对比 Full Attention 的“保存全部 K/V”和 DeltaNet 的“维护一张状态矩阵”。
3. 用 `2×2` 状态演示“不要重复累加，先算差值再修正”。
4. 加入 `β`，说明一次修正多少。
5. 加入 `α`，说明为什么更新前还要整体遗忘。
6. 再引入 Q、K、V 的读写职责。
7. 把因果卷积放在 Delta Rule 前面，解释独立的 conv state。
8. 代入 Qwen3.5 的真实 shape。
9. 用同一张图分别走一遍 Prefill Chunk 和单 token Decode。
10. 最后比较 Full Attention，并说明 3+1 混合结构。

正文第一遍只需要五行递推公式。论文中的 Householder 矩阵、WY 表示和 Chunk 反向传播适合放在“实现补充”或参考资料中，不应挡在原理主线前面。

建议配四张图：

- `2×2` 状态的第一次写入和同 key 修正图。
- `α` 整体衰减、`β` 定向修正、`z` 输出调节的三阶段图。
- Qwen3.5 单层 shape 流程图。
- Prefill Chunk 与单 token Decode 共享同一递推语义的双栏图。

## 13. 常见错误清单

| 容易写错的说法 | 更准确的说法 |
|---|---|
| Gated DeltaNet 是另一种 Softmax Attention | 它是矩阵状态递推，没有对全部历史 token 做 Softmax |
| recurrent state 就是压缩后的 KV Cache | 它是另一种学习到的固定状态，不是历史 K/V 的无损压缩 |
| `a` 就是 `α` | `a` 是原始投影，先生成负的 `g`，递归中再使用 `α=exp(g)` |
| `α`、`β`、`z` 都是同一个门 | `α` 衰减旧状态，`β` 控制定向修正，`z` 调节当前输出 |
| Delta Rule 删除旧 value 后再写入新 value | 它计算当前返回值与目标 v 的差，并沿 k 方向进行软修正 |
| 状态是一张精确 Key-Value 字典 | key 是连续向量，方向可能干扰，状态容量有限 |
| 因果卷积会缩短 token 序列 | 它沿时间混合当前位置和局部历史，输出 token 数不变 |
| 因果卷积就是 Gated DeltaNet 的位置编码 | 顺序同时来自因果卷积和递归更新，不能只归因于卷积 |
| Chunk Kernel 每个 Chunk 独立，边界会清空状态 | 前一 Chunk 的最终状态会传给后一 Chunk |
| Prefill 真正逐 token 执行同一个 Decode Kernel | 硬件高效实现用 Chunk 算法把递推改写为较大的矩阵运算 |
| Gated DeltaNet 层不需要缓存 | 它不保存逐 token K/V，但要保存 recurrent state 和 conv state |
| 固定状态意味着不受请求数影响 | 状态不随 `T` 增长，但每个活动序列或 Beam 都需要自己的状态 |
| 线性 Attention 一定比 Full Attention 快 | 实际速度取决于状态尺寸、Kernel、Batch、硬件和上下文 |
| Qwen3.5 的 24 个 DeltaNet 层共享一份状态 | 每个层都有自己的 recurrent state 和 conv state |

## 来源

- Songlin Yang, Jan Kautz, Ali Hatamizadeh, [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), ICLR 2025。
- Songlin Yang et al., [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), NeurIPS 2024。
- Qwen, [Qwen3.5-9B-Base 模型卡, revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)。
- Qwen, [Qwen3.5-9B-Base 配置, revision `68c46c4`](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)。
- Hugging Face Transformers, [Qwen3.5 模型实现, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
- Hugging Face Transformers, [Linear Attention Cache 实现, revision `9436284`](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py)。
- FLA Authors, [Flash Linear Attention Gated Delta Rule, revision `3c4c54a`](https://github.com/fla-org/flash-linear-attention/tree/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule)。
- NVIDIA Research, [GatedDeltaNet 官方 PyTorch 实现, revision `b53d6d3`](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921)。
