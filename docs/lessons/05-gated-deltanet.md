# 第 5 课：Gated DeltaNet 的状态读写

第 4 课讲过，Qwen3.5-9B 的 32 个 Decoder Layer 中，8 层使用 Full Attention，另外 24 层使用 Gated DeltaNet。两种层的 RMSNorm、残差连接和 FFN 相同，区别在 Token Mixer：

```text
Full Attention：保存每个历史位置的 K/V，当前 Q 再逐位置读取
Gated DeltaNet：每来一个 token 就修改一张固定大小的状态矩阵
```

Qwen3.5-9B 每 4 层中有 3 个 Gated DeltaNet 层和 1 个 Full Attention 层，这组排列重复 8 次。

![Qwen3.5 的混合 Token Mixer](../assets/05-hybrid-layout.svg)

后面的公式都从一张 `2×2` 状态矩阵开始。先看一次读写，再逐个加入三个控制量，最后换成 Qwen3.5 的真实 shape。

## 1. 从一排 K/V 到一张状态矩阵

Full Attention 保留逐 token 历史：

```text
k1/v1  k2/v2  k3/v3  ...  kt/vt
```

当前 Query 可以分别与每个历史 Key 打分，再按权重汇总 Value。上下文越长，这排 K/V 越长。

Gated DeltaNet 不保存这排历史。每个头维护一张状态矩阵：

```text
S：[Dk,Dv]
```

序列继续增长时，`S` 的 shape 不变，矩阵里的数值会被反复更新。Q、K、V 在这里的分工是：

| 向量 | 在 Gated DeltaNet 中做什么 |
| --- | --- |
| K | 定位状态中要检查和修改的方向 |
| V | 给出这个方向当前希望记录的内容 |
| Q | 从更新后的状态中读取本次输出 |

可以暂时把 `S` 看成一组连续数值的 Key 到 Value 关联。它不是 Python Dictionary，也没有为某个 token 单独保留一行。不同 Key 的方向不完全独立时，多个关联会在同一张矩阵里混合和干扰。

递归状态也不是 token 的隐藏状态。隐藏状态是某个 token 在层与层之间传递的向量；递归状态属于某个 Gated DeltaNet 层，会跨 token 保留并原地更新。

## 2. 手算一次状态更新

只看一个头，把 K、Q、V 都缩成 2 维。为了看清读写位置，先用容易手算的 `k=[1,0]`，并从全零状态开始：

$$
S_0=
\begin{bmatrix}
0&0\\
0&0
\end{bmatrix}
$$

第一个 token 产生：

$$
k_1=[1,0],\qquad v_1=[3,4]
$$

### 2.1 先读旧值

用 Key 读取当前状态：

$$
\hat v_1=k_1S_0=[0,0]
$$

`k_1=[1,0]` 在这个教学例子中选中了状态的第一行。状态还没有写过，所以读到 `[0,0]`。

### 2.2 只写入差值

当前希望记录 `[3,4]`，旧状态返回 `[0,0]`，需要修正：

$$
\Delta_1=v_1-\hat v_1=[3,4]
$$

`k_1^T` 的 shape 是 `[2,1]`，`\Delta_1` 的 shape 是 `[1,2]`。两者做外积，得到与状态同 shape 的更新矩阵：

$$
\begin{aligned}
k_1^T\Delta_1&=
\begin{bmatrix}1\\0\end{bmatrix}
\begin{bmatrix}3&4\end{bmatrix}
\\
&=
\begin{bmatrix}3&4\\0&0\end{bmatrix}
\end{aligned}
$$

加回旧状态：

$$
S_1=S_0+k_1^T\Delta_1=
\begin{bmatrix}3&4\\0&0\end{bmatrix}
$$

如果当前 Query 也是 `q_1=[1,0]`，它会从新状态读出：

$$
q_1S_1=[3,4]
$$

### 2.3 同一个 Key 出现新 Value

第二个 token 仍然产生 `k_2=[1,0]`，但希望记录的新 Value 是：

$$
v_2=[5,1]
$$

状态当前返回 `[3,4]`。这次无需把 `[5,1]` 整体再加一次，只需写入两者的差：

$$
\Delta_2=[5,1]-[3,4]=[2,-3]
$$

$$
S_2=S_1+k_2^T\Delta_2=
\begin{bmatrix}5&1\\0&0\end{bmatrix}
$$

如果直接累加新的 `k_2^Tv_2`，第一行会变成 `[8,5]`。Delta Rule 把旧记录 `[3,4]` 修正成 `[5,1]`。名称中的 Delta 指的就是这段误差。

![Delta Rule 先读旧值，再按误差修正](../assets/05-delta-update.svg)

实际 Q/K 是连续向量，不是 `[1,0]` 这样的 one-hot。一次更新通常会影响状态的多行，相近的 Key 也可能相互干扰。这个二维例子只用于看清计算顺序。

## 3. β 控制这次修正多少

真实模型不一定把全部误差写回。`β` 位于 0 和 1 之间，用来缩放当前修正量：

$$
e_t=\beta_t(v_t-\hat v_t)
$$

仍从 `S_1` 开始，第二次更新的误差是 `[2,-3]`。如果 `β=0.5`，只写入一半：

$$
e_2=0.5[2,-3]=[1,-1.5]
$$

$$
\begin{aligned}
S_2&=
\begin{bmatrix}3&4\\0&0\end{bmatrix}
+
\begin{bmatrix}1&-1.5\\0&0\end{bmatrix}
\\
&=
\begin{bmatrix}4&2.5\\0&0\end{bmatrix}
\end{aligned}
$$

`β` 接近 0 时，这次 Value 几乎不改变状态；接近 1 时，模型更积极地把当前返回值推向新 Value。Qwen3.5 先用 Linear 产生 `b`，再通过 Sigmoid 得到 `β`：

$$
\beta=\mathrm{sigmoid}(b)=\frac{1}{1+e^{-b}}
$$

## 4. α 先决定旧状态保留多少

`β` 只控制当前 Key 方向的修正。Gated DeltaNet 还会在每次更新前，用 `α` 缩放整个旧状态：

$$
\bar S_t=\alpha_tS_{t-1}
$$

`α` 也位于 0 和 1 之间。接近 1 时，大部分旧状态得到保留；接近 0 时，旧状态快速减弱。

继续使用前面 `β=1` 时得到的状态：

$$
S_2=
\begin{bmatrix}5&1\\0&0\end{bmatrix}
$$

第三个 token 取：

$$
\alpha_3=0.5,\quad \beta_3=0.5,\quad
k_3=[1,0],\quad v_3=[3,3]
$$

先衰减旧状态：

$$
\bar S_3=0.5S_2=
\begin{bmatrix}2.5&0.5\\0&0\end{bmatrix}
$$

Key 读到 `[2.5,0.5]`，与目标 `[3,3]` 的误差是 `[0.5,2.5]`。`β=0.5` 写入一半误差：

$$
S_3=\bar S_3+0.5k_3^T[0.5,2.5]=
\begin{bmatrix}2.75&1.75\\0&0\end{bmatrix}
$$

`α` 和 `β` 的作用位置不同：`α` 先缩放整个旧状态，`β` 再控制当前 Key 方向修正多少。

### 4.1 完整的状态读写公式

把前面的步骤写在一起：

$$
\bar S_t=\alpha_tS_{t-1}
$$

$$
\hat v_t=k_t\bar S_t
$$

$$
e_t=\beta_t(v_t-\hat v_t)
$$

$$
S_t=\bar S_t+k_t^Te_t
$$

$$
o_t=\frac{q_tS_t}{\sqrt{D_k}}
$$

前四行完成状态更新，最后一行用 Query 读取输出。这里使用行向量记法，状态 shape 是 `[Dk,Dv]`。论文有时采用转置后的状态布局，公式左右会随之互换，数学含义相同。

Qwen3.5 的代码先计算负数 `g`，递归计算实际使用 `α=exp(g)`。代码中的原始投影 `a` 不是 `α`。`β` 则由 `sigmoid(b)` 得到。

## 5. z 只调节当前输出

状态读出的 `o_t` 不会直接送入 `out_proj`。Qwen3.5 还为当前 token 产生 `z_t`：

```text
状态读出 o_t
→ RMSNorm
→ 与 SiLU(z_t) 逐元素相乘
→ 拼回所有头
→ out_proj
→ H 维输出
```

`z` 只调节本次送出哪些特征，不修改递归状态。它与 `α`、`β` 的分工如下：

| 控制量 | 作用位置 | 控制什么 |
| --- | --- | --- |
| `α` | 更新开始前 | 旧状态保留多少 |
| `β` | 误差写回时 | 当前修正写入多少 |
| `z` | 状态读出后 | 哪些输出特征送往后续模块 |

`SiLU(z)` 不限于 0 到 1，因此 `z` 不是只能开关的二值门。它既可以压低特征，也可以改变特征的尺度和符号。

![α、β 和 z 分别作用在状态更新与输出的哪个位置](../assets/05-three-controls.svg)

## 6. 因果卷积先读取最近几个位置

前面的公式只描述递归状态。Qwen3.5 在状态更新前，还会让 Q/K/V 的每个通道先混入很短的局部历史：

```text
隐藏状态 X
→ Linear 得到混合 Q/K/V 通道
→ 沿 token 轴做因果卷积
→ SiLU
→ 拆成 Q、K、V
→ 更新递归状态
```

“因果”表示当前位置只能读取自己和左侧位置。假设某个通道的输入是：

```text
位置：p1  p2  p3  p4
数值： 2   5   1   4
```

用长度为 3 的教学窗口和权重 `[0.2,0.3,0.5]` 计算 `p3`：

$$
0.2\times2+0.3\times5+0.5\times1=2.4
$$

`p4` 仍属于未来，不能参与这次计算。

![计算 p3 时，因果卷积只读取 p1、p2 和 p3](../assets/05-causal-convolution.svg)

Qwen3.5 的真实窗口宽度是 4，卷积权重由训练得到。这里使用深度卷积，即每个通道分别沿 token 轴卷积；这一步不会改变 token 数量，也不会在通道之间相互混合。

Decode 每次只有一个新 token，但长度为 4 的卷积还需要最近几个投影值。因此每个 Gated DeltaNet 层还要保存 `conv_state`。它与 Delta Rule 使用的 `recurrent_state` 是两份状态：

```text
conv_state       保存最近几个位置的局部窗口
recurrent_state  保存长期更新的状态矩阵
```

因果卷积不是 RoPE 的替代品。Gated DeltaNet 的顺序信息同时来自局部因果卷积和按 token 顺序更新的递归状态。

## 7. 一个 token 经过 Gated DeltaNet 的完整过程

前面的部件现在可以接成一条数据流：

1. `X` 经过 Linear，产生混合 Q/K/V、`a`、`b` 和 `z`。
2. 混合 Q/K/V 与 `conv_state` 组成局部窗口，经过因果卷积和 SiLU。
3. 结果拆成 Q、K、V。Q/K 做 L2 归一化，并扩展到与 Value 头数量一致。
4. `a` 经过变换得到 `g` 和 `α=exp(g)`，`b` 经过 Sigmoid 得到 `β`。
5. 用 `α`、`β`、K 和 V 更新 `recurrent_state`，再用 Q 读取输出。
6. 读出结果经过 RMSNorm，与 `SiLU(z)` 逐元素相乘。
7. 所有头拼回后通过 `out_proj`，输出重新回到 `[B,T,H]`。

![Gated DeltaNet 的完整数据流](../assets/05-gated-deltanet-flow.svg)

Q/K 做 L2 归一化，是把每条向量缩放到长度约为 1。这样状态的读写方向主要由各维度的相对比例决定，不会因为整条 Q 或 K 同时放大几倍而突然增强。Qwen3.5 还在 Query 读出时乘以 `1/sqrt(Dk)`，用于控制数值尺度。

## 8. Prefill 和 Decode 使用同一条递归规则

状态更新在数学上有先后关系：`S_t` 依赖 `S_{t-1}`。Prefill 如果真的为每个 Prompt token 单独启动一次小 Kernel，GPU 利用率会很差。

Qwen3.5 的实现采用两条执行路径：

| 输入情况 | Kernel 组织方式 | 最终状态 |
| --- | --- | --- |
| 多个已知位置，或没有可复用状态 | Chunk Gated Delta Rule | 返回最后一个位置后的状态 |
| 已有状态且 `T=1` | Recurrent Gated Delta Rule | 原地更新单步状态 |

Chunk Kernel 把一段已知序列改写成更大的矩阵计算，并在 Chunk 之间传递最终状态。它改变计算组织方式，不会在 Chunk 边界清空历史，也不会改变逐 token 递推的数学结果。

![Prefill 与 Decode 怎样更新固定状态](../assets/05-prefill-decode-state.svg)

这里的 Chunk Kernel 是算子内部算法。第 4 课的 Chunked Prefill 是服务端把长 Prompt 分配到多个调度轮次。两者都使用 Chunk 这个词，但切分层级不同。

## 9. Qwen3.5-9B 的真实 shape

Qwen3.5-9B 的配置是：

```text
Hidden Size H            = 4096
Q/K 头数量               = 16
V 头数量                 = 32
Q/K 头维 Dk              = 128
V 头维 Dv                = 128
因果卷积宽度             = 4
```

输入 `X:[B,T,4096]` 后，各分支的 shape 是：

| 数据 | shape | 说明 |
| --- | --- | --- |
| 混合 Q/K/V | `[B,T,8192]` | `2048 + 2048 + 4096` |
| Q，扩头前 | `[B,T,16,128]` | 16 个 Q 头 |
| K，扩头前 | `[B,T,16,128]` | 16 个 K 头 |
| V | `[B,T,32,128]` | 32 个 V 头 |
| Q/K，扩头后 | `[B,T,32,128]` | 每个 Q/K 头供 2 个 V 头使用 |
| `β` | `[B,T,32]` | 每个 V 头一个修正幅度 |
| `g` | `[B,T,32]` | 每个 V 头一个对数衰减值 |
| `z` | `[B,T,32,128]` | 对每个输出特征做门控 |
| `recurrent_state` | `[B,32,128,128]` | 每层固定 shape |
| `conv_state` | `[B,8192,4]` | 每层固定窗口 |
| `out_proj` 后 | `[B,T,4096]` | 回到残差接口宽度 |

混合 Q/K/V 的 8192 维来自：

$$
16\times128+16\times128+32\times128=8192
$$

Q/K 从 16 个头扩展到 32 个头，是为了与 32 个 Value 头一一对应。这是计算视图中的重复，不表示模型训练了两套相同权重。

每个 Gated DeltaNet 层、每个请求有 32 张 `128×128` 状态矩阵。当前参考实现中，递归状态按 FP32 计算约为 2 MiB/层；24 层约 48 MiB。卷积状态若按 BF16 计算，24 层约 1.5 MiB。两类状态合计约 49.5 MiB/请求，不含分配、对齐、临时张量和 8 个 Full Attention 层的 KV Cache。

## 10. 与 Full Attention 对照

| 对比项 | Full Attention | Gated DeltaNet |
| --- | --- | --- |
| 历史形式 | 每个位置一份 K/V | 每头一张固定状态矩阵 |
| 当前 token 怎样读取 | Q 与所有历史 K 打分，再汇总 V | Q 直接读取状态矩阵 |
| 状态容量 | 随上下文长度增长 | 与上下文长度无关 |
| 旧信息怎样变化 | 历史 K/V 保持原值 | 会被衰减、覆盖和混合 |
| 单层 Decode 的历史读取量 | 随上下文增长 | 固定 |
| Softmax 权重 | 有 | 没有 |
| 顺序信息 | Q/K 使用 RoPE 和因果遮罩 | 因果卷积与递归更新顺序 |

固定 shape 不表示无损保存无限历史。许多 token 共用同一张状态矩阵，旧信息会衰减、覆盖，也会出现 Key 方向之间的干扰。Full Attention 保存逐位置 K/V，读取成本更高，但当前 Query 可以直接对不同历史位置分别打分。

这组结构差异会影响推理系统：

- 上下文变长时，8 层 Full Attention 的 KV Cache 继续增长，24 层 Gated DeltaNet 状态保持固定 shape。
- Prefix Cache 要恢复完整前缀状态，除了 Full Attention 的 K/V，还要匹配各 Gated DeltaNet 层的卷积状态和递归状态。
- 每条活动序列或 Beam 都需要自己的状态。Batch 重排时，runtime 必须让状态跟随正确请求。
- 固定状态仍有计算和访存成本。每个新 token 都要完成投影、卷积、矩阵状态读写、门控和输出投影。

因此，Gated DeltaNet 不是“压缩后的 KV Cache”，也不能只根据固定 shape 就断言它在所有硬件和 Batch 下更快。

## 11. 练习：完成一次状态更新

只看一个头，给定：

$$
S_{t-1}=
\begin{bmatrix}
2&1\\
0&3
\end{bmatrix},\quad
\alpha=0.5,\quad \beta=0.5
$$

$$
k=[1,0],\quad v=[4,2],\quad q=[1,0],\quad D_k=2
$$

按下面的顺序计算：

1. 衰减后的状态 `\bar S_t`。
2. Key 读到的旧值 `\hat v_t`。
3. 乘过 `β` 的修正量 `e_t`。
4. 更新后的状态 `S_t`。
5. Query 读出的 `o_t`。

然后回答两个判断题：

1. 上下文从 4096 增长到 8192 时，`recurrent_state` 的 shape 会不会翻倍？
2. 复用 Qwen3.5 的前缀时，只复用 8 个 Full Attention 层的 KV Cache 是否足够？

<details>
<summary>查看答案</summary>

旧状态先衰减：

$$
\bar S_t=0.5S_{t-1}=
\begin{bmatrix}1&0.5\\0&1.5\end{bmatrix}
$$

Key 读取旧值：

$$
\hat v_t=k\bar S_t=[1,0.5]
$$

计算误差并写入一半：

$$
e_t=0.5([4,2]-[1,0.5])=[1.5,0.75]
$$

$$
S_t=\bar S_t+k^Te_t=
\begin{bmatrix}2.5&1.25\\0&1.5\end{bmatrix}
$$

Query 读取并按 `\sqrt{D_k}` 缩放：

$$
o_t=\frac{qS_t}{\sqrt{2}}
=\frac{[2.5,1.25]}{\sqrt{2}}
\approx[1.768,0.884]
$$

上下文翻倍时，`recurrent_state` 的数值会继续更新，shape 不会翻倍。Qwen3.5 仍有 8 个 Full Attention 层保存随长度增长的 KV Cache，所以整个请求状态并非常数大小。

只复用 Full Attention 的 K/V 不足以恢复完整前缀。24 个 Gated DeltaNet 层的 `conv_state` 和 `recurrent_state` 也必须与该前缀对应。

</details>

## 参考资料

以下 Qwen3.5、Transformers 和 FLA 实现于 2026-08-08 按固定 revision 复核：

- [Qwen3.5-9B-Base 模型卡，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/README.md)
- [Qwen3.5-9B-Base `config.json`，revision 68c46c4](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/68c46c4b3498877f3ef123c856ecfde50c39f404/config.json)
- [Transformers：Qwen3.5 模型实现，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Transformers：Linear Attention Cache，revision 9436284](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py)
- [Flash Linear Attention：Gated Delta Rule，revision 3c4c54a](https://github.com/fla-org/flash-linear-attention/tree/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule)
- [NVIDIA：GatedDeltaNet 官方实现，revision b53d6d3](https://github.com/NVlabs/GatedDeltaNet/tree/b53d6d3a161267432a79c1c04af69fa52bddc921)

算法原理参考：

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)

---

[上一课：Prefill、Decode 与 KV Cache](04-prefill-decode-kv-cache.md) · [返回课程路线](../roadmap.md) · [下一课：Dense FFN 与 MoE](06-dense-and-moe.md)
