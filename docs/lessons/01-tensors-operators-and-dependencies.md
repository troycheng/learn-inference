# 第一课：张量、算子、依赖与 Linear 的成本直觉

## 本课目标

学完本课后，应该能够：

1. 用张量、算子和依赖描述一次模型计算；
2. 解释 shape、dtype、layout 和 device 的含义；
3. 计算一个 Linear 的权重大小和计算量；
4. 区分 `FLOPs` 和 `FLOP/s`；
5. 解释为什么有效 token 数会改变硬件行为；
6. 解释 Prompt token 可以并行，而普通生成 token 必须逐步产生。

## 1. 推理的最小本质

先暂时拿掉 PyTorch、TensorRT、vLLM 和 GPU 型号。一次模型计算最基本的表达是：

```text
张量 → 算子 → 新张量
```

多个算子按照数据依赖连接起来，就形成计算图：

```text
输入张量
  → 算子 A
  → 中间张量
  → 算子 B
  → 输出张量
```

分析任何关键节点时，先回答四个问题：

1. 输入和输出是什么形状？
2. 需要多少计算？
3. 需要搬运多少数据？
4. 它依赖谁，能否并行、缓存或重算？

## 2. 张量包含什么

工程上，一个张量至少包含以下属性：

```text
shape + dtype + layout + device
```

例如：

```text
X: [T, H], FP16, row-major, GPU 0
```

表示：

- `T`：本次执行中同时参与计算的有效 token 或样本数量；
- `H`：每个 token 的 hidden dimension；
- `FP16`：每个元素占 2 Bytes；
- `row-major`：元素在内存中的排列方式；
- `GPU 0`：数据所在设备。

相同的数学计算，如果 shape、dtype 或 layout 不同，Runtime 可能选择完全不同的 Kernel，实际性能也会不同。

## 3. 用一个 Linear 建立成本直觉

考虑一个简化的线性层：

$$
Y=XW
$$

张量形状为：

```text
X: [T, H]
W: [H, 4H]
Y: [T, 4H]
```

这里最关键的变量之一是 `T`。在 Transformer 推理中，它可以表示本次迭代参加 Linear 计算的有效 token 数：

- 单请求、普通单 token Decode：`T = 1`；
- 32 个运行中请求一起 Decode：`T = 32`；
- 一个 512-token Prompt 一次完成 Prefill：`T = 512`；
- 多个 Prompt packed 在一起：`T` 是所有有效 Prompt token 数之和；
- 混合批次：`T` 是 Decode token 与本轮 Prefill chunk token 数之和。

这个 Linear 的矩阵乘法计算量近似为：

$$
F=2\times T\times H\times 4H
$$

其中一次乘法和一次加法按 2 FLOPs 计算。

权重存储大小为：

$$
B_W=H\times4H\times dtypeBytes
$$

两个量随 `T` 的变化不同：

- 计算量随 `T` 线性增加；
- 权重存储大小不随 `T` 变化；
- `T` 越大，同一份权重被更多输入行复用。

## 4. 代入一个实际尺度

令：

```text
H = 4096
W = [4096, 16384]
dtype = FP16
```

### 4.1 权重大小

权重元素数量为：

$$
4096\times16384=67,108,864
$$

FP16 每个元素占 2 Bytes：

$$
67,108,864\times2=134,217,728\ Bytes
$$

换算后：

```text
134,217,728 Bytes
= 128 MiB
≈ 0.134 GB
```

`GB` 是十进制单位，`MiB/GiB` 是二进制单位，因此也可以写成 `0.125 GiB`。

### 4.2 当 T = 1

计算量为：

$$
F=2\times1\times4096\times16384
=134,217,728\ FLOPs
\approx0.134\ GFLOPs
$$

只计算权重读取时，算术强度约为：

$$
AI_W=
\frac{134,217,728\ FLOPs}
{134,217,728\ Bytes}
\approx1\ FLOP/Byte
$$

这个结果也可以直接推导：

$$
AI_W=
\frac{2\times T\times H\times4H}
{H\times4H\times dtypeBytes}
=\frac{2T}{dtypeBytes}
$$

FP16 下 `dtypeBytes=2`，所以权重算术强度近似为：

$$
AI_W\approx T
$$

当 `T=1` 时，每读取 2 Bytes 的一个 FP16 权重，只进行大约一次乘法和一次加法。计算核心可能没有足够工作，时间更容易由权重搬运决定。

### 4.3 当 T = 512

计算量变成：

$$
F=2\times512\times4096\times16384
\approx68.7\ GFLOPs
$$

权重仍然是 128 MiB。只考虑权重读取时：

$$
AI_W\approx512\ FLOPs/Byte
$$

同一份权重被 512 个 token 复用，计算相对数据搬运显著增加，因此更可能接近计算能力上限。

这个分析忽略了输入输出读写、缓存、tiling、其他算子和通信，只用于建立数量级与方向判断。真实算术强度必须使用实际内存流量。

## 5. FLOPs 与 FLOP/s

这两个术语必须严格区分：

- `FLOPs`：完成某次任务需要的浮点计算次数，是计算量；
- `FLOP/s`：设备每秒完成的浮点计算次数，是计算速率。

计算时间近似为：

$$
T_{compute}=
\frac{\text{计算量（FLOPs）}}
{\text{计算速率（FLOP/s）}}
$$

因此：

- 计算量增加，计算时间倾向于增加；
- 计算速率提高，计算时间倾向于降低。

如果总时间主要由数据搬运决定，提高理论计算速率不一定显著降低整体延迟：

$$
T\approx
\max\left(
\frac{F}{P},
\frac{B}{BW}
\right)+T_{overhead}
$$

提高 `P` 只会降低 `F/P`。如果 `B/BW` 更大，总时间仍然主要由带宽决定。

## 6. 为什么 Prefill 和 Decode 行为不同

不要只记忆“Prefill 偏 compute-bound、Decode 偏 bandwidth-bound”，而要理解推导过程。

当本次有效 token 数 `T` 较小时：

```text
权重缺少复用
→ 每做少量计算就需要读取大量权重
→ 算术强度低
→ 更容易受带宽、延迟和 launch overhead 限制
```

当 `T` 增大时：

```text
权重被更多 token 复用
→ 单位权重读取对应更多计算
→ 算术强度提高
→ 更容易接近计算上限
```

这是一种趋势，不是固定分类。实际瓶颈还取决于模型结构、上下文长度、batch、硬件、Kernel 和通信。

## 7. 为什么混合 Prefill chunk 和 Decode token

假设当前有：

```text
Decode：32 个 token
Prefill chunk：256 个 token
```

如果分开执行，可以概念化为：

```text
加载权重 → 计算 32 行
再次加载权重 → 计算 256 行
```

如果 Runtime 能把相应 Linear 工作合并为一个 `T=288` 的批次，同一份权重可以服务更多输入行，从而提高整体算术强度。

Decode 提供需要及时推进的请求，Prefill chunk 提供更多可并行计算。组合后可能同时改善带宽利用和计算利用。

但 Prefill chunk 不能无限增大，因为同一 iteration 中的 Decode 要等待这轮工作结束：

```text
Prefill chunk 太大
→ iteration 变长
→ Decode 等待变长
→ ITL/TPOT 可能恶化
```

因此 Chunked Prefill 的关键是控制每轮 token budget，在吞吐、TTFT 和 ITL 之间取舍。

## 8. 为什么理论算力提高不一定带来同比例 Decode 提升

考虑一个简化示例：

```text
模型权重：16 GB
显存带宽：1 TB/s
```

如果低 batch Decode 每一步需要从 HBM 读取大部分权重，只考虑权重读取的时间下界约为：

$$
\frac{16\ GB}{1\ TB/s}=16\ ms
$$

假设原计算时间为 3 ms：

$$
T\approx\max(3ms,16ms)=16ms
$$

理论计算速率翻倍后，计算时间降至约 1.5 ms：

$$
T\approx\max(1.5ms,16ms)=16ms
$$

因此只提高 `FLOP/s`，没有改变当前更大的带宽时间项。现实中还要考虑缓存、量化、分片和实际有效带宽，示例只说明瓶颈决定优化收益。

## 9. 为什么小模型、小 batch 容易受固定开销影响

一次 GPU Kernel 的执行包含：

```text
CPU 准备和提交
→ GPU 接收工作
→ Kernel 执行
→ 必要的同步或后续提交
```

总时间可以概念化为：

$$
T_{total}=
\sum_{i=1}^{N}
\left(T_{launch,i}+T_{kernel,i}\right)
+T_{gap}
$$

当模型或 batch 很大时，Kernel 计算时间较长，launch 的相对占比较小。小模型、小 batch 会缩短 Kernel 本身的执行时间，但 CPU 提交、launch 和同步等固定开销不会同比缩短。

一个 Transformer 层通常会产生多个 Linear、Norm、Attention、激活和残差相关 Kernel，再乘以很多层。单次固定开销即使不大，累计后也可能在 GPU timeline 上形成明显空隙。

因此：

- Kernel Fusion 减少 Kernel 数量和中间数据读写；
- CUDA Graph 减少 CPU 重复提交开销；
- Persistent Kernel 或 MegaKernel 尝试进一步减少 launch、同步和中间数据往返；
- 增大 batch 让一次提交承载更多有效工作。

是否值得采用这些方法，仍要由 timeline 和 profile 证明。

## 10. 两种并行不能混淆

### 一次 forward 内部的并行

同一 Linear 中，不同 token 可以组成一个更大的矩阵：

```text
多个已知 token → 一个较大的 GEMM
```

### 多个生成步骤之间的依赖

自回归生成过程是：

```text
token 1 = sample(model(prompt))
token 2 = sample(model(prompt + token 1))
token 3 = sample(model(prompt + token 1 + token 2))
```

计算 token 2 前必须知道 token 1，因此普通 Decode 无法预先构造未来 100 个输出 token 对应的输入矩阵。

Prompt 不同：Prompt token 已由用户给出，值是已知的。Causal mask 限制每个位置只能读取之前位置，但各位置在同一层的计算仍可并行安排。

Speculative Decoding 会先猜测若干未来 token，再由目标模型并行验证。它没有消除依赖，而是用候选生成和验证减少目标模型的串行执行次数。

## 11. 理解检查

### 问题 1

4 个请求正在普通 Decode，每个请求本轮处理一个 token。这一层 Linear 的 `T` 大约是多少？

**答案：** `T=4`。前提是四个请求都处于运行状态，没有混入 Prefill 或 speculative token。

### 问题 2

4 个 Prompt 的长度分别为 100、200、300、400。若进行一次 packed prefill，`T` 大约是多少？

**答案：**

$$
T=100+200+300+400=1000
$$

如果全部 padding 到最长的 400，则会处理 `4×400=1600` 个位置，其中 600 个位置是 padding。

### 问题 3

`T` 从 8 增大到 64，Linear 的计算量和权重大小分别怎样变化？

**答案：**

- 计算量 `FLOPs` 增加 8 倍；
- 权重元素数量和存储大小不变；
- 输入输出激活大小增加 8 倍；
- 实际计算速率 `FLOP/s` 可能因利用率提高而上升，但不保证提高 8 倍；
- 延迟也不一定增加 8 倍，因为固定权重读取成本被更多输入摊薄。

### 问题 4

为什么不能把同一个请求未来生成的 100 个 token 直接当成 `T=100` 做普通并行计算？

**答案：** 后一个 token 的输入依赖前一个 token 的模型输出和采样结果。未来 token 尚未确定，因此普通自回归生成必须逐步推进。

## 12. 常见疑问速查

### 0.134 GB 从哪里来？

它是 `W=[4096,16384]` 的 FP16 权重大小：

$$
4096\times16384\times2\ Bytes
=134,217,728\ Bytes
\approx0.134\ GB
$$

同一个数用二进制单位表示是 `128 MiB`。

### 为什么混合 Prefill chunk 和 Decode token 可能提高利用率？

Linear 可以把两类 token 组成更大的输入矩阵，让一次权重读取服务更多输入行，提高权重复用和算术强度。代价是 Prefill chunk 会延长当前 iteration，过大时会损害 Decode 的 ITL/TPOT。

### 为什么更高的理论算力不一定带来同比例 Decode 提升？

理论计算速率 `FLOP/s` 只影响计算时间 `F/P`。如果更大的时间项是权重或 KV 的搬运时间 `B/BW`，只提升计算速率不会消除当前瓶颈。

### 为什么小模型、小 batch 更容易受 Kernel launch 和 CPU 调度影响？

小 workload 会缩短 Kernel 的计算时间，但提交、launch 和同步等近似固定开销不会同比缩短。当模型包含很多短 Kernel 时，这些固定成本会累积并在 GPU timeline 上形成空隙。

### FLOPs 和 FLOP/s 有什么区别？

`FLOPs` 是完成任务所需的计算量；`FLOP/s` 是设备执行计算的速率。前者增加会增加工作量，后者提高才会降低同等工作量的计算时间。

## 13. 本课结论

本课最重要的不是记住例子中的数字，而是建立以下关系：

1. `T` 是本次执行中参与计算的有效 token/样本数量；
2. Linear 的计算量随 `T` 增长，权重存储大小不变；
3. 增大 `T` 会增加计算量，也会提高权重复用；
4. `FLOPs` 是计算量，`FLOP/s` 是计算速率；
5. 性能由当前最大的时间项决定，理论算力只是其中一项；
6. 已知输入可以在 forward 内并行，未知的自回归结果构成串行依赖。

下一课将分别建立计算账、容量账和数据搬运账，并把这些公式应用到一个完整模型。

## 参考资料

- [ONNX Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
