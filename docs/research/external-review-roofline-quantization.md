# 外部建议核验：Roofline、量化与并行

**范围。** 本文只核验转述给第 3、8、9 课的八组外部建议；不修改课程正文。结论只依赖原论文、NVIDIA 官方规格/文档或框架源码与官方文档。这里的“采纳”指可作为通用课程的表述；不是采纳某个未经给出配置的性能数字。

## 结论总表

| # | 外部建议的核心论点 | 结论 | 课程级判断 |
| --- | --- | --- | --- |
| 1 | 用 Roofline 解释算力、带宽与 V100/A100/H100 拐点 | 改写后采纳 | 定义和同口径的三张卡数字可用；峰值 precision、稀疏性和卡型必须写清。 |
| 2 | 用 A6000/H20 的拐点数直接说明“哪张卡更适合 Decode” | 不采纳 | 常见的约 `50` 与约 `37` FLOP/Byte 混用了 FP32 CUDA 峰值、BF16 Tensor 峰值和不同 SKU；不能比较。 |
| 3 | Decode 的 AI 约为 `1 FLOP/Byte` | 改写后采纳 | 这是 BF16/FP16、batch=1、一次读取每个活跃权重的 weight-only 近似，不是 Decode 的通用常数。 |
| 4 | FP8/INT8 同时带来三类收益且精度通常只损失固定比例 | 改写后采纳 | 可以讲容量、带宽、低精度 Tensor Core/Kernels 三条路径；固定精度损失百分比不成立。 |
| 5 | 用一个固定的 `B_sat` 与一个固定的 KV 读取交叉点指导 batching | 改写后采纳 | 可给条件化的 Roofline 推导；没有跨模型、上下文和 Kernel 通用的阈值。 |
| 6 | MFU/MBU 与“70B 在 H100 上 24 tok/s”说明 Decode 利用率低 | 改写后采纳 | 利用率定义可以引入；`24 tok/s` 没有并行度、dtype、上下文和测量边界时不可采用。 |
| 7 | GQA/MLA 都能按 KV bytes 解释 | 采纳 | 公式可靠，但 MLA 必须由该模型的 latent/rope 维度计算，不能套 GQA 的头数公式。 |
| 8 | KV eviction/quantization、NVFP4/TurboQuant 和跨机 TP 都是通用的“提速项” | 改写后采纳 | 方法与通信机制可讲；各论文数值只能带实验条件，NVFP4 与 TurboQuant 的不同实现/基线不能合成同一倍率。 |

## 1. Roofline 与 V100/A100/H100

### 判定：改写后采纳

原始 Roofline 定义的横轴是相对于主存流量的 operational intensity（此处可记为 `AI`），不是“模型参数量”或某张 GPU 的指标。忽略其他 ceiling 时：

$$
P_{attainable}\leq \min(P_{peak},\ AI\times BW_{peak})
$$

因此 ridge point / machine balance 为：

$$
AI_{ridge}=\frac{P_{peak}}{BW_{peak}}.
$$

原始论文明确将模型定义为浮点性能、operational intensity 和可持续内存带宽的上界关系，而不是实测性能预测器。[Williams, Waterman, Patterson, *Roofline*](https://doi.org/10.1145/1498765.1498785) [作者公开稿](https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S09/handouts/papers/RooflineVyNoYellow.pdf)

若选择 **SXM 规格、dense BF16/FP16 Tensor Core 峰值**，可复算为：

| GPU | `P_peak` | `BW_peak` | `P_peak/BW_peak` |
| --- | ---: | ---: | ---: |
| V100 SXM2 | 125 TFLOP/s | 900 GB/s | 138.9 FLOP/Byte |
| A100 SXM4 80 GB | 312 TFLOP/s | 2,039 GB/s | 153.0 FLOP/Byte |
| H100 SXM5 | 989 TFLOP/s | 3,350 GB/s | 295.2 FLOP/Byte |

规格来源：[V100 官方产品页/数据表](https://www.nvidia.com/en-us/data-center/v100/)、[A100 官方数据表](https://resources.nvidia.com/en-us-gpu-resources/nvidia-a100-datashee-1)、[H100 官方数据表](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)。H100 表中还列 FP8、稀疏和不同卡型峰值；它们不能与上表 dense BF16/FP16 任意互换。

**风险。**

- 使用 PCIe 而不是 SXM、FP32 CUDA 而不是 BF16 Tensor Core、或把 2:4 sparsity 峰值当 dense 峰值，都会得到另一组 ridge point。
- `BW_peak` 是规格上界；Roofline 原文建议用可持续带宽 microbenchmark 建模。课程不应把上表称为实际 Kernel 的实测拐点。
- 操作的 FLOP 统计、dtype、稀疏模式必须与所选 `P_peak` 对齐。

## 2. A6000/H20 “拐点”数字

### 判定：不采纳

NVIDIA 的 RTX A6000 官方资料给出 768 GB/s 显存带宽、38.7 FP32 TFLOP/s，并另列 309.7 Tensor TFLOP/s（稀疏口径）。所以至少可得到 `38.7/0.768≈50.4 FLOP/Byte`（FP32 CUDA），或需先明确 dense/sparse 与 Tensor 格式后才可计算另一个值。[NVIDIA 专业 GPU 官方 line card](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/design-visualization/quadro-product-literature/rtx-6000-l40-linecard-nvidia-us-2653097-r7-web.pdf)

H20 的公开 NVIDIA 文档可确认有 96 GB/141 GB 等 SKU 和 Hopper/FP8 支持，但本次未找到能同时给出对应卡型、Tensor precision、dense/sparse 峰值与 HBM 带宽的公开官方数据表。因此“`H20≈37 FLOP/Byte`”即使可由流传的 `148 TFLOP/s ÷ 4 TB/s` 算出，也不满足本核验要求的可追溯一手规格。[NVIDIA H20 vGPU SKU 文档](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper.html)

**风险。** `50` 与 `37` 即使各自算术正确，也若一个来自 FP32 CUDA、另一个来自 BF16 Tensor Core，就回答了不同问题。不能据此写“某卡 Decode 一定更计算受限/带宽受限”，更不能跨卡比较。

## 3. Decode 的 `AI≈1 FLOP/Byte`

### 判定：改写后采纳

对一个 dense Linear，batch=1，假设本轮每个活跃 BF16/FP16 权重从 HBM 读取一次、忽略激活/KV/scale/输出流量：

$$
F\approx2H_{in}H_{out},\qquad Q_{weight}\approx sH_{in}H_{out},\qquad
AI_{weight}\approx\frac{2}{s}.
$$

故 `s=2 Byte` 时为约 `1 FLOP/Byte`；INT8/FP8 weight-only 的理想编码字节 `s=1` 时是约 `2 FLOP/Byte`，但它不保证 Kernel、累加和实际读取字节也相应变化。该近似正是第 8 课现有 BF16 Linear 例子的适用边界。

把一轮的 token 数记为 `B`，并理想化地认为权重被复用且仍只从 HBM 搬一次，则只有 weight 项时：

$$
AI_{weight}(B)\approx\frac{2B}{s}.
$$

这个推导与 Roofline 原则一致；它不需要也不应伪造一个“所有 Decode 都是 1”的论文实测数。PagedAttention 论文也说明 KV 是随 batch 和序列长度增长的额外状态，故真实 Decode 的分母不能只保留权重。[vLLM/PagedAttention 原论文](https://arxiv.org/abs/2309.06180)

**风险。** MoE 只读被路由专家、权重缓存命中、量化 scale/反量化、LM head、attention KV 读取、TP 通信和 Kernel launch 都会改变该比值；长上下文下 KV 项尤其不能省略。

## 4. FP8/INT8：收益路径与精度

### 判定：改写后采纳

可保留的三条机制是：

1. **存储容量。** FP8 和 INT8 的编码主体均为 1 Byte/element，理想上是 BF16/FP16（2 Byte）的二分之一；实际还须加 scale、zero-point、对齐和未量化张量。
2. **HBM/互连流量。** 被量化并真正以低比特读写的权重、激活或 KV 可减少字节；这是可能降低带宽时间、或释放显存以容纳更多 KV/并发的条件，而非端到端速度保证。
3. **算力路径。** 有对应硬件和融合 Kernel 时，低精度 GEMM 可使用不同的 Tensor Core 路径。NVIDIA Transformer Engine 明确将 FP8 用于 Hopper/Ada/Blackwell 上更低内存和更高性能，也同时指出速度依赖矩阵 shape；其官方 benchmark 也展示小 GEMM 可能无收益或变慢。[Transformer Engine 总览](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/) [形状依赖的官方 benchmark 说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html)

不应写“FP8/INT8 通常只损失 X% 精度”。量化会引入 rounding、clipping 与 scale 误差；TensorRT 对 INT8 要求校准或显式 dynamic range，对 FP8 要求显式量化 scale。[TensorRT 量化文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html) Transformer Engine 也明确说明并非每个算子都适合 FP8。[Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-1.6/user-guide/examples/fp8_primer.html)

**风险。** “模型文件少一半”“Tensor Core 峰值翻倍”“质量下降小”是三个不同层级的命题。课程应要求固定模型、校准/量化方案、输入分布、上下文、硬件、runtime 与质量集后实测质量、TTFT/TPOT/吞吐、峰值显存和 fallback。

## 5. `B_sat` 与 KV 读取交叉点

### 判定：改写后采纳

`B_sat` 不是统一定义的行业常数；可以作为**显式简化模型**中的推导符号。对一个 Decode iteration，设：

- `F`：每个新增 token 的 FLOPs；
- `W`：本轮仅搬一次的权重字节；
- `K(T)`：每 token 的 KV 与其他随上下文 `T` 变化的读取字节；
- `B`：本轮 token 数；
- `R=P_peak/BW_peak`：同 precision 的 machine balance。

则理想化 AI 为：

$$
AI(B,T)=\frac{BF}{W+BK(T)}.
$$

当 `F>R K(T)` 时，解 `AI≥R` 得到：

$$
B_{sat}(T)=\frac{RW}{F-RK(T)}.
$$

若 `F≤RK(T)`，这个模型下无有限 batch 能把该路径推过计算 roof；随着上下文变长，`K(T)` 增长，`B_sat` 会变大或消失。另一个仅比较流量的交叉点是：

$$
B_{KV=weight}(T)=\frac{W}{K(T)},
$$

它只表示 iteration 内 KV bytes 与权重 bytes 相等，**不**等于性能拐点。

Roofline 原论文支持把 FLOPs/bytes 与硬件上界联立，但不支持把这个符号、上述近似或某个数值当成通用定律。[Roofline 原文](https://doi.org/10.1145/1498765.1498785)

**风险。** 权重未必恰好只读一次；KV layout、cache hit、block table、Attention kernel、MoE 路由、TP/EP 通信与排队都会破坏这个简化。课程如采用，应称为“用于提出 profiling 假设”，随后用 Kernel trace 验证，而不是 batching 配置公式。

## 6. MFU/MBU 与 70B/H100 的 24 tok/s 示例

### 判定：改写后采纳

只要先定义口径，利用率是有用的诊断量：

$$
MFU=\frac{F_{per\ token}\times throughput}{P_{peak}},\qquad
MBU=\frac{Q_{per\ token}\times throughput}{BW_{peak}}.
$$

其中分子必须是同一 workload、同一 rank/整组和同一时间窗口的实测或明确估算；分母必须匹配实际 precision、卡型、稀疏和可持续（或理论峰值）口径。NVIDIA 的 Megatron Core 源码中，行并行线性层在 forward 显式执行 TP group all-reduce，说明端到端时间并非只由 `F/P_peak` 与 `Q/BW_peak` 决定。[Megatron Core tensor-parallel layers](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py) [NCCL AllReduce 定义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

“70B 在一张 H100 上 24 tok/s”不能作为课程例子。仅 BF16 权重 payload 就约 `70e9×2=140 GB`（约 130.4 GiB），大于 H100 80 GB SKU 的显存，且还未计 KV 和 runtime；它至少需要说明量化/卸载，或多卡 TP/PP/分片。NVIDIA 的 NIM 支持矩阵也把大模型的配置按 GPU 数和 precision 分开列出。[H100 官方规格页](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) [NVIDIA NIM 支持矩阵](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html)

即使暂按 `F≈2×70B FLOPs/token`，24 tok/s 也只是 `3.36 TFLOP/s`；除以哪一个 H100 峰值、是单卡还是 TP group、是否包含 attention/KV 和通信，会给出完全不同的 MFU。MBU 更不能由 token/s 单独推出。

**风险。** `MFU` 在公开材料中常指 Model FLOPs Utilization，`MBU` 有时被用为 memory-bandwidth utilization，但后者并非统一标准名。课程若引入，应直接给上述公式和采样边界，不给脱离配置的“低利用率”结论。

## 7. GQA 与 MLA 的 KV bytes

### 判定：采纳

对标准 MHA/GQA，每层、每 token 的逻辑 KV payload 是：

$$
KV_{GQA}=2N_{kv}D\,s.
$$

`2` 是 K 与 V；GQA 只将 `N_{kv}` 减至小于 query 头数 `N_q`，所以相对同样 `N_q,D,s` 的 MHA 缩小比例为 `N_{kv}/N_q`。GQA 的一手论文正是以多 query heads 共用 K/V heads 定义该结构。[Ainslie et al., *GQA*](https://arxiv.org/abs/2305.13245)

MLA 不是把这个公式中的 `N_kv` 改成另一个常数。DeepSeek-V2 将 K/V 压为 latent `c_t^{KV}`，同时为 decoupled RoPE 保留 key 部分；因此每层每 token 的缓存宽度应从模型配置/论文符号写成：

$$
KV_{MLA}=(d_c+d_h^R)s,
$$

其中 `d_c` 为 KV latent 宽度，`d_h^R` 为 RoPE key 宽度。DeepSeek-V2 的原表还给出其配置 `d_c=4d_h`、`d_h^R=d_h/2`，即 MLA 的缓存宽度为 `4.5d_h`；这是容量上等效于 `n_{kv}=2.25` 的 GQA，不是可部署的“2.25 个头”。该式描述缓存编码，不表示必然有同样比例的端到端加速。[DeepSeek-V2 技术报告 §2.1.2、Table 1](https://arxiv.org/html/2405.04434#S2.T1)

**风险。** 两式都是每层、每 token 的**逻辑有效载荷**；乘以 attention 层数、序列长度与并发后才是请求量。TP 下还要按 runtime 的本地头/latent 分片和可能复制重新计算，不能先算全局值再机械除以 TP。

## 8. KV 管理、NVFP4/TurboQuant 与跨机 TP

### 判定：改写后采纳

### 8.1 KV eviction 与 KV quantization 是不同策略

- **Eviction / token selection**：减少保留的历史 token，改变可供未来 Attention 读取的信息。H2O 以 recent tokens 与 heavy hitters 的保留策略为核心；StreamingLLM 保留 attention sinks 与最近窗口。二者都有各自模型/任务/缓存预算下的实验结果，不是无损的通用内存管理。[H2O 原论文](https://arxiv.org/abs/2306.14048) [StreamingLLM 原论文](https://arxiv.org/abs/2309.17453)
- **KV quantization**：保留 token 位置但用低比特表示 K/V，误差来自量化而非直接删 token。KIVI 的 2-bit 方案对 Key 使用 per-channel、Value 使用 per-token 的非对称粒度；其“2.6× 总 peak memory、至多 4× batch、2.35–3.47× 吞吐”是 Llama/Falcon/Mistral、其 kernel 与工作负载下的论文结果，不能改写为所有 runtime 的保证。[KIVI 原论文](https://arxiv.org/abs/2402.02750)

### 8.2 NVFP4 与 TurboQuant 不能合并数字

NVFP4 是 NVIDIA Transformer Engine 的 4-bit 浮点 recipe；它有 block scale、随机 Hadamard transform 等格式与硬件条件。它可以用于低比特 GEMM 的权重/激活路径，也有 NVIDIA 的 **NVFP4 KV cache** 实现，但两者都不是 KV eviction。[NVIDIA Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) [NVIDIA NVFP4 KV cache 官方文](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)

这份 NVFP4 KV 实现相对 **FP8 KV** 的格式主体可少约 50% KV memory；其 attention 路径当前先将 NVFP4 dequantize 到 FP8，再进行 attention/context math，故 “4 bit” 不等于 attention 全程原生 FP4。该官方文在 Qwen3-Coder-480B-A35B/Blackwell 等指定实验中报告“最多 3× 更低 TTFT、最多 20% 更高 cache hit、相对 BF16/FP8 的准确率差异低于 1%”；这些都是模型、缓存压力、GPU、评测与比较基线限定的观察，不是通用的 NVFP4 数字。

TurboQuant 是 Google Research 的在线**向量 KV 量化**工作，不是 NVFP4 数据格式或 Tensor Core 路径。原论文的 MSE 量化使用随机旋转加 per-coordinate scalar quantization；内积版本再以 1-bit QJL 编码 residual。其作者实验在 3.5 bits/channel 下报告 absolute quality neutrality、在 2.5 bits/channel 下报告 marginal quality degradation；非整数 bit 是按 channel 分配 bitwidth 的平均值。LongBench 等结果只适用于论文列明的 Llama-3.1-8B-Instruct、Ministral-7B-Instruct、bitwidth、kernel 和 benchmark。[TurboQuant 原论文](https://arxiv.org/html/2504.19874)

课程可写的安全结论是：**NVFP4 与 TurboQuant 都可能改变 KV 表示，但格式、量化算法、kernel 和实验基线不同；NVFP4 也可能另用于低比特 GEMM。两者的压缩/吞吐数字不能相乘或互相替代，组合收益须端到端实测。** 不建议写入任何未同时给出模型、GPU、context、batch、实现版本、比较基线和质量任务的 “X 倍” 数字。

### 8.3 TP 可以跨机，但通信在每层关键路径

典型 TP 将列并行投影的局部结果留在本 rank，将行并行投影的局部结果通过 all-reduce（或配合 sequence parallel 的 all-gather/reduce-scatter）合并。Megatron Core 的 mapping 源码直接调用 `torch.distributed.all_reduce` 和 `all_gather_into_tensor`；NCCL 定义 all-reduce 为每个 rank 取得规约后的完整结果。因此 TP group 的 rank 可以位于不同节点，但每个 transformer block 的 collective 必经跨节点互联，低 batch Decode 特别容易受其延迟影响。[Megatron TP mappings 源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py) [NCCL collectives 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

**风险。** “可以跨机”不等于“适合跨机”。实际取决于 TP group 映射、NVLink/NVSwitch、PCIe、NIC、InfiniBand/RoCE、NCCL 拓扑和是否能 overlap；需对同模型同 batch 比较每层 collective 时间、TPOT、TTFT 与吞吐。若目标是扩展到很多节点，PP/DP/EP 或混合并行常是不同的设计问题，不能从 TP 的机制直接推出配置建议。

## 写入课程时的最小安全边界

1. 每一个硬件 ridge point 都标出 GPU SKU、`P_peak` precision、dense/sparse 与 `BW` 口径。
2. `AI≈1`、`B_sat`、MFU/MBU 都标出假设、分子分母和 workload；它们是判断工具，不是模型常数。
3. 量化数字仅在同一原论文/官方 benchmark 的完整实验条件下引用；通用正文讲机制、收益条件、质量风险与验证方法。
4. GQA/MLA、权重量化/KV 量化、KV eviction、TP 通信分别说明改变的对象，禁止互相替代或合并倍率。
