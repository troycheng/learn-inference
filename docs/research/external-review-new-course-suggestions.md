# 外部建议核验：新一轮课程补充

## 结论先行

本记录只核对课程建议能否由论文、官方技术报告、官方模型配置、官方源码或官方文档直接支持；不修改课程正文。引用的 Qwen3.5、Qwen3.8 与 FLA 源码均固定到链接中的 revision，避免把某个运行时的行为写成模型定义。

| 建议 | 结论 | 可以进入课程的边界 |
| --- | --- | --- |
| GDN 状态 FP32 的原因 | 改写后采纳 | 可陈述参考实现的 dtype 与递推行为；不能把 FP32 的动机断言为“专为避免 delta-rule 反馈误差放大”。 |
| Chunk Gated Delta Rule 的块大小 | 改写后采纳 | Chunk 是把已知序列的递推改组为块内矩阵计算的实现参数；可讨论并行/临时工作区权衡，但不得把 `64` 写成 Qwen 架构常数或记忆截断点。 |
| MoE EP dispatch/combine | 改写后采纳 | dispatch 与 combine 是稳定的逻辑边界；All-to-All 是常见实现而非所有引擎的唯一 collective。 |
| QSA token budget、indexer state 与 131k | 改写后采纳 | `2048` 是候选 token 预算，indexer 原始 Key 随 `T` 增长；131,072 可作算术示例，不能说是官方报告给出的精确对比点。 |
| N-gram 16×160 与约 5 KiB | 改写后采纳 | 在该配置的 BF16 有效载荷下成立；必须标明读取/传输方向和缓存、dtype、预取的条件。 |
| GR 的 FP8 残差与 `4H` | 改写后采纳 | 官方报告支持 FP8 分支与“几乎无质量损失”；`4H` 是每个位置的分支载荷，不等于跨请求 KV Cache，也不是完整 kernel 流量。 |
| Capstone 的 FP8 KV 与长上下文验证 | 改写后采纳 | 加入 scale/布局/页对齐的实测与校准要求；质量要做 BF16 对照和长上下文检索评测，不能预设 FP8 无损。 |

## 1. Qwen3.5 Gated DeltaNet：状态 dtype 与数值解释

**结论：改写后采纳。**

固定 revision 的 Transformers Python fallback 会将 `query`、`key`、`value`、`beta`、`g` 都转换为 FP32；递归状态在该路径中随之以 FP32 创建或转换，输出才转换回输入 dtype。[`torch_recurrent_gated_delta_rule`，L342–380](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L342-L380) 官方 FLA fused recurrent kernel 也明确以 `torch.float32` 分配 `final_state`。[FLA，L210–217](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/fused_recurrent.py#L210-L217)

`conv_state` 不是这张递归矩阵。它保存短因果卷积需要的投影历史；cache 初始化直接使用传入 `conv_states.dtype`，而卷积更新只暂时转为权重 dtype 后再转回输入 dtype。[卷积更新，L199–216](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L199-L216) [LinearAttention cache，L998–1029](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L998-L1029) 默认只保留卷积窗口；`record_past=True` 是为了回滚而保留更长暂存历史的例外。[L1048–1070](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/cache_utils.py#L1048-L1070)

可用表述：

> 在 Transformers 的 Qwen3.5 Python fallback 和 FLA 的这条 fused recurrent 实现中，Gated DeltaNet 的最终递归状态采用 FP32；短卷积状态通常随输入投影 dtype 保存。递推会反复读、衰减并改写同一状态，因此低精度会改变数值轨迹；是否必须 FP32、以及具体收益，仍取决于所用 kernel 与 runtime。

不采纳“FP32 **是为**避免 delta-rule 反馈误差放大”的因果归因：源码给出 dtype，Qwen 代码只明确说明 `A_log.float()` 用于避免 FP16 下 `A` 变为 `-inf`，[L497–500](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L497-L500)；这些一手材料没有把最终状态 FP32 的唯一目的归因为该机制。也不得由此泛化到所有 fused kernel、vLLM 或其他服务 runtime。

## 2. Chunk Gated Delta Rule：chunk size 的语义

**结论：改写后采纳。**

Qwen3.5 在已有状态且 `seq_len=1` 时选择 recurrent 路径，其他情况选择 chunk 路径。[Qwen 前向分支，L504–534](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L504-L534) FLA 的公开实现默认 `chunk_size=64`，且只接受 `16/32/64`；同一参数传给块内 WY 表示、块间状态和块内输出计算。[FLA chunk 调用链](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/chunk.py#L33-L123) [参数校验，L534–536](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/chunk.py#L534-L536) 原论文的结论是这种改写把 DeltaNet 沿序列长度并行化，并追求适合 GPU 的矩阵计算，而不是取消递推的因果语义。[论文](https://arxiv.org/abs/2406.06484)

源码也足以支持“调优权衡”这句话：块内 `A` 的索引同时含两个 `BT` 维度，[FLA WY kernel，L68–76](https://github.com/fla-org/flash-linear-attention/blob/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule/wy_fast.py#L68-L76) 因而更大的块会改变块内并行工作与临时块状工作区；实际收益还受头维、SRAM、Triton autotune 和硬件影响。这是从源码结构作出的工程推论，不是论文承诺的通用最优点。

可用表述：

> `chunk_size` 是把连续 token 更新组织成块内矩阵工作、块间传递最终状态的 kernel 参数：增大它通常提供更粗粒度的块内并行，但也改变块内临时工作与数据复用；必须在目标 GPU 上测量。FLA 当前实现默认 64，不代表模型每 64 token 清空状态，也不是 Qwen 配置中固定的架构超参数。

## 3. MoE EP：dispatch/combine、All-to-All 与小 GEMM

**结论：改写后采纳。**

在单设备参考前向中，Qwen3.5 将 token 按 expert mask gather，运行 expert 后以 `index_add_` 写回原 token 位置；这就是 dispatch 与 combine 的算法语义。[官方 Transformers MoE forward，L720–757](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L720-L757) 官方 EP plan 将 router 标为 `ep_router`、routed experts 标为 grouped GEMM；shared expert 没有作为 expert-sharded 项出现。[配置，L83–88](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L83-L88)

因此可以讲：token 的 Top-k assignment 要到持有对应 expert 的 rank，expert 输出按原 token 与路由权重合并；`n_e` 随 batch 和内容改变，热点 rank、发送量不均以及很小的 expert micro-batch 都会降低吞吐。以 Qwen3.5-35B-A3B 的 `256 experts / top-8` 为例，Decode 的少量 token assignment 很容易分散到许多 expert，这是数学上的必然后果，而非某个固定性能数字。[模型配置](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)

不采纳“EP 就是两次 All-to-All”。Transformers 官方 EP 文档描述的原生路径是本地 expert、router 与最终 All-Reduce，[官方文档](https://huggingface.co/docs/transformers/expert_parallelism)；Megatron Core 则公开提供 All-to-All、All-Gather、Flex/DeepEP 等 dispatcher。[官方 API](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.token_dispatcher.html) All-to-All 是常见部署实现，不是模型或 EP 的唯一语义。

## 4. QSA：2048 budget、随 `T` 增长的 indexer Key 与 131k

**结论：改写后采纳。**

官方配置是 `indexer_budget=2048`、`indexer_compress_ratio=4`、4 个 indexer query heads、1 个 shared key head、`indexer_head_dim=128`。[Qwen3.8-Flash-Next 配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) 技术报告明确给出：每个 query 最多选择 `2048/4=512` 个完整微块，并额外保留最后一个未满块的 tail。[技术报告 §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf) 参考实现同样分配 `budget + ratio - 1` 的选择槽位。[源码，L620–702](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L620-L702)

`raw_keys` 每个 token 先产生，再通过 `update_indexer` 接到历史；Dynamic cache 的形状就是 `[B,total_len,index_head_dim]`。[QSA 调用，L644–655](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L644-L655) [cache 定义](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/cache_utils.py#L319-L351) 因而 QSA 减少的是核心 Attention 的候选读取，不是把完整正式 K/V 或原始 indexer Key 缩成固定大小。

在 `T=131,072` 的**算术示例**中，2048 个完整候选 token 是 `T/64`；最多加 3 个 tail token。索引器仍需要面对约 `T/4=32,768` 个完整微块，且请求状态仍含全部正式 K/V 与逐 token raw Key。此处 `64×` 仅是“每个 query 的完整候选数相对所有历史位置”的上界比，不能外推为端到端加速或显存减少 64 倍。

不采纳“131k 对比来自官方报告”：报告的长上下文表按 `≤128K`、`128–256K` 等区间报告 RULER，并在 128K、256K、512K、1M 报告 MRCR；没有 131,072 的单点性能数据。报告可支持采用 RULER + 多针 MRCR 的验证方法，不能支持把 131k 写成其官方测量结果。

## 5. N-gram Embedding：16×160、约 5 KiB 与预取边界

**结论：改写后采纳。**

配置给出 `ngram_size=3`、`heads_per_ngram=8`、`ple_embed_dim=2560`、模型 dtype 为 BF16。[配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) 参考实现据此计算 `(ngram_size-1)*heads_per_ngram=16` 个头，并令每头维度为 `2560/16=160`；生成两组各 8 个 hash ID，查表后展平为 2560 维。[N-gram 源码，L1019–1051](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1019-L1051) [L1069–1114](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1069-L1114)

所以精确说法是“每个 token 查询 16 行、每行 160 个元素”，不是“读取一个 `16×160` 表行”。若表权重为 BF16，逻辑有效载荷为 `16×160×2=5120 bytes=5 KiB`，主机表读取与把命中向量交给 GPU 都各以此为**理想有效负载**量级。它不含索引、CPU/GPU 缓存行放大、DMA 描述符、预取缓冲、批量去重，也不适用于 FP8/INT8 或已驻留 GPU 的表。

“放第 2 层以给主机预取留时间”有官方报告依据：Qwen 报告说明最终放在 Layer 2，使 host-memory prefetch 能与第 1 层计算重叠；README 也明确称表可 offload 并通过 asynchronous prefetch overlap。[技术报告 §2.3.1](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf) [官方 README](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/README.md) 但 Transformers reference 的可见路径是“embedding 在其自身 device 上 lookup，随后 `.to(ngram_ids.device)`”，[L1112–1114](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1112-L1114) 它没有实现完整服务 runtime 的异步流水。因此不能声称主机预取、PCIe/NVLink 传输一定被隐藏；要以目标 runtime 的 trace 验证。

## 6. Gated Residual：FP8 residual 与 `4H` 公式

**结论：改写后采纳。**

官方技术报告明确：GR 使用 `n_r=4` 个分支；推理成本受 widened residual state 的内存流量主导。报告称 GDN、gated attention 与 GR 的门将写入值限制在较窄范围，因而分支以 FP8 保存相对 BF16 将 residual state 搬运字节减半，且“almost no loss in quality”；GR Read 与 Write 各自融合成一个 kernel、每个 block 每个方向遍历 widened stream 一次。[技术报告 §2.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf) `hc_count=4` 也由模型配置和参考实现的 `hc_count * hidden_size` 接口直接确认。[配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) [实现，L941–969](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L941-L969)

因此可用载荷公式是：每个位置在一个层间 GR 边界携带 `n_rH=4H` 个元素；若该边界的四分支**存储格式为 FP8**，裸 payload 是 `4H×1 byte=4H bytes/位置`，BF16 则为 `8H bytes/位置`。对 `H=2560`，这是 10,240 B（10 KiB）与 20,480 B（20 KiB）。

限制必须写清：它是“一个分支状态快照的裸数据量”，不是每层的完整读写流量，更不是跨 token 持久请求状态。实际流量还取决于 read/write 次数、融合、寄存器复用、量化 scale/布局、对齐与 batch；训练时保留的反向激活又是另一种口径。官方报告支持 FP8 方案本身，不足以支持任何硬件上的固定吞吐比例。

## 7. Capstone：FP8 KV 元数据与长上下文质量门槛

**结论：改写后采纳。**

Capstone 的 FP8 KV 建议应从“每元素 1 byte，所以容量减半”改为以下可验证要求：

1. 先固定 engine、attention backend、FP8 格式和 scale 粒度，再按该实现的实际页布局统计 KV pool；把 data payload、K/V scale、页表/块尾空余和对齐分开记录。
2. 校准不是可选装饰。vLLM 官方 FP8 KV 文档列出 per-tensor scale（每个 Q/K/V tensor 一个）与 per-head scale（按 head），默认无校准时 scale 为 1.0，并推荐数据集校准以获得最高质量。[官方文档](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md) vLLM 的配置代码也明确警告：量化 KV 在缺少合适 scale 时可能造成 accuracy drop。[源码，L324–342](https://github.com/vllm-project/vllm/blob/main/vllm/config/kv_cache.py#L324-L342)
3. 对 Qwen3.5 的目标长度与并发，保持同一模型/runtime/采样配置，以 BF16 KV 为对照，记录每 rank 的真实 pool 使用、scale/metadata、分页空洞、OOM/抢占/重算、TTFT/TPOT 以及端到端完成时间；不得只用逻辑元素数批准上线。
4. 质量至少分两层：固定输入的输出与 logprob/生成一致性对照，以及接近目标长度的检索任务。Qwen 官方 QSA 报告使用 RULER（4K–1000K）和 8-needle MRCR（128K–1M）比较 full attention 与 QSA，[技术报告 Table 3 / §2.1.2](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf) 可作为“长度分档 + 检索压力”的一手方法依据，但它**不是** Qwen3.5 FP8 KV 的质量证明。

这使课程结论保持诚实：FP8 KV 是待校准、待容量实测、待长上下文质量对照的候选方案，而不是“理论减半后自然无损”的上线结论。

## 资料范围

- Qwen Team，[Qwen3.8-Next Architecture 技术报告，revision `6988587`](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf)。
- Qwen Team，[Qwen3.8-Flash-Next 官方配置，revision `de4b8e4`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) 与 [官方 README，revision `6988587`](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/README.md)。
- Hugging Face Transformers，[Qwen3.5 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[Qwen3.5 MoE 实现](https://github.com/huggingface/transformers/blob/943628458a1691f8af09c47ea9fc6e314734722f/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)、[Qwen4 experimental/Qwen3.8 参考实现，revision `281dd53`](https://github.com/huggingface/transformers/blob/281dd533060988a1de8d063c4c1ea72b304a2bb8/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。
- Flash Linear Attention，[Gated Delta Rule，revision `3c4c54a`](https://github.com/fla-org/flash-linear-attention/tree/3c4c54ae7397d37130d7101edd0f4eb596af896d/fla/ops/gated_delta_rule)；Y. Yang et al., [*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*](https://arxiv.org/abs/2406.06484)。
- vLLM，[Quantized KV Cache](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md) 与 [KV cache configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/kv_cache.py)。
