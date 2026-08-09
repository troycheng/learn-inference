"""复算第 8 课中与请求容量和上下文长度有关的几个数字。"""

from dataclasses import dataclass


KIB = 1024
MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint_payload_bytes: int
    full_attention_layers: int
    query_heads: int
    kv_heads: int
    head_dimension: int
    gated_deltanet_layers: int


QWEN_9B = ModelConfig(
    name="Qwen3.5-9B",
    checkpoint_payload_bytes=19_306_216_416,
    full_attention_layers=8,
    query_heads=16,
    kv_heads=4,
    head_dimension=256,
    gated_deltanet_layers=24,
)

QWEN_35B = ModelConfig(
    name="Qwen3.5-35B-A3B",
    checkpoint_payload_bytes=71_903_655_008,
    full_attention_layers=10,
    query_heads=16,
    kv_heads=2,
    head_dimension=256,
    gated_deltanet_layers=30,
)


def logical_kv_bytes_per_token(config: ModelConfig, element_bytes: int) -> int:
    """整个模型中，一个请求每新增一个位置所需的逻辑 K/V 字节数。"""
    return 2 * config.full_attention_layers * config.kv_heads * config.head_dimension * element_bytes


def kv_bytes_per_rank_token(config: ModelConfig, element_bytes: int, tensor_parallel: int) -> int:
    """按第 8 课引用的 vLLM 规则，估算单个 TP Rank 保存的 K/V。"""
    kv_heads_per_rank = max(1, config.kv_heads // tensor_parallel)
    return 2 * config.full_attention_layers * kv_heads_per_rank * config.head_dimension * element_bytes


def attention_flops(config: ModelConfig, cached_tokens: int) -> int:
    """单步 Decode 中，全部 Full Attention 层 QK 和 AV 的近似 FLOPs。"""
    return 4 * config.full_attention_layers * config.query_heads * config.head_dimension * cached_tokens


def gated_deltanet_state_bytes(config: ModelConfig) -> int:
    """按 BF16 卷积状态和 FP32 递归状态估算单请求固定状态。"""
    conv_state_per_layer = 8192 * 4 * 2
    recurrent_state_per_layer = 32 * 128 * 128 * 4
    return config.gated_deltanet_layers * (conv_state_per_layer + recurrent_state_per_layer)


for model in (QWEN_9B, QWEN_35B):
    logical_kv = logical_kv_bytes_per_token(model, element_bytes=2)
    rank_kv = kv_bytes_per_rank_token(model, element_bytes=2, tensor_parallel=8)
    fixed_state = gated_deltanet_state_bytes(model)

    print(model.name)
    print(f"  检查点权重有效载荷：{model.checkpoint_payload_bytes / GIB:.2f} GiB")
    print(f"  BF16 逻辑 KV：{logical_kv / KIB:.0f} KiB / 请求 / 位置")
    print(f"  TP=8 时每 Rank KV：{rank_kv / KIB:.0f} KiB / 请求 / 位置")
    print(f"  Gated DeltaNet 固定状态：{fixed_state / MIB:.1f} MiB / 请求")
    for length in (4096, 131072):
        flops = attention_flops(model, cached_tokens=length)
        kv_total = logical_kv * length
        kv_unit = f"{kv_total / MIB:.0f} MiB" if kv_total < GIB else f"{kv_total / GIB:.2f} GiB"
        print(f"  历史长度 {length:>6}：逻辑 KV {kv_unit}，Attention 长度项约 {flops / 1e9:.2f} GFLOPs")


assert logical_kv_bytes_per_token(QWEN_9B, 2) == 32 * KIB
assert logical_kv_bytes_per_token(QWEN_35B, 2) == 20 * KIB
assert QWEN_9B.checkpoint_payload_bytes == 19_306_216_416
assert QWEN_35B.checkpoint_payload_bytes == 71_903_655_008
assert kv_bytes_per_rank_token(QWEN_9B, 2, tensor_parallel=8) == 8 * KIB
assert kv_bytes_per_rank_token(QWEN_35B, 2, tensor_parallel=8) == 10 * KIB
assert gated_deltanet_state_bytes(QWEN_9B) == int(49.5 * MIB)
assert gated_deltanet_state_bytes(QWEN_35B) == int(61.875 * MIB)
assert attention_flops(QWEN_9B, 4096) == 536_870_912
assert attention_flops(QWEN_35B, 131072) == 21_474_836_480
