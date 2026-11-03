"""复算第 8、9 课的 Roofline 教学例子；不是 GPU benchmark。"""

import math


PEAK_FLOPS = 100e12
BANDWIDTH = 1e12
BALANCE = PEAK_FLOPS / BANDWIDTH
H_IN, H_OUT = 4096, 12288
WEIGHT_BYTES = 2 * H_IN * H_OUT
ACTIVATION_BYTES_PER_TOKEN = 2 * (H_IN + H_OUT)
FLOPS_PER_TOKEN = 2 * H_IN * H_OUT


def linear_estimate(token_count):
    flops = FLOPS_PER_TOKEN * token_count
    byte_count = WEIGHT_BYTES + ACTIVATION_BYTES_PER_TOKEN * token_count
    return flops / byte_count, flops / PEAK_FLOPS, byte_count / BANDWIDTH


assert WEIGHT_BYTES == 96 * 2**20
assert ACTIVATION_BYTES_PER_TOKEN == 32768
for tokens, expected_ai in [(1, 1.00), (32, 31.67), (128, 122.88)]:
    intensity, compute, memory = linear_estimate(tokens)
    assert round(intensity, 2) == expected_ai
    print(f"M={tokens:3}: AI={intensity:.2f}, compute={compute * 1000:.3f} ms, memory={memory * 1000:.3f} ms")

linear_crossover = BALANCE * WEIGHT_BYTES / (
    FLOPS_PER_TOKEN - BALANCE * ACTIVATION_BYTES_PER_TOKEN
)
assert math.ceil(linear_crossover) == 104
assert linear_estimate(103)[0] < BALANCE < linear_estimate(104)[0]
print(f"Linear crossover M={linear_crossover:.2f}")

# Lesson 9's synthetic combined model, all quantities in FLOPs and bytes.
# KV uses the model-logical scope, without TP replication.
shared_weights = 16e9
linear_flops = 16e9
for length in (4096, 131072):
    kv_bytes = 32768 * length
    total_flops = linear_flops + 4 * kv_bytes
    denominator = total_flops - BALANCE * kv_bytes
    saturation = BALANCE * shared_weights / denominator if denominator > 0 else math.inf
    limit = total_flops / kv_bytes
    print(f"T_kv={length}: KV={kv_bytes / 2**20:g} MiB, F={total_flops / 1e9:.3f} GFLOPs, M_sat={saturation:.2f}, AI_limit={limit:.2f}")
    if length == 4096:
        assert math.ceil(saturation) == 514
        assert round(saturation, 2) == 513.63
    else:
        assert math.isinf(saturation)
        assert round(limit, 2) == 7.73

# Qwen3.5-9B: logical KV Attention AI vs TP=8, with replicated KV heads.
assert 131072 / 32768 == 4
assert 16384 / 8192 == 2
