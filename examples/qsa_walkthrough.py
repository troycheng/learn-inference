"""Reproduce micro-block selection and sparse Attention from lesson 11."""

import math


def softmax(values):
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def weighted_sum(weights, vectors):
    width = len(vectors[0])
    return [sum(weight * vector[index] for weight, vector in zip(weights, vectors)) for index in range(width)]


def block_positions(block_id, block_size):
    """Zero-based block id, one-based token positions, as in the lesson."""
    return [block_id * block_size + offset + 1 for offset in range(block_size)]


def selected_count(visible_length, block_size=4, token_budget=2048):
    """Count complete selected blocks plus the unconditionally retained tail."""
    complete_blocks, tail = divmod(visible_length, block_size)
    return min(complete_blocks, token_budget // block_size) * block_size + tail


block_size = 2
token_budget = 4
indexer_scores = [0.2, 1.7, 0.6, 1.1]
blocks = [block_positions(index, block_size) for index in range(4)]
assert blocks[1] == [3, 4]
raw_keys = {3: [1, 3], 4: [3, 1]}
pooled_key = [sum(raw_keys[position][axis] for position in blocks[1]) / block_size for axis in range(2)]
assert pooled_key == [2, 2]  # Before RMSNorm.
pooled_rms = math.sqrt(sum(value * value for value in pooled_key) / len(pooled_key))
normalized_key = [value / pooled_rms for value in pooled_key]
assert normalized_key == [1, 1]  # Teaching example: epsilon=0, learned scale=1.

block_budget = token_budget // block_size
selected_block_ids = sorted(
    range(len(indexer_scores)), key=lambda index: indexer_scores[index], reverse=True
)[:block_budget]
selected_token_ids = [token for block_id in selected_block_ids for token in blocks[block_id]]

attention_scores = [0.4, 1.2, -0.3, 0.7]
values = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]
attention_weights = softmax(attention_scores)
output = weighted_sum(attention_weights, values)

print(f"selected blocks = {selected_block_ids}")
print(f"selected token positions = {selected_token_ids}")
print("attention weights =", [round(value, 5) for value in attention_weights])
print("attention output =", [round(value, 3) for value in output])

assert selected_block_ids == [1, 3]
assert selected_token_ids == [3, 4, 7, 8]
assert [round(value, 5) for value in attention_weights] == [0.19716, 0.43879, 0.09791, 0.26614]
assert [round(value, 3) for value in output] == [0.827, 0.975]

# Indexer scores select positions; only the main Attention scores weight V.
main_score_by_position = dict(zip(selected_token_ids, attention_scores))
value_by_position = dict(zip(selected_token_ids, values))


def output_for_indexer_scores(scores):
    chosen_blocks = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:block_budget]
    chosen_positions = [position for block_id in chosen_blocks for position in blocks[block_id]]
    weights = softmax([main_score_by_position[position] for position in chosen_positions])
    return chosen_positions, weighted_sum(weights, [value_by_position[position] for position in chosen_positions])


changed_indexer_scores = [score * 2 + 0.1 for score in indexer_scores]
chosen_positions, changed_output = output_for_indexer_scores(changed_indexer_scores)
assert chosen_positions == selected_token_ids
assert changed_output == output
print("Changed indexer scores, unchanged selected positions: main Attention output unchanged")

# Check short contexts, every tail length, and a context beyond the block budget.
for visible_length in range(12):
    assert selected_count(visible_length) == visible_length
for tail in range(4):
    assert selected_count(2048 + tail) == 2048 + tail
    assert selected_count(4096 + tail) == 2048 + tail
assert selected_count(131072) == 2048
assert selected_count(131075) == 2051

visible_length = 131072
element_bytes = 2
kv_bytes_per_position = 2 * 2 * 256 * element_bytes
index_key_bytes = 128 * element_bytes
payloads = {
    "dense K/V": visible_length * kv_bytes_per_position,
    "raw indexer Keys": visible_length * index_key_bytes,
    "pooled indexer Keys": (visible_length // 4) * index_key_bytes,
    "selected K/V": selected_count(visible_length) * kv_bytes_per_position,
}
assert list(payloads.values()) == [256 * 2**20, 32 * 2**20, 8 * 2**20, 4 * 2**20]
for name, byte_count in payloads.items():
    print(f"{name}: {byte_count / 2**20:g} MiB (logical BF16 payload, not measured HBM traffic)")
