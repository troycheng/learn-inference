"""Reproduce lesson 12 hash, parameter sizing, and local convolution history.

The convolution check uses supplied two-channel inputs. It is not a full PLE
implementation: learned projections, gates, RMSNorm and residual injection are
not reproduced here.
"""

import math


table_size = 11
token_t_minus_2 = 2
token_t_minus_1 = 5
token_t = 7

hash_2gram = ((token_t * 3) ^ (token_t_minus_1 * 5)) % table_size
hash_3gram = ((token_t * 3) ^ (token_t_minus_1 * 5) ^ (token_t_minus_2 * 7)) % table_size

heads_per_ngram = 8
ngram_orders = 2
ngram_heads = heads_per_ngram * ngram_orders
head_width = 160
base_rows_per_head = 20_000_000

active_ngram_parameters = ngram_heads * head_width
total_ngram_parameters = ngram_heads * base_rows_per_head * head_width
bf16_gib = total_ngram_parameters * 2 / 1024**3

print(f"2-gram toy table row = {hash_2gram}")
print(f"3-gram toy table row = {hash_3gram}")
print(f"lookup heads per token = {ngram_heads}")
print(f"active N-gram parameters per token = {active_ngram_parameters}")
print(f"approximate total parameters = {total_ngram_parameters / 1e9:.1f}B")
print(f"approximate BF16 payload = {bf16_gib:.1f} GiB")

assert hash_2gram == 1
assert hash_3gram == 2
assert active_ngram_parameters == 2560
assert total_ngram_parameters == 51_200_000_000


def short_conv(inputs, history, weights, dilation):
    """Causal depthwise convolution + SiLU, with a fixed-length input cache.

Rows are positions and columns are channels. Like PyTorch Conv1d, kernel
weights run from the oldest tap to the current-position tap.
"""
    state_length = (len(weights[0]) - 1) * dilation
    joined = history + inputs
    outputs = []
    for position in range(len(history), len(joined)):
        row = []
        for channel, kernel in enumerate(weights):
            value = 0.0
            for tap, weight in enumerate(kernel):
                source = position - (len(kernel) - 1 - tap) * dilation
                value += weight * (joined[source][channel] if source >= 0 else 0.0)
            row.append(value / (1 + math.exp(-value)))
        outputs.append(row)
    return outputs, joined[-state_length:]


conv_inputs = [[(position % 5 - 2) / 2, (position % 7 - 3) / 3] for position in range(20)]
conv_weights = [[0.1, 0.2, -0.3, 0.4], [-0.2, 0.3, 0.1, 0.5]]
conv_dilation = 3
state_length = (len(conv_weights[0]) - 1) * conv_dilation
assert state_length == 9
full_outputs, full_history = short_conv(conv_inputs, [], conv_weights, conv_dilation)

for chunk_sizes in ([20], [1] * 20, [4, 1, 6, 9]):
    history = []
    streamed_outputs = []
    start = 0
    for size in chunk_sizes:
        rows, history = short_conv(conv_inputs[start:start + size], history, conv_weights, conv_dilation)
        streamed_outputs.extend(rows)
        start += size
        assert len(history) <= 9
    assert start == len(conv_inputs)
    assert streamed_outputs == full_outputs
    assert history == full_history

# Token IDs alone do not provide these historical vector inputs.
without_history, _ = short_conv(conv_inputs[10:11], [], conv_weights, conv_dilation)
assert without_history[0] != full_outputs[10]
print("PLE local convolution: full sequence == chunked == one-position steps; 9-position history required")
