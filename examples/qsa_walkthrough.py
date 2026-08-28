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


block_size = 2
token_budget = 4
indexer_scores = [0.2, 1.7, 0.6, 1.1]
blocks = [[1, 2], [3, 4], [5, 6], [7, 8]]

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
print("attention weights =", [round(value, 3) for value in attention_weights])
print("attention output =", [round(value, 3) for value in output])

assert selected_block_ids == [1, 3]
assert selected_token_ids == [3, 4, 7, 8]
assert all(abs(actual - expected) < 0.001 for actual, expected in zip(attention_weights, [0.197, 0.439, 0.098, 0.266]))
assert all(abs(actual - expected) < 0.002 for actual, expected in zip(output, [0.827, 0.976]))
