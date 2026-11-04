"""用第 3 课的三位置示例跑一遍单头因果 Attention。"""

from math import cos, exp, sin, sqrt


Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]

K = [
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]

V = [
    [2.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0],
]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax(values: list[float]) -> list[float]:
    """减去最大值，避免 exp 在大数上溢出。"""
    finite_values = [value for value in values if value != float("-inf")]
    row_max = max(finite_values)
    numerators = [0.0 if value == float("-inf") else exp(value - row_max) for value in values]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def print_matrix(name: str, matrix: list[list[float]]) -> None:
    print(name)
    for row in matrix:
        shown = ["-inf" if value == float("-inf") else f"{value:.3f}" for value in row]
        print("  [" + ", ".join(shown) + "]")


def attention(queries: list[list[float]], keys: list[list[float]], values: list[list[float]]):
    """单头、无历史缓存的因果 Attention；行号就是位置编号。"""
    scale = sqrt(len(queries[0]))
    scores = [[dot(query, key) / scale for key in keys] for query in queries]
    masked_scores = [
        [score if key_position <= query_position else float("-inf") for key_position, score in enumerate(row)]
        for query_position, row in enumerate(scores)
    ]
    weights = [softmax(row) for row in masked_scores]
    output = [
        [
            sum(row_weights[position] * values[position][feature] for position in range(len(values)))
            for feature in range(len(values[0]))
        ]
        for row_weights in weights
    ]
    return scores, masked_scores, weights, output


scores, masked_scores, weights, output = attention(Q, K, V)

print_matrix("缩放后的 QK^T / sqrt(D):", scores)
print_matrix("加上因果遮罩:", masked_scores)
print_matrix("逐行 Softmax:", weights)
print_matrix("Attention 输出 A @ V:", output)

for row in weights:
    assert abs(sum(row) - 1.0) < 1e-12

assert all(abs(actual - expected) < 5e-4 for actual, expected in zip(output[2], [0.745, 1.255]))

# GQA 示例：另一个查询头在位置 3 使用同一组 K/V，但 Q 不同。
_, _, shared_kv_weights, shared_kv_output = attention(Q[:2] + [[0.0, 1.0]], K, V)
assert shared_kv_weights[2] != weights[2]
assert all(abs(actual - expected) < 5e-4 for actual, expected in zip(shared_kv_output[2], [0.797, 1.203]))

# 保持 Q/K 和遮罩不变，只改位置 2 的 V。权重不变，输出变化可直接计算。
changed_v = [V[0][:], [2.0, 2.0], V[2][:]]
_, _, same_weights, changed_output = attention(Q, K, changed_v)
assert same_weights == weights
assert abs(changed_output[2][0] - output[2][0] - weights[2][1] * 2.0) < 1e-12
assert changed_output[2][1] == output[2][1]
assert all(abs(actual - expected) < 5e-4 for actual, expected in zip(changed_output[2], [1.752, 1.255]))
print_matrix("只改 V2 后，位置 3 的输出:", [changed_output[2]])

# 位置 1、2 读不到位置 3，所以修改位置 3 的 K/V 不会改变前两行输出。
changed_k = K[:2] + [[3.0, -4.0]]
changed_future_v = V[:2] + [[-2.0, 5.0]]
_, _, _, changed_future_output = attention(Q, changed_k, changed_future_v)
assert changed_future_output[:2] == output[:2]
assert changed_future_output[2] != output[2]

# 追加未来位置也不改变已有位置的输出。
_, _, _, extended_output = attention(Q + [[2.0, -1.0]], K + [[1.0, -3.0]], V + [[-4.0, 2.0]])
assert extended_output[:3] == output

# 练习：V 完全相同时，加权结果相同，并不要求权重相等。
exercise_q = Q[:2] + [[1.0, -1.0]]
_, _, unequal_weights, same_value_output = attention(exercise_q, K, [[1.0, 1.0] for _ in V])
assert len(set(unequal_weights[2])) == 3
assert all(abs(value - 1.0) < 1e-12 for value in same_value_output[2])


def rotate(vector: list[float], angle: float) -> list[float]:
    x, y = vector
    return [x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)]


# RoPE 的共同平移结论要求原始 Q/K 不变，不要求它们最初同向。
query, key, theta = [1.0, 0.5], [-0.5, 2.0], 0.3
before_shift = dot(rotate(query, 3 * theta), rotate(key, 2 * theta))
after_shift = dot(rotate(query, 13 * theta), rotate(key, 12 * theta))
relative_rotation = dot(query, rotate(key, (2 - 3) * theta))
assert abs(before_shift - after_shift) < 1e-12
assert abs(before_shift - relative_rotation) < 1e-12
print("对照通过：Q/K 与 V 分工、因果遮罩、同值 V 和 RoPE 共同平移。")
