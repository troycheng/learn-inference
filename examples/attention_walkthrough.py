"""用第 3 课的三位置示例跑一遍单头因果 Attention。"""

from math import exp, sqrt


Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
]

K = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.0],
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


head_dimension = len(Q[0])
scale = sqrt(head_dimension)

scores = [[dot(query, key) / scale for key in K] for query in Q]

masked_scores = [
    [score if key_position <= query_position else float("-inf") for key_position, score in enumerate(row)]
    for query_position, row in enumerate(scores)
]

weights = [softmax(row) for row in masked_scores]

output = [
    [
        sum(row_weights[position] * V[position][feature] for position in range(len(V)))
        for feature in range(len(V[0]))
    ]
    for row_weights in weights
]

print_matrix("缩放后的 QK^T / sqrt(D):", scores)
print_matrix("加上因果遮罩:", masked_scores)
print_matrix("逐行 Softmax:", weights)
print_matrix("Attention 输出 A @ V:", output)

for row in weights:
    assert abs(sum(row) - 1.0) < 1e-12
