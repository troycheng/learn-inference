"""复算第 4 课的三次 Gated DeltaNet 状态更新。"""

from math import sqrt


Matrix = list[list[float]]


def vector_matrix(vector: list[float], matrix: Matrix) -> list[float]:
    return [
        sum(vector[row] * matrix[row][column] for row in range(len(vector)))
        for column in range(len(matrix[0]))
    ]


def outer(left: list[float], right: list[float]) -> Matrix:
    return [[left_value * right_value for right_value in right] for left_value in left]


def scale_matrix(factor: float, matrix: Matrix) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def add_matrix(left: Matrix, right: Matrix) -> Matrix:
    return [[a + b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def update(
    state: Matrix,
    key: list[float],
    value: list[float],
    alpha: float,
    beta: float,
) -> tuple[Matrix, list[float], list[float]]:
    decayed_state = scale_matrix(alpha, state)
    old_value = vector_matrix(key, decayed_state)
    error = [target - old for target, old in zip(value, old_value)]
    correction = scale_matrix(beta, outer(key, error))
    return add_matrix(decayed_state, correction), old_value, error


def show_matrix(name: str, matrix: Matrix) -> None:
    print(name)
    for row in matrix:
        print("  [" + ", ".join(f"{value:.3f}" for value in row) + "]")


state_0 = [[0.0, 0.0], [0.0, 0.0]]
state_1, old_1, error_1 = update(state_0, [1.0, 0.0], [3.0, 4.0], alpha=1.0, beta=1.0)
state_2, old_2, error_2 = update(state_1, [1.0, 0.0], [5.0, 1.0], alpha=1.0, beta=1.0)
state_3, old_3, error_3 = update(state_2, [1.0, 0.0], [3.0, 3.0], alpha=0.5, beta=0.5)

print(f"第 1 次旧记录：{old_1}，误差：{error_1}")
show_matrix("S1", state_1)
print(f"第 2 次旧记录：{old_2}，误差：{error_2}")
show_matrix("S2", state_2)
print(f"第 3 次旧记录：{old_3}，误差：{error_3}")
show_matrix("S3", state_3)

query = [1.0, 0.0]
readout = [value / sqrt(2.0) for value in vector_matrix(query, state_3)]
normalized_readout = [value / sqrt(sum(item * item for item in readout) / len(readout)) for value in readout]
gated_output = [value * gate for value, gate in zip(normalized_readout, [0.5, 0.8])]

print("状态读出: [" + ", ".join(f"{value:.3f}" for value in readout) + "]")
print("RMSNorm 后: [" + ", ".join(f"{value:.3f}" for value in normalized_readout) + "]")
print("输出门控后: [" + ", ".join(f"{value:.3f}" for value in gated_output) + "]")

assert state_1 == [[3.0, 4.0], [0.0, 0.0]]
assert state_2 == [[5.0, 1.0], [0.0, 0.0]]
assert state_3 == [[2.75, 1.75], [0.0, 0.0]]
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(readout, [1.945, 1.237]))
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(gated_output, [0.597, 0.607]))
