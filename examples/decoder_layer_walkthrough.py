"""复算第 2 课的 RMSNorm 与 SwiGLU 小例子。"""

from math import exp, sqrt


def rmsnorm(vector: list[float]) -> list[float]:
    """忽略 epsilon 和学习缩放，只保留 RMSNorm 的核心计算。"""
    rms = sqrt(sum(value * value for value in vector) / len(vector))
    return [value / rms for value in vector]


def linear(weight: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(coefficient * value for coefficient, value in zip(row, vector)) for row in weight]


def silu(value: float) -> float:
    return value / (1.0 + exp(-value))


def show(name: str, vector: list[float]) -> None:
    print(f"{name}: [" + ", ".join(f"{value:.3f}" for value in vector) + "]")


rms_input = [3.0, 4.0]
rms_output = rmsnorm(rms_input)
show("RMSNorm([3, 4])", rms_output)

y_norm = [1.0, 2.0]
w_gate = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
w_up = [
    [1.0, 1.0],
    [1.0, -1.0],
    [0.0, 1.0],
]
w_down = [
    [1.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
]


def ffn(vector: list[float]) -> list[float]:
    """对一条已归一化向量执行正文的 SwiGLU，不包含残差相加。"""
    coefficients = [silu(value) for value in linear(w_gate, vector)]
    features = linear(w_up, vector)
    return linear(w_down, [coefficient * value for coefficient, value in zip(coefficients, features)])


gate = linear(w_gate, y_norm)
activated_gate = [silu(value) for value in gate]
up = linear(w_up, y_norm)
gated_features = [gate_value * up_value for gate_value, up_value in zip(activated_gate, up)]
ffn_output = linear(w_down, gated_features)

show("gate_proj(y_norm)", gate)
show("SiLU(gate_proj(y_norm))", activated_gate)
show("up_proj(y_norm)", up)
show("逐元素门控结果", gated_features)
show("down_proj 输出", ffn_output)

assert all(abs(actual - expected) < 1e-9 for actual, expected in zip(rms_output, [0.8485281374, 1.1313708499]))
assert all(abs(actual - expected) < 1e-12 for actual, expected in zip(gate, [1.0, 2.0, 3.0]))
assert all(abs(actual - expected) < 1e-12 for actual, expected in zip(up, [3.0, -1.0, 2.0]))
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(ffn_output, [2.193, 1.977]))

# 同一套参数下，输入改变时，特征组合与门控系数都会重新计算。
changed_input = [2.0, 1.0]
changed_output = ffn(changed_input)
show("相同参数，FFN([2, 1])", changed_output)
assert linear(w_up, changed_input) == [3.0, 1.0, 1.0]
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(changed_output, [5.285, 1.794]))

# 固定 FFN 的第 2 行输入，只改第 1 行；第 2 行输出不受影响。
rows_before = [y_norm, [2.0, 1.0]]
rows_after = [[-1.0, 2.0], rows_before[1][:]]
outputs_before = [ffn(row) for row in rows_before]
outputs_after = [ffn(row) for row in rows_after]
assert outputs_before[0] != outputs_after[0]
assert outputs_before[1] == outputs_after[1]

# 若是单个无偏置 Linear，F(2x) 应等于 2F(x)；本例 SwiGLU 不满足。
twice_input_output = ffn([2.0 * value for value in y_norm])
twice_output = [2.0 * value for value in ffn_output]
assert any(abs(actual - expected) > 1e-3 for actual, expected in zip(twice_input_output, twice_output))
print("对照通过：FFN 逐行独立，门控随输入变化，整体不是线性映射。")
