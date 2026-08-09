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
