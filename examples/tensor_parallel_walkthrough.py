"""在 CPU 上复算 TP 切分，模拟两张卡的局部结果，不执行真实通信。"""

from math import exp, isclose


def linear(weight, x):
    return [sum(w * v for w, v in zip(row, x)) for row in weight]


def main():
    weight = [[1, 2, 0, -1], [0, 1, 3, 2]]
    x = [1, 2, 3, 4]
    reference = linear(weight, x)
    # 输出维度切分：每张卡算不同的输出坐标。
    output_0 = linear(weight[:1], x)
    output_1 = linear(weight[1:], x)
    assert output_0 + output_1 == reference == [1, 19]
    # 输入维度切分：每张卡算每个输出坐标的一部分乘加。
    partial_0 = linear([row[:2] for row in weight], x[:2])
    partial_1 = linear([row[2:] for row in weight], x[2:])
    assert partial_0 == [5, 2] and partial_1 == [-4, 17]
    assert [a + b for a, b in zip(partial_0, partial_1)] == reference
    print(f"输出切分: {output_0} concat {output_1} = {reference}")
    print(f"输入切分: {partial_0} + {partial_1} = {reference}")

    # gate/up 按同一组中间坐标切分，逐元素运算保持在本地。
    gate = [[1, 0], [0, 1], [1, 1], [1, -1]]
    up = [[1, 1], [1, -1], [0, 1], [1, 0]]
    down = weight
    ffn_input = [1, 2]

    def gated_features(gate_weight, up_weight):
        g = linear(gate_weight, ffn_input)
        u = linear(up_weight, ffn_input)
        return [a / (1 + exp(-a)) * b for a, b in zip(g, u)]

    full = linear(down, gated_features(gate, up))
    parts = []
    for start in (0, 2):
        local = gated_features(gate[start:start + 2], up[start:start + 2])
        parts.append(linear([row[start:start + 2] for row in down], local))
    combined = [a + b for a, b in zip(*parts)]
    assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
               for a, b in zip(full, combined))
    print("SwiGLU 中间结果不拼接，down_proj 后求和:", combined)
    print("Linear 两种切分与 SwiGLU 的 TP=2 等价性: PASS")


if __name__ == "__main__":
    main()
