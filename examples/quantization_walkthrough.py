"""对称 INT4 教学量化：Scale、舍入、截断、分组及 Linear 误差。

整数用 Python int 表示，没有真正打包成 INT4，也没有模拟硬件低精度 Kernel。
"""

from math import isclose


WEIGHT = [[0.2, -0.4, 0.6, 3.5], [0.1, 0.3, -0.5, 0.7]]
INPUT = [1.0, 2.0, -1.0, 0.5]


def quantize(values, scale):
    assert scale > 0
    # 本例使用对称的 -7...7，空出 INT4 可编码的 -8；半整数向偶数舍入。
    return [max(-7, min(7, round(value / scale))) for value in values]


def reconstruct(codes, scale):
    return [code * scale for code in codes]


def quantize_group(values):
    largest = max(abs(value) for value in values)
    scale = largest / 7 if largest else 1.0
    codes = quantize(values, scale)
    return scale, codes, reconstruct(codes, scale)


def linear(weight, x):
    return [sum(w * v for w, v in zip(row, x)) for row in weight]


def close(actual, expected):
    assert len(actual) == len(expected)
    assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
               for a, b in zip(actual, expected))


def main():
    flat = [value for row in WEIGHT for value in row]
    scale, codes, restored = quantize_group(flat)
    restored_matrix = [restored[:4], restored[4:]]
    original = linear(WEIGHT, INPUT)
    whole = linear(restored_matrix, INPUT)
    close(original, [0.55, 1.55])
    close(whole, [0.25, 1.75])
    assert scale == 0.5
    assert codes == [0, -1, 1, 7, 0, 1, -1, 1]
    assert quantize([0.6, 4.2, -4.2], 0.5) == [1, 7, -7]
    assert quantize_group([0.0, 0.0]) == (1.0, [0, 0], [0.0, 0.0])
    assert quantize([1.5, 2.5, -1.5, -2.5], 1) == [2, 2, -2, -2]

    print("原始 Linear:", original)
    print("整张矩阵共用 Scale:", scale)
    print("整数编码:", codes)
    print("反量化权重:", restored_matrix)
    print("反量化后 Linear:", whole)
    # 验证 output_error = weight_error @ input，而非只比较权重误差。
    weight_error = [[b - a for a, b in zip(row, restored_row)]
                    for row, restored_row in zip(WEIGHT, restored_matrix)]
    close(linear(weight_error, INPUT), [b - a for a, b in zip(original, whole)])

    grouped = []
    for row in WEIGHT:
        restored_row = []
        for start in (0, 2):
            values = row[start:start + 2]
            group_scale, group_codes, group_values = quantize_group(values)
            print(f"分组 {values}: scale={group_scale:.6f}, q={group_codes}, "
                  f"restored={[round(v, 6) for v in group_values]}")
            restored_row.extend(group_values)
        grouped.append(restored_row)
    result = linear(grouped, INPUT)
    close(result, [0.6785714285714286, 1.5357142857142858])
    print("每行每 2 个数一组，Linear 输出:", result)
    print("舍入、截断、零值组、输出误差与分组量化: PASS")


if __name__ == "__main__":
    main()
