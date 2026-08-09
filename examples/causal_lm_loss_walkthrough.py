"""复算第 1 课的 Softmax 概率与正确 token 交叉熵损失。"""

from math import exp, log


def softmax(values: list[float]) -> list[float]:
    largest = max(values)
    numerators = [exp(value - largest) for value in values]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def cross_entropy_for_target(logits: list[float], target_index: int) -> float:
    probabilities = softmax(logits)
    return -log(probabilities[target_index])


logits = [-1.0, 0.0, 3.0, 1.0]
probabilities = softmax(logits)

high_probability_loss = cross_entropy_for_target(logits, target_index=2)
low_probability_loss = cross_entropy_for_target(logits, target_index=3)

print("Softmax 概率：       [" + ", ".join(f"{value:.3f}" for value in probabilities) + "]")
print(f"正确答案为第 3 项：  p={probabilities[2]:.3f}，Loss={high_probability_loss:.3f}")
print(f"正确答案为第 4 项：  p={probabilities[3]:.3f}，Loss={low_probability_loss:.3f}")

assert abs(sum(probabilities) - 1.0) < 1e-12
assert abs(probabilities[2] - 0.8309526605) < 1e-9
assert abs(high_probability_loss - 0.1851825) < 1e-6
assert abs(low_probability_loss - 2.1851825) < 1e-6
assert high_probability_loss < low_probability_loss
