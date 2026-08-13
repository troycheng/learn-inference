"""复算第 5 课的 Router、Top-2 和 Shared Expert 小例子。"""

from math import exp


def softmax(values: list[float]) -> list[float]:
    largest = max(values)
    numerators = [exp(value - largest) for value in values]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def weighted_sum(weights: list[float], vectors: list[list[float]]) -> list[float]:
    return [
        sum(weight * vector[feature] for weight, vector in zip(weights, vectors))
        for feature in range(len(vectors[0]))
    ]


router_logits = [0.2, -0.8, 1.4, 0.0]
router_probabilities = softmax(router_logits)

selected_experts = sorted(
    range(len(router_logits)),
    key=lambda expert_id: router_probabilities[expert_id],
    reverse=True,
)[:2]

selected_probabilities = [router_probabilities[expert_id] for expert_id in selected_experts]
selected_total = sum(selected_probabilities)
routing_weights = [probability / selected_total for probability in selected_probabilities]

expert_outputs = {
    2: [2.0, 0.0],
    0: [0.0, 3.0],
}
routed_output = weighted_sum(routing_weights, [expert_outputs[expert_id] for expert_id in selected_experts])

shared_expert_output = [0.5, 0.5]
shared_gate = 0.4
gated_shared_output = [shared_gate * value for value in shared_expert_output]
moe_output = [routed + shared for routed, shared in zip(routed_output, gated_shared_output)]

print("Router Softmax：       [" + ", ".join(f"{value:.3f}" for value in router_probabilities) + "]")
print(f"Top-2 Expert IDs：    {selected_experts}")
print("归一化后的路由权重： [" + ", ".join(f"{value:.3f}" for value in routing_weights) + "]")
print("Routed 输出：          [" + ", ".join(f"{value:.3f}" for value in routed_output) + "]")
print("加上 Shared Expert：  [" + ", ".join(f"{value:.3f}" for value in moe_output) + "]")

assert selected_experts == [2, 0]
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(routing_weights, [0.769, 0.231]))
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(routed_output, [1.537, 0.694]))
assert all(abs(actual - expected) < 1e-3 for actual, expected in zip(moe_output, [1.737, 0.894]))
