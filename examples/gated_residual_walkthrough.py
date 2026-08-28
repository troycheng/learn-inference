"""Reproduce the small Gated Residual read/write example from lesson 10."""


def elementwise_mul(left, right):
    return [a * b for a, b in zip(left, right)]


def add(left, right):
    return [a + b for a, b in zip(left, right)]


def scale(value, vector):
    return [value * item for item in vector]


normalized_branches = [[2.0, 4.0], [6.0, 2.0]]
read_gates = [[0.8, 0.2], [0.3, 0.9]]

gated_branches = [
    elementwise_mul(gate, branch) for gate, branch in zip(read_gates, normalized_branches)
]
mixed_input = [sum(values) / len(normalized_branches) for values in zip(*gated_branches)]

raw_branches = [[2.0, 4.0], [6.0, 2.0]]
block_output = [0.6, -1.0]
write_gates = [1.5, 0.4]
updated_branches = [
    add(branch, scale(gate, block_output)) for branch, gate in zip(raw_branches, write_gates)
]

print("GR Read")
for index, value in enumerate(gated_branches, start=1):
    print(f"  G{index} * Rhat{index} = {value}")
print(f"  mixed input x = {mixed_input}")

print("GR Write")
for index, value in enumerate(updated_branches, start=1):
    print(f"  R{index}' = {value}")

assert mixed_input == [1.7, 1.3]
assert updated_branches == [[2.9, 2.5], [6.24, 1.6]]
