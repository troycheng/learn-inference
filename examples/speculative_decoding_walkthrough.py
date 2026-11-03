"""第 9 课的贪心推测解码计数：不实现随机 Speculative Sampling。"""


def verify_greedy(past, pending, candidates, target_next):
    """pending is confirmed, but not yet in the target model cache."""
    inputs = [pending, *candidates]
    # Causal forward: each position produces a distribution for the NEXT token.
    predictions = [target_next(past + inputs[:index + 1]) for index in range(len(inputs))]
    accepted = 0
    for candidate, prediction in zip(candidates, predictions):
        if candidate != prediction:
            break
        accepted += 1
    new_outputs = candidates[:accepted] + [predictions[accepted]]
    retained_cache = past + inputs[:accepted + 1]
    return inputs, new_outputs, retained_cache, accepted


past = ["p1", "p2"]
pending = "y0"
continuation = ["c1", "c2", "c3", "b"]
known_predictions = {
    tuple(past + [pending] + continuation[:index]): token
    for index, token in enumerate(continuation)
}


def target_next(prefix):
    # A different draft prefix yields unusable predictions after rejection.
    return known_predictions.get(tuple(prefix), "wrong-prefix-output")


for accepted_count in range(4):
    candidates = continuation[:3].copy()
    if accepted_count < 3:
        candidates[accepted_count] = "rejected-candidate"
    inputs, outputs, cache, accepted = verify_greedy(past, pending, candidates, target_next)
    assert len(inputs) == 4  # k+1, not k.
    assert accepted == accepted_count
    assert outputs == continuation[:accepted + 1]
    assert cache == past + [pending] + outputs[:-1]
    assert len(outputs) == accepted + 1
    assert "rejected-candidate" not in cache
    assert "wrong-prefix-output" not in outputs
    print(f"accepted={accepted}: new outputs={outputs}, retained cache={cache}, next pending={outputs[-1]}")

# Reproduce the figure's specific rejection example: c3 is replaced by r.
rejection_predictions = {
    tuple(past + [pending]): "c1",
    tuple(past + [pending, "c1"]): "c2",
    tuple(past + [pending, "c1", "c2"]): "r",
}
_, outputs, cache, accepted = verify_greedy(
    past, pending, ["c1", "c2", "c3"],
    lambda prefix: rejection_predictions.get(tuple(prefix), "wrong-prefix-output"),
)
assert outputs == ["c1", "c2", "r"]
assert cache == past + [pending, "c1", "c2"]
assert accepted == 2

round_ms = 2 + 7 + 1  # Drafter, target verification, acceptance/state work.
ordinary_ms_per_token = 5
for average_accepted, expected_speedup in [(2, 1.5), (0.5, 0.75)]:
    outputs_per_round = average_accepted + 1
    ms_per_output = round_ms / outputs_per_round
    speedup = ordinary_ms_per_token / ms_per_output
    assert abs(speedup - expected_speedup) < 1e-12
    print(f"average accepted={average_accepted}: {ms_per_output:.2f} ms/output, speedup={speedup:g}x")
