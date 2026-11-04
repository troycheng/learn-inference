"""两层教学 Decoder：完整前向、逐步生成及 KV Cache 等价性检查。

只依赖标准库。参数是固定的教学数值，未经训练，不代表 Qwen 的输出。
列表省略 B=1 的批次轴；每层 Cache 使用 [头, 位置, 头内维度]。
"""

from argparse import ArgumentParser
from math import cos, exp, sin, sqrt


H, I, HEADS, D, VOCAB, LAYERS = 4, 6, 2, 2, 6, 2
EPS = 1e-6
EMBEDDING = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.5, -0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5, -0.5],
]


def fixed_weight(rows, columns, offset):
    """生成可复现的非随机参数，矩阵按 [输出维度, 输入维度] 保存。"""
    return [
        [(((r * columns + c + offset) * 7) % 19 - 9) / 20
         for c in range(columns)]
        for r in range(rows)
    ]


PARAMETERS = [
    {name: fixed_weight(rows, columns, 31 * layer + offset)
     for name, rows, columns, offset in [
         ("q", H, H, 1), ("k", H, H, 3), ("v", H, H, 5),
         ("o", H, H, 7), ("gate", I, H, 9), ("up", I, H, 11),
         ("down", H, I, 13),
     ]}
    for layer in range(LAYERS)
]
LM_HEAD = fixed_weight(VOCAB, H, 10)


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def linear(weight, x):
    return [dot(row, x) for row in weight]


def add(a, b):
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]


def rmsnorm(x):
    # 所有 RMSNorm 的学习缩放 gamma 固定为 1。
    rms = sqrt(sum(v * v for v in x) / len(x) + EPS)
    return [v / rms for v in x]


def softmax(scores):
    values = [exp(s - max(scores)) for s in scores]
    return [v / sum(values) for v in values]


def silu(x):
    return x / (1 + exp(-x))


def rotate(pair, position):
    # D=2，只有一组二维坐标，标准 RoPE 的该组频率为 1 rad/位置。
    a, b = pair
    return [a * cos(position) - b * sin(position),
            a * sin(position) + b * cos(position)]


def empty_cache():
    return [{"k": [[] for _ in range(HEADS)],
             "v": [[] for _ in range(HEADS)]} for _ in range(LAYERS)]


def forward(token_ids, cache=None):
    """对本轮新 token 前向；返回 Logits、新缓存和最后位置的逐层记录。"""
    assert token_ids and all(0 <= t < VOCAB for t in token_ids)
    old_cache = empty_cache() if cache is None else cache
    assert len(old_cache) == LAYERS
    past = len(old_cache[0]["k"][0])
    for layer_cache in old_cache:
        assert all(len(head) == past for key in ("k", "v")
                   for head in layer_cache[key])
    x = [EMBEDDING[t][:] for t in token_ids]
    trace = [("Embedding", x[-1][:])]
    new_cache = []

    for layer, weights in enumerate(PARAMETERS):
        norm = [rmsnorm(row) for row in x]
        q = [linear(weights["q"], row) for row in norm]
        k = [linear(weights["k"], row) for row in norm]
        v = [linear(weights["v"], row) for row in norm]
        # 仅对新 Q/K 应用当前位置的 RoPE；旧 K 已旋转，直接复用。
        q_heads, k_heads, v_heads = [], [], []
        for head in range(HEADS):
            start = head * D
            q_heads.append([rotate(row[start:start + D], past + t)
                            for t, row in enumerate(q)])
            k_heads.append(old_cache[layer]["k"][head] + [
                rotate(row[start:start + D], past + t)
                for t, row in enumerate(k)])
            v_heads.append(old_cache[layer]["v"][head] + [
                row[start:start + D] for row in v])
        new_cache.append({"k": k_heads, "v": v_heads})

        attended = []
        for t in range(len(token_ids)):
            concat = []
            for head in range(HEADS):
                # 因果遮罩：已有前缀 + 本轮截至 t 的位置。
                visible = past + t + 1
                scores = [dot(q_heads[head][t], key) / sqrt(D)
                          for key in k_heads[head][:visible]]
                probability = softmax(scores)
                concat.extend([
                    sum(probability[j] * v_heads[head][j][d]
                        for j in range(visible))
                    for d in range(D)
                ])
            attended.append(linear(weights["o"], concat))

        after_attention = [add(row, update) for row, update in zip(x, attended)]
        ffn_norm = [rmsnorm(row) for row in after_attention]
        ffn = []
        for row in ffn_norm:
            gate = linear(weights["gate"], row)
            up = linear(weights["up"], row)
            mixed = [silu(g) * u for g, u in zip(gate, up)]
            ffn.append(linear(weights["down"], mixed))
        x = [add(row, update) for row, update in zip(after_attention, ffn)]
        trace.extend([
            (f"Layer {layer + 1} / Attention 前 RMSNorm", norm[-1]),
            (f"Layer {layer + 1} / Attention 输出投影", attended[-1]),
            (f"Layer {layer + 1} / 第一次残差相加", after_attention[-1]),
            (f"Layer {layer + 1} / FFN 前 RMSNorm", ffn_norm[-1]),
            (f"Layer {layer + 1} / FFN 输出", ffn[-1]),
            (f"Layer {layer + 1} / 第二次残差相加", x[-1]),
        ])

    final = [rmsnorm(row) for row in x]
    logits = [linear(LM_HEAD, row) for row in final]
    trace.extend([("最终 RMSNorm", final[-1]), ("LM Head", logits[-1])])
    return logits, new_cache, trace


def max_error(a, b):
    """比较任意同 shape 嵌套列表。"""
    assert len(a) == len(b)
    if not a:
        return 0.0
    if isinstance(a[0], list):
        return max(max_error(x, y) for x, y in zip(a, b))
    return max(abs(x - y) for x, y in zip(a, b))


def greedy(logits):
    return max(range(len(logits)), key=logits.__getitem__)


def check_equivalence():
    sequence = [0, 1, 2, 4, 3]
    reference, reference_cache, _ = forward(sequence)
    largest_error = 0.0
    # 穷举 5 个位置之间的 16 种切分，覆盖多种 Prefill/Chunk/Decode 边界。
    for cuts in range(1 << (len(sequence) - 1)):
        ends = [i + 1 for i in range(len(sequence) - 1) if cuts & (1 << i)]
        cache, collected, start = None, [], 0
        for end in ends + [len(sequence)]:
            logits, cache, _ = forward(sequence[start:end], cache)
            collected.extend(logits)
            start = end
        largest_error = max(largest_error, max_error(collected, reference))
        for actual, expected in zip(cache, reference_cache):
            for name in ("k", "v"):
                largest_error = max(largest_error, max_error(actual[name], expected[name]))
    assert largest_error < 1e-10

    # 改变未来 token，不应改变前面位置的 Logits。
    changed, _, _ = forward([0, 1, 2, 5, 5])
    assert max_error(reference[:3], changed[:3]) < 1e-10
    # 缓存受损必须能被比较发现，避免“只检查生成 ID”漏掉错误。
    _, cache, _ = forward(sequence[:3])
    cache[0]["v"][0][0][0] += 1.0
    corrupted, _, _ = forward(sequence[3:], cache)
    assert max_error(corrupted, reference[3:]) > 1e-5
    return largest_error


def check_input_contrasts():
    """固定权重与位置编号，分别改变前文和追加未来位置。"""
    original_ids, changed_ids = [0, 1, 2], [4, 5, 2]
    original, original_cache, original_trace = forward(original_ids)
    changed, _, changed_trace = forward(changed_ids)
    # 两段输入的末位置都是 ID 2，Embedding 相同，前文不同。
    assert original_trace[0][1] == changed_trace[0][1] == [0, 0, 1, 0]
    assert max_error(original_trace[-3][1], changed_trace[-3][1]) > 1e-5
    assert max_error(original[-1], changed[-1]) > 1e-5
    expected_changed = [-0.163507, 0.806940, -0.261768, 0.708679, -0.295274, 0.610419]
    assert max_error(changed[-1], expected_changed) < 5.1e-7
    # Logits 已改变，但这个例子的最大项仍相同，不能只比较生成 ID。
    assert greedy(original[-1]) == greedy(changed[-1]) == 1

    extended, extended_cache, _ = forward(original_ids + [4])
    prefix_error = max_error(original, extended[:len(original_ids)])
    assert prefix_error < 1e-10
    for before, after in zip(original_cache, extended_cache):
        for name in ("k", "v"):
            for head in range(HEADS):
                assert max_error(before[name][head],
                                 after[name][head][:len(original_ids)]) < 1e-10
    return original[-1], changed[-1], prefix_error


def main(show_trace=False):
    history = [0, 1, 2]
    pending = history[:]
    cache = None
    for step in range(3):
        before = 0 if cache is None else len(cache[0]["k"][0])
        logits, cache, trace = forward(pending, cache)
        expected_logits = [
            [-0.113440, 0.819185, -0.212208, 0.720417, -0.287349, 0.621649],
            [-0.853404, 0.269412, -0.961859, 0.160957, 0.758579, 0.052503],
            [0.253075, 0.348595, 0.259690, 0.355210, -0.772653, 0.361825],
        ]
        assert max_error(logits[-1], expected_logits[step]) < 5.1e-7
        reference, _, _ = forward(history)
        error = max_error(logits[-1], reference[-1])
        assert error < 1e-10
        chosen = greedy(logits[-1])
        assert chosen == [1, 4, 5][step]
        label = "Prefill" if step == 0 else f"Decode {step}"
        print(f"{label}: input={pending}, positions={list(range(before, len(history)))}")
        print(f"  shape: X=[1,{len(pending)},4], Q=[1,2,{len(pending)},2], "
              f"K/V=[1,2,{len(history)},2], Logits=[1,{len(pending)},6]")
        print("  Logits:", [round(v, 6) for v in logits[-1]])
        print(f"  next_id={chosen}, cache_length={len(cache[0]['k'][0])}, error={error:.2e}")
        if step == 0 and show_trace:
            print("  最后输入位置的逐层结果：")
            for name, row in trace:
                print(f"    {name}: {[round(v, 6) for v in row]}")
        history.append(chosen)
        pending = [chosen]
    original, changed, prefix_error = check_input_contrasts()
    print("改变前文，末位置 Embedding 均为 [0,0,1,0]：")
    print("  [0,1,2] 末位置 Logits:", [round(v, 6) for v in original])
    print("  [4,5,2] 末位置 Logits:", [round(v, 6) for v in changed])
    print("  两者贪心选择均为 ID 1，但 Logits 不同")
    print(f"追加未来 token 后，原有位置 Logits 最大误差: {prefix_error:.2e}")
    print(f"16 种切分的 Logits 与缓存最大误差: {check_equivalence():.2e}")
    print("前文影响、因果约束、缓存等价性、缓存损坏检测: PASS")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="同时打印 Prefill 最后位置的逐层数值")
    main(show_trace=parser.parse_args().trace)
