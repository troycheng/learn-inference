"""复算结业案例的请求状态与可复用前缀。"""


KIB = 1024
MIB = 1024**2
GIB = 1024**3

prompt_tokens = 4096
max_output_tokens = 256
concurrency = 32
shared_prefix_tokens = 2048
shared_prefix_ratio = 0.8

kv_bytes_per_position = 32 * KIB
gated_deltanet_state_per_request = int(49.5 * MIB)
reserved_context_tokens = prompt_tokens + max_output_tokens

kv_per_request = reserved_context_tokens * kv_bytes_per_position
state_per_request = kv_per_request + gated_deltanet_state_per_request
state_for_concurrency = concurrency * state_per_request
average_reused_prompt_tokens = shared_prefix_ratio * shared_prefix_tokens

print(f"预留上下文长度：       {reserved_context_tokens} token")
print(f"逻辑 KV：              {kv_per_request / MIB:.1f} MiB / 请求")
print(f"加固定状态：           {state_per_request / MIB:.1f} MiB / 请求")
print(f"并发 {concurrency} 的模型状态： {state_for_concurrency / GIB:.2f} GiB")
print(f"平均可复用 Prompt：    {average_reused_prompt_tokens:.1f} token / 请求")

assert reserved_context_tokens == 4352
assert kv_per_request == 136 * MIB
assert state_per_request == int(185.5 * MIB)
assert abs(state_for_concurrency / GIB - 5.796875) < 1e-12
assert average_reused_prompt_tokens == 1638.4
