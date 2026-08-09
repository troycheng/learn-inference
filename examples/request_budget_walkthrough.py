"""复算综合评审中的请求状态、TP 下物理 KV 与可复用前缀。"""


KIB = 1024
MIB = 1024**2
GIB = 1024**3

prompt_tokens = 4096
max_output_tokens = 256
concurrency = 32
shared_prefix_tokens = 2048
shared_prefix_ratio = 0.8
full_attention_layers = 8
global_kv_heads = 4
head_dimension = 256
kv_element_bytes = 2
tensor_parallel_size = 8

kv_bytes_per_position = (
    2
    * full_attention_layers
    * global_kv_heads
    * head_dimension
    * kv_element_bytes
)
local_kv_heads = max(1, global_kv_heads // tensor_parallel_size)
kv_bytes_per_rank_per_position = (
    2
    * full_attention_layers
    * local_kv_heads
    * head_dimension
    * kv_element_bytes
)
physical_kv_bytes_all_ranks_per_position = (
    tensor_parallel_size * kv_bytes_per_rank_per_position
)
gated_deltanet_state_per_request = int(49.5 * MIB)
reserved_context_tokens = prompt_tokens + max_output_tokens

kv_per_request = reserved_context_tokens * kv_bytes_per_position
state_per_request = kv_per_request + gated_deltanet_state_per_request
state_for_concurrency = concurrency * state_per_request
ideal_average_reused_prompt_tokens = shared_prefix_ratio * shared_prefix_tokens

print(f"预留上下文长度：       {reserved_context_tokens} token")
print(f"逻辑 KV：              {kv_bytes_per_position / KIB:.1f} KiB / 位置")
print(f"TP=8 每 Rank KV：      {kv_bytes_per_rank_per_position / KIB:.1f} KiB / 位置")
print(f"8 个 Rank 物理 KV：   {physical_kv_bytes_all_ranks_per_position / KIB:.1f} KiB / 位置")
print(f"逻辑 KV：              {kv_per_request / MIB:.1f} MiB / 请求")
print(f"加固定状态：           {state_per_request / MIB:.1f} MiB / 请求")
print(f"并发 {concurrency} 的模型状态： {state_for_concurrency / GIB:.2f} GiB")
print(
    "理想命中条件下，按请求比例折算的可复用前缀： "
    f"{ideal_average_reused_prompt_tokens:.1f} token / 请求"
)

assert reserved_context_tokens == 4352
assert kv_bytes_per_position == 32 * KIB
assert local_kv_heads == 1
assert kv_bytes_per_rank_per_position == 8 * KIB
assert physical_kv_bytes_all_ranks_per_position == 64 * KIB
assert kv_per_request == 136 * MIB
assert state_per_request == int(185.5 * MIB)
assert abs(state_for_concurrency / GIB - 5.796875) < 1e-12
assert ideal_average_reused_prompt_tokens == 1638.4
