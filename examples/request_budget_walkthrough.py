"""复算长上下文扩容案例中的逻辑 KV 与 TP 下每 Rank KV。"""


KIB = 1024
MIB = 1024**2
GIB = 1024**3

current_prompt_tokens = 8192
current_max_output_tokens = 1024
target_prompt_tokens = 65536
target_max_output_tokens = 4096
concurrency = 32
replica_count = 2
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
current_context_tokens = current_prompt_tokens + current_max_output_tokens
target_context_tokens = target_prompt_tokens + target_max_output_tokens

current_logical_kv_per_request = current_context_tokens * kv_bytes_per_position
target_logical_kv_per_request = target_context_tokens * kv_bytes_per_position
target_logical_state_per_request = (
    target_logical_kv_per_request + gated_deltanet_state_per_request
)
target_logical_state_for_concurrency = concurrency * target_logical_state_per_request

current_rank_kv_per_request = current_context_tokens * kv_bytes_per_rank_per_position
target_rank_kv_per_request = target_context_tokens * kv_bytes_per_rank_per_position
current_rank_kv_for_concurrency = concurrency * current_rank_kv_per_request
target_rank_kv_for_concurrency = concurrency * target_rank_kv_per_request
target_rank_kv_with_two_replicas = (
    concurrency // replica_count * target_rank_kv_per_request
)

print(f"现网上下文上限：       {current_context_tokens} 个位置")
print(f"目标上下文上限：       {target_context_tokens} 个位置")
print(f"逻辑 KV：              {kv_bytes_per_position / KIB:.1f} KiB / 位置")
print(f"TP=8 每 Rank KV：      {kv_bytes_per_rank_per_position / KIB:.1f} KiB / 位置")
print(f"8 个 Rank 物理 KV：   {physical_kv_bytes_all_ranks_per_position / KIB:.1f} KiB / 位置")
print(f"现网逻辑 KV：          {current_logical_kv_per_request / MIB:.1f} MiB / 请求")
print(f"目标逻辑 KV：          {target_logical_kv_per_request / MIB:.1f} MiB / 请求")
print(f"目标逻辑状态：         {target_logical_state_per_request / MIB:.1f} MiB / 请求")
print(
    f"目标并发 {concurrency} 的逻辑状态： "
    f"{target_logical_state_for_concurrency / GIB:.2f} GiB"
)
print(
    f"现网并发 {concurrency} 的每 Rank KV： "
    f"{current_rank_kv_for_concurrency / GIB:.2f} GiB"
)
print(
    f"目标并发 {concurrency} 的每 Rank KV： "
    f"{target_rank_kv_for_concurrency / GIB:.2f} GiB"
)
print(
    f"两个 TP=8 副本平均分流后的每 Rank KV： "
    f"{target_rank_kv_with_two_replicas / GIB:.2f} GiB"
)

assert current_context_tokens == 9216
assert target_context_tokens == 69632
assert kv_bytes_per_position == 32 * KIB
assert local_kv_heads == 1
assert kv_bytes_per_rank_per_position == 8 * KIB
assert physical_kv_bytes_all_ranks_per_position == 64 * KIB
assert current_logical_kv_per_request == 288 * MIB
assert target_logical_kv_per_request == 2176 * MIB
assert target_logical_state_per_request == int(2225.5 * MIB)
assert abs(target_logical_state_for_concurrency / GIB - 69.546875) < 1e-12
assert current_rank_kv_per_request == 72 * MIB
assert target_rank_kv_per_request == 544 * MIB
assert current_rank_kv_for_concurrency == int(2.25 * GIB)
assert target_rank_kv_for_concurrency == 17 * GIB
assert target_rank_kv_with_two_replicas == int(8.5 * GIB)
