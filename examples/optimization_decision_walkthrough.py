"""复算第 9 课评审案例中 Full Attention Kernel 对 TTFT 的影响。"""


def amdahl_speedup(covered_fraction: float, local_speedup: float) -> float:
    """Return the end-to-end speedup when only one time fraction is accelerated."""
    return 1.0 / ((1.0 - covered_fraction) + covered_fraction / local_speedup)


full_attention_fraction_of_prefill_gpu = 0.24
full_attention_kernel_speedup = 1.8
queue_seconds = 1.08
input_and_scheduling_seconds = 0.12
prefill_gpu_seconds = 1.05
first_decode_and_return_seconds = 0.10
target_ttft_seconds = 1.5

prefill_gpu_speedup_upper_bound = amdahl_speedup(
    full_attention_fraction_of_prefill_gpu,
    full_attention_kernel_speedup,
)
current_ttft_seconds = (
    queue_seconds
    + input_and_scheduling_seconds
    + prefill_gpu_seconds
    + first_decode_and_return_seconds
)
optimized_prefill_gpu_seconds = prefill_gpu_seconds / prefill_gpu_speedup_upper_bound
projected_ttft_with_queue_unchanged = (
    queue_seconds
    + input_and_scheduling_seconds
    + optimized_prefill_gpu_seconds
    + first_decode_and_return_seconds
)
saved_seconds_with_queue_unchanged = (
    current_ttft_seconds - projected_ttft_with_queue_unchanged
)

print(f"Prefill GPU 时间的 Amdahl 上限： {prefill_gpu_speedup_upper_bound:.3f}x")
print(f"当前 trace 的 TTFT：             {current_ttft_seconds:.3f} s")
print(f"假设排队不变，优化后的 TTFT：    {projected_ttft_with_queue_unchanged:.3f} s")
print(f"这条 trace 直接节省：            {saved_seconds_with_queue_unchanged:.3f} s")
print(f"目标 TTFT：                      {target_ttft_seconds:.3f} s")

assert abs(prefill_gpu_speedup_upper_bound - 1.119402985074627) < 1e-12
assert abs(current_ttft_seconds - 2.35) < 1e-12
assert abs(projected_ttft_with_queue_unchanged - 2.238) < 1e-12
assert projected_ttft_with_queue_unchanged > target_ttft_seconds
