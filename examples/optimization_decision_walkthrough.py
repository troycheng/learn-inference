"""复算第 9 课优化案例中的端到端收益上限。"""


def amdahl_speedup(covered_fraction: float, local_speedup: float) -> float:
    """Return the end-to-end speedup when only one time fraction is accelerated."""
    return 1.0 / ((1.0 - covered_fraction) + covered_fraction / local_speedup)


full_attention_fraction_of_prefill_gpu = 0.24
full_attention_kernel_speedup = 1.8
current_ttft_seconds = 2.35
target_ttft_seconds = 1.5

prefill_gpu_speedup_upper_bound = amdahl_speedup(
    full_attention_fraction_of_prefill_gpu,
    full_attention_kernel_speedup,
)
required_end_to_end_speedup = current_ttft_seconds / target_ttft_seconds

print(f"Prefill GPU 时间的 Amdahl 上限： {prefill_gpu_speedup_upper_bound:.3f}x")
print(f"TTFT 达标所需的整体加速：       {required_end_to_end_speedup:.3f}x")

assert abs(prefill_gpu_speedup_upper_bound - 1.119402985074627) < 1e-12
assert abs(required_end_to_end_speedup - 1.5666666666666667) < 1e-12
assert prefill_gpu_speedup_upper_bound < required_end_to_end_speedup
