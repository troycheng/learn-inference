"""Reproduce the toy N-gram hash and real parameter sizing from lesson 12."""


table_size = 11
token_t_minus_2 = 2
token_t_minus_1 = 5
token_t = 7

hash_2gram = ((token_t * 3) ^ (token_t_minus_1 * 5)) % table_size
hash_3gram = ((token_t * 3) ^ (token_t_minus_1 * 5) ^ (token_t_minus_2 * 7)) % table_size

heads_per_ngram = 8
ngram_orders = 2
ngram_heads = heads_per_ngram * ngram_orders
head_width = 160
base_rows_per_head = 20_000_000

active_ngram_parameters = ngram_heads * head_width
total_ngram_parameters = ngram_heads * base_rows_per_head * head_width
bf16_gib = total_ngram_parameters * 2 / 1024**3

print(f"2-gram toy table row = {hash_2gram}")
print(f"3-gram toy table row = {hash_3gram}")
print(f"lookup heads per token = {ngram_heads}")
print(f"active N-gram parameters per token = {active_ngram_parameters}")
print(f"approximate total parameters = {total_ngram_parameters / 1e9:.1f}B")
print(f"approximate BF16 payload = {bf16_gib:.1f} GiB")

assert hash_2gram == 1
assert hash_3gram == 2
assert active_ngram_parameters == 2560
assert total_ngram_parameters == 51_200_000_000
