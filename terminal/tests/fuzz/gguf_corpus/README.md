# GGUF Fuzzing & Mutation Corpus

Corpus directory containing corrupted, truncated, overflowed, and crafted GGUF headers used for regression fault-injection testing.
- `overflow_dim.gguf`: Contains dimensions that cause 64-bit integer overflow.
- `zero_shard.gguf`: Missing shards in sequence.
- `bad_magic.gguf`: Non-GGUF magic bytes.
- `tampered_sha.gguf`: Payload modified after hash computation.
