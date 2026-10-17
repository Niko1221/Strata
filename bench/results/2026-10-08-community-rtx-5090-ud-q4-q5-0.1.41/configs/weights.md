# Model files

Checked with [../scripts/verify_weights.py](../scripts/verify_weights.py) `--sha`: every byte hashed on this machine and compared with the Hugging Face LFS SHA-256 of the same-named file on `main`. The UD-Q5_K_XL and Q8_0 rows were checked on 2026-10-08, when unsloth/Qwen3.8-Flash-Next-GGUF `main` was `766911a6b7369840a91dbcd95f9f997acaab6cd6`; their per-file hashes are in [weights-q5-q8.json](weights-q5-q8.json). The other rows come from an earlier run of the same check on this machine.

| File | Bytes | SHA-256 matches Hugging Face |
| --- | ---: | --- |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF: `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | yes |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF: `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | yes |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF: `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf` | 10,946,618 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf` | 682,434,912 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf` | 54,400,261,312 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf` | 49,990,245,824 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf` | 49,882,416,192 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `UD-Q5_K_XL/Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf` | 3,320,101,792 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf` | 10,946,624 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf` | 682,434,912 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf` | 54,400,261,312 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf` | 49,446,841,216 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf` | 49,668,930,400 | yes |
| unsloth/Qwen3.8-Flash-Next-GGUF: `Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf` | 34,015,618,784 | yes |
