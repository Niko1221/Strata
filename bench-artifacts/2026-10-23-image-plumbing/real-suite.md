# real calibration suite (QFN live)

| qtype | dataset | acc raw→cal | NLL raw→cal | ECE15 raw→cal | flips | fit |
|---|---|---|---|---|---|---|
| noul-pope | POPE adversarial | 88.3%→88.3% | 0.2087→0.2018 | 0.0993→0.1107 | 0 | {"a": 1.145657, "c": -0.076814, "nll": 0.2686282323117881, "nll_identity": 0.2711070928538245, "improvement": 0.009143, "flat": false, "boundary": false} |
| choice-cvb | CV-Bench | 65.0%→65.0% | 0.9428→0.8581 | 0.1848→0.0957 | 0 | {"T": 1.534636, "nll": 0.8728698995191673, "nll_identity": 0.9502878726849003, "improvement": 0.081468, "flat": false, "boundary": false} |
| score-countbench | CountBenchQA | 100.0%→100.0% | 0.1136→0.0421 | 0.0992→0.0377 | 0 | {"T": 0.62753, "nll": 0.33572028090183403, "nll_identity": 0.40304715784946266, "improvement": 0.167045, "flat": false, "boundary": false} |
