# real calibration suite (QFN live)

| qtype | dataset | acc raw→cal | NLL raw→cal | ECE15 raw→cal | flips | fit |
|---|---|---|---|---|---|---|
| noul-pope | POPE adversarial | 76.7%→78.3% | 0.6604→0.4327 | 0.1867→0.1288 | 7 | {"a": 0.435457, "c": 0.818939, "nll": 0.499821075397024, "nll_identity": 0.8538632274284486, "improvement": 0.414636, "flat": false, "boundary": false} |
| choice-cvb | CV-Bench | 35.0%→35.0% | 2.5186→1.3653 | 0.597→0.3596 | 0 | {"T": 5.027691, "nll": 1.3806052679155107, "nll_identity": 2.429639589353826, "improvement": 0.431765, "flat": false, "boundary": false} |
| score-countbench | CountBenchQA | 90.0%→90.0% | 0.3695→0.4067 | 0.081→0.1564 | 0 | {"T": 1.846229, "nll": 0.7685237361324757, "nll_identity": 1.0738988667971685, "improvement": 0.284361, "flat": false, "boundary": false} |
