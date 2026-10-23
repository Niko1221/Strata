# Bench prompts (verbatim)

Three Korean task prompts; a unique `[문서 N]` / `[로그 N]` tag (and two varying operands in the short task) is
generated per request from a seeded RNG (`random.seed(1234)`), so consecutive requests never share a prefix
(no KV-prefix reuse inside a run). The repeated context sentence `SENT` (~115 Korean chars) is pasted 12x
(medium) or 150x (long) after the tag. The medium/long prompts begin with a duplicated-word typo
(`아아 아래` / `아아 위`); it was in every measured request and is kept verbatim here — it cost nothing to fix
and honesty required not silently editing a prompt that produced numbers.

## short — arithmetic + parity explanation (max_tokens 200, ~60 prompt tokens)
```
{r%97}과 {r%89}을 더한 값과 그 곱을 계산하고, 각 결과가 짝수인지 홀수인지 설명해줘.
```

## medium — 3-line summary of a pasted document (max_tokens 350, ~1,500 prompt chars ≈ 900 tokens)
```
아아 아래 문서의 핵심 주장을 세 줄로 요약해줘.

[문서 N] + SENT ×12
```

## long — power-rail log extraction (max_tokens 250, ~15,600 prompt chars ≈ 8,700 tokens)
```
아아 위 로그에서 전원/접지 관련 경고만 골라 번호 목록으로 정리해줘.

[로그 N] + SENT ×150
```

`SENT` verbatim:

```
서버 랙의 전원 분배 회로를 점검할 때는 부하 중심과 상쇄 선로 저항을 함께 고려해야 하며, 특히 급전 경로가 서로 다른 계통과 만나면 접지 전위차가 흘러 센서 신호에 오프셋이 생긴다.
```

(Translation of the three tasks: short = "compute the sum and the product of X and Y, explain whether each
result is even or odd"; medium = "summarize the main claim of the document below in three lines"; long =
"from the log above, pick out only power/ground-related warnings as a numbered list". Temperature 0.3,
thinking ON, both reasoning and content deltas counted.)
