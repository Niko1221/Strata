# Reserve a lane for a permission classifier

The optional `classifier_model` run setting names a request model ID that uses
the last batch lane. For example, with `parallel: 4` and
`classifier_model: "local-classifier"`, ordinary requests use three lanes and
Anthropic Messages requests naming `local-classifier` use the fourth. They use the same loaded
weights. A gateway must preserve this request model ID rather than rewrite it.
Choose an alias distinct from the primary model ID.

With three lanes, two ordinary conversations can keep their prefixes while the
third lane serves classifier requests. Two lanes support one ordinary request
and one classifier. These are scheduling capacities: each extra lane also needs
engine state. Validate the memory cost on the actual card before enabling it.

A pending classifier asks an ordinary prompt read to yield at the next engine
boundary, even when its own prompt is longer. New ordinary prompt reads wait
until the classifier completes or cancels. Already decoding requests continue
alongside it. The reserved lane keeps its conversation prefix for later checks.

This is scheduling, not a classifier implementation. The API does not generate
permission verdicts itself, alter classifier replies, shorten their policies,
or approve failed checks. The client still evaluates the actual model response.
Multiple classifier requests share one lane and can still exceed their client
deadlines. An already running engine operation is not interrupted mid-kernel.
Continuous classifier traffic can delay ordinary prompt reads.

The option requires one batched Strata engine with at least two supported lanes.
It cannot be combined with replica engines or coalesced prefills. If the engine
reports fewer than two batch slots, Messages requests to the classifier alias
receive HTTP 503 with `classifier_slot_unavailable`. Ordinary requests use the
available engine capacity. The reservation takes effect again when the engine
starts with two or more slots. Without the option, requests keep the existing
slot selection and scheduling behavior.

CPU regression checks from the repository root:

```sh
python3 -m unittest serve.test_classifier_lane serve.test_parallel serve.test_parallel_yield serve.test_server -q
```

These use the real HTTP service with a simulated engine protocol. They check
reservation, prompt yielding, cancellation and the unchanged default path.
The classifier tests cover two, three and four lanes, two ordinary requests
beside a classifier, and interleaved cached-prefix selection for two projects.
They do not establish model classification quality or GPU latency; those need
separate measurements with the actual client and engine.
