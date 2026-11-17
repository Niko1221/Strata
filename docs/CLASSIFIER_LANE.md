# Reserve a lane for a permission classifier

`messages_classifier_alias` reserves the last batch lane for requests to
`POST /v1/messages` whose `model` exactly matches the configured alias. Add
these keys to the [model run config](DETAILS.md):

```json
{
  "parallel": 3,
  "aliases": ["local-classifier"],
  "messages_classifier_alias": "local-classifier"
}
```

Choose an alias distinct from the primary model ID. A gateway must preserve
the `model` value. The same loaded weights serve all requests. Chat
Completions and Responses requests use ordinary scheduling.

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
reports fewer than two batch slots, all requests use the ordinary queue. The
reservation takes effect again when the engine starts with two or more slots.
Without the option, requests keep the existing slot selection and scheduling
behavior.

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
