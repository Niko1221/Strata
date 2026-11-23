# stpar 20261123-161217 — parallel-slots A/B (config: strata-inkling-sq20-farm.json, engine: strata 0.1.66)

Per variant: 1× officiële suite (chat/prefill/agent) met --clients N. Aggregate = alle streams opgeteld (burst-wall-time).
| 4 | 4 | ? | ? | ? | ? | ? | ? |
| 8 | 8 | ? | ? | ? | ? | ? | ? |
| 16 | 16 | ? | ? | ? | ? | ? | ? |
| 32 | 32 | ? | ? | ? | ? | ? | ? |

Vergelijking: aggregate stijgt boven solo-waarde = echte throughput-winst voor agent-farms.
Per-stream daalt <20% bij 2 streams = parallelisen is 'gratis'; daalt >40% = expert-cache thrash.
Losse suites: /home/user/bin/stpar-out/20261123-161217/suite-*/ (report.md per variant).
