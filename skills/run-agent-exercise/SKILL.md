---
name: run-agent-exercise
description: "Design and score agent exercises for continuation, helper coordination, or evidence-driven completion."
---

# Run Agent Exercise

Pick a realistic synthetic task with explicit ground truth and a verifiable artifact. Pin the model, tool surface, input and baseline. Choose one disturbance such as a live correction, restart, changed file, helper stop or context limit. Record what the agent observed and when, actual tool calls and final state.

Score task correctness, response to the disturbance, evidence quality, and unnecessary repeated work. Keep training examples separate from evaluation cases. Preserve failed runs and avoid reporting success based only on the agent's final message.
