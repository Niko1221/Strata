# Claude Code With Local Strata

This optional profile does not replace your current provider automatically.
Install Claude Code separately, start Strata, and run `CLAUDE-STRATA.bat` from
the project you want to work on. The launcher passes this folder's profile
using `--settings`, keeps the current directory, and forwards your arguments.
It expects `%USERPROFILE%\.local\bin\claude.exe`; other installations can run
`claude --settings <absolute-path-to-settings.local.json>` instead.

The API speaks Anthropic's Messages protocol. The Claude model name in the
profile is a compatibility alias: the model doing inference is the loaded
Strata model, not Anthropic Claude.
History shows the real serving model; Details retains the client's requested
alias as `requested_model`. Claude Code's own UI can still display that alias.
The profile's token is a non-secret placeholder for loopback servers without
`--api-key`. If yours requires a key, put it in your private copy and never
commit it. Nothing here changes the public gateway or dashboard login. Client
cost estimates based on the alias are not evidence of an Anthropic bill.

All three Claude model aliases and the subagent default point at the same local
server. Requests identify themselves as Claude Code in Strata's History tab.
The launcher leaves global settings, plugins, permissions and sign-in files
alone. Existing plugins may use other services; this profile does not audit them.

The profile requests 132,000 total output tokens and sets no separate thinking
budget. Thinking and answer text share that finite budget, not 132,000 each.
Claude Code can clamp that request to its own model-dependent output cap;
Strata's server-side limit and fit-to-context behavior remain unchanged.
Claude Code 2.1.158 sent a maximum of 128,000 tokens in the verified requests.
The four-hour client timeout accommodates slow local generation, not a change
to model sampling or output quality.

To make this the default, first back up your private
`%USERPROFILE%\.claude\settings.json`, then merge the profile's `model` and `env`
entries without deleting unrelated settings. Project settings or environment
variables can override user settings. Restore your backup to undo the change.
Backups can contain credentials and must never be committed or shared.

On another device, use the public server's root URL rather than `/v1` for
`ANTHROPIC_BASE_URL`, and the private gateway key as `ANTHROPIC_AUTH_TOKEN`.
Keep the hosting PC and Strata running. This profile uses the hosting PC's
loopback address by default. See [the complete setup guide](../docs/LOCAL_FEATURES.md).

References: [gateway settings](https://code.claude.com/docs/en/llm-gateway),
[environment variables](https://code.claude.com/docs/en/env-vars),
[configuration precedence](https://code.claude.com/docs/en/settings).
