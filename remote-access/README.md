# Strata Remote API

After enabling Funnel, use your device's public OpenAI-compatible base URL:

https://<your-device>.<your-tailnet>.ts.net/v1

An API-key protected gateway forwards requests to the existing Strata server
at `127.0.0.1:8080`. Tailscale Funnel publishes the gateway over HTTPS. Editors
on other devices do not need Tailscale installed.

## Editor settings

Open `%LOCALAPPDATA%\StrataRemote\editor-settings.txt` for the base URL, API key,
and model ID. Choose an OpenAI-compatible provider in your editor. Editors vary
in their custom-provider support; a public URL does not add unsupported models
to editors that restrict their provider choices.

The gateway accepts Bearer authorization and Anthropic's `x-api-key` header.
It forwards model listing, Chat Completions, Responses, Messages, and token
counting. Strata's web UI, history, and model management routes are not exposed.
Streaming responses pass through immediately. Prompts are limited to 32 MiB.
The gateway does not log prompts or API keys. Credentials are outside the repo
in a directory restricted to the current Windows user and SYSTEM.

## Start and stop

Add `"remote_access": true` to your model's `strata-<model>.json` to opt in to
automatic startup. Install Node.js and Tailscale, sign in to Tailscale and enable
Funnel for your tailnet first. Public access is off by default; simply cloning
these helpers does not publish an endpoint or hold the PC awake.

Run your normal `run-<model>.bat` launcher or `START-HERE.bat`. After the
model server is ready on port 8080, it automatically starts the authenticated
gateway and restores the public Funnel endpoint. The helpers are kept here
inside the Strata installation. Keep this folder with the app.

The startup hook applies to opted-in Windows servers using the Strata engine on
port 8080. Set `"remote_access": false` and restart to disable the hook. Remote startup
failures are reported without stopping the local model; details are appended
to `%LOCALAPPDATA%\StrataRemote\autostart.log`.

The hook now keeps a background watchdog running for the life of Strata.
Every 30 seconds it checks the authenticated gateway and Tailscale Funnel
configuration, restarting a failed owned gateway or restoring a missing
tunnel. Every minute it checks public ingress using external DNS and normal
HTTPS certificate validation, not this PC's private Tailscale address. Two
consecutive public failures cause the owned Funnel endpoint to be republished.
An unavailable DNS resolver does not trigger a tunnel reset. Checks use model
metadata only; they do not run inference or keep the GPU busy.

While remote access is enabled, an app-scoped Windows sleep guard also holds
the PC awake between requests. The display can still turn off. This does not
change the Windows power plan or block a manually requested sleep or shutdown.
The guard is released when the monitor ends or remote access is paused.
Only one watchdog is allowed per Windows user/session. Its status changes are
recorded in `%LOCALAPPDATA%\StrataRemote\watchdog.log` without prompts or keys.

Startup waits up to two minutes for Tailscale to connect and retries transient
Funnel failures. A temporary `NoState` during sign-in or service startup no
longer makes it give up immediately. Missing Tailscale sign-in or device
approval is reported separately from connection timeouts. The server allows
three minutes for this background startup, without blocking the local model.

Before reusing a gateway, startup verifies that it accepts the saved API key.
An owned gateway with stale credentials is restarted with the existing key;
an unrelated service on the gateway port is never stopped. The public URL and
editor key stay unchanged.

Stop-Remote.ps1 writes a protected `disabled` marker so the watchdog respects
an intentional stop. Start-Remote.ps1 or restarting Strata resumes remote
access. Internal recovery attempts never clear that marker.

The following commands manage remote access independently of the model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote-access\Start-Remote.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\remote-access\Stop-Remote.ps1
```

Add `-InstallStartup` to Start-Remote.ps1 to configure startup at Windows
sign-in. Add `-RemoveStartup` to Stop-Remote.ps1 to remove that startup entry.
Funnel's background setting survives restarts. Startup does not start the
model: Strata must be running separately.

The PC must remain powered on, awake, and connected to the internet. Tailscale
must remain signed in. Its device key can expire; check the Tailscale admin
console if connectivity stops. Renaming the device or tailnet changes the URL.
Funnel availability and bandwidth depend on Tailscale.

The watchdog repairs local gateway and tunnel failures; it cannot keep requests
working during a real internet outage, forced PC sleep, power loss, expired
Tailscale credentials, or a loss of connectivity on the other device. A client
whose connection dropped still needs to retry its request.

## Key rotation

Stop remote access. Delete only `%LOCALAPPDATA%\StrataRemote\gateway.json`, then
start remote access again to generate a new key. Update each editor with the
new key from editor-settings.txt. If Strata later gets its own local API key,
set `upstreamApiKey` in gateway.json to that key before starting the gateway.

## Troubleshooting

Public DNS registration can take time after first enabling Funnel. Local
Tailscale DNS access does not prove public access. Check the hostname using an
external DNS resolver and test against the public ingress IPs.

If Funnel configuration is present but public access fails after registration,
Windows service restart is a reported workaround. Run Restart-Tunnel.ps1 from
an administrator PowerShell. It briefly interrupts Tailscale connections and
retains the Funnel configuration. The local Strata server is unaffected.

## Verification

Verified model listing, missing-key rejection, and real streamed completions
through both public ingress IPs on 2026-10-13, with normal TLS certificate
validation. Chat Completions, Responses, Anthropic Messages, and token counting
also passed through the local authenticated gateway. Malformed URLs return 400,
and oversized prompts return 413. JavaScript lint and PowerShell syntax checks
passed. Windows sign-in startup is configured; a full PC reboot was not tested.

Automatic remote startup was verified from a fresh Strata server process and
by recovering a deliberately stopped gateway. Three focused startup checks
passed, including normal startup without helpers and a failed helper that does
not stop the local model. The startup change introduces no new Python lint
findings; the existing server file retains 41 unrelated findings.

On 2026-10-14, six focused checks passed for delayed Tailscale readiness,
transient Funnel errors, missing sign-in, readiness timeout, wrong gateway
credentials, and valid saved credentials. Repeated startup reused a healthy
gateway. After stopping the gateway and disabling Funnel, the actual Strata
startup hook restored both. Authenticated model listing and missing-key
rejection passed through both public ingress IPs; a real streamed completion
returned `READY`. PowerShell Script Analyzer 1.25.0 reported zero findings,
JavaScript lint passed, and the Python timeout change added no findings. A full
model-process restart was not performed in this verification.

Continuous recovery was verified on 2026-10-14. A deliberately terminated
gateway was replaced automatically in 26.6 seconds; a removed Funnel setting
was restored in 4.1 seconds, with the model process unchanged. An intentional
stop stayed stopped for 35 seconds and then resumed through Start-Remote.ps1.
After a 180-second interval without editor inference, both public ingress
addresses returned authenticated model lists and rejected missing keys, and a
real streamed request returned `READY`. Twelve focused watchdog checks and the
six existing sleep-guard checks passed. Native singleton and parent-process
handle checks also passed. The final watchdog was activated for the already
running Strata process; startup-hook delegation was checked separately. A full
app restart/reboot and the other device's editor were not tested in this run.
Python watchdog and PowerShell lint passed; the server change introduced no
new findings beyond its 41 pre-existing findings.

## Architecture

```text
Editor -- HTTPS + API key --> Tailscale Funnel
       --> 127.0.0.1:8787 gateway --> 127.0.0.1:8080 Strata
```

References:
https://tailscale.com/docs/features/tailscale-funnel
https://github.com/tailscale/tailscale/issues/19508

Lint the JavaScript helpers:

```powershell
npx --yes eslint@latest --no-config-lookup --global process --global console --global Buffer --global URL --rule 'no-unused-vars:error' --rule 'no-undef:error' --rule 'no-unreachable:error' .\remote-access\gateway.mjs
```
