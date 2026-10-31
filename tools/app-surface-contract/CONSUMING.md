# Consuming The Contract

These lessons describe how a companion should consume the optional snapshot and
unmodified asset export. They come from static inspection of App 0.2.1 source
and installed App 0.2.2 metadata/source-identity acceptance, not live API integration
validation. The helper implements no production bridge.
Route descriptors identify owners; they grant no permission to call an API.

## Model Identity And Context

Keep these states separate:

| State | Meaning |
| --- | --- |
| Prepared native download | Files or a download recommendation; no proof of a loaded model |
| Saved request/settings | Requested Context, Idle or selection; no proof it took effect |
| Observed native instance | The Host/Control's reported loaded instance and effective settings |
| External API selection | A provider connection and model ID; no native instance implied |

The catalog supplies allowed Context choices and recommendations. `NATIVE_FIELDS`
supplies Idle bounds. The helper exports only that official metadata. Neither the
initial native download Context nor a saved Context request proves effective
allocation. A companion should show the actual loaded model and effective Context
from the owning runtime when available, and leave them unknown otherwise. Do not
turn a catalog maximum into a claim about available memory or a client's context.

Preserve physical IDs: native instance ID, model/variant ID, API model ID and GPU
IDs have different roles. Keep an external provider's connection ID plus model ID
as a pair. Friendly labels are display text, not identifiers. The Host's Chat path
uses `nativeModelId` for a native instance or `remoteModel` for a provider choice;
the descriptors alone do not validate these payloads.

## Monitor And Lifecycle

Native Host telemetry and a shared external API's request log have different
owners. App 0.2.1's `scopedMonitor` in
[`runtime-models.ts`](../../apps/host/src/lib/runtime-models.ts) replaces
`live.requests` with the selected serving log in real mode. An empty serving log
can therefore discard externally projected requests. Check the actual page's data
flow before claiming requests are visible; do not fabricate native instances to
make an external API appear in native Monitor. Unknown hardware and throughput
stay unknown, and synthetic fixture values are not measurements.

Judge load, unload, default selection and Idle behavior from subsequent observed
owner state. An accepted operation or saved setting alone is not lifecycle
success. Leave job progress, cancellation, instance conflicts and Idle unloading
with official App/Control. The helper starts no timers or operations.

## Native Features And Owners

| Surface | Consumer responsibility |
| --- | --- |
| Chat And Approvals | Preserve native streams, session/call IDs and approve/deny decisions; no synthetic approval success |
| Tools And Search | Use the native tool configuration and its permission flow; do not infer that Search is available merely because Chat works |
| Research | The reviewed `/research` message goes through native Chat/Harness; it is not a separate invented research API |
| Skills And Tasks | Preserve official records, enablement, schedules and approvals; an exported route list does not imply a task ran |
| Workspaces And Files | Preserve registered workspace roots, file boundaries and attachment/session ownership |
| Image Generation | Image-model configuration and generation availability have their own owner; input-picture understanding is a separate capability |
| Effort And Updates | Keep official Composer controls and official App/Control checks, install and restart behavior; no quick-install or replacement Effort control |

Input-picture support is not proof that image generation exists, and an image
model's presence is not proof of picture-reading support. Keep both conditional
on their official capability/state metadata. This contract does not test either.

## Authentication, History And Revalidation

The exported frontend is unchanged. A companion remains responsible for its own
owner authentication, authorization and CSRF boundary before forwarding any native
controls. Preserve the Host's checks and provider credential ownership. Do not
interpret this local metadata snapshot as a public API access policy.

Keep existing account, chat history, research history and storage ownership intact.
Do not replace or clear history during an asset refresh. Any private integration
stays outside the official React root and needs its own reviewed data boundaries.
The export helper implements none of those account or history integrations.

Require a reviewed surface schema/version and a fresh App/Control fingerprint;
verify candidate bytes with the saved manifest pin. Changed or missing anchors,
metadata, API envelopes or UI layouts need validation. Re-export after official
updates, and validate the affected payloads and rendered pages separately. String
anchor checks and passing fixture tests cannot establish complete API semantics,
live model behavior, authentication correctness or future layout compatibility.
