# Upstream Adapters Boundary

This directory houses thin bridge adapters that interface fork-specific extensions (Spin(8) Triality, TurboQuant dialects, Query-v3 sidecars) with upstream Strata's documented CLI, configuration, and owner runtime APIs.

## Boundary Principles
1. **No Fake "Official Plugin SDK"**: Upstream Strata does not have an official plugin SDK. All interactions are strictly confined to documented CLI arguments, run configs, and verified internal bridges pinned to exact upstream commits (U1: `7dd5cf18e68c639a8773e096c5c79f971ea2a9f5`, C1: `c487c875d05075113965bc61d822b27179f70639`).
2. **Clean OFF State**: When extensions are disabled (`spin8_mode="off"`, no sidecars), zero additional arguments or memory allocations are introduced.
3. **Immutability & Snapshot Integrity**: Configurations and sidecar bindings are resolved once into an immutable snapshot (`EffectiveExtension`) prior to process launch.
