# nlohmann/json

Version: 3.12.0. License: MIT, retained in `LICENSE.MIT` and the header.

`json.hpp` was copied from an existing llama.cpp vendor directory.
The new weight source has no runtime dependency on that checkout.
The copied file's hash was also checked against the upstream v3.12.0 raw header
on 2026-10-09; the hashes match.

SHA-256 of the vendored header:
`aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`.

Upstream release source:
https://github.com/nlohmann/json/blob/v3.12.0/single_include/nlohmann/json.hpp

The wrapper in `src/weights/json_checked.hpp` uses the public SAX API to reject
duplicate keys and excessive nesting, then the ordinary DOM parser. It does not
patch this vendor header. The DOM callback parser in this release scans existing
parent members after every child object; using it for duplicate detection made
the large safetensors headers needlessly expensive.
