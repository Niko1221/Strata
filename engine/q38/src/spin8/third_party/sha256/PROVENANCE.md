# SHA-256 provenance

Copied from the local pinned Prism tree at commit
`73bfe6cc485554822f80c473a3c0c4f25e80f41b`, `vendor/hash/sha256` and
`vendor/hash/rotate-bits`. No network fetch or package install was performed.
The SHA implementation is Igor Pavlov's 2010-06-11 public-domain implementation,
based on Wei Dai's public-domain code. Rotate-bits retains its MIT LICENSE.md.
Original licenses are included unchanged.

Original SHA-256 content hashes:

| Original file | SHA-256 |
| --- | --- |
| sha256/sha256.c | ea70a42189b4c798657cb2fb334ad6afb456245b13d3d36d0ff9fd4f9d9e8f62 |
| sha256/sha256.h | 697b3138aa7590d4c86f332b80e241990674c8344d47171a259b16353309b056 |
| sha256/LICENSE | 24836bd0f33996623060eabeb765fb0b091bcbca47e6e6e9493385df8ce9e30a |
| rotate-bits/rotate-bits.h | 633d6f97ababf28a562b820fc49c6a8788ea56349568cbe908c051ac032f0685 |
| rotate-bits/LICENSE.md | 6d1a68e8ae5085d15a5c4f8dc1363b7bf256756b3a49cbaf29fe7f7a49eae4e3 |

Local modifications: every `sha256_` identifier in sha256.c/.h has the prefix
`spin8_sidecar_sha256_`, including the context type and static functions.
The SHA header guard is renamed to `SPIN8_SIDECAR_SHA256_H`. This prevents
external SHA symbol collisions when the adapter and Prism are linked together.
Line endings are normalized by the repository. Rotate-bits and license contents
are retained apart from that normalization. The SHA digest algorithm has no
local changes.

The focused test checks published empty, abc and million-a vectors; Python
hashlib independently checks multi-block sidecar integrity and full content
identity through the C++ executable. These checks establish tested output
agreement, not a cryptographic certification or publisher authentication.
