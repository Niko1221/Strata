# third_party/wex: wex4.py, frozen

The reference reader and writer of WEX4, the lossless expert container (`docs/WEX4.md`). `terminal/tools/iq_pack.py --experts-wex4` writes
a container with it, and `terminal/tools/wex4_check.py` checks every record of one with it. Python standard library only. The engine does
not use this file: it reads WEX4 with its own small reader (`engine/src/core/wex4_experts.cpp`, RAW records only).

pin: fd8cd359290aebd1192c2aa26594017f7477b4cc

Source: https://github.com/KairaCode/Wex-WRN2 at that commit: `wex4/wex4.py`, `LICENSE` and `NOTICE`. Copyright 2026 Ludovico Zanichelli.
Licence: Apache License 2.0; the full text is in `LICENSE`, and `NOTICE` is the upstream NOTICE file, kept as Apache-2.0 section 4(d)
asks. Apache-2.0 code may be part of an MIT project when its licence and notice travel with it, as here. There are no Strata edits: the
three files are byte-identical to the upstream files (LF line ends).

`wex4.py` also holds the WNR3 and WBP1 codecs of the format. Strata calls only its RAW path (`Tensor`, `Record`, `write`, `Reader`).

`python tools/check_vendored.py` verifies the list below.

<!-- vendored-files:begin -->
```
# kind sha256 upstream-sha256 path (relative to third_party/wex)
upstream cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 LICENSE
upstream 01af9e4bebaaa316f946b045a22af7bb95ba1d25582fd8186ed9ec77afa98efa 01af9e4bebaaa316f946b045a22af7bb95ba1d25582fd8186ed9ec77afa98efa NOTICE
upstream 683d9652d3b5058fbf406439259a96fd48003d70b6c2546f9eb266118e06fe90 683d9652d3b5058fbf406439259a96fd48003d70b6c2546f9eb266118e06fe90 wex4.py
```
<!-- vendored-files:end -->
