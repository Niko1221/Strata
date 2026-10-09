# Video path measurements

What the video path has been measured doing, and on what. Every number names its build and its input, because a
different card, CPU or FFmpeg build can move them. None of these is a pass threshold.

## The runs

| | |
| --- | --- |
| Host | Linux 6.12.107+deb13-amd64, Intel i9-12900K (24 threads), 93 GiB RAM, RTX 3090 + RTX 5080 |
| Code | `feat/video-support` at `b620e923`, rebased on `main` `fb58e0bd` |
| Engine | Strata 0.1.41, CUDA 13.4.92, arch 86;89;120, `strata` SHA256 `aa001d26…b8bdd8e6` |
| Encoder | `strata-vision`, CPU-only build, SHA256 `35d0c017…938249fd`, llama.cpp `3cf03257` |
| Decoder | FFmpeg/ffprobe n8.1.3-14-g330caae0c1, a static build named in `vision.video.ffmpeg`/`ffprobe` |
| Reference | Transformers `770e4c40`, checkpoint `f5d08274`, PyTorch 2.8.0+cpu, Python 3.13.5 |

The encoder is the CPU build in every run below. The two cards belong to the language-model engine that answers the
requests further down.

The tests, the reference trace, the repeat encode and the byte-for-byte re-encode below were run on 2026-10-08 on
the head commit. The encoder-parity comparison and the server requests were taken on 2026-10-07 at `ff16667a`, the
pre-rebase twin of the head commit; `serve/video.py`, `serve/video_source.py`, `serve/video_encoder.py`,
`serve/media.py`, `src/program/media_embeddings.cpp` and `tools/vision/media_export.cpp` are byte-identical between
the two, and the 2026-10-08 re-encode below shows the head encoder producing the same rows.

## Frames are chosen by time, and decode to the pixels they should

Two generated clips, one word card per frame, sampled on the grid 0, 0.5, 1, 1.5, 2, 2.5, 3 and 3.5 s. The
variable-rate clip's frames sit at 0, 0.2, 0.8, 1, 1.68, 2, 2.08, 3 and 3.52 s, so its 0.5, 1.5 and 2.5 s grid points
have to fall back to the nearest frame that exists.

| Clip | source frames chosen | their PTS (s) |
| --- | --- | --- |
| constant rate, 8 frames at 0.5 s | 0, 1, 2, 3, 4, 5, 6, 7 | 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 |
| variable rate, 9 frames | 0, 1, 3, 4, 5, 6, 7, 8 | 0, 0.2, 1, 1.68, 2, 2.08, 3, 3.52 |

Every chosen frame's RGB matched the PNG the clip was built from, byte for byte, on both clips. A second
variable-rate clip, identical except for the card at 2 s (WILLOW replaced by FALCON), checks that changing one
frame changes one temporal group.

## Encoder rows match the reference visual tower

Rows from the CPU encoder against the pinned checkpoint's visual tower - 333 visual tensors read strictly from the
checkpoint, BF16 weights computed in FP32 on CPU - on the same decoded frames.

Smoke clip: 12 source frames, 3.75 s, variable rate, 7 frames chosen, 128 wide by 64 high, four groups of 8 rows of
width 2560.

| group | frame times (s) | rows | mean row cosine | minimum | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 0, 0.5 | 8 | 0.9998463 | 0.9997357 | 0.0009655 |
| 1 | 1, 1.5 | 8 | 0.9986330 | 0.9931709 | 0.0030836 |
| 2 | 2, 3 | 8 | 0.9998567 | 0.9997972 | 0.0010400 |
| 3 | 3.5, 3.5 | 8 | 0.9998924 | 0.9998652 | 0.0008343 |

All 32 rows: mean cosine 0.9995571, minimum 0.9931709, RMSE 0.0017477, mean absolute difference 0.0010282, largest
0.0329685; row RMS 0.053285 native against 0.053340 reference.

The three QA clips, 240 rows each:

| Clip | rows | mean row cosine | minimum | RMSE | relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant rate | 240 | 0.999905609 | 0.993993108 | 0.000439710 | 1.22988% |
| variable rate A | 240 | 0.999924659 | 0.996627284 | 0.000321461 | 0.902571% |
| variable rate B | 240 | 0.999926929 | 0.996627284 | 0.000351521 | 0.985412% |

Computing the reference over the whole clip or over separate two-frame groups gives bitwise identical reference
embeddings for these clips, so the comparison is not sensitive to how the reference is batched.

The head build gives those rows. Re-encoding the three clips with the head-commit CPU encoder - `READY 2560`, the
profile above, `VSET` from the default policy, `ENCV` per clip, one thread, CPU - returned `VOK 240 4` for each and
bundles identical to the ones recorded on 2026-10-07: 240 rows, 2,461,888 bytes, SHA256 `f2ca4b8c…`, `73f60812…` and
`0a5a127b…`. The encoder exited 0 and released its disk quota.

## The same clip twice gives the same bytes

Two encodes of the smoke clip: 32 rows, 81,920 floats, 327,680 bytes, SHA256 `212a067e…d9bde4` both times, RMS
0.05328527880428726 and largest value 0.3364448845386505 both times.

## Positions and prompt tokens follow the reference

`tools/vision/video_reference_trace.py --positions --tokenizer-reference`, run on the head commit: six cases - one
frame, two frames, three frames, five frames, a ten-second clip, and an image and a video in one prompt - all
report `PASS_CPU_TORCH` for the absolute positions and for the position after the prompt, and the same six encode to
the token IDs the checkpoint's own `tokenizer.json` gives (`PASS_RUST_TOKENIZERS`, SHA256 `0997f410…29b9f3`). The
C++ position planner and the Python planner produce the same positions and the same row bindings for these prompts.

## Tests on the 2026-10-08 head

```
ctest --test-dir <build> -R '^media_embeddings_test$'                              1/1 passed
media_embeddings_test tools/vision/fixtures/media-v2.hex                           5011 checks passed
python -m unittest serve.test_media serve.test_media_process serve.test_media_profile \
  serve.test_video serve.test_video_encoder serve.test_video_frontend \
  serve.test_video_policy serve.test_video_reference serve.test_video_server       69 tests, 0 skipped, OK
```

`MEDIA_TEST_EXE` pointed at the C++ test, so the cross-language transport tests ran, and `STRATA_FFMPEG` and
`STRATA_FFPROBE` pointed at the static FFmpeg above, so the real decoder tests ran instead of skipping.

## Frame budget sizing on 2026-10-09

Separate **synthetic CPU-only** runs for the configurable 1,024-frame policy, on the same i9-12900K. The input
repeats the 8-frame 192x320 RGB QA spool; repeating frames measures resource cost, not natural-video accuracy.
The encoder ran with its default 12 CPU threads, `--max-tokens 300`, without `--gpu`.

| selected frames | visual rows | encoder wall | SVE2 bytes | SVE2 MiB |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 240 | 1.207 s | 2,461,888 | 2.35 |
| 64 | 1,920 | 9.603 s | 19,694,744 | 18.78 |
| 128 | 3,840 | 19.126 s | 39,389,464 | 37.56 |
| 1,024 | 30,720 | 153.117 s | 315,117,192 | 300.52 |

The 8/64/128 rows used the earlier CPU encoder (`35d0c017…938249fd`); its peak process RSS for the three runs
was 1,008 MiB. The 1,024 row used the new CPU encoder built from `8c4c8fa7` (`c84b2bb2…59cc464`), with peak
process RSS 1,297 MiB. Both returned `VOK`; a repeat 128-frame encode with the new binary had the same
SHA256 as the old one and took 19.092 s. The 1,024-frame artifact's SHA256 is `ac5b6f07…f964`.

Separately, FFmpeg 8.1 decoded generated 320x192, 512 s `testsrc2` MP4s at 2 and 30 FPS: 1,024 and 15,360
source frames, respectively, both selecting 1,024. Both produced 30,720 visual rows, 188,743,680 retained RGB
bytes and 251,658,240 streamed RGBA bytes. Staging, probing and decoding took 0.659 s and 1.711 s, respectively,
after the source files existed. The encoder timing above and these decoder timings used **different synthetic
inputs**. The configured 600 s deadline is a safety budget, not a latency guarantee on other CPUs, source formats
or resolutions. The actual selected-frame costs are checked
before decode; exceeding any row, wire, RGB, decoder-output, duration or disk budget rejects the request.

At `8c4c8fa7`, the host transport test passed 5,013 checks, CTest passed 1/1, and 73 Python video/media tests
passed with zero skips (`MEDIA_TEST_EXE`, `STRATA_FFMPEG` and `STRATA_FFPROBE` set). The host C++ Qwen-profile
reader also accepted the 1,024-frame artifact: 512 spans, 30,720 rows under the engine's new limits. A separate
CUDA 13.4 engine build linked and reported version 0.1.41, but **was not run on a GPU**. These checks add no
model-answer or natural-video accuracy claim to the earlier live-server results.

## Requests to a running server

The same clips, asked of a running server on loopback: model `strata-0139-ud-iq4-duo` (IQ4_XS, 262,144 context),
`video.available` true with profile `qwen4_exp_16x2x2_2560_v1` before and after. Each answer came back with
finish reason `stop`.

| Request | clip (SHA256 prefix) | asked for | answered | prompt tokens | wall |
| --- | --- | --- | --- | ---: | ---: |
| variable rate, card at 2 s | `506efc35` | WILLOW | WALNUT | 309 | 2.91 s |
| variable rate, counterfactual | `8f6fb59b` | FALCON | FALCON | 309 | 2.88 s |
| constant rate, card at 1.5 s | `177408c8` | PEACH | KIWI | 309 | 2.80 s |
| constant rate, name every card | `177408c8` | eight cards | `plum, grape, melon, apple` | 308 | 1.38 s |

One of the three single-card probes matched. Both constant-rate probes use a clip whose card changes every 0.5 s, so
each two-frame group carries two different cards and its timestamp label is the mean of the pair; the fourth request
lists four of the eight cards, in order.

## Prompt tokens where text meets video

Where the question runs into the clip, the prompt is not reference-equivalent. In the constant-rate probe the
text ends `only.` and the wrapper starts `<0.2 seconds>`: the engine carries IDs `[13, 27]` (`.` then `<`) where the
reference, which expands the wrapper before tokenizing, carries `[15294]` (`.<`). 309 tokens against 308, and the
decoded text is the same either way. The two variable-rate probes put the clip before the question and are unaffected.

## Re-running the host checks

```sh
cmake -S . -B /tmp/strata-media -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_MEDIA_TESTS=ON -DSTRATA_WERROR=ON
cmake --build /tmp/strata-media --target media_embeddings_test -j4
ctest --test-dir /tmp/strata-media -R '^media_embeddings_test$' --output-on-failure
MEDIA_TEST_EXE=/tmp/strata-media/media_embeddings_test \
  STRATA_FFMPEG=<ffmpeg> STRATA_FFPROBE=<ffprobe> \
  python -m unittest serve.test_media serve.test_media_process serve.test_media_profile \
    serve.test_video serve.test_video_encoder serve.test_video_frontend \
    serve.test_video_policy serve.test_video_reference serve.test_video_server

python tools/strata_tokenizer.py --gguf <model.gguf> --out /tmp/strata-tok
python tools/vision/video_reference_trace.py --sources <pinned reference sources> \
  --checkpoint <checkpoint directory> --tokenizer /tmp/strata-tok/tokenizer \
  --positions --tokenizer-reference
```

The trace checks the reference sources and the checkpoint's JSON metadata against pinned hashes before it runs, and
refuses a checkpoint whose `tokenizer.json` is not the pinned one. It never authorizes a GPU: it clears
`CUDA_VISIBLE_DEVICES` before importing PyTorch.

The re-encode starts the resident encoder with `--threads 1` and no `--gpu`, reads `READY 2560`, sends `CAPS`, then
`VSET` with the policy's six budgets, then one `ENCV` per clip:

```text
VSET <max_group_tokens> <max_frames> <max_tokens> <max_rgb_bytes> <max_embedding_bytes> <max_duration_s>
ENCV <frames.svf1> <out.sve2>          ->  "VOK <rows> <groups> <ms>"
```

The frame spool comes from `serve.video_source.decode_video` on the clip file.
