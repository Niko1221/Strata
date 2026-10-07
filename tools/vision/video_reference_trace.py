#!/usr/bin/env python3
"""Trace the pinned processor without loading checkpoint weights or enabling video.

Hash-verified reference method bodies are executed with small metadata/tensor
fixtures, not imported as full Transformers models. --positions additionally
requires real PyTorch and runs the reference position methods on CPU only.
Without it, position/reference-embedding gates remain explicitly unrun.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve.media import (MediaBundle, MediaKind, MediaLimits, VisualSpan, build_positions)  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

REFERENCE_REVISION = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
CHECKPOINT_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
SOURCE_HASHES = {
    "processing_qwen3_vl.py": "02d50224d9dc9ce38690cfa6a0106be897f619444ebe3ff7ba38cbcdda9e3ef2",
    "video_processing_qwen3_vl.py": "62a47c40e0d69a28915fde8fa6a492adc175051e167f3ceb30bf534766a09107",
    "modeling_qwen4_exp.py": "797a18fd6dd76c574d237a5643759acdeb1c4d0f1c2508693f8fdabce0a19057",
}
METADATA_HASHES = {
    "config.json": "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}


def verified_files(directory, hashes):
    result = {}
    for name, expected in hashes.items():
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{name}: pinned SHA256 mismatch")
        result[name] = raw.decode("utf-8")
    return result


def reference_class(source, name):
    return next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == name)


def bind_methods(obj, node, names, globals_):
    """Execute only verified method bodies; annotations/decorators need no model imports."""
    methods = []
    for name in names:
        method = copy.deepcopy(next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name))
        method.decorator_list = []
        method.returns = None
        arguments = method.args.posonlyargs + method.args.args + method.args.kwonlyargs
        arguments += [a for a in (method.args.vararg, method.args.kwarg) if a is not None]
        for arg in arguments:
            arg.annotation = None
        methods.append(method)
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = dict(globals_)
    exec(compile(module, "<hash-verified reference methods>", "exec"), namespace)
    for name in names:
        setattr(obj, name, MethodType(namespace[name], obj))


def literal_attribute(node, name):
    for statement in node.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError(f"missing reference attribute {name}")


def tokenizer(directory, config):
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in ("vocab.json", "merges.txt", "token_type.json")}
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    for symbol, key in (("<|image_pad|>", "image_token_id"), ("<|video_pad|>", "video_token_id"),
                        ("<|vision_start|>", "vision_start_token_id"), ("<|vision_end|>", "vision_end_token_id")):
        if vocab.get(symbol) != config[key]:
            raise ValueError(f"{symbol}: checkpoint/tokenizer mismatch")
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = ST.Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((directory / "token_type.json").read_text(encoding="utf-8")))
    return tok, hashes, len(vocab)


def case_trace(name, total_frames, fps, mixed, processor, video_processor, tok, config, vocab_size, torch):
    metadata = SimpleNamespace(total_num_frames=total_frames, fps=fps)
    indices = video_processor.sample_frames(metadata).tolist()
    metadata.frames_indices = list(indices)  # Reference timestamp method pads this list in place.
    merge = video_processor.merge_size
    grid_h, grid_w = 4, 6
    grid_t = (len(indices) + video_processor.temporal_patch_size - 1) // video_processor.temporal_patch_size
    video_inputs = {"video_grid_thw": np.array([[grid_t, grid_h, grid_w]]), "video_metadata": [metadata]}
    text = processor.replace_video_token(video_inputs, 0)
    timestamps = processor._calculate_timestamps(list(indices), fps, video_processor.temporal_patch_size)
    if mixed:
        image_inputs = {"image_grid_thw": np.array([[1, grid_h, grid_w]])}
        image = processor.replace_image_token(image_inputs, 0)
        text = "First image: <|vision_start|>" + image + "<|vision_end|> Then clip: " + text
    else:
        text = "Describe this clip: " + text
    text += " End."
    tokens = tuple(tok.encode(text, parse_special=True))
    kinds = {config["image_token_id"]: MediaKind.IMAGE, config["video_token_id"]: MediaKind.VIDEO}
    nx, ny = grid_w // merge, grid_h // merge
    rows = nx * ny
    spans, descriptors = [], []
    cell = 0
    while cell < len(tokens):
        if tokens[cell] not in kinds:
            cell += 1
            continue
        pad, start = tokens[cell], cell
        while cell < len(tokens) and tokens[cell] == pad:
            cell += 1
        if cell - start != rows:
            raise ValueError(f"{name}: reference wrapper pad count mismatch")
        kind = kinds[pad]
        positions = tuple((0, i // nx, i % nx) for i in range(rows))
        spans.append(VisualSpan(start, pad, kind, max(nx, ny), positions, bytes(rows * 4),
                                nx if kind == MediaKind.IMAGE else 0, ny if kind == MediaKind.IMAGE else 0))
        descriptors.append({"start": start, "rows": rows, "kind": kind.name.lower(), "pad_id": pad})
    if sum(s.kind == MediaKind.VIDEO for s in spans) != grid_t or sum(s.kind == MediaKind.IMAGE for s in spans) != int(mixed):
        raise ValueError(f"{name}: reference span count mismatch")
    trace = {"name": name, "source_frames": total_frames, "source_fps": fps, "sampled_indices": indices,
             "padded_indices": metadata.frames_indices, "merged_timestamps": timestamps, "expanded_text": text,
             "token_ids": tokens, "visual_spans": descriptors, "video_grid_thw": [grid_t, grid_h, grid_w],
             "position_reference": "NOT_RUN"}
    if torch is not None:
        model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=merge)))
        bind_methods(model, reference_class(processor.model_source, "Qwen4ExpModel"),
                     ("get_vision_position_ids", "get_rope_index"), {"torch": torch, "itertools": itertools})
        types = [0] * len(tokens)
        for span in spans:
            types[span.start:span.start + rows] = [int(span.kind)] * rows
        positions, delta = model.get_rope_index(
            torch.tensor([tokens], dtype=torch.int64, device="cpu"),
            torch.tensor([types], dtype=torch.int64, device="cpu"),
            image_grid_thw=torch.tensor([[1, grid_h, grid_w]], device="cpu") if mixed else None,
            video_grid_thw=torch.tensor([[grid_t, grid_h, grid_w]], device="cpu"))
        expected = tuple(zip(*positions[:, 0].tolist()))
        # Width-1 zero rows are host probes, never checkpoint/projector embeddings.
        bundle = MediaBundle(1, tokens, tuple(spans))
        plan = build_positions(bundle, len(tokens) + 2,
                               MediaLimits(expected_width=1, vocab_size=vocab_size, allowed_pad_ids=tuple(kinds)))
        next_position = len(tokens) + int(delta[0, 0].item())
        if expected != plan.positions[:len(tokens)] or plan.positions[len(tokens)] != (next_position,) * 3:
            raise ValueError(f"{name}: host/reference position mismatch")
        trace.update(position_reference="PASS_CPU_TORCH", positions=expected, next_position=next_position)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="existing pinned JSON metadata, not weights")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--positions", action="store_true", help="require real CPU PyTorch; failure is not a skip")
    args = parser.parse_args()
    try:
        sources = verified_files(args.sources, SOURCE_HASHES)
        metadata = verified_files(args.checkpoint, METADATA_HASHES)
        config = json.loads(metadata["config.json"])
        image_config = json.loads(metadata["preprocessor_config.json"])
        video_config = json.loads(metadata["video_preprocessor_config.json"])
        if config["model_type"] != "qwen4_exp" or video_config["processor_class"] != "Qwen3VLProcessor":
            raise ValueError("unsupported pinned checkpoint processor")
        tok, tok_hashes, vocab_size = tokenizer(args.tokenizer, config)
        video_node = reference_class(sources["video_processing_qwen3_vl.py"], "Qwen3VLVideoProcessor")
        video = SimpleNamespace(**{name: literal_attribute(video_node, name) for name in ("fps", "min_frames", "max_frames")},
                                merge_size=video_config["merge_size"], temporal_patch_size=video_config["temporal_patch_size"])
        globals_ = {"np": np, "logger": logging.getLogger("video_reference")}
        bind_methods(video, video_node, ("sample_frames",), globals_)
        processor = SimpleNamespace(video_processor=video, image_processor=SimpleNamespace(merge_size=image_config["merge_size"]),
                                    image_token="<|image_pad|>", video_token="<|video_pad|>",
                                    vision_start_token="<|vision_start|>", vision_end_token="<|vision_end|>",
                                    model_source=sources["modeling_qwen4_exp.py"])
        bind_methods(processor, reference_class(sources["processing_qwen3_vl.py"], "Qwen3VLProcessor"),
                     ("replace_image_token", "replace_video_token", "_calculate_timestamps"), globals_)
        torch = None
        if args.positions:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This tool never authorizes a GPU run.
            import torch
        cases = [("one_frame", 1, 30.0, False), ("two_frames", 2, 30.0, False),
                 ("odd_three_frames", 3, 2.0, False), ("odd_five_frames", 5, 2.0, False),
                 ("ten_second_clip", 300, 30.0, False), ("mixed_image_video", 5, 2.0, True)]
        traces = [case_trace(*case, processor, video, tok, config, vocab_size, torch) for case in cases]
        result = {"schema": 1, "reference_revision": REFERENCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
                  "source_sha256": SOURCE_HASHES, "metadata_sha256": METADATA_HASHES, "tokenizer_sha256": tok_hashes,
                  "sampling_defaults": {"fps": video.fps, "min_frames": video.min_frames, "max_frames": video.max_frames},
                  "embedding_reference": "NOT_RUN", "decoder_mtmd_comparison": "NOT_RUN", "cases": traces}
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ValueError, ImportError, StopIteration, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
