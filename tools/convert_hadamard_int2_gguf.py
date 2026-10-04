#!/usr/bin/env python3
"""Convert routed expert matrices in a GGUF model to Strata Hadamard-INT2."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile, GGUF_MAGIC, BLOCK_GEOMETRY  # noqa: E402
from hadamard_int2 import BLOCK_BYTES, BLOCK_SIZE, TYPE_ID, quantize_rows  # noqa: E402

DEFAULT_ALIGNMENT = 32
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d+)-of-(?P<count>\d+)\.gguf$")
TARGET_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<role>gate|up|down)_exps\.weight$")
META_PREFIX = "strata.had2."
META_VALUES = {
    "version": ("u32", 1),
    "block_size": ("u32", BLOCK_SIZE),
    "bits": ("u32", 2),
    "codebook": ("array:f32", [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]),
    "rotation": ("string", "normalized_sylvester_fwht_splitmix64_input_sign"),
    "scale": ("string", "nonnegative_fp16_per_128_values"),
    "packing": ("string", "four_2bit_codes_per_byte_lsb_first"),
    "tensor_scope": ("string", "blk.*.ffn_{gate,up,down}_exps.weight"),
}

META_IDS = {
    "u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
    "f32": 6, "bool": 7, "string": 8, "array": 9, "u64": 10,
    "i64": 11, "f64": 12,
}
SCALAR_FORMATS = {
    "u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I",
    "i32": "<i", "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d",
}


@dataclass
class TensorPlan:
    info: object
    source_size: int
    output_size: int
    convert: bool
    offset: int = 0


def _align(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _scalar(kind: str, value) -> bytes:
    if kind == "string":
        return _gguf_string(value)
    return struct.pack(SCALAR_FORMATS[kind], value)


def _metadata_record(key: str, type_name: str, value) -> bytes:
    out = bytearray(_gguf_string(key))
    if type_name.startswith("array:"):
        element_type = type_name.split(":", 1)[1]
        out += struct.pack("<IIQ", META_IDS["array"], META_IDS[element_type], len(value))
        for item in value:
            out += _scalar(element_type, item)
    else:
        out += struct.pack("<I", META_IDS[type_name])
        out += _scalar(type_name, value)
    return bytes(out)


def _collect_shards(input_path: pathlib.Path) -> tuple[list[pathlib.Path], bool, str]:
    match = SHARD_RE.match(input_path.name)
    if not match:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        return [input_path], False, input_path.stem
    count = int(match.group("count"))
    stem = match.group("stem")
    width = len(match.group("index"))
    shards = [input_path.with_name(f"{stem}-{i:0{width}d}-of-{count:0{len(match.group('count'))}d}.gguf")
              for i in range(1, count + 1)]
    missing = [p.name for p in shards if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"split GGUF is missing shard(s): {', '.join(missing)}")
    return shards, True, stem


def _source_tensor_size(gguf: GGUFFile, tensor, file_size: int, next_offset: int | None) -> int:
    geom = BLOCK_GEOMETRY.get(tensor.type_name)
    if geom is not None:
        block_elements, block_bytes = geom
        if tensor.elements % block_elements:
            raise ValueError(f"{tensor.name}: element count is not divisible by the {tensor.type_name} block size")
        size = tensor.elements // block_elements * block_bytes
    elif next_offset is not None:
        size = next_offset - tensor.offset
    else:
        size = file_size - gguf.data_start - tensor.offset
    if size < 0 or gguf.data_start + tensor.offset + size > file_size:
        raise ValueError(f"{tensor.name}: tensor data extends past the end of {gguf.path.name}")
    if next_offset is not None and tensor.offset + size > next_offset:
        raise ValueError(f"{tensor.name}: tensor data overlaps the next tensor")
    return size


def _plan_file(path: pathlib.Path, alignment: int, seed: int) -> tuple[GGUFFile, list[TensorPlan]]:
    gguf = GGUFFile(path)
    file_size = path.stat().st_size
    ordered_offsets = sorted(t.offset for t in gguf.tensors)
    if len(set(ordered_offsets)) != len(ordered_offsets):
        raise ValueError(f"{path.name}: multiple tensors share a data offset")
    next_for = {offset: (ordered_offsets[i + 1] if i + 1 < len(ordered_offsets) else None)
                for i, offset in enumerate(ordered_offsets)}
    plans = []
    for tensor in gguf.tensors:
        if tensor.type_id == TYPE_ID or tensor.type_name == "HADAMARD_INT2":
            raise ValueError(f"{path.name}: input already contains Hadamard-INT2 tensor {tensor.name}")
        convert = bool(TARGET_RE.fullmatch(tensor.name))
        source_size = _source_tensor_size(gguf, tensor, file_size, next_for[tensor.offset])
        if convert:
            if tensor.type_name not in ("F16", "BF16", "F32"):
                raise ValueError(f"{tensor.name}: target tensor is {tensor.type_name}; convert from F16, BF16, or F32")
            if len(tensor.shape) < 2 or tensor.shape[0] <= 0 or tensor.shape[0] % BLOCK_SIZE:
                raise ValueError(f"{tensor.name}: GGUF dim 0 must be a positive multiple of {BLOCK_SIZE}")
            if source_size != tensor.elements * (4 if tensor.type_name == "F32" else 2):
                raise ValueError(f"{tensor.name}: source byte count does not match {tensor.type_name} shape")
            out_size = tensor.elements // BLOCK_SIZE * BLOCK_BYTES
        else:
            out_size = source_size
        plans.append(TensorPlan(tensor, source_size, out_size, convert))
    return gguf, plans


def _validate_target_set(shard_plans: list[tuple]) -> None:
    by_layer: dict[int, dict[str, tuple[int, ...]]] = {}
    for _, _, _, plans, _ in shard_plans:
        for plan in plans:
            if not plan.convert:
                continue
            match = TARGET_RE.fullmatch(plan.info.name)
            if match is None:
                raise ValueError(f"{plan.info.name}: internal target-name mismatch")
            layer, role = int(match.group("layer")), match.group("role")
            shape = tuple(int(d) for d in plan.info.shape)
            if len(shape) != 3 or any(d <= 0 for d in shape):
                raise ValueError(f"{plan.info.name}: routed expert tensors must have three positive dimensions")
            roles = by_layer.setdefault(layer, {})
            if role in roles:
                raise ValueError(f"duplicate routed {role} tensor in layer {layer}")
            roles[role] = shape

    if not by_layer:
        raise ValueError("no routed expert tensors were planned")
    for layer in range(max(by_layer) + 1):
        roles = by_layer.get(layer)
        if roles is None or set(roles) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer}: expected exactly one gate, up, and down expert tensor")
        gate, up, down = roles["gate"], roles["up"], roles["down"]
        if gate != up:
            raise ValueError(f"layer {layer}: gate and up expert shapes differ: {gate} vs {up}")
        expected_down = (gate[1], gate[0], gate[2])
        if down != expected_down:
            raise ValueError(f"layer {layer}: down expert shape is {down}, expected {expected_down}")
        if any(shape[2] != gate[2] for shape in roles.values()):
            raise ValueError(f"layer {layer}: gate/up/down expert counts differ")
    expert_counts = {roles["gate"][2] for roles in by_layer.values()}
    if len(expert_counts) != 1:
        raise ValueError("routed expert count differs across layers")


def _header_bytes(gguf: GGUFFile, plans: list[TensorPlan], alignment: int, seed: int) -> bytes:
    present = {entry.key for entry in gguf.metadata_entries}
    collision = sorted(key for key in present if key.startswith(META_PREFIX))
    if collision:
        raise ValueError(f"{gguf.path.name}: reserved Hadamard metadata already exists: {', '.join(collision)}")
    records = [entry.encoded for entry in gguf.metadata_entries]
    records.extend(_metadata_record(META_PREFIX + name, kind, value) for name, (kind, value) in META_VALUES.items())
    records.append(_metadata_record(META_PREFIX + "seed", "u64", seed))
    if any(entry.key == "general.alignment" for entry in gguf.metadata_entries):
        # Retain the source's alignment declaration exactly; tensor data is laid out at that alignment.
        pass
    else:
        records.append(_metadata_record("general.alignment", "u32", alignment))

    out = bytearray(struct.pack("<I I Q Q", GGUF_MAGIC, 3, len(plans), len(records)))
    out.extend(b"".join(records))
    for plan in plans:
        tensor = plan.info
        out.extend(_gguf_string(tensor.name))
        out.extend(struct.pack("<I", len(tensor.shape)))
        out.extend(struct.pack(f"<{len(tensor.shape)}Q", *tensor.shape))
        out.extend(struct.pack("<I Q", TYPE_ID if plan.convert else tensor.type_id, plan.offset))
    out.extend(b"\0" * ((_align(len(out), alignment)) - len(out)))
    return bytes(out)


def _decode_chunk(raw: bytes, type_name: str, rows: int, width: int) -> np.ndarray:
    if type_name == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(rows, width).astype(np.float32, copy=False)
    if type_name == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(rows, width).astype(np.float32)
    if type_name == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").reshape(rows, width).astype(np.uint32) << 16
        return bits.view(np.float32)
    raise ValueError(f"unsupported source type {type_name}")


def _copy_exact(source: BinaryIO, destination: BinaryIO, size: int) -> None:
    left = size
    while left:
        block = source.read(min(left, 8 * 1024 * 1024))
        if not block:
            raise EOFError("unexpected end of source tensor data")
        destination.write(block)
        left -= len(block)


def _convert_tensor(source: BinaryIO, destination: BinaryIO, plan: TensorPlan, seed: int,
                    rows_per_chunk: int, scale_iters: int, imatrix) -> dict[str, float | int | str]:
    tensor = plan.info
    width = tensor.shape[0]
    rows = tensor.elements // width
    row_bytes = width * (4 if tensor.type_name == "F32" else 2)
    importance = None
    if imatrix is not None and tensor.name in imatrix.files:
        importance = np.asarray(imatrix[tensor.name], dtype=np.float32).reshape(-1)
        if importance.shape != (width,):
            raise ValueError(f"imatrix[{tensor.name!r}] must contain {width} values, got {importance.shape}")
        if not np.isfinite(importance).all() or np.any(importance < 0):
            raise ValueError(f"imatrix[{tensor.name!r}] must contain finite non-negative second moments")

    # The caller positions the stream at the tensor start before entering this function.
    bytes_written = 0
    squared_error = 0.0
    signal_squared = 0.0
    weighted_error_sum = 0.0
    weighted_denominator = 0.0
    for first in range(0, rows, rows_per_chunk):
        n_rows = min(rows_per_chunk, rows - first)
        raw = source.read(n_rows * row_bytes)
        if len(raw) != n_rows * row_bytes:
            raise EOFError(f"{tensor.name}: source tensor ended during row {first}")
        matrix = _decode_chunk(raw, tensor.type_name, n_rows, width)
        packed, stats = quantize_rows(matrix, seed=seed, importance=importance, scale_iters=scale_iters)
        destination.write(packed)
        bytes_written += len(packed)
        count = n_rows * width
        squared_error += stats["weight_mse"] * count
        signal_squared += float(np.mean(matrix * matrix)) * count
        omega_sum = float(np.sum(importance)) if importance is not None else float(width)
        weighted_error_sum += stats["importance_weighted_mse"] * n_rows * omega_sum
        weighted_denominator += n_rows * omega_sum

    if bytes_written != plan.output_size:
        raise ValueError(f"{tensor.name}: wrote {bytes_written} bytes, expected {plan.output_size}")
    mse = squared_error / (rows * width)
    return {
        "name": tensor.name,
        "source_type": tensor.type_name,
        "shape": tensor.shape,
        "rows": rows,
        "input_bytes": plan.source_size,
        "output_bytes": bytes_written,
        "weight_mse": mse,
        "weight_nmse": mse / max(signal_squared / (rows * width), 1e-30),
        "importance_weighted_mse": weighted_error_sum / max(weighted_denominator, 1e-30),
        "importance_used": importance is not None,
    }


def _convert_file(source_path: pathlib.Path, output_path: pathlib.Path, gguf: GGUFFile,
                  plans: list[TensorPlan], alignment: int, seed: int, rows_per_chunk: int,
                  scale_iters: int, imatrix) -> list[dict]:
    cursor = 0
    for plan in plans:
        plan.offset = cursor
        cursor = _align(cursor + plan.output_size, alignment)
    header = _header_bytes(gguf, plans, alignment, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    metrics = []
    try:
        with source_path.open("rb") as source, temp.open("wb") as destination:
            destination.write(header)
            for plan in plans:
                current = destination.tell() - len(header)
                if current > plan.offset:
                    raise ValueError(f"{plan.info.name}: output data offset calculation overlapped")
                destination.write(b"\0" * (plan.offset - current))
                source.seek(gguf.data_start + plan.info.offset)
                if plan.convert:
                    metrics.append(_convert_tensor(source, destination, plan, seed, rows_per_chunk,
                                                   scale_iters, imatrix))
                else:
                    _copy_exact(source, destination, plan.source_size)
                pad = _align(plan.output_size, alignment) - plan.output_size
                if pad:
                    destination.write(b"\0" * pad)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp, output_path)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    return metrics


def _write_report(path: pathlib.Path, report: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="GGUF file or any shard of a split GGUF")
    parser.add_argument("output", type=pathlib.Path,
                        help="output GGUF path, or output directory when input is split")
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0, help="unsigned 64-bit rotation seed (default: 0)")
    parser.add_argument("--imatrix", type=pathlib.Path,
                        help="optional NPZ with one rotated-basis second-moment vector per tensor name")
    parser.add_argument("--rows-per-chunk", type=int, default=128, help="streaming rows per conversion chunk")
    parser.add_argument("--scale-iters", type=int, default=8, help="weighted scale-refinement iterations")
    parser.add_argument("--report", type=pathlib.Path, help="JSON report path")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    if args.seed < 0 or args.seed >= 1 << 64:
        parser.error("--seed must be in [0, 2^64)")
    if args.rows_per_chunk < 1 or args.scale_iters < 1:
        parser.error("--rows-per-chunk and --scale-iters must be positive")
    shards, split, stem = _collect_shards(args.input)
    output = args.output
    if split:
        if output.exists() and not output.is_dir():
            parser.error("for a split input, output must be a directory")
        output.mkdir(parents=True, exist_ok=True)
        output_paths = [output / shard.name for shard in shards]
    else:
        output_paths = [output]
    if any(src.resolve() == dst.resolve() for src, dst in zip(shards, output_paths)):
        parser.error("input and output paths must differ")
    report_path = args.report or (output / f"{stem}.hadamard-int2.json" if split
                                  else output.with_suffix(output.suffix + ".hadamard-int2.json"))
    protected_paths = {path.resolve() for path in shards + output_paths}
    if report_path.resolve() in protected_paths:
        parser.error("the report path must differ from every input and output GGUF path")

    imatrix = None
    try:
        if args.imatrix:
            imatrix = np.load(args.imatrix, allow_pickle=False)
            if not isinstance(imatrix, np.lib.npyio.NpzFile):
                raise ValueError("--imatrix must be an NPZ archive")
        shard_plans = []
        seen = set()
        total_targets = 0
        for path in shards:
            gguf = GGUFFile(path)
            alignment = gguf.alignment
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(f"{path.name}: GGUF alignment must be a positive power of two")
            gguf, plans = _plan_file(path, alignment, args.seed)
            for plan in plans:
                if plan.info.name in seen:
                    raise ValueError(f"duplicate tensor name across split shards: {plan.info.name}")
                seen.add(plan.info.name)
            total_targets += sum(plan.convert for plan in plans)
            shard_plans.append((path, output_paths[len(shard_plans)], gguf, plans, alignment))
        if not total_targets:
            raise ValueError("no routed expert tensors matched blk.N.ffn_{gate,up,down}_exps.weight")
        _validate_target_set(shard_plans)
        existing = [str(path) for path in output_paths if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"output already exists (pass --overwrite to replace): {', '.join(existing)}")
        if report_path.exists() and not args.overwrite:
            raise FileExistsError(f"report already exists (pass --overwrite to replace): {report_path}")

        all_metrics = []
        for source_path, output_path, gguf, plans, alignment in shard_plans:
            metrics = _convert_file(source_path, output_path, gguf, plans, alignment, args.seed,
                                    args.rows_per_chunk, args.scale_iters, imatrix)
            all_metrics.extend(metrics)
            print(f"wrote {output_path} ({len(metrics)} tensors converted)")

        total_in = sum(int(item["input_bytes"]) for item in all_metrics)
        total_out = sum(int(item["output_bytes"]) for item in all_metrics)
        report = {
            "format": "Strata Hadamard-INT2 GGUF extension",
            "version": 1,
            "seed": args.seed,
            "block_size": BLOCK_SIZE,
            "type_id": TYPE_ID,
            "block_bytes": BLOCK_BYTES,
            "converted_tensor_count": len(all_metrics),
            "converted_input_bytes": total_in,
            "converted_output_bytes": total_out,
            "converted_size_ratio": total_out / total_in if total_in else None,
            "tensors": all_metrics,
            "quality_note": "Weight reconstruction error only; this report does not measure end-to-end model quality.",
        }
        _write_report(report_path, report)
        print(f"report: {report_path}")
        return 0
    finally:
        if isinstance(imatrix, np.lib.npyio.NpzFile):
            imatrix.close()


if __name__ == "__main__":
    raise SystemExit(main())
