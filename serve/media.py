"""Host-only SVE1/SVE2 media transport and rotary-position planning.

SVE1 stays the image encoder's existing format. SVE2 binds explicit visual spans
and relative (time, height, width) positions to token IDs. See docs/MEDIA_FORMAT.md.
This module neither decodes video nor runs inference.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO


class MediaError(ValueError):
    pass


class MediaKind(IntEnum):
    IMAGE = 1
    VIDEO = 2


Position = tuple[int, int, int]
HEADER = struct.Struct("<4sHHIIQQQQQQ")
SPAN = struct.Struct("<QQIiQIIQQQ")
LEGACY_HEADER = struct.Struct("<4siiii")
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class MediaLimits:
    max_tokens: int = 1 << 20
    max_spans: int = 128
    max_rows: int = 16384
    max_width: int = 16384
    max_bytes: int = 256 << 20
    max_position: int = INT32_MAX
    vocab_size: int = INT32_MAX + 1
    expected_width: int = 0
    allowed_pad_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class VisualSpan:
    start: int
    pad_id: int
    kind: MediaKind
    advance: int
    positions: tuple[Position, ...]
    embeddings: bytes
    nx: int = 0
    ny: int = 0


@dataclass(frozen=True)
class MediaBundle:
    width: int
    tokens: tuple[int, ...]
    spans: tuple[VisualSpan, ...]


@dataclass(frozen=True)
class LegacyImage:
    width: int
    nx: int
    ny: int
    embeddings: bytes


@dataclass(frozen=True)
class PositionPlan:
    positions: tuple[Position, ...]
    # (span index, row index), or None for text and generation cells.
    rows: tuple[tuple[int, int] | None, ...]


DEFAULT_LIMITS = MediaLimits()


def _require(ok: bool, message: str):
    if not ok:
        raise MediaError(message)


def _integer(value: int, lo: int, hi: int, label: str):
    _require(type(value) is int and lo <= value <= hi, f"invalid {label}")


def _limits(limits: MediaLimits):
    _require(isinstance(limits, MediaLimits), "invalid media limits")
    for name in ("max_tokens", "max_spans", "max_rows", "max_width", "max_bytes"):
        _integer(getattr(limits, name), 1, (1 << 63) - 1, name)
    _integer(limits.max_position, 0, INT32_MAX, "max_position")
    _integer(limits.vocab_size, 1, INT32_MAX + 1, "vocab_size")
    _integer(limits.expected_width, 0, limits.max_width, "expected_width")
    _require(isinstance(limits.allowed_pad_ids, tuple), "invalid allowed_pad_ids")
    for pad in limits.allowed_pad_ids:
        _integer(pad, 0, limits.vocab_size - 1, "allowed pad ID")


def _width(width: int, limits: MediaLimits):
    _integer(width, 1, min(limits.max_width, INT32_MAX), "embedding width")
    _require(not limits.expected_width or width == limits.expected_width, "embedding width mismatch")


def _floats(payload: bytes, count: int):
    _require(isinstance(payload, bytes) and len(payload) == count * 4, "embedding length mismatch")
    _require(all(math.isfinite(f) for (f,) in struct.iter_unpack("<f", payload)), "non-finite embedding")


def _size(tokens: int, spans: int, rows: int, width: int) -> int:
    return HEADER.size + tokens * 4 + spans * SPAN.size + rows * (12 + width * 4)


def _validate_structure(bundle: MediaBundle, limits: MediaLimits):
    _limits(limits)
    _require(isinstance(bundle, MediaBundle), "expected a MediaBundle")
    _width(bundle.width, limits)
    _require(isinstance(bundle.tokens, tuple) and isinstance(bundle.spans, tuple), "media records must be immutable")
    _require(len(bundle.tokens) <= limits.max_tokens, "token budget exceeded")
    _require(len(bundle.spans) <= limits.max_spans, "span budget exceeded")
    for token in bundle.tokens:
        _integer(token, 0, limits.vocab_size - 1, "token ID")
    rows, end, base = 0, 0, 0
    for span in bundle.spans:
        _require(isinstance(span, VisualSpan), "expected a VisualSpan")
        _require(isinstance(span.positions, tuple), "positions must be immutable")
        count = len(span.positions)
        _require(count > 0, "empty visual span")
        rows += count
        _require(rows <= limits.max_rows, "row budget exceeded")
        _require(_size(len(bundle.tokens), len(bundle.spans), rows, bundle.width) <= limits.max_bytes,
                 "media byte budget exceeded")
        _integer(span.start, end, len(bundle.tokens), "span start")
        _require(span.start + count <= len(bundle.tokens), "span extends past tokens")
        _integer(span.pad_id, 0, limits.vocab_size - 1, "pad ID")
        _require(not limits.allowed_pad_ids or span.pad_id in limits.allowed_pad_ids, "unsupported pad ID")
        _require(span.kind in (MediaKind.IMAGE, MediaKind.VIDEO) and isinstance(span.kind, MediaKind), "invalid media kind")
        _integer(span.advance, 1, limits.max_position + 1, "position advance")
        _integer(span.nx, 0, INT32_MAX, "grid width")
        _integer(span.ny, 0, INT32_MAX, "grid height")
        if span.kind == MediaKind.IMAGE:
            _require(span.nx > 0 and span.ny > 0 and span.nx * span.ny == count, "invalid image grid")
            _require(span.advance == max(span.nx, span.ny), "invalid image position advance")
        else:
            _require(span.nx == span.ny == 0, "video grid must be zero; explicit positions are authoritative")
        base += span.start - end
        extent = 0
        for row, pos in enumerate(span.positions):
            _require(isinstance(pos, tuple) and len(pos) == 3, "invalid position")
            for coord in pos:
                _integer(coord, 0, limits.max_position, "relative position")
                _require(base + coord <= limits.max_position, "absolute position budget exceeded")
            extent = max(extent, *pos)
            if span.kind == MediaKind.IMAGE:
                _require(pos == (0, row // span.nx, row % span.nx), "image positions do not match grid")
            _require(bundle.tokens[span.start + row] == span.pad_id, "span pad tokens do not match")
        _require(span.advance > extent, "position advance does not cover span")
        base += span.advance
        end = span.start + count
    _require(_size(len(bundle.tokens), len(bundle.spans), rows, bundle.width) <= limits.max_bytes,
             "media byte budget exceeded")
    _require(base + len(bundle.tokens) - end <= limits.max_position + 1, "text position budget exceeded")


def validate(bundle: MediaBundle, limits: MediaLimits = DEFAULT_LIMITS):
    _validate_structure(bundle, limits)
    for span in bundle.spans:
        _floats(span.embeddings, len(span.positions) * bundle.width)


def build_positions(bundle: MediaBundle, capacity: int, limits: MediaLimits = DEFAULT_LIMITS) -> PositionPlan:
    validate(bundle, limits)
    _integer(capacity, len(bundle.tokens), limits.max_tokens, "position capacity")
    next_pos = len(bundle.tokens) - sum(len(s.positions) for s in bundle.spans) + sum(s.advance for s in bundle.spans)
    _require(next_pos + capacity - len(bundle.tokens) <= limits.max_position + 1, "generation position budget exceeded")
    positions, rows = [], []
    cell, base = 0, 0
    for index, span in enumerate(bundle.spans):
        while cell < span.start:
            positions.append((base, base, base))
            rows.append(None)
            cell, base = cell + 1, base + 1
        for row, pos in enumerate(span.positions):
            positions.append(tuple(base + v for v in pos))
            rows.append((index, row))
        cell += len(span.positions)
        base += span.advance
    while cell < capacity:
        positions.append((base, base, base))
        rows.append(None)
        cell, base = cell + 1, base + 1
    return PositionPlan(tuple(positions), tuple(rows))


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    _require(isinstance(data, bytes) and len(data) == size, "truncated media file")
    return data


def _write(stream: BinaryIO, data: bytes):
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if not isinstance(written, int) or not 0 < written <= len(view):
            raise OSError("short media write")
        view = view[written:]


def read_bundle(stream: BinaryIO, limits: MediaLimits = DEFAULT_LIMITS) -> MediaBundle:
    _limits(limits)
    magic, version, header_size, flags, width, nt, ns, nr, pb, eb, total = HEADER.unpack(_read(stream, HEADER.size))
    _require((magic, version, header_size, flags) == (b"SVE2", 2, HEADER.size, 0), "unsupported media header")
    _width(width, limits)
    _require(nt <= limits.max_tokens and ns <= limits.max_spans and nr <= limits.max_rows, "media count budget exceeded")
    _require(ns <= nr and (ns != 0 or nr == 0), "invalid span/row counts")
    _require(pb == nr * 12 and eb == nr * width * 4 and total == _size(nt, ns, nr, width), "media byte counts mismatch")
    _require(total <= limits.max_bytes, "media byte budget exceeded")
    tokens = tuple(v for (v,) in struct.iter_unpack("<i", _read(stream, nt * 4)))
    records, pos_off, emb_off, end = [], 0, 0, 0
    for _ in range(ns):
        start, count, kind, pad, advance, nx, ny, po, eo, reserved = SPAN.unpack(_read(stream, SPAN.size))
        _require(count > 0 and count <= nr and start >= end and start + count <= nt, "invalid span range")
        _require(po == pos_off and eo == emb_off and reserved == 0, "noncanonical span offsets or flags")
        pos_off += count * 12
        emb_off += count * width * 4
        _require(pos_off <= pb and emb_off <= eb, "span payload exceeds budget")
        _require(kind in (1, 2), "invalid media kind")
        records.append((start, count, MediaKind(kind), pad, advance, nx, ny))
        end = start + count
    _require((pos_off, emb_off) == (pb, eb), "unused media payload")
    positions = [tuple(struct.iter_unpack("<iii", _read(stream, count * 12)))
                 for _, count, *_ in records]
    spans = tuple(VisualSpan(start, pad, kind, advance, pos, b"", nx, ny)
                  for (start, _, kind, pad, advance, nx, ny), pos in zip(records, positions))
    _validate_structure(MediaBundle(width, tokens, spans), limits)
    spans = tuple(VisualSpan(s.start, s.pad_id, s.kind, s.advance, s.positions,
                             _read(stream, len(s.positions) * width * 4), s.nx, s.ny) for s in spans)
    _require(stream.read(1) == b"", "trailing media bytes")
    result = MediaBundle(width, tokens, tuple(spans))
    validate(result, limits)
    return result


def write_bundle(stream: BinaryIO, bundle: MediaBundle, limits: MediaLimits = DEFAULT_LIMITS):
    validate(bundle, limits)
    nt, ns, nr = len(bundle.tokens), len(bundle.spans), sum(len(s.positions) for s in bundle.spans)
    _write(stream, HEADER.pack(b"SVE2", 2, HEADER.size, 0, bundle.width, nt, ns, nr, nr * 12,
                              nr * bundle.width * 4, _size(nt, ns, nr, bundle.width)))
    for start in range(0, nt, 4096):
        chunk = bundle.tokens[start:start + 4096]
        _write(stream, struct.pack(f"<{len(chunk)}i", *chunk))
    po, eo = 0, 0
    for span in bundle.spans:
        count = len(span.positions)
        _write(stream, SPAN.pack(span.start, count, span.kind, span.pad_id, span.advance, span.nx, span.ny, po, eo, 0))
        po += count * 12
        eo += count * bundle.width * 4
    for span in bundle.spans:
        for pos in span.positions:
            _write(stream, struct.pack("<iii", *pos))
    for span in bundle.spans:
        _write(stream, span.embeddings)


def decode(data: bytes, limits: MediaLimits = DEFAULT_LIMITS) -> MediaBundle:
    return read_bundle(io.BytesIO(data), limits)


def encode(bundle: MediaBundle, limits: MediaLimits = DEFAULT_LIMITS) -> bytes:
    out = io.BytesIO()
    write_bundle(out, bundle, limits)
    return out.getvalue()


def _legacy(image: LegacyImage, limits: MediaLimits):
    _require(isinstance(image, LegacyImage), "expected a LegacyImage")
    _width(image.width, limits)
    _integer(image.nx, 1, INT32_MAX, "image grid width")
    _integer(image.ny, 1, INT32_MAX, "image grid height")
    count = image.nx * image.ny
    _require(count <= min(limits.max_rows, INT32_MAX), "image row budget exceeded")
    _require(LEGACY_HEADER.size + count * image.width * 4 <= limits.max_bytes, "image byte budget exceeded")
    _floats(image.embeddings, count * image.width)


def read_legacy_images(stream: BinaryIO, limits: MediaLimits = DEFAULT_LIMITS) -> tuple[LegacyImage, ...]:
    _limits(limits)
    images, rows, used = [], 0, 0
    while True:
        magic = stream.read(4)
        if magic == b"":
            return tuple(images)
        _require(magic == b"SVE1", "invalid legacy image header")
        count, nx, ny, width = struct.unpack("<iiii", _read(stream, 16))
        _width(width, limits)
        _require(nx > 0 and ny > 0 and nx * ny == count, "invalid legacy image grid")
        rows += count
        used += LEGACY_HEADER.size + count * width * 4
        _require(rows <= limits.max_rows and len(images) < limits.max_spans, "image count budget exceeded")
        _require(used <= limits.max_bytes, "image byte budget exceeded")
        image = LegacyImage(width, nx, ny, _read(stream, count * width * 4))
        _legacy(image, limits)
        images.append(image)


def write_legacy_images(stream: BinaryIO, images: tuple[LegacyImage, ...], limits: MediaLimits = DEFAULT_LIMITS):
    _limits(limits)
    _require(len(images) <= limits.max_spans, "image count budget exceeded")
    rows, used = 0, 0
    for image in images:
        _legacy(image, limits)
        rows += image.nx * image.ny
        used += LEGACY_HEADER.size + len(image.embeddings)
    _require(rows <= limits.max_rows and used <= limits.max_bytes, "image budget exceeded")
    for image in images:
        _write(stream, LEGACY_HEADER.pack(b"SVE1", image.nx * image.ny, image.nx, image.ny, image.width))
        _write(stream, image.embeddings)


def adapt_legacy_images(images: tuple[LegacyImage, ...], tokens: tuple[int, ...], pad_id: int,
                        limits: MediaLimits = DEFAULT_LIMITS) -> MediaBundle:
    _limits(limits)
    _integer(pad_id, 0, limits.vocab_size - 1, "pad ID")
    _require(bool(images) and len(images) <= limits.max_spans, "invalid legacy image count")
    _require(len(tokens) <= limits.max_tokens, "token budget exceeded")
    spans, cell, index = [], 0, 0
    while cell < len(tokens):
        if tokens[cell] != pad_id:
            cell += 1
            continue
        _require(index < len(images), "more image tokens than records")
        image = images[index]
        _legacy(image, limits)
        _require(image.width == images[0].width, "image widths differ")
        count = image.nx * image.ny
        _require(cell + count <= len(tokens), "image extends past tokens")
        positions = tuple((0, j // image.nx, j % image.nx) for j in range(count))
        spans.append(VisualSpan(cell, pad_id, MediaKind.IMAGE, max(image.nx, image.ny), positions,
                                image.embeddings, image.nx, image.ny))
        cell, index = cell + count, index + 1
    _require(index == len(images), "more image records than tokens")
    _require(not tokens or tokens[-1] != pad_id, "prompt cannot end in an image")
    result = MediaBundle(images[0].width, tokens, tuple(spans))
    validate(result, limits)
    return result


def span_fingerprint(bundle: MediaBundle, index: int, limits: MediaLimits = DEFAULT_LIMITS) -> int:
    validate(bundle, limits)
    _integer(index, 0, len(bundle.spans) - 1, "span index")
    span = bundle.spans[index]
    count = len(span.positions)
    if span.kind == MediaKind.IMAGE:
        parts = [struct.pack("<qqq", count, span.nx, span.ny)]
    else:
        parts = [b"SVE2", struct.pack("<IIiQQQ", bundle.width, span.kind, span.pad_id,
                                     span.start, count, span.advance)]
        parts.extend(struct.pack("<iii", *p) for p in span.positions)
    parts.append(span.embeddings)
    value = 1469598103934665603  # Existing image-cache seed; retain it, including for negative zero.
    for part in parts:
        for byte in part:
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value
