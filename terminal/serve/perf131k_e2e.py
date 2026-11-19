"""Opt-in, bounded server-path evidence. No GPU calls, workers, or prompt text."""
import hashlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import struct
import threading
import time

_local = threading.local()
_LIMIT = (1 << 63) - 1


def token_hash(ids, *, limit=32768):
    if len(ids) > limit or any(type(t) is not int or not 0 <= t < (1 << 31) for t in ids):
        raise ValueError("invalid bounded token fixture")
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


class Settings:
    def __init__(self, directory, run_id, prompt_sha256, bindings_sha256, *,
                 prompt_tokens=4096, series="normal", detail_trace="OFF", max_requests=64, max_samples=256):
        directory = Path(directory).resolve()
        if directory.drive.upper() != "H:" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
            raise ValueError("H: destination and simple run ID required")
        if any(not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{64}", s)
               for s in (prompt_sha256, bindings_sha256)):
            raise ValueError("fixture and external binding manifest hashes required")
        if (type(prompt_tokens) is not int or prompt_tokens not in (4096, 32768) or
                series not in ("normal", "diagnostic") or detail_trace not in ("OFF", "ON") or
                type(max_requests) is not int or not 1 <= max_requests <= 64 or
                type(max_samples) is not int or not 1 <= max_samples <= 4096):
            raise ValueError("invalid bounded measurement settings")
        self.directory = directory / run_id
        self.directory.mkdir()  # exclusive: never replace another run's evidence
        self.prompt_sha256, self.bindings_sha256 = prompt_sha256, bindings_sha256
        self.prompt_tokens, self.series, self.detail_trace = prompt_tokens, series, detail_trace
        self.max_requests, self.max_samples = max_requests, max_samples
        self.config_sha256 = None
        self.lock = threading.Lock()
        self.started = self.completed = self.dropped_requests = self.write_errors = 0
        with (self.directory / "summary.json").open("x", encoding="utf-8") as out:
            out.write("{}\n")
        self._summary()

    def _summary(self):
        data = {k: getattr(self, k) for k in ("started", "completed", "dropped_requests", "write_errors")}
        data["request_limit"] = self.max_requests
        try:
            with (self.directory / "summary.json").open("r+", encoding="utf-8") as out:
                out.seek(0)
                out.write(json.dumps(data, allow_nan=False) + "\n")
                out.truncate()
        except OSError:
            self.write_errors = min(self.write_errors + 1, _LIMIT)

    def claim(self):
        with self.lock:
            if self.started >= self.max_requests:
                self.dropped_requests = min(self.dropped_requests + 1, _LIMIT)
                self._summary()
                return None
            self.started += 1
            self._summary()
            return self.started

    def publish(self, record):
        with self.lock:
            try:
                raw = json.dumps(record, allow_nan=False)
                if len(raw.encode("utf-8")) > 131072:
                    raise ValueError("record exceeds bound")
                with (self.directory / f"request-{record['sequence']:06d}.json").open("x", encoding="utf-8") as out:
                    out.write(raw + "\n")
                self.completed += 1
            except (OSError, ValueError):
                self.write_errors = min(self.write_errors + 1, _LIMIT)
            self._summary()


def load_settings(path):
    if path is None:
        return None
    with Path(path).open("rb") as source:
        raw = source.read(16385)
    if len(raw) > 16384:
        raise ValueError("measurement config exceeds 16KiB")
    settings = Settings(**json.loads(raw))
    settings.config_sha256 = hashlib.sha256(raw).hexdigest()
    return settings


def current():
    return getattr(_local, "observer", None)


def begin(settings, path, req):
    if settings is None or path != "/v1/chat/completions":
        return None
    sequence = settings.claim()
    if sequence is None:
        return None
    obs = Observer(settings, sequence)
    _local.observer = obs
    if req is not None:
        obs.request(req)
    return obs


def end(obs):
    if obs is not None:
        try:
            obs.finish()
        finally:
            _local.observer = None


class Observer:
    def __init__(self, settings, sequence):
        self.settings, self.sequence = settings, sequence
        self.start = time.perf_counter_ns()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.native_pid = None
        self.reasons = set()
        self.segment = None
        self.prepared_seen = self.http_ended = self.http_terminal = self.finished = False
        self._guard = threading.RLock()
        self.segment_chain = hashlib.sha256(b"perf131k.segment.v1")
        self.segment_samples = []
        self.segment_detail_dropped = 0
        self.native_segments = self.native_terminals = self.native_t = self.delivered_t = self.synthetic = 0
        self.native_generated = 0
        self.token_times = []
        self.sample_dropped = self.http_chunks = 0
        self.first_http_delta_s = None
        self.usage = None
        self.http_finish = self.http_response_id = None
        self.requested_reserve = self.actual_capacity = None
        self.effective_reserve = self.actual_prompt_tokens = self.actual_prompt_hash = None
        self.native_prompt_ms = self.native_decode_ms = 0.0
        self.native_finish = None
        self.service_finish = None
        self.service_terminals = 0
        self.optional_counters = {"drafts_accepted": None, "drafts_offered": None, "reused": None}

    def fail(self, reason):
        with self._guard:
            if not self.finished:  # later watcher notifications are outside the sealed server-path observation
                self.reasons.add(reason)  # fixed vocabulary; never request/error strings

    def elapsed(self):
        return (time.perf_counter_ns() - self.start) / 1e9

    def count(self, name, amount=1):
        value = getattr(self, name) + amount
        if value > _LIMIT:
            self.fail("counter_overflow")
        setattr(self, name, min(value, _LIMIT))

    def request(self, req):
        reserve = req.get("max_completion_tokens") or req.get("max_tokens")
        self.requested_reserve = reserve if type(reserve) is int and 0 <= reserve <= _LIMIT else None
        if type(reserve) is not int or reserve != 256:
            self.fail("requested_reserve_mismatch")
        if req.get("strata_mcp") or req.get("n", 1) != 1:
            self.fail("unsupported_request_route")

    def prepared(self, ids, reserve, capacity):
        self.prepared_seen = True
        self.effective_reserve, self.actual_prompt_tokens = reserve, len(ids)
        self.actual_capacity = capacity
        if type(reserve) is not int or reserve != 256:
            self.fail("effective_reserve_mismatch")
        if type(capacity) is not int or capacity != 131072:
            self.fail("capacity_mismatch")
        try:
            self.actual_prompt_hash = token_hash(ids)
        except ValueError:
            self.fail("invalid_input_tokens")
        if len(ids) != self.settings.prompt_tokens or self.actual_prompt_hash != self.settings.prompt_sha256:
            self.fail("input_fixture_mismatch")

    def native_begin(self, ids, reserve, batch=False, embeddings=None, pid=None):
        if type(pid) is int and pid > 0:
            if self.native_pid is not None and self.native_pid != pid:
                self.fail("native_process_changed")
            self.native_pid = pid
        if self.segment is not None:
            self.fail("missing_native_terminal")
        if batch or embeddings:
            self.fail("unsupported_native_route")
        try:
            input_hash = token_hash(ids, limit=131072)
        except ValueError:
            input_hash = None
            self.fail("invalid_segment_input")
        if self.native_segments == 0:
            matches = self.prepared_seen and input_hash == self.actual_prompt_hash
            if not matches:
                self.fail("native_input_mismatch")
            if reserve != self.effective_reserve:
                self.fail("native_reserve_mismatch")
        self.count("native_segments")
        if type(reserve) is not int or not 0 <= reserve <= _LIMIT:
            self.fail("invalid_native_reserve")
            reserve = None
        self.segment = {"index": self.native_segments, "prompt": len(ids), "tokens": 0,
                        "input_sha256": input_hash, "reserve": reserve, "pid": self.native_pid}
        self.bind_segment("begin", self.segment)

    def bind_segment(self, phase, fields):
        raw = json.dumps([phase, fields], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.segment_chain.update(struct.pack("<I", len(raw)))
        self.segment_chain.update(raw)

    def native_token(self, token, delivered=True):
        if type(token) is not int or not 0 <= token < (1 << 31):
            self.fail("invalid_native_token")
        self.count("native_t")
        self.count("delivered_t", int(delivered))
        if self.segment is None:
            self.fail("token_without_segment")
        else:
            self.segment["tokens"] = min(self.segment["tokens"] + 1, _LIMIT)
        if len(self.token_times) < self.settings.max_samples:
            self.token_times.append(self.elapsed())
        else:
            self.count("sample_dropped")
            self.fail("timestamp_overflow")

    def drain_token(self, line):
        try:
            self.native_token(int(line[2:]), delivered=False)
        except ValueError:
            self.fail("invalid_native_token")

    def native_done(self, line):
        if len(line) > 4096:
            self.fail("invalid_native_terminal")
            return
        try:
            fields = line.split()
            produced, prompt = int(fields[1]), int(fields[2])
            phases = [float(fields[3]), float(fields[4])]
            finish = fields[5]
            if (fields[0] != "DONE" or not 0 <= produced <= _LIMIT or not 0 <= prompt <= _LIMIT or
                    any(not math.isfinite(v) or v < 0 for v in phases)):
                raise ValueError()
        except (IndexError, ValueError):
            self.fail("invalid_native_terminal")
            return
        if self.segment is None:
            self.fail("duplicate_native_terminal")
        elif (produced != self.segment["tokens"] or prompt != self.segment["prompt"] or
              self.segment["reserve"] is None or produced > self.segment["reserve"]):
            self.fail("native_count_mismatch")
        if finish not in ("stop", "length"):
            self.fail("native_unsuccessful_finish")
        self.native_finish = finish if finish in ("stop", "length", "cancel", "nonfinite", "error") else "UNKNOWN"
        terminal_hash = hashlib.sha256(line.encode()).hexdigest()
        self.bind_segment("done", {"index": self.segment["index"] if self.segment else None,
                                   "terminal_sha256": terminal_hash})
        if self.segment is not None:
            if len(self.segment_samples) < 8:
                self.segment_samples.append({"index": self.segment["index"],
                    "prompt_tokens": self.segment["prompt"], "input_sha256": self.segment["input_sha256"],
                    "reserve": self.segment["reserve"], "terminal_sha256": terminal_hash})
            else:
                self.count("segment_detail_dropped")
        self.native_prompt_ms += phases[0]
        self.native_decode_ms += phases[1]
        if not math.isfinite(self.native_prompt_ms + self.native_decode_ms):
            self.fail("native_clock_overflow")
        for index, key in enumerate(self.optional_counters, 6):
            try:
                value = int(fields[index])
                if not 0 <= value <= _LIMIT:
                    raise ValueError()
            except IndexError:
                value = None
            except ValueError:
                value = None
                self.fail("invalid_native_optional_counter")
            prior = self.optional_counters[key]
            if self.native_terminals and value is not None and prior is not None:
                if prior + value > _LIMIT:
                    self.fail("counter_overflow")
                    value = None  # an unknown exact total, not silently saturated precision
                else:
                    value += prior
            elif self.native_terminals:
                value = None
            self.optional_counters[key] = value
        self.count("native_generated", produced)
        self.count("native_terminals")
        self.segment = None

    def synthetic_token(self):
        self.count("synthetic")

    def service_terminal(self, finish):
        self.count("service_terminals")
        self.service_finish = finish if finish in (
            "stop", "length", "cancel", "disconnect", "timeout", "error") else "UNKNOWN"
        if finish not in ("stop", "length"):
            self.fail("service_unsuccessful_finish")

    def note(self, **values):
        if values.get("error") or values.get("outcome") == "disconnected":
            self.fail("http_error_or_disconnect")
        if values.get("http_status", 200) != 200:
            self.fail("http_status_error")

    def http_event(self, event):
        self.count("http_chunks")
        if event.get("error"):
            self.fail("http_error_or_disconnect")
        response_id = event.get("id")
        if isinstance(response_id, str) and re.fullmatch(r"chatcmpl-[0-9a-f]{24}", response_id):
            if self.http_response_id is not None and self.http_response_id != response_id:
                self.fail("http_identity_mismatch")
            self.http_response_id = response_id
        for choice in event.get("choices", []):
            if choice.get("finish_reason") is not None:
                self.http_terminal = True
                self.http_finish = choice["finish_reason"] if choice["finish_reason"] in (
                    "stop", "length", "tool_calls") else "UNKNOWN"
                if choice["finish_reason"] not in ("stop", "length"):
                    self.fail("http_unsuccessful_finish")
            delta = choice.get("delta") or choice.get("message") or {}
            if self.first_http_delta_s is None and (delta.get("content") or delta.get("reasoning_content")):
                self.first_http_delta_s = self.elapsed()
        if "usage" in event:
            self.usage = event["usage"]

    def http_end(self):
        self.http_ended = True

    def finish(self):
        with self._guard:
            if self.finished:
                return
            record = self._finish_record()
            self.finished = True
        self.settings.publish(record)

    def _finish_record(self):
        if not self.prepared_seen:
            self.fail("missing_prepared_input")
        if not self.native_segments or self.native_terminals != self.native_segments or self.segment is not None:
            self.fail("missing_native_terminal")
        if not self.http_ended or not self.http_terminal:
            self.fail("missing_http_terminal")
        if not self.service_terminals:
            self.fail("missing_service_terminal")
        if not isinstance(self.usage, dict):
            self.fail("missing_http_usage")
        elif (type(self.usage.get("prompt_tokens")) is not int or
              self.usage["prompt_tokens"] != self.actual_prompt_tokens or
              type(self.usage.get("completion_tokens")) is not int or
              self.usage["completion_tokens"] != self.delivered_t + self.synthetic):
            self.fail("http_count_mismatch")
        if type(self.effective_reserve) is int and self.delivered_t + self.synthetic > self.effective_reserve:
            self.fail("reservation_exceeded")
        record = {"schema": "perf131k.e2e.v1", "sequence": self.sequence,
                  "server_pid": os.getpid(), "native_pid": self.native_pid,
                  "started_utc": self.started_utc, "server_clock_start_ns": self.start,
                  "status": "INCOMPLETE" if self.reasons else "COMPLETE_OBSERVED_SERVER_PATH",
                  "reasons": sorted(self.reasons), "run_id": self.settings.directory.name,
                  "capacity": self.actual_capacity, "expected_capacity": 131072,
                  "requested_reserve": self.requested_reserve, "expected_reserve": 256,
                  "effective_reserve": self.effective_reserve, "prompt_tokens": self.actual_prompt_tokens,
                  "expected_prompt_tokens": self.settings.prompt_tokens,
                  "expected_prompt_sha256": self.settings.prompt_sha256,
                  "prompt_sha256": self.actual_prompt_hash, "native_segments": self.native_segments,
                  "native_terminals": self.native_terminals, "native_t": self.native_t,
                  "segment_chain_sha256": self.segment_chain.hexdigest(),
                  "segment_samples": self.segment_samples, "segment_detail_dropped": self.segment_detail_dropped,
                  "native_delivered_t": self.delivered_t,
                  "native_generated": self.native_generated if self.native_terminals else None,
                  "synthetic_tokens": self.synthetic, "token_times_s": self.token_times,
                  "native_finish": self.native_finish, "optional_native_counters": self.optional_counters,
                  "service_finish": self.service_finish, "service_terminals": self.service_terminals,
                  "native_host_prompt_ms_sum": self.native_prompt_ms if self.native_terminals and
                       math.isfinite(self.native_prompt_ms) else None,
                  "native_host_decode_ms_sum": self.native_decode_ms if self.native_terminals and
                       math.isfinite(self.native_decode_ms) else None,
                  "token_samples_dropped": self.sample_dropped, "http_chunks": self.http_chunks,
                  "http_response_id": self.http_response_id, "http_finish": self.http_finish,
                  "http_server_write_complete": self.http_ended,
                  "http_prompt_tokens": self.usage.get("prompt_tokens") if isinstance(self.usage, dict)
                       and type(self.usage.get("prompt_tokens")) is int else None,
                  "http_completion_tokens": self.usage.get("completion_tokens") if isinstance(self.usage, dict)
                       and type(self.usage.get("completion_tokens")) is int else None,
                  "first_http_delta_s": self.first_http_delta_s, "end_s": self.elapsed(),
                  "clock_domain": "server perf_counter_ns; HTTP writes are not client arrival",
                  "observation_cutoff": "sealed server-handler snapshot; later watcher notifications excluded",
                  "client_arrival_times": None, "GPU_execution_time": None,
                  "external_bindings_sha256": self.settings.bindings_sha256,
                  "config_sha256": self.settings.config_sha256,
                  "external_bindings_verified": False, "series": self.settings.series,
                  "detail_trace": self.settings.detail_trace}
        return record
