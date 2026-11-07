#!/usr/bin/env python3
"""Merge-safe contracts for the L2 fork. No running engine is required."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source(name):
    return (ROOT / name).read_text(encoding="utf-8")


class SourceContracts(unittest.TestCase):
    def assert_tokens(self, name, *tokens):
        text = source(name)
        for token in tokens:
            self.assertIn(token, text, f"{name}: missing fork contract {token!r}")

    def test_engine_command_and_reply_surface(self):
        self.assert_tokens(
            "src/program/generate.cpp",
            "SAVEINC <path>", "RESTOREP <path>", "RESTOREPP <path> <ids>", "PROBE <ids>",
            "SAVEINC'D", "RESTOREDP", "PROBED", "VERIFIED", "L2W q",
        )

    def test_file_format_and_writer_surface(self):
        self.assert_tokens(
            "src/core/conversation_file_v2.inc",
            "session_file_saveinc", "session_file_saveinc_peek", "session_file_read_parked",
            "session_file_read_prefix", "v2_verify_structure", "STRSESS\\x02",
        )
        self.assert_tokens(
            "src/core/conversation_file_l2.inc",
            "class WriterPool", "hook_split", "hook_post_saveinc", "hook_post_read",
            "inplace=1", "gap_tokens == 0", "next_gen", "WriterPool_submit_alias", "L2C ", "L2W q",
            "copy_bytes=0", "copy_ms=", "saveinc_ms=", "writer_ms=", "growth_bytes=", "copy_used=0",
            "stale mini ignored", "mini diverges from main live", "job.opt.progress = {};", "job.opt.phase = {};",
        )
        self.assertNotIn("std::filesystem::copy_file(job.path", source("src/core/conversation_file_l2.inc"))

    def test_resume_and_observability_surface(self):
        self.assert_tokens(
            "src/program/generate.cpp",
            "lcp_length", "BPE-boundary LCP borrow", "STRATA_G2_TRACE", "G2WIN",
            "STRATA_G3_HIST", "G3HIST", "strata G1 chunk:",
        )
        self.assert_tokens("src/core/conversation_state.cpp", "tolerant_ckpts", "prefix_ok")
        self.assert_tokens("include/strata/core/conversation_cache.hpp", "conversation_live_lcp", "if (live && n == 0)")
        self.assert_tokens("src/prefill/prefill.cpp", "strata prefill warmer:")

    def test_dflash2_snapshot_draft_is_optional(self):
        text = source("src/program/generate.cpp")
        self.assertGreaterEqual(text.count("use_mtp ? &mtp.kv_state() : nullptr"), 5)

    def test_single_server_source(self):
        self.assert_tokens(
            "serve/server.py",
            '"saveinc"', '"restorep"', '"restorepp"', '"verify"', "/v1/cache/probe", "PROBED",
            "restore_events", "restore_for_request", "X-Strata-Request-Id", "X-Strata-L2-Trace",
            "verified_prompt_prefixes", "prefix_tokens", "prefix_rejected",
            "/v1/cache/stats", "/v1/cache/list", "l2_cache_get", "l2_cache_post", "l2_extend_fields",
            "L2W ", "L2C ", "l2_writer_pending", "l2_commits", "async_pending", 'body["gen"] = r["gen"]',
        )
        # One server source only: the old synchronized copy and legacy gateway runtimes stay out.
        for legacy in ("l2/serve-server.py", "l2/session-gateway.py", "l2-gateway", "regression-l2.py", "session-gateway"):
            self.assertFalse((ROOT / legacy).exists(), f"legacy runtime remains: {legacy}")
        self.assert_tokens(
            "l2/cache_control.py",
            "CACHE STATS", "CACHE LIST", "cache_set_pinned", "restorepp", "/v1/cache/preload",
            "serial",
        )
        self.assert_tokens(
            "src/program/generate.cpp", "l2_cache_control.hpp",
            "RESTOREDP %zu %zu %.1f %d serial=",
        )
        self.assert_tokens(
            "src/program/l2_cache_control.hpp", "CACHE ", "STATS", "LIST", "parked_best",
            "l2_set_pinned", "l2_drop",
        )
        self.assert_tokens(
            "include/strata/core/conversation_cache_l2.inc",
            "EntryInfo", "l2_stats", "l2_list", "l2_set_pinned", "l2_drop", "l2_touch",
            "l2_last_serial", "l2_set_source",
        )

    def test_build_and_monitor_surface(self):
        self.assert_tokens("CMakeLists.txt", "STRATA_LZ4_LIBRARY", "conversation_file_test")
        self.assert_tokens("serve/web/app.js", "restore_ms", "r.prompt_tokens", "requestTrace", "reqTraceOpen", "l2_read",
                           "requestKeyMetrics", "requestCachePath", "live-continuation", "metricRows", "data-req-expand",
                           "Request · all metrics", "L2 / Gateway · all metrics", "cachePath", "cost-a", "cost-b", "cost-c",
                           "cost-ratio", "cachePct", "cache-low", "value / 1000", "r.restore_ms / 1000")
        self.assert_tokens("serve/web/index.html", "monitor-overview", ">Path</th>", ">Queue</th>", ">Decision</th>",
                           ">A</th>", ">B</th>", ">C</th>", "C / Best", "L2 Read", "L2 Reuse")
        self.assert_tokens("serve/web/app.css", ".monitor-overview", "grid-template-columns: 280px", ".cache-low", ".req-expand")
        self.assert_tokens("serve/web/monitor.js", "detail-cache", "detail-restore", "detail-l2", "l2Facts")
        self.assert_tokens("l2/gateway2.py", "restore-feedback.jsonl", "sched-metrics.jsonl", "engine-cgroup.jsonl",
                           "anc_cov = chain_prefix_length", "X-Strata-Request-Id", "X-Strata-L2-Trace",
                           "/__gw/trace", "STRATA_POOL_BUDGET_MIB", "STRATA_SCHED_DECODE_MS_PER_TOKEN",
                           "timing_source", "restore_cost_policy", "snapshot_prefix_plan", "STRATA_RESTORE_MIN_ABD",
                           "l2_read", "prefix_tokens", "save-commits.jsonl", "save_commit_state",
                            "commit_matches", "wait_l2_commit", "ENGINE_SNAP_DIR",
                            "pool_workset_metrics", "engine_main_metrics", "process_read_bytes",
                            "restore_read_mib_s", "restore_n_read", "copy_bytes",
                            "main_path", "tok_path", "_pool_budget_mib", "GC_DRY_RUN", "_log_gc_plan",
                            "inplace-latest", "_gc_plan_seen", "protected = {e.get(\"file\")",
                            "normalize_inplace_aliases")
        cfg = json.loads(source("strata-iq4_xs.json"))
        self.assertEqual(cfg["env"]["STRATA_V2_READAHEAD"], "4")
        self.assertEqual(cfg["env"]["STRATA_L2_STATE_SINK_TOKENS"], "0")
        args = cfg["user_args"]
        self.assertEqual(args[args.index("--conversation-cache-mib") + 1], "32768")
        

@unittest.skipUnless(hasattr(signal, "SIGUSR1"), "gateway imports SIGUSR1 on the Linux test host")
class GatewayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("l2_gateway_under_test", ROOT / "l2" / "gateway2.py")
        cls.gw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gw)

    def setUp(self):
        names = (
            "SNAP_DIR", "ENGINE_SNAP_DIR", "HOT_DIR", "ARCHIVE_DIR", "INDEX_FILE", "DECISIONS_LOG", "SAVEINC_LOG",
            "SAVE_COMMIT_LOG", "COMMIT_WAIT_S", "PENDING_MAX_S", "last_seen", "_pending_saves",
            "GC_DRY_RUN", "GC_PLAN_LOG",
            "fam_idx", "live_fp", "live_chain", "live_hashes", "live_dirty", "_saveinc_ok",
            "resolve_snap", "snap_path", "engine_req", "wait_l2_commit", "l2_reclaim",
        )
        self.saved = {name: getattr(self.gw, name) for name in names}
        self.gw.fam_idx = {}
        self.gw.last_seen = {}
        self.gw._pending_saves = {}
        self.gw.live_fp = None
        self.gw.live_chain = None
        self.gw.live_hashes = []
        self.gw.live_dirty = False

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(self.gw, name, value)

    def request(self, family="family", content="x" * 4000, max_tokens=16, hashes=None):
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
        return self.gw.SchedRequest(family, "chain", body, list(hashes or []), rid="test-rid")

    def test_probe_parses_parked_inventory_fields(self):
        def fake_engine(method, path, body=None, timeout=60, rid=""):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/cache/probe")
            return 200, {"reusable_tokens": 900, "live_lcp": 800, "prompt_tokens": 1000,
                         "parked_best": 850, "parked_serial": 17, "prefix_tokens": {}}, None
        self.gw.engine_req = fake_engine
        reusable, live, prompt, ms, prefixes, parked, serial = self.gw.engine_probe(
            {"messages": [{"role": "user", "content": "x"}]}, rid="r")
        self.assertEqual((reusable, live, prompt, parked, serial), (900, 800, 1000, 850, 17))
        self.assertEqual(prefixes, {})

    def test_cache_source_lookup_and_preload_pin_protocol(self):
        calls = []
        def fake_engine(method, path, body=None, timeout=60, rid=""):
            calls.append((method, path, body))
            if path == "/v1/cache/list":
                return 200, {"entries": [{"serial": 21, "source": "/tmp/sessions/snap"}]}, None
            if path == "/v1/cache/preload":
                return 200, {"serial": 21, "n_restored": 10, "n_read": 20,
                             "timings": {"restore_ms": 5}, "partial": True}, None
            if path in ("/v1/cache/pin", "/v1/cache/unpin"):
                return 200, {"ok": True}, None
            return 500, None, None
        self.gw.engine_req = fake_engine
        self.assertEqual(self.gw.cache_has_source("snap"), 21)
        ok, body = self.gw.cache_preload("snap", {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "t"}],
        }, rid="r", pin=True)
        self.assertTrue(ok)
        self.assertEqual(body["serial"], 21)
        preload = next(c for c in calls if c[1] == "/v1/cache/preload")
        self.assertEqual(preload[2]["filename"], "snap")
        self.assertTrue(preload[2]["pin"])
        self.assertEqual(preload[2]["tools"], [{"name": "t"}])
        self.gw.release_preload_pin({"preload_serial": 21})
        self.assertTrue(any(c[1] == "/v1/cache/unpin" for c in calls))

    def test_cache_source_ignores_shallow_partial_same_file(self):
        def fake_engine(method, path, body=None, timeout=60, rid=""):
            return 200, {"entries": [
                {"serial": 21, "source": "/tmp/sessions/snap", "tokens": 1000},
                {"serial": 22, "source": "/tmp/sessions/snap", "tokens": 5000},
            ]}, None
        self.gw.engine_req = fake_engine
        self.assertEqual(self.gw.cache_has_source("snap", min_tokens=4000), 22)
        self.assertIsNone(self.gw.cache_has_source("snap", min_tokens=6000))

    def test_restore_metrics_use_actual_bytes_and_engine_timings(self):
        decision = {}
        self.gw._record_restore_metrics(decision, {
            "n_read": 1048576, "n_restored": 1234, "partial": True,
            "timings": {"restore_ms": 1000},
        })
        self.assertEqual(decision["restore_n_read"], 1048576)
        self.assertEqual(decision["restore_n_restored"], 1234)
        self.assertEqual(decision["restore_read_mib_s"], 1.0)
        self.assertTrue(decision["restore_partial"])

    def test_pool_and_engine_main_metrics_are_read_only_views(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"; engine = Path(td) / "engine"
            cold.mkdir(); engine.mkdir()
            self.gw.SNAP_DIR = str(cold)
            self.gw.ENGINE_SNAP_DIR = str(engine)
            Path(cold, "snap.tok").write_text("1000", encoding="utf-8")
            Path(engine, "snap").write_bytes(b"x" * 100)
            Path(engine, "other.l2tmp").write_bytes(b"y" * 20)
            self.gw.fam_idx = {"family": {"chain": {"file": "snap", "tok": 1000, "ts": 1}}}
            self.gw.last_seen = {"family": time.time()}
            workset = self.gw.pool_workset_metrics()
            self.assertEqual(workset["tokens"], 1000)
            self.assertEqual(workset["families"], 1)
            main = self.gw.engine_main_metrics()
            self.assertEqual(main["files"], 2)
            self.assertEqual(main["bytes"], 120)
            self.assertEqual(main["temp_bytes"], 20)

    def test_save_commit_log_preserves_writer_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            self.gw.SAVE_COMMIT_LOG = str(Path(td) / "commits.jsonl")
            job = {"rid": "r", "family": "f", "chain": "c" * 16, "prev": "p" * 16, "gen": 4}
            self.gw._log_save_commit(job, "committed", 25, 100,
                                     rec={"copy_bytes": 500, "copy_ms": 10, "saveinc_ms": 5,
                                          "writer_ms": 16, "growth_bytes": 12, "copy_used": 1})
            row = json.loads(Path(self.gw.SAVE_COMMIT_LOG).read_text(encoding="utf-8"))
            self.assertEqual(row["copy_bytes"], 500)
            self.assertEqual(row["writer_ms"], 16)
            self.assertEqual(row["growth_bytes"], 12)

    def test_restore_waits_for_pending_generation(self):
        self.gw._pending_saves = {("family", "snap"): {"prev": "snap", "gen": 7}}
        calls = []
        def fake_wait(prev, gen, rid="", wait_s=None):
            calls.append((prev, gen))
            return "committed", 12, {"tokens": 100}
        self.gw.wait_l2_commit = fake_wait
        ok, _ = self.gw._wait_restore_file_idle("snap", rid="r")
        self.assertTrue(ok)
        self.assertEqual(calls, [("snap", 7)])

    def test_pool_budget_reads_engine_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "engine.json"
            cfg.write_text(json.dumps({"user_args": ["--conversation-cache-mib", "32768"]}), encoding="utf-8")
            old_env = os.environ.pop("STRATA_POOL_BUDGET_MIB", None)
            old_cfg = os.environ.get("STRATA_ENGINE_CONFIG")
            os.environ["STRATA_ENGINE_CONFIG"] = str(cfg)
            try:
                self.assertEqual(self.gw._pool_budget_mib(), 32768)
            finally:
                if old_env is not None: os.environ["STRATA_POOL_BUDGET_MIB"] = old_env
                if old_cfg is None: os.environ.pop("STRATA_ENGINE_CONFIG", None)
                else: os.environ["STRATA_ENGINE_CONFIG"] = old_cfg

    def test_gc_dry_run_keeps_main_and_logs_plan(self):
        with tempfile.TemporaryDirectory() as td:
            engine = Path(td) / "engine"; cold = Path(td) / "cold"; hot = Path(td) / "hot"
            for d in (engine, cold, hot): d.mkdir()
            self.gw.ENGINE_SNAP_DIR = str(engine); self.gw.SNAP_DIR = str(cold)
            self.gw.HOT_DIR = str(hot); self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.GC_DRY_RUN = True
            self.gw.GC_PLAN_LOG = str(Path(td) / "gc.jsonl")
            self.gw.engine_req = lambda *a, **k: (0, None, None)
            Path(engine, "snap").write_bytes(b"main")
            Path(cold, "snap.tok").write_text("5", encoding="utf-8")
            self.assertFalse(self.gw.l2_reclaim("snap", "test"))
            self.assertTrue(Path(engine, "snap").exists())
            row = json.loads(Path(self.gw.GC_PLAN_LOG).read_text(encoding="utf-8"))
            self.assertTrue(row["dry_run"])
            self.assertEqual(row["target"], "snap")

    def test_inplace_aliases_are_normalized_on_load(self):
        self.gw.INDEX_FILE = str(Path(tempfile.mkdtemp()) / "index.json")
        self.gw.fam_idx = {"f": {
            "a": {"file": "shared", "tok": 10},
            "b": {"file": "shared", "tok": 20},
            "c": {"file": "single", "tok": 30},
        }}
        self.assertEqual(self.gw.normalize_inplace_aliases(), 2)
        self.assertTrue(self.gw.fam_idx["f"]["a"]["inplace"])
        self.assertTrue(self.gw.fam_idx["f"]["b"]["inplace"])
        self.assertNotIn("inplace", self.gw.fam_idx["f"]["c"])

    def test_roll_forward_updates_all_aliases_of_previous_file(self):
        with tempfile.TemporaryDirectory() as td:
            engine = Path(td) / "engine"; cold = Path(td) / "cold"; engine.mkdir(); cold.mkdir()
            self.gw.ENGINE_SNAP_DIR = str(engine); self.gw.SNAP_DIR = str(cold)
            self.gw.HOT_DIR = str(Path(td) / "hot"); self.gw.INDEX_FILE = str(cold / "index.json")
            Path(engine, "prev").write_bytes(b"main")
            self.gw.fam_idx = {"family": {
                "prev": {"file": "prev", "tok": 10, "ts": 1, "hashes": ["a"]},
                "alias": {"file": "prev", "tok": 20, "ts": 2, "hashes": ["a", "b"]},
            }}
            self.gw._advance_snapshot("family", "next", "prev", ["a", "b"], 20, move_prev=True)
            for entry in self.gw.fam_idx["family"].values():
                self.assertEqual(entry["file"], "next")
            self.assertTrue(Path(engine, "next").exists())
            self.assertFalse(Path(engine, "prev").exists())

    def test_family_gc_drops_shared_aliases_without_deleting_main(self):
        self.gw.fam_idx = {"family": {
            "k1": {"file": "shared", "ts": 1, "tok": 10},
            "k2": {"file": "shared", "ts": 2, "tok": 10},
            "k3": {"file": "shared", "ts": 3, "tok": 10},
            "k4": {"file": "shared", "ts": 4, "tok": 10},
            "k5": {"file": "shared", "ts": 5, "tok": 10},
        }}
        calls = []
        self.gw.l2_reclaim = lambda *args, **kwargs: calls.append((args, kwargs)) or True
        self.gw.fam_gc("family")
        self.assertEqual(len(self.gw.fam_idx["family"]), 3)
        self.assertEqual(calls, [], "shared aliases must not delete the physical main")

    def test_semantic_generation_eviction_removes_all_family_refs(self):
        self.gw.fam_idx = {
            "a": {"c1": {"file": "bad"}, "c2": {"file": "good"}},
            "b": {"c3": {"file": "bad"}},
        }
        self.gw.INDEX_FILE = str(Path(tempfile.mkdtemp()) / "index.json")
        self.gw._evict_generation("bad", "test")
        self.assertEqual(self.gw.fam_idx["a"], {"c2": {"file": "good"}})
        self.assertNotIn("bad", str(self.gw.fam_idx))

    def test_hot_ensure_uses_engine_main_root(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"; engine = Path(td) / "engine"; hot = Path(td) / "hot"
            archive = cold / ".state-archive"
            for d in (cold, engine, hot, archive): d.mkdir()
            self.gw.SNAP_DIR = str(cold); self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot); self.gw.ARCHIVE_DIR = str(archive)
            Path(engine, "snap").write_bytes(b"main")
            Path(archive, "snap.state").write_bytes(b"mini")
            self.assertTrue(self.gw.hot_ensure("snap"))
            self.assertEqual(Path(hot, "snap.state").read_bytes(), b"mini")

    def test_family_ignores_first_user_and_chain_tracks_content(self):
        body = {
            "tools": [{"name": "tool"}],
            "messages": [
                {"role": "system", "content": "stable system"},
                {"role": "user", "content": "first version"},
            ],
        }
        changed = {**body, "messages": [body["messages"][0], {"role": "user", "content": "edited"}]}
        self.assertEqual(self.gw.fingerprint(body), self.gw.fingerprint(changed))
        other = {**body, "messages": [{"role": "system", "content": "other"}] + body["messages"][1:]}
        self.assertNotEqual(self.gw.fingerprint(body), self.gw.fingerprint(other))
        hashes = self.gw.chain_hashes(body["messages"])
        self.assertNotEqual(self.gw.chain_id(hashes), self.gw.chain_id(hashes + ["next"]))

    def test_lineage_prefers_deepest_common_prefix_not_token_count(self):
        incoming = ["system", "u1", "a1", "u2"]
        self.gw.fam_idx = {
            "family": {
                "shallow-big": {"file": "shallow", "tok": 999999, "ts": 9, "hashes": ["system"]},
                "deep-old": {"file": "deep", "tok": 10, "ts": 1, "hashes": ["system", "u1", "a1"]},
            }
        }
        self.assertEqual(self.gw.resolve_snap("family", "new-chain", incoming), "deep")
        self.assertEqual(self.gw.resolve_snap("family", "deep-old", incoming), "deep")

    def test_divergent_deep_ancestor_does_not_shield_current_branch(self):
        current = ["system", "u1"] + [f"current-{i}" for i in range(18)]
        divergent = ["system", "u1"] + [f"other-{i}" for i in range(22)]
        common = self.gw.chain_prefix_length(divergent, current)
        self.assertEqual(common, 2)
        self.assertGreater(len(current), common + 2,
                           "a deep divergent branch must not suppress the current branch's bootstrap save")
        self.assertLessEqual(len(current), len(divergent) + 2,
                             "the former total-depth comparison would incorrectly skip this save")

    def test_cost_model_and_scheduler_priorities(self):
        costs = self.gw.layer_costs(10000, 10000, 0, 0)
        self.assertEqual(costs["A"], 0)
        self.assertEqual(set(costs), {"A", "B", "C", "D"})
        fast = self.request(max_tokens=1)
        slow = self.request(max_tokens=100)
        self.assertLess(fast.est_cost, slow.est_cost)
        now = time.time()
        slow.enqueue_ts = now - 2
        os.environ["STRATA_SCHED_AGING"] = "1000"
        try:
            self.assertLess(slow.priority(now), slow.est_cost)
        finally:
            os.environ.pop("STRATA_SCHED_AGING", None)
        hashes = ["system", "u1"]
        self.gw.live_fp = "family"
        self.gw.live_hashes = hashes
        continuation = self.request(hashes=hashes + ["a1"])
        self.assertEqual(continuation.est_cost, -1e9)
        self.assertTrue(continuation.is_live)
        continuation.enqueue_ts = now - 1000
        os.environ["STRATA_SCHED_STARVE_S"] = "0"
        try:
            self.assertEqual(continuation.priority(now), -1e9)
            self.assertFalse(continuation.is_starved, "live must not be misclassified as starve")
        finally:
            os.environ.pop("STRATA_SCHED_STARVE_S", None)
        slow.enqueue_ts = now - 1000
        os.environ["STRATA_SCHED_STARVE_S"] = "0"
        try:
            self.assertEqual(slow.priority(now), -1e10)
            self.assertTrue(slow.is_starved)
        finally:
            os.environ.pop("STRATA_SCHED_STARVE_S", None)
        self.assertEqual(slow.rid, "test-rid")
        self.assertGreater(slow.est_prompt, 0)
        self.assertGreater(slow.decode_est, 0)

    def test_p1_min_abd_policy_and_read_reuse_cost_separation(self):
        costs = {"A": 13000, "B": 6554, "C": 897, "D": 13469}
        old, new, best = self.gw.restore_cost_policy(costs, 17000, 29286)
        self.assertFalse(old, "the Y30-r2 case was rejected by the old PARKED-50% gate")
        self.assertTrue(new)
        self.assertEqual(best, 6554)
        small_borrow = self.gw.layer_costs(29224, 0, 0, 29203, l2_read=277000)
        self.assertGreater(small_borrow["C"], 5800)
        self.assertGreater(small_borrow["C"], self.gw.layer_costs(29224, 0, 0, 29203)["C"])
        no_reuse = self.gw.layer_costs(29224, 0, 0, 0, l2_read=277000)
        self.assertGreater(no_reuse["C"], no_reuse["D"])

    def test_snapshot_prefix_plan_uses_only_exact_or_verified_watermarks(self):
        self.gw.fam_idx = {"family": {
            "deep": {"file": "deep", "tok": 1000, "hashes": ["s", "u", "a", "divergent"],
                     "prefix_tokens": {"3": 123}},
            "deep-uncached": {"file": "deep-uncached", "tok": 1000, "hashes": ["s", "u", "a", "other"]},
            "boundary": {"file": "boundary", "tok": 120, "hashes": ["s", "u", "a"]},
        }}
        entry, common, reuse = self.gw.snapshot_prefix_plan("family", "deep", ["s", "u", "a", "new"])
        self.assertEqual(entry["file"], "deep")
        self.assertEqual((common, reuse), (3, 123))
        entry, common, reuse = self.gw.snapshot_prefix_plan("family", "deep-uncached", ["s", "u", "a", "different"])
        self.assertEqual((common, reuse), (3, 120))
        entry, common, reuse = self.gw.snapshot_prefix_plan("family", "boundary", ["s", "u", "a", "new"])
        self.assertEqual((common, reuse), (3, 120))

    def test_commit_match_uses_chain_basename_and_generation(self):
        self.assertTrue(self.gw.commit_matches(
            {"path": "/tmp/sessions/prev", "gen": 3}, "prev", 3))
        self.assertTrue(self.gw.commit_matches(
            {"path": "/tmp/hot/prev", "gen": 3}, "prev", 3))
        self.assertFalse(self.gw.commit_matches(
            {"path": "/tmp/sessions/prev", "gen": 2}, "prev", 3))
        self.assertFalse(self.gw.commit_matches(
            {"path": "/tmp/sessions/other", "gen": 3}, "prev", 3))

    def test_pending_saveinc_finalizes_without_sync_rescue(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"
            hot = Path(td) / "hot"
            cold.mkdir(); hot.mkdir()
            engine = Path(td) / "engine"
            engine.mkdir()
            self.gw.SNAP_DIR = str(cold)
            self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot)
            self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.INDEX_FILE = str(cold / "index.json")
            self.gw.SAVEINC_LOG = str(Path(td) / "saveinc.jsonl")
            self.gw.SAVE_COMMIT_LOG = str(Path(td) / "commits.jsonl")
            prev = "prev-chain"
            (engine / prev).write_bytes(b"previous")
            self.gw.fam_idx = {"family": {prev: {"file": prev, "ts": 1, "tok": 5,
                                                  "hashes": ["system"]}}}
            self.gw._saveinc_ok = True
            self.gw.COMMIT_WAIT_S = 1.0
            calls = []
            def fake_engine(method, path, _body=None, timeout=60, rid=""):
                calls.append((method, path))
                if method == "POST" and "action=saveinc" in path:
                    return 200, {"incremental": True, "gen": 9, "async_pending": 1,
                                 "n_saved": 5, "n_written": 12,
                                 "timings": {"save_ms": 5}}, None
                if method == "GET" and path == "/slots":
                    return 200, [{"l2_commits": [{
                        "path": "/tmp/sessions/" + prev,
                        "gen": 9, "status": 1, "tokens": 5}]}], None
                return 0, None, None
            self.gw.engine_req = fake_engine
            decision = {}
            self.assertTrue(self.gw.do_save("family", "next-chain", ["system", "u1"],
                                            rid="rid-async", decision=decision))
            self.assertEqual(decision["save_commit_state"], "pending")
            deadline = time.time() + 3
            while time.time() < deadline and not ((engine / "next-chain").exists()
                                                 and Path(self.gw.SAVE_COMMIT_LOG).exists()):
                time.sleep(0.02)
            self.assertTrue((engine / "next-chain").exists())
            self.assertIn("next-chain", self.gw.fam_idx["family"])
            self.assertFalse(any("action=save" in p and "saveinc" not in p
                                 for _, p in calls), "a pending commit must not trigger full rescue")
            self.assertTrue(Path(self.gw.SAVE_COMMIT_LOG).exists())

    def test_pending_commit_survives_first_wait_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"
            engine = Path(td) / "engine"
            hot = Path(td) / "hot"
            cold.mkdir(); engine.mkdir(); hot.mkdir()
            self.gw.SNAP_DIR = str(cold)
            self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot)
            self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.INDEX_FILE = str(cold / "index.json")
            self.gw.SAVEINC_LOG = str(Path(td) / "saveinc.jsonl")
            self.gw.SAVE_COMMIT_LOG = str(Path(td) / "commits.jsonl")
            self.gw.COMMIT_WAIT_S = 0.05
            self.gw.PENDING_MAX_S = 2.0
            prev = "prev-timeout"
            (engine / prev).write_bytes(b"previous")
            self.gw.fam_idx = {"family": {prev: {"file": prev, "ts": 1, "tok": 5,
                                                  "hashes": ["system"]}}}
            self.gw._saveinc_ok = True
            calls = {"post": 0, "get": 0}
            def fake_engine(method, path, _body=None, timeout=60, rid=""):
                if method == "POST" and "action=saveinc" in path:
                    calls["post"] += 1
                    return 200, {"incremental": True, "gen": 3, "async_pending": 1,
                                 "n_saved": 5, "n_written": 12,
                                 "timings": {"save_ms": 5}}, None
                if method == "GET" and path == "/slots":
                    calls["get"] += 1
                    status = 0 if calls["get"] <= 4 else 1
                    return 200, [{"l2_commits": [{
                        "path": str(engine / prev), "gen": 3,
                        "status": status, "tokens": 5}]}], None
                return 0, None, None
            self.gw.engine_req = fake_engine
            self.assertTrue(self.gw.do_save("family", "next-timeout", ["system", "u1"],
                                            rid="rid-timeout"))
            # A duplicate request for the same target must not enqueue another save.
            self.assertTrue(self.gw.do_save("family", "next-timeout-2", ["system", "u1"],
                                            rid="rid-timeout-2"))
            self.assertEqual(calls["post"], 1)
            deadline = time.time() + 4
            while time.time() < deadline and not (engine / "next-timeout").exists():
                time.sleep(0.02)
            self.assertTrue((engine / "next-timeout").exists())
            rows = [json.loads(line) for line in Path(self.gw.SAVE_COMMIT_LOG).read_text().splitlines()]
            self.assertTrue(any(r["state"] == "committed" for r in rows))

    def test_saveinc_full_fallback_keeps_ancestor_as_published_file(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"
            engine = Path(td) / "engine"
            hot = Path(td) / "hot"
            cold.mkdir(); engine.mkdir(); hot.mkdir()
            self.gw.SNAP_DIR = str(cold)
            self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot)
            self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.INDEX_FILE = str(cold / "index.json")
            self.gw.SAVEINC_LOG = str(Path(td) / "saveinc.jsonl")
            self.gw.SAVE_COMMIT_LOG = str(Path(td) / "commits.jsonl")
            prev = "prev-fallback"
            (engine / prev).write_bytes(b"ancestor")
            (cold / (prev + ".tok")).write_text("5", encoding="utf-8")
            self.gw.fam_idx = {"family": {prev: {"file": prev, "ts": 1, "tok": 5,
                                                  "hashes": ["system"]}}}
            self.gw._saveinc_ok = True
            def fake_engine(method, path, _body=None, timeout=60, rid=""):
                self.assertEqual(method, "POST")
                self.assertIn("action=saveinc", path)
                # No "incremental": the engine declined saveinc and saved in full
                # under the requested ancestor filename.
                return 200, {"filename": prev, "n_saved": 7,
                             "n_written": 123456789,
                             "timings": {"save_ms": 9}}, None
            self.gw.engine_req = fake_engine
            decision = {}
            self.assertTrue(self.gw.do_save("family", "next-fallback",
                                            ["system", "u1"], decision=decision))
            self.assertTrue(decision["save_fallback"])
            self.assertFalse(decision["save_incremental"])
            self.assertEqual(decision["save_published_file"], prev)
            self.assertTrue((engine / prev).exists())
            self.assertFalse((engine / "next-fallback").exists())
            self.assertTrue((cold / (prev + ".tok")).exists())
            self.assertFalse((cold / "next-fallback.tok").exists())
            self.assertEqual(self.gw.fam_idx["family"]["next-fallback"]["file"], prev)
            self.assertNotIn(prev, self.gw.fam_idx["family"])

    def test_save_trace_contains_rid_action_gen_and_write_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"
            hot = Path(td) / "hot"
            cold.mkdir(); hot.mkdir()
            self.gw.SNAP_DIR = str(cold)
            self.gw.HOT_DIR = str(hot)
            self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.INDEX_FILE = str(cold / "index.json")
            self.gw.SAVEINC_LOG = str(Path(td) / "saveinc.jsonl")
            (cold / "prev").write_bytes(b"previous")
            self.gw.fam_idx = {"family": {"prev": {"file": "prev", "ts": 1, "tok": 5, "hashes": ["system"]}}}
            self.gw._saveinc_ok = True
            calls = []
            def save_reply(_method, _path, _body, timeout=60, rid=""):
                calls.append(rid)
                return 200, {"incremental": True, "gen": 7, "async_pending": 0,
                             "n_saved": 12, "n_written": 34, "timings": {"save_ms": 5}}, None
            self.gw.engine_req = save_reply
            decision = {}
            self.assertTrue(self.gw.do_save("family", "next", ["system", "u1"],
                                            rid="rid-save", decision=decision))
            self.assertEqual(calls, ["rid-save"])
            self.assertEqual(decision["save_action"], "saveinc")
            self.assertEqual(decision["save_gen"], 7)
            self.assertEqual(decision["save_n_written"], 34)
            self.assertEqual(decision["save_async_pending"], 0)

    def test_gate_single_flight_sjf_and_one_admission_per_request(self):
        with tempfile.TemporaryDirectory() as td:
            self.gw.DECISIONS_LOG = str(Path(td) / "decisions.jsonl")
            scheduler = self.gw.Scheduler()
            slow = self.request(max_tokens=100)
            fast = self.request(max_tokens=1)
            slow.est_cost = 5000
            fast.est_cost = 100
            scheduler.queue.extend([slow, fast])
            warm_calls = []
            scheduler._warm_next = lambda: warm_calls.append(time.time())
            thread = threading.Thread(target=scheduler.gate_loop, daemon=True)
            thread.start()
            try:
                self.assertTrue(fast.turn_event.wait(3), "SJF did not admit the cheaper request")
                self.assertFalse(slow.turn_event.is_set(), "single-flight gate admitted two requests")
                fast.done_event.set()
                self.assertTrue(slow.turn_event.wait(3), "second request remained blocked")
                slow.done_event.set()
                deadline = time.time() + 3
                while time.time() < deadline and (scheduler.busy or scheduler.queue):
                    time.sleep(0.01)
                self.assertFalse(scheduler.busy)
                self.assertEqual(scheduler.stats["reordered"], 1)
                self.assertEqual(len(warm_calls), 2, "an admission was duplicated")
                rows = [json.loads(line) for line in Path(self.gw.DECISIONS_LOG).read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(row["rid"], "test-rid")
                    for key in ("est_prompt", "cover", "prefill_est", "restore_est", "decode_est", "sched_cost"):
                        self.assertIn(key, row)
            finally:
                slow.done_event.set()
                fast.done_event.set()

    def test_gate_live_has_distinct_action_and_stat(self):
        with tempfile.TemporaryDirectory() as td:
            self.gw.DECISIONS_LOG = str(Path(td) / "decisions.jsonl")
            self.gw.live_fp = "family"
            self.gw.live_hashes = ["system", "u1"]
            live = self.request(hashes=["system", "u1", "a1"])
            scheduler = self.gw.Scheduler()
            scheduler.queue.append(live)
            scheduler._warm_next = lambda: None
            thread = threading.Thread(target=scheduler.gate_loop, daemon=True)
            thread.start()
            try:
                self.assertTrue(live.turn_event.wait(3))
                self.assertEqual(live.sched_action, "live_continuation")
                self.assertEqual(scheduler.stats["live_admitted"], 1)
                self.assertEqual(scheduler.stats["starved_rescue"], 0)
                row = json.loads(Path(self.gw.DECISIONS_LOG).read_text().splitlines()[0])
                self.assertTrue(row["is_live"])
                self.assertFalse(row["is_starved"])
            finally:
                live.done_event.set()

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "requires posix_fadvise")
    def test_warm_next_uses_local_snapshot_without_engine(self):
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "snapshot"
            snap.write_bytes(b"test")
            self.gw.resolve_snap = lambda *_args: "snapshot"
            self.gw.snap_path = lambda _name: str(snap)
            scheduler = self.gw.Scheduler()
            scheduler.queue.append(self.request())
            scheduler._warm_next()  # Must not raise or call the engine.

    def test_boot_scan_preserves_orphan_and_removes_partial_temp(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"; engine = Path(td) / "engine"; hot = Path(td) / "hot"
            for d in (cold, engine, hot): d.mkdir()
            self.gw.SNAP_DIR = str(cold); self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot); self.gw.ARCHIVE_DIR = str(cold / ".state-archive")
            self.gw.INDEX_FILE = str(cold / "index.json")
            orphan = "a" * 16
            (engine / orphan).write_bytes(b"orphan")
            (engine / ("b" * 16)).write_bytes(b"main")
            (engine / ("b" * 16 + ".l2tmp")).write_bytes(b"partial")
            (engine / ("c" * 16 + ".rescue")).write_bytes(b"rescued")
            self.gw.fam_idx = {"dead": {"chain": {"file": "missing"}}}
            self.gw.l2_boot_scan()
            self.assertTrue((engine / orphan).exists())
            self.assertTrue((cold / (orphan + ".tok")).exists())
            self.assertFalse((engine / ("b" * 16 + ".l2tmp")).exists())
            self.assertTrue((engine / ("c" * 16)).exists())
            self.assertFalse(self.gw.fam_idx["dead"])

    def test_reclaim_protects_unarchived_state_then_removes_whole_group(self):
        with tempfile.TemporaryDirectory() as td:
            cold = Path(td) / "cold"; engine = Path(td) / "engine"
            hot = Path(td) / "hot"; archive = cold / ".state-archive"
            for d in (cold, engine, hot, archive): d.mkdir()
            self.gw.SNAP_DIR = str(cold); self.gw.ENGINE_SNAP_DIR = str(engine)
            self.gw.HOT_DIR = str(hot); self.gw.ARCHIVE_DIR = str(archive)
            self.gw.INDEX_FILE = str(cold / "index.json")
            self.gw.GC_DRY_RUN = False
            self.gw.engine_req = lambda *_args, **_kwargs: (0, None, None)
            name = "d" * 16
            main = engine / name
            mini = hot / (name + ".state")
            main.write_bytes(b"main")
            mini.write_bytes(b"mini")
            self.gw.fam_idx = {"family": {name: {"file": name}}}
            self.assertFalse(self.gw.l2_reclaim(name, "test"))
            self.assertTrue(main.exists())
            marker = Path(self.gw._archived_marker(name + ".state"))
            marker.write_text("archived", encoding="utf-8")
            (archive / (name + ".state")).write_bytes(b"backup")
            self.assertTrue(self.gw.l2_reclaim(name, "test"))
            self.assertFalse(main.exists())
            self.assertFalse(mini.exists())
            self.assertFalse((archive / (name + ".state")).exists())
            self.assertFalse(self.gw.fam_idx["family"])


if __name__ == "__main__":
    unittest.main()
