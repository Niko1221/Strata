"""Deterministic branching, retention, tier failure and eviction contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serve.branch_store import BranchStore
from serve.checkpoint_store import CheckpointStore, CacheFull
from serve.responses import ResponsesError, new_id


class History(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = BranchStore(self.tmp.name, execution_identity={"model": "weights-a", "mtp": 4})

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def record(self, req=None):
        resolved, context = self.store.resolve(req or {"input": "hello"}, "m")
        response = dict(id=new_id("resp"), status="in_progress", model="m", output=[])
        self.store.begin(response, context)
        self.store.record_prompt(response["id"], [1, 2, 3])
        response.update(status="completed", output=[dict(id=new_id("msg"), role="assistant", type="message",
                                                       content=[dict(type="output_text", text="recorded answer")])])
        self.store.record_generated(response["id"], [4, 5])
        self.store.finish(response)
        return response, resolved

    def test_four_boundaries_are_two_adjacent_canonical_prefixes(self):
        response, _ = self.record()
        nodes = self.store.boundaries(response["id"])["nodes"]
        self.assertEqual(nodes[0]["after"], nodes[1]["before"])
        for index, side, length in ((0,"before",0),(0,"after",1),(1,"before",1),(1,"after",2)):
            req, context = self.store.resolve({"input": [], "branch_from": dict(
                response_id=response["id"], node_id=nodes[index]["id"], side=side)}, "m")
            self.assertEqual(len(req["input"]), length)
            self.assertEqual(context["base"], nodes[index][side]["prefix_tip_node_id"])

    def test_bookmarked_ancestry_survives_cleanup_but_other_history_replays(self):
        root, _ = self.record()
        child, _ = self.record({"input":"protected child", "previous_response_id":root["id"]})
        ordinary, _ = self.record({"input":"ordinary"})
        self.store.bookmark(child["id"], True)
        self.assertEqual(self.store.protected_checkpoint_owners(), {root["id"],child["id"]})
        cache = CheckpointStore(Path(self.tmp.name)/'cache', budget_bytes=1<<20, reserve_bytes=0)
        try:
            source=Path(self.tmp.name)/'native-session'
            source.write_bytes(b'a'*4096)
            kept=cache.admit(source,'f',[1],restore_s=.1)
            cache.attach(kept,root['id'])
            source.write_bytes(b'b'*4096)
            dropped=cache.admit(source,'f',[2],restore_s=.1)
            cache.attach(dropped,ordinary['id'])
            self.store.on_shutdown=cache.restart_cleanup
            self.store.close()
            self.store=BranchStore(self.tmp.name,execution_identity={"model":"weights-a","mtp":4})
            self.assertEqual([s['id'] for s in cache._rows()],[kept])
            request,_=self.store.resolve({'input':'continue','previous_response_id':ordinary['id']},'m')
            self.assertIn('recorded answer',json.dumps(request))
        finally:
            cache.close()

    def test_branch_does_not_mutate_parent_or_replay_tools(self):
        r, _ = self.record({"input": [{"type":"function_call", "call_id":"x", "name":"delete_everything", "arguments":"{}"},
                                     {"type":"function_call_output", "call_id":"x", "output":"already ran"}]})
        original = self.store.boundaries(r["id"])
        child, resolved = self.record({"input":"continue", "previous_response_id":r["id"]})
        self.assertIn("already ran", json.dumps(resolved))
        self.assertEqual(self.store.boundaries(r["id"]), original)
        self.assertGreater(len(self.store.boundaries(child["id"])["nodes"]), len(original["nodes"]))

    def test_fingerprint_change_requires_explicit_new_branch(self):
        r, _ = self.record()
        self.store.close()
        self.store = BranchStore(self.tmp.name, execution_identity={"model":"weights-b", "mtp":0})
        with self.assertRaises(ResponsesError) as error:
            self.store.resolve({"input":"next", "previous_response_id":r["id"]}, "m")
        self.assertEqual(error.exception.code, "history_migration_required")
        child, _ = self.record({"input":"next", "previous_response_id":r["id"], "migrate_history":True})
        self.assertNotEqual(child["id"], r["id"])
        migrated,_=self.store.resolve({"input":"next", "previous_response_id":r["id"],
                                      "migrate_history":True},"another-model")
        self.assertIn("recorded answer",json.dumps(migrated))

    def test_settings_bound_and_embedded_assets_survive_restart(self):
        self.store.asset_loader = lambda url: "data:image/png;base64,YQ=="
        r, _ = self.record({"instructions":"fixed framing", "input":[{"role":"user", "content":[
            {"type":"input_image", "image_url":"https://example.invalid/image"}]}]})
        self.store.close()
        self.store = BranchStore(self.tmp.name, execution_identity={"model":"weights-a", "mtp":4})
        req, _ = self.store.resolve({"input":"more", "previous_response_id":r["id"]}, "m")
        self.assertIn("data:image/png", json.dumps(req))
        self.assertNotIn("https://example.invalid", json.dumps(req))
        self.assertEqual(req["instructions"], "fixed framing")
        with self.assertRaises(ResponsesError):
            self.store.resolve({"input":"more", "previous_response_id":r["id"], "instructions":"changed"}, "m")

    def test_retention_activity_bookmark_and_shared_ancestry(self):
        self.store.retention_s = 30
        with patch("time.time", return_value=100):
            root, _ = self.record()
            protected, _ = self.record({"input":"keep", "previous_response_id":root["id"]})
            self.store.bookmark(protected["id"], True)
            cold, _ = self.record({"input":"cold", "previous_response_id":root["id"]})
        with patch("time.time", return_value=129):
            self.store.get(cold["id"])  # inspections must not renew activity
            self.record({"input":"fresh", "previous_response_id":root["id"]})
        with patch("time.time", return_value=131):
            with self.assertRaises(ResponsesError):
                self.store.get(cold["id"])
            self.store.get(root["id"])
        with patch("time.time", return_value=1000):
            self.store.get(protected["id"])
            self.assertIn("hello", json.dumps(self.store.boundaries(protected["id"])))
            self.store.bookmark(protected["id"], False)
            with self.assertRaises(ResponsesError):
                self.store.get(protected["id"])
            self.assertEqual(self.store.db.execute("SELECT count(*) FROM history_nodes").fetchone()[0], 0)

    def test_failed_finish_rolls_back_nodes_and_parent_activity(self):
        r, _ = self.record()
        req, ctx = self.store.resolve({"input":"x", "previous_response_id":r["id"]}, "m")
        child = dict(id=new_id("resp"), status="in_progress", model="m", output=[])
        self.store.begin(child,ctx)
        before = self.store.boundaries(child["id"])
        child.update(status="completed", output=[{"id":"huge", "content":"x"*10000}])
        self.store.max_bytes = 1024
        with self.assertRaises(ResponsesError):
            self.store.finish(child)
        self.assertEqual(self.store.boundaries(child["id"]), before)
        self.assertEqual(self.store.get(child["id"])["status"], "in_progress")

    def test_branch_input_pagination_excludes_removed_tail_and_allows_its_ids(self):
        r,_=self.record()
        nodes=self.store.boundaries(r["id"])["nodes"]
        reused_id=nodes[-1]["item"]["id"]
        child,_=self.record({"branch_from":{"response_id":r["id"],"node_id":nodes[0]["id"],"side":"after"},
            "input":[{"id":reused_id,"role":"user","content":"replacement"}]})
        page=self.store.input_items(child["id"],{"order":["asc"]})
        self.assertEqual(len(page["data"]),2)
        self.assertNotIn("recorded answer",json.dumps(page))


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/"archive").mkdir()
        self.store = CheckpointStore(self.root/"local", budget_bytes=1<<20, reserve_bytes=0,
            archive=self.root/"archive", archive_budget_bytes=1<<20, chunk_bytes=4096,
            free_bytes=lambda p: 1<<30)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self, data=b"a"*8192, tokens=None, fingerprint="f", pinned=False):
        source = self.root/"source"
        source.write_bytes(data)
        return self.store.admit(source, fingerprint, tokens or [1,2,3], restore_s=.1, pinned=pinned)

    def test_group_reclaims_bytes_that_individuals_cannot(self):
        a,b = self.add(tokens=[1]), self.add(tokens=[2])
        rows = self.store._rows()
        self.assertEqual(self.store._freed({a},rows,"local"),0)
        self.assertEqual(self.store._freed({b},rows,"local"),0)
        self.assertGreater(self.store._freed({a,b},rows,"local"),0)
        self.assertIn(sorted([a,b]), [c["ids"] for c in self.store.candidates()])

    def test_old_popular_prefix_survives_over_new_oneoff(self):
        a = self.add(tokens=[1])
        b = self.add(b"b"*8192, tokens=[2])
        for _ in range(20):
            self.store.observe("f",[1,9],10)
        self.store.observe("f",[2,9],10)
        choice = self.store.candidates()[0]
        self.assertEqual(choice["ids"],[b])
        self.assertNotEqual(choice["ids"],[a])

    def test_best_accounts_for_restore_overhead_and_identity(self):
        a = self.add(tokens=[1,2])
        self.add(b"c"*8192, tokens=[1,2,3], fingerprint="wrong")
        row,cost = self.store.best("f",[1,2,3],3)
        self.assertEqual(row["id"],a)
        self.assertAlmostEqual(cost,1.1)
        self.store.restored(a,5)
        self.assertIsNone(self.store.best("f",[1,2,3],3)[0])

    def test_pinned_pressure_preserves_history_reserve(self):
        a = self.add(pinned=True)
        self.store.reserve=100
        self.store.free_bytes=lambda p: 105
        with self.assertRaises(CacheFull):
            self.store.enforce(incoming=10)
        self.assertEqual(self.store._rows()[0]["id"],a)

    def test_verified_archive_roundtrip_and_outage(self):
        a = self.add()
        self.store.demote([a])
        self.assertEqual(self.store.used(),0)
        with self.store.materialize(a) as path:
            self.assertEqual(path.read_bytes(),b"a"*8192)
            self.assertFalse(self.store.candidates("archive"))
        archive=self.store.paths["archive"]
        archive.rename(archive.with_name("offline"))
        self.assertIsNone(self.store.best("f",[1,2,3],10)[0])

    def test_failed_archive_copy_keeps_local_source(self):
        a=self.add()
        with patch.object(self.store,"_copy_block",side_effect=OSError("pool offline")):
            with self.assertRaises(OSError):
                self.store.demote([a])
        with self.store.materialize(a) as path:
            self.assertEqual(path.read_bytes(),b"a"*8192)

    def test_corrupt_block_is_not_restored_and_lease_released(self):
        a=self.add()
        block=self.store._rows()[0]["blocks"][0][0]
        (self.store.paths["local"]/block).write_bytes(b"corrupt")
        with self.assertRaises(OSError):
            with self.store.materialize(a):
                self.fail("must not expose corrupt state to the engine")
        self.assertFalse(self.store.leases)

    def test_catalog_reopen_recovers_statistics_before_restart_cleanup(self):
        self.add()
        self.store.observe("f",[1,2,3],10)
        self.store.observe("f",[1,2,3],10,successful=False)
        before=self.store.db.execute("SELECT * FROM demand").fetchall()
        self.store.close()
        self.store=CheckpointStore(self.root/"local",budget_bytes=1<<20,reserve_bytes=0,
                                  archive=self.root/"archive",archive_budget_bytes=1<<20)
        self.assertEqual(self.store.db.execute("SELECT * FROM demand").fetchall(),before)

    def test_restart_cleanup_keeps_bookmarked_shared_blocks_only_and_resets_stats(self):
        kept=self.add(tokens=[1])
        shared=self.add(tokens=[2])
        self.store.attach(kept,'protected')
        self.store.attach(shared,'ordinary')
        transient=self.add(b'b'*8192,tokens=[3],pinned=True)
        self.store.attach(transient,'ordinary')
        self.store.observe('f',[1],10)
        self.store.restored(kept,.1)
        self.store.restart_cleanup({'protected'})
        self.assertEqual([r['id'] for r in self.store._rows()],[kept])
        self.assertEqual(self.store.used(),4096)  # repeated chunks occupy one physical block
        for table in ('demand','successful_restores','decisions'):
            self.assertEqual(self.store.db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        with self.store.materialize(kept) as p:
            self.assertEqual(p.read_bytes(),b'a'*8192)
        self.store.restart_cleanup(set())
        self.assertEqual(self.store.used(),0)

    def test_dirty_boot_removes_unprotected_and_abandoned_staging_files(self):
        import uuid
        self.add()
        tmp=self.store.root/(uuid.uuid4().hex+'.save.tmp')
        tmp.write_bytes(b'interrupted save')
        self.store.close()  # catalog alone deliberately omits graceful owner cleanup
        self.store=CheckpointStore(self.root/'local',budget_bytes=1<<20,reserve_bytes=0)
        self.assertFalse(tmp.exists())
        self.assertTrue(self.store._rows())
        self.store.restart_cleanup(set())
        self.assertEqual(self.store._rows(),[])
        self.assertEqual(self.store.used(),0)

    def test_restart_cleanup_rejects_active_restore(self):
        sid=self.add()
        with self.store.materialize(sid):
            with self.assertRaises(RuntimeError):
                self.store.restart_cleanup(set())
        self.assertTrue(self.store._rows())

    def test_dependency_group_matches_exhaustive_two_snapshot_oracle(self):
        import itertools
        self.store.clock=lambda:100
        a,b=self.add(tokens=[1]),self.add(tokens=[2])
        for _ in range(20):
            self.store.observe("f",[1],10)
        self.store.observe("f",[2],10)
        rows=self.store._rows()
        queries=self.store._weights()
        exact=[]
        for n in (1,2):
            for group in itertools.combinations([a,b],n):
                freed=self.store._freed(set(group),rows,"local")
                if freed:
                    remaining=[r for r in rows if r["id"] not in group]
                    lost=sum(q["weight"]*(self.store.cost(q,remaining)[0]-self.store.cost(q,rows)[0]) for q in queries)
                    exact.append((lost/freed,sorted(group)))
        selected=self.store.candidates(allow_demote=False)[0]
        self.assertEqual((selected["rank"],selected["ids"]),min(exact))

    def test_deduplication_does_not_evict_or_allocate_another_copy(self):
        a=self.add()
        used=self.store.used()
        self.store.budgets['local']=used
        self.assertEqual(self.add(),a)
        self.assertEqual(self.store.used(),used)

    def test_expiring_one_owner_preserves_shared_checkpoint(self):
        a=self.add()
        self.store.attach(a,'old')
        self.store.attach(a,'kept')
        self.store.expire_owners({'kept'})
        self.assertTrue(self.store._rows())
        self.store.expire_owners(set())
        self.assertEqual(self.store._rows(),[])
        self.assertEqual(self.store.used(),0)

    def test_restart_with_missing_archive_falls_back_to_replay(self):
        sid=self.add()
        self.store.demote([sid])
        self.store.close()
        (self.root/'archive').rename(self.root/'offline')
        self.store=CheckpointStore(self.root/'local',budget_bytes=1<<20,reserve_bytes=0,
                                  archive=self.root/'archive',archive_budget_bytes=1<<20)
        self.assertEqual(self.store.best('f',[1,2,3],10),(None,10))
        self.assertFalse((self.root/'archive').exists())

    def test_two_servers_on_one_pool_cannot_collect_each_others_blocks(self):
        sid=self.add()
        self.store.demote([sid])
        other=CheckpointStore(self.root/'other',budget_bytes=1<<20,reserve_bytes=0,
                              archive=self.root/'archive',archive_budget_bytes=1<<20)
        try:
            self.assertNotEqual(other.paths['archive'],self.store.paths['archive'])
            other.collect_orphans()
            with self.store.materialize(sid) as p:
                self.assertEqual(p.read_bytes(),b'a'*8192)
        finally:
            other.close()



class NativeMetadata(unittest.TestCase):
    def test_indexes_committed_mtp_tail_and_internal_ancestor_without_loading_state(self):
        import struct
        from serve.session_metadata import session_prefixes
        def checkpoint(ids):
            return struct.pack('<Q',len(ids))+struct.pack('<'+'i'*len(ids),*ids)+struct.pack('<Q',0)+bytes(6*8)
        payload=bytes(21*8)+checkpoint([1,2,3,4,5])+struct.pack('<Q',1)+checkpoint([1,2,3])+struct.pack('<Q',0)
        header=b'STRSESS\x01'+struct.pack('<IIQQQQQQ',1,64,0,0,len(payload),0,0,0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'session'
            p.write_bytes(header+payload+bytes(16))
            self.assertEqual(session_prefixes(p,8),[[1,2,3,4,5],[1,2,3]])
            with self.assertRaises(ValueError):
                session_prefixes(p,4)
            p.write_bytes(header[:8]+struct.pack('<I',2)+header[12:]+payload+bytes(16))
            with self.assertRaises(ValueError):
                session_prefixes(p,8)


if __name__ == "__main__":
    unittest.main()
