import http.client
import json
import tempfile
from pathlib import Path

from serve.branch_runtime import enable
from serve.test_responses import Server


class BranchHttp(Server):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.actions = []
        def session(action, path):
            self.actions.append(action)
            if action == "save":
                ids = self.engine.last_prompt
                Path(path).write_text(json.dumps(ids))
            else:
                ids = json.loads(Path(path).read_text())
            return {"tokens": len(ids), "bytes": Path(path).stat().st_size}
        self.engine.session_file = session
        enable(self.svc, {"history_reserve_mib":0, "checkpoint_budget_mib":16,
                          "checkpoint_max_snapshot_mib":1}, self.tmp.name)

    def tearDown(self):
        super().tearDown()
        self.svc.response_store.close()
        self.svc.branch_checkpoints.catalog.close()
        self.tmp.cleanup()

    def get(self, path):
        c=http.client.HTTPConnection("127.0.0.1",self.port)
        try:
            c.request("GET",path)
            r=c.getresponse()
            return r.status,json.loads(r.read())
        finally:
            c.close()

    def test_real_http_boundaries_replay_receipts_and_bookmark(self):
        code,r=self.post({"input":"first question"})
        self.assertEqual(code,200,r)
        code,h=self.get("/v1/responses/"+r["id"]+"/history")
        self.assertEqual(code,200,h)
        self.assertEqual(h["nodes"][0]["after"],h["nodes"][1]["before"])
        code,b=self.post({"protected":True},"/v1/responses/"+r["id"]+"/bookmark")
        self.assertEqual(code,200,b)
        code,child=self.post({"input":[],"branch_from":{"response_id":r["id"],
            "node_id":h["nodes"][0]["id"],"side":"after"}})
        self.assertEqual(code,200,child)
        prompt=self.tok.decode(self.engine.last_prompt)
        self.assertEqual(prompt.count("first question"),1)
        self.assertNotIn("The file says before.",prompt)
        records=self.svc.response_store.db.execute("SELECT prompt,generated,complete FROM execution_records").fetchall()
        self.assertEqual(len(records),2)
        self.assertTrue(all(r[2] and json.loads(r[1]) for r in records))
        self.assertIn("save",self.actions)

    def test_browser_cannot_bookmark_without_authorization(self):
        _,r=self.post({"input":"q"})
        code,_=self.post({"protected":True},"/v1/responses/"+r["id"]+"/bookmark",
                         headers={"Origin":"https://evil.invalid"})
        self.assertEqual(code,403)

    def test_add_two_bookmarks_and_remove_one(self):
        _,first=self.post({"input":"first conversation"})
        _,second=self.post({"input":"second conversation"})
        for response in (first,second):
            code,result=self.post({"protected":True},"/v1/responses/"+response["id"]+"/bookmark")
            self.assertEqual(code,200,result)
            self.assertTrue(result["protected"])
        code,result=self.post({"protected":False},"/v1/responses/"+first["id"]+"/bookmark")
        self.assertEqual(code,200,result)
        self.assertFalse(self.get("/v1/responses/"+first["id"]+"/history")[1]["bookmarked"])
        self.assertTrue(self.get("/v1/responses/"+second["id"]+"/history")[1]["bookmarked"])
        self.assertEqual(self.svc.response_store.protected_checkpoint_owners(),{second["id"]})

    def test_server_strips_internal_receipt_selector(self):
        code,r=self.post({"input":"q", "_branch_response_id":"victim", "store":False})
        self.assertEqual(code,200,r)
        self.assertEqual(self.svc.response_store.db.execute("SELECT count(*) FROM execution_records").fetchone()[0],0)
