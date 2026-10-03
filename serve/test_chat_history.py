"""Disk history, data preservation and authenticated route regressions; no GPU."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

from serve.chat_history import ChatHistory, HistoryError, EMPTY_MEMORY, encoded
from serve.server import ByteTokenizer, MockEngine, Service, make_handler
from serve.frontend import ChatTemplate
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def msg(text, role="user", **extra):
    return {"role": role, "text": text, **extra}


class DiskHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = ChatHistory(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, messages, id="first", **extra):
        return self.repo.save({"id": id, "revision": 0, "total": len(messages), "start": 0, "messages": messages,
                              "memory": EMPTY_MEMORY, "draft": "", **extra})

    def test_catalog_reads_metadata_only_and_selected_chat_returns_a_bounded_page(self):
        records = [msg(f"record {i}", "user" if i % 2 == 0 else "assistant", reasoning="trace " * 2000) for i in range(120)]
        self.save(records)
        self.save([msg("other transcript")], id="other")
        with mock.patch("serve.chat_history.unpacked", side_effect=AssertionError("catalog decompressed a transcript")):
            catalog = self.repo.catalog()
        self.assertEqual(catalog["total"], 2)
        self.assertNotIn("messages", catalog["chats"][0])
        self.assertNotIn("state", catalog["chats"][0])
        page = self.repo.page("first")
        self.assertEqual(page["offset"], 80)
        self.assertEqual(page["messages"], records[80:])
        self.assertEqual(self.repo.page("first", 40)["messages"], records[40:80])
        self.assertLess(self.repo.stats()["compressedBytes"], self.repo.stats()["originalBytes"] / 10)

    def test_large_message_pages_are_byte_bounded_and_contiguous(self):
        records = [msg(str(i), reasoning="a" * (1024 * 1024)) for i in range(10)]
        self.save(records)
        recent = self.repo.page("first")
        self.assertEqual(recent["messages"], records[recent["offset"]:])
        self.assertLessEqual(len(encoded(recent["messages"])), 4 * 1024 * 1024)
        older = self.repo.page("first", end=recent["offset"])
        self.assertEqual(older["messages"] + recent["messages"], records[older["offset"]:])
        one = self.repo.page("first", 0, 1, False)
        self.assertNotIn("draft", one)
        self.assertEqual(one["messages"], records[:1])

    def test_draft_update_writes_no_messages_and_tail_patch_preserves_old_records(self):
        records = [msg(f"keep-{i}") for i in range(100)]
        initial = self.save(records)
        revision = initial["chat"]["revision"]
        draft = self.repo.save({"id": "first", "revision": revision, "total": 100, "memory": EMPTY_MEMORY, "draft": "typing"})
        self.assertEqual(draft["writtenMessages"], 0)
        same = self.repo.save({"id": "first", "revision": draft["chat"]["revision"], "total": 100, "memory": EMPTY_MEMORY, "draft": "typing"})
        self.assertEqual(same["chat"]["revision"], draft["chat"]["revision"])
        appended = self.repo.save({"id": "first", "revision": draft["chat"]["revision"], "total": 102, "start": 100,
            "messages": [msg("new"), msg("answer", "assistant")], "memory": EMPTY_MEMORY, "draft": ""})
        self.assertEqual(appended["writtenMessages"], 2)
        self.assertEqual(self.repo.export_book("first")["chats"][0]["messages"][:100], records)
        with self.assertRaises(HistoryError) as error:
            self.repo.save({"id": "first", "revision": revision, "total": 0, "start": 0, "messages": []})
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(self.repo.page("first")["messageCount"], 102)

    def test_replayed_save_survives_a_lost_reply_without_overwriting_later_changes(self):
        body = {"id": "first", "revision": 0, "writeId": "request-1", "start": 0, "total": 1, "messages": [msg("initial")]}
        first = self.repo.save(body)
        self.repo.save({"id": "first", "revision": first["chat"]["revision"], "total": 1, "draft": "later draft"})
        self.assertTrue(self.repo.save(body)["replayed"])
        self.assertEqual(self.repo.page("first")["draft"], "later draft")
        with self.assertRaises(HistoryError):
            self.repo.save({**body, "messages": [msg("changed request")]})

    def test_migration_keeps_exact_backup_active_id_and_distinct_conversations(self):
        records = [msg("original", files=[{"name": "notes.txt", "text": "exact file contents"}], reasoning="original trace")]
        book = {"version": 1, "active": "old-b", "chats": [
            {"id": "old-a", "messages": records, "createdAt": 10, "updatedAt": 20},
            {"id": "old-b", "messages": records, "memory": {"through": 1, "summary": "fact", "count": 1}, "draft": "draft"}]}
        raw = json.dumps(book, ensure_ascii=False).encode()
        result = self.repo.import_book(raw, migrate=True)
        self.assertEqual(result["active"], "old-b")
        self.assertEqual(result["mapped_count"], 2)
        self.assertEqual(result["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(gzip.decompress((Path(self.tmp.name) / "backups" / result["backup"]).read_bytes()), raw)
        self.assertEqual(self.repo.import_book(raw, migrate=True)["added"], 0)
        self.assertEqual(self.repo.catalog()["total"], 2)
        self.assertEqual(self.repo.page("old-b")["draft"], "draft")
        self.assertEqual(self.repo.page("old-a")["createdAt"], 10)

    def test_compressed_backup_restores_every_message_and_attachment(self):
        records = [msg("attachment", files=[{"name": "memo.txt", "text": "complete body"}],
                       images=[{"name": "photo.png", "url": "data:image/png;base64,YWJj"}]),
                   msg("done", "assistant", tools=[{"id": "tool1", "result": "original tool output"}])]
        self.save(records, memory={"through": 1, "summary": "remember", "count": 2}, draft="unfinished",
                  draftAttachments=[{"kind": "file", "name": "draft.txt", "text": "unsent attachment"}])
        raw = encoded(self.repo.export_book())
        with tempfile.TemporaryDirectory() as second:
            restored = ChatHistory(second)
            restored.import_book(gzip.decompress(gzip.compress(raw)))
            page = restored.page("first")
            self.assertEqual(page["messages"], records)
            self.assertEqual(page["memory"], {"through": 1, "summary": "remember", "count": 2})
            self.assertEqual(page["draftAttachments"][0]["text"], "unsent attachment")
            self.assertEqual(restored.import_book(raw)["added"], 0)

    def test_invalid_import_is_atomic_and_external_image_urls_are_rejected(self):
        self.save([msg("keep")])
        invalid = {"version": 1, "chats": [{"id": "valid", "messages": [msg("new")]},
            {"id": "invalid", "messages": [msg("remote", images=[{"name": "x", "url": "https://example.com/private"}])]}]}
        with self.assertRaises(HistoryError):
            self.repo.import_book(encoded(invalid))
        self.assertEqual(self.repo.catalog()["total"], 1)
        self.assertEqual(self.repo.page("first")["messages"][0]["text"], "keep")

    def test_interrupted_draft_recovery_keeps_the_other_tabs_update_and_original_prefix(self):
        records = [msg(f"record-{i}") for i in range(80)]
        first = self.save(records)
        snapshot = {"id": "first", "revision": first["chat"]["revision"], "recoveryId": "last-keystrokes",
            "offset": 40, "messages": records[40:], "memory": {"through": 60, "summary": "preserved memory", "count": 1},
            "draft": "typed just before closing"}
        self.repo.save({"id": "first", "revision": first["chat"]["revision"], "total": 80, "start": 79,
            "messages": [msg("other tab's new answer")]})
        recovered = self.repo.recover(snapshot)
        self.assertTrue(recovered["recoveredSeparately"])
        self.assertEqual(self.repo.page("first")["messages"][-1]["text"], "other tab's new answer")
        restored = self.repo.export_book(recovered["chat"]["id"])["chats"][0]
        self.assertEqual(restored["messages"], records)
        self.assertEqual(restored["draft"], "typed just before closing")
        self.assertEqual(self.repo.recover(snapshot)["chat"]["id"], recovered["chat"]["id"])
        self.assertEqual(self.repo.catalog()["total"], 2)

    def test_recovering_an_already_saved_window_does_not_create_a_duplicate(self):
        first = self.save([msg("keep")], draft="saved draft")
        recovered = self.repo.recover({"id": "first", "revision": 0, "recoveryId": "closed-browser",
            "offset": 0, "messages": [msg("keep")], "memory": EMPTY_MEMORY, "draft": "saved draft"})
        self.assertTrue(recovered["alreadySaved"])
        self.assertEqual(recovered["chat"]["revision"], first["chat"]["revision"])
        self.assertEqual(self.repo.catalog()["total"], 1)

    def test_recall_is_bounded_excludes_recent_active_turns_and_handles_japanese(self):
        self.save([msg("プロジェクトAuroraの保存先は D:\\Aurora\\notes です。", "assistant"),
                   msg("最近の保存先の発言")], id="active")
        self.save([msg("保存先のバックアップを確認しました。", "assistant")], id="archive")
        result = self.repo.recall("Auroraの保存先を教えてください", "active", 1)
        self.assertTrue(any("D:\\Aurora\\notes" in hit["text"] for hit in result["hits"]))
        self.assertFalse(any(hit["chatId"] == "active" and hit["seq"] >= 1 for hit in result["hits"]))
        self.assertLessEqual(len(result["hits"]), 4)
        self.assertLessEqual(result["characters"], 6000)
        self.assertEqual(self.repo.recall('" OR DROP TABLE chats; --', "active", 0)["hits"], [])
        self.assertEqual(self.repo.catalog()["total"], 2)


class HistoryRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "mock", 4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.api_key = "history-test-key"
        self.svc.history_directory = Path(self.tmp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.svc))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.tmp.cleanup()

    def call(self, path, body=None, headers=None):
        h = {"Authorization": "Bearer history-test-key", **({"Content-Type": "application/json"} if body is not None else {}), **(headers or {})}
        req = urllib.request.Request(self.base + path, data=encoded(body) if body is not None and not isinstance(body, bytes) else body, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                raw = response.read()
                return response.status, response.headers, raw
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def test_auth_origin_metadata_and_gzip_export_import(self):
        self.assertEqual(self.call("/api/history", headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.call("/api/history", headers={"Origin": "http://foreign.example"})[0], 403)
        body = {"id": "chat", "revision": 0, "start": 0, "total": 1, "messages": [msg("local-only")], "writeId": "save1"}
        self.assertEqual(self.call("/api/history/chat", body, {"Origin": "http://foreign.example"})[0], 403)
        self.assertEqual(self.call("/api/history/chat", body)[0], 200)
        code, h, raw = self.call("/api/history")
        self.assertEqual(code, 200); self.assertEqual(h["Cache-Control"], "no-store")
        self.assertNotIn("messages", json.loads(raw)["chats"][0])
        code, h, archive = self.call("/api/history/export")
        self.assertEqual(h["Content-Type"], "application/gzip")
        self.assertEqual(json.loads(gzip.decompress(archive))["chats"][0]["messages"][0]["text"], "local-only")
        self.assertEqual(self.call("/api/history/import", archive, {"Content-Type": "application/gzip"})[0], 200)
        self.assertEqual(json.loads(self.call("/api/history")[2])["total"], 1)
        self.assertEqual(self.call("/api/history/chat?id=chat&limit=101")[0], 400)
        self.assertEqual(self.call("/api/history/chat", [])[0], 400)

    def test_gzip_expansion_limit_leaves_saved_history_intact(self):
        archive = gzip.compress(b"x" * 2048)
        with mock.patch("serve.chat_history.MAX_BYTES", 1024):
            self.assertEqual(self.call("/api/history/import", archive, {"Content-Type": "application/gzip"})[0], 413)
        self.assertEqual(json.loads(self.call("/api/history")[2])["total"], 0)


if __name__ == "__main__":
    unittest.main()
