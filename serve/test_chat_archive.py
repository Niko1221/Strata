"""Synthetic local SQLite archive tests; never access the user's runtime database."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from serve.chat_archive import ChatArchive, MAX_RECALL_CHARS


def fixture(identity="chat", text="保留精确路径 C:\\repo 和 Native Review 决策"):
    return {"id": identity, "title": "Review", "activeBranchId": "chosen", "source": {"format": "llama", "unknown": [1, {"raw": True}]},
            "branches": [{"id": "main", "title": "Original", "messages": [{"role": "system", "text": "Keep decisions", "reasoning": ""}], "context": None},
                         {"id": "chosen", "title": "Chosen", "messages": [{"role": "user", "text": text, "reasoning": "", "files": [{"name": "note", "text": "中文检索字串"}]}],
                          "context": {"summary": "Decisions retained", "through": 1}}]}


def stored(value, revision=1):
    return {**copy.deepcopy(value), "revision": revision}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.sqlite3"
        self.archive = ChatArchive(self.path)

    def tearDown(self):
        self.archive.close()
        self.temp.cleanup()

    def test_persistence_active_branch_source_detached(self):
        source = fixture()
        before = copy.deepcopy(source)
        saved = self.archive.save(source, activate=True)
        saved["source"]["unknown"].append("changed")
        self.assertEqual(source, before)
        state = ChatArchive(self.path).list()
        self.assertEqual(state, {"sessions": [stored(before)], "activeId": "chat"})
        self.assertEqual(state["sessions"][0]["activeBranchId"], "chosen")

    def test_import_atomic_validation_and_no_overwrite(self):
        broken = fixture("broken")
        broken["activeBranchId"] = "missing"
        with self.assertRaises(ValueError):
            self.archive.import_sessions([fixture(), broken])
        self.assertEqual(self.archive.list()["sessions"], [])
        result = self.archive.import_sessions([fixture(), fixture("second")])
        self.assertEqual(result, {"added": 2, "skipped": 0, "firstId": "chat"})
        changed = fixture(text="Do not overwrite")
        self.assertEqual(self.archive.import_sessions([changed]), {"added": 0, "skipped": 1, "firstId": None})
        self.assertEqual(self.archive.list()["sessions"][0], stored(fixture()))

    def test_import_atomic_capacity_failure(self):
        first, second = fixture("one"), fixture("two")
        limit = len(json.dumps(first, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 10
        with patch("serve.chat_archive.MAX_ARCHIVE_BYTES", limit):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                self.archive.import_sessions([first, second])
        self.assertEqual(self.archive.list()["sessions"], [])

    def test_failed_save_keeps_payload_index_and_active(self):
        self.archive.save(fixture(), activate=True)
        value = fixture(text="invalid")
        value["branches"][1]["context"]["through"] = 99
        with self.assertRaises(ValueError):
            self.archive.save(value, activate=True)
        self.assertEqual(self.archive.list()["sessions"], [stored(fixture())])
        self.assertEqual(len(self.archive.search("精确路径")["hits"]), 1)

    def test_chinese_search_literal_sql_symbols_scoped_hits(self):
        self.archive.save(fixture())
        self.archive.save(fixture("other", "literal % _ \\ and ' OR 1=1 --"))
        hits = self.archive.search("中文检索")["hits"]
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["branch_id"], "chosen")
        self.assertEqual(hits[0]["message_index"], 0)
        self.assertEqual(len(self.archive.search("精确路径", chat_id="other")["hits"]), 0)
        for query in ("%", "_", "' OR 1=1 --"):
            self.assertEqual([h["chat_id"] for h in self.archive.search(query)["hits"]], ["other"])
        self.assertEqual([h["chat_id"] for h in self.archive.search("\\", chat_id="other")["hits"]], ["other"])
        limited = self.archive.search("中文", limit=1)
        self.assertEqual(len(limited["hits"]), 1)
        self.assertTrue(limited["truncated"])

    def test_recall_roles_files_no_images_or_executable_tools(self):
        value = fixture()
        calls = [{"id": "c", "type": "function", "function": {"name": "not_executed", "arguments": "{}"}}]
        value["branches"][1]["messages"] += [
            {"role": "assistant", "text": "", "reasoning": "Thought", "wire": [{"role": "assistant", "content": "", "tool_calls": calls}]},
            {"role": "tool", "text": "Tool result", "wire": [{"role": "tool", "tool_call_id": "c", "content": "Tool result"}]},
            {"role": "user", "text": "", "images": [{"name": "image", "url": "data:image/png;base64," + "A" * 20000}],
             "wire": [{"role": "user", "content": [{"type": "text", "text": "Image question"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 20000}}]}]}
        ]
        self.archive.save(value)
        result = self.archive.recall("chat", "chosen", count=4)
        self.assertEqual([record["role"] for record in result["records"]], ["user", "assistant", "tool", "user"])
        self.assertEqual(result["records"][0]["files"][0]["text"], "中文检索字串")
        self.assertEqual(result["records"][1]["reasoning"], "Thought")
        self.assertEqual(result["records"][2]["wire"][0]["tool_call_id"], "c")
        self.assertNotIn("base64", json.dumps(result))
        self.assertEqual(self.archive.list()["sessions"][0], stored(value))

    def test_recall_bounded_large_escaped_text_pagination(self):
        value = fixture(text='"\\\n中文' * 20000)
        value["branches"][1]["messages"].append({"role": "assistant", "text": "Next"})
        self.archive.save(value)
        result = self.archive.recall("chat", "chosen", count=2)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), MAX_RECALL_CHARS)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["records"][0]["truncated"])
        self.assertEqual(result["next_start"], 0)
        self.assertGreater(result["next_offset"], 0)
        self.assertEqual(self.archive.recall("chat", "chosen", start=1)["records"][0]["text"], "Next")

    def test_legacy_once_and_existing_session_wins(self):
        source = fixture()["branches"][1]["messages"]
        original = copy.deepcopy(source)
        self.assertTrue(self.archive.seed_legacy(source, None)["copied"])
        self.assertFalse(self.archive.seed_legacy([{"role": "user", "text": "later"}], None)["copied"])
        self.assertEqual(source, original)
        self.assertEqual(self.archive.list()["sessions"][0]["branches"][0]["messages"], original)
        another = ChatArchive(Path(self.temp.name) / "another.sqlite3")
        existing = fixture("strata:legacy")
        another.save(existing, activate=True)
        self.assertFalse(another.seed_legacy(source, None)["copied"])
        self.assertEqual(another.list()["sessions"], [stored(existing)])
        another.close()

    def test_recall_large_file_pages_exact_projection(self):
        value = fixture(text="Read file")
        value["branches"][1]["messages"][0]["files"] = [{"name": "large.txt", "text": "中文内容" * 10000}]
        self.archive.save(value)
        pages, offset = [], 0
        while True:
            result = self.archive.recall("chat", "chosen", offset=offset)
            self.assertEqual(result["records"][0]["page_format"], "archived_record_json")
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), MAX_RECALL_CHARS)
            pages.append(result["records"][0]["text"])
            if result["next_offset"] is None:
                break
            self.assertEqual(result["next_start"], 0)
            self.assertGreater(result["next_offset"], offset)
            offset = result["next_offset"]
        record = json.loads("".join(pages))
        self.assertEqual(record["files"][0]["text"], value["branches"][1]["messages"][0]["files"][0]["text"])
        self.assertEqual(self.archive.list()["sessions"][0], stored(value))

    def test_recall_large_assistant_marker_near_end_retrieved_without_loss(self):
        original = "Start " + ('中文 \" \\ \n' * 5000) + " END_MARKER_927"
        value = fixture()
        value["branches"][1]["messages"] = [{"role": "assistant", "text": original, "reasoning": "Keep reason"}, {"role": "user", "text": "Next"}]
        self.archive.save(value)
        pages, offset = [], 0
        while True:
            result = self.archive.recall("chat", "chosen", count=2, offset=offset)
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), MAX_RECALL_CHARS)
            pages.append(result["records"][0]["text"])
            if result["next_offset"] is None:
                self.assertEqual(result["next_start"], 1)
                break
            self.assertEqual(result["next_start"], 0)
            offset = result["next_offset"]
        record = json.loads("".join(pages))
        self.assertEqual(record["text"], original)
        self.assertIn("END_MARKER_927", record["text"])
        self.assertEqual(record["reasoning"], "Keep reason")
        self.assertEqual(self.archive.recall("chat", "chosen", start=1)["records"][0]["text"], "Next")
        for bad in (-1, True, 100000001, len("".join(pages)) + 1):
            with self.assertRaises(ValueError):
                self.archive.recall("chat", "chosen", offset=bad)

    def test_validation_duplicates_context_wire_nonfinite_and_limits(self):
        for mutate in (
            lambda value: value["branches"].append(copy.deepcopy(value["branches"][0])),
            lambda value: value["branches"][0]["messages"][0].update(role="bad"),
            lambda value: value["branches"][0].update(context={"summary": "s", "through": True}),
            lambda value: value["branches"][0]["messages"][0].update(wire=[{"role": "tool", "content": "missing ID"}]),
            lambda value: value.update(updated=float("nan")),
            lambda value: value["branches"][0]["messages"][0].update(wire=[{"role": "user", "content": [{"type": "input_audio"}]}]),
        ):
            with self.subTest(mutate=mutate):
                value = fixture(); mutate(value)
                with self.assertRaises(ValueError):
                    self.archive.save(value)
        for kwargs in ({"start": -1}, {"count": 0}, {"count": 21}, {"start": True}):
            with self.assertRaises(ValueError):
                self.archive.recall("chat", "chosen", **kwargs)
        with self.assertRaises(ValueError):
            self.archive.search("x", limit=21)
        with self.assertRaises(ValueError):
            self.archive.search("")

    def test_concurrent_import_existing_wins(self):
        other = ChatArchive(self.path)
        outcomes, failures = [], []
        def run(archive):
            try:
                outcomes.append(archive.import_sessions([fixture()]))
            except Exception as error:
                failures.append(error)
        threads = [threading.Thread(target=run, args=(archive,)) for archive in (self.archive, other)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(sum(result["added"] for result in outcomes), 1)
        self.assertEqual(sum(result["skipped"] for result in outcomes), 1)
        other.close()

    def test_stale_tab_cannot_erase_newer_context_messages_or_activation(self):
        saved = self.archive.save(fixture(), activate=True)
        first_tab, stale_tab = copy.deepcopy(saved), copy.deepcopy(saved)
        first_tab["branches"][1]["messages"].append({"role": "assistant", "text": "New decision retained"})
        first_tab["branches"][1]["context"] = {"summary": "New summary retained", "through": 2}
        current = self.archive.save(first_tab)
        self.assertEqual(current["revision"], 2)
        other = self.archive.save(fixture("other"), activate=True)
        stale_tab["branches"][1]["messages"] = []
        stale_tab["branches"][1]["context"] = None
        with self.assertRaisesRegex(ValueError, "changed in another tab.*Reload"):
            self.archive.save(stale_tab, activate=True)
        self.assertEqual(self.archive.list(), {"sessions": [current, other], "activeId": "other"})
        self.assertEqual(len(self.archive.search("New decision")["hits"]), 1)
        current["title"] = "Reloaded and updated"
        newer = self.archive.save(current, activate=True)
        self.assertEqual(newer["revision"], 3)
        self.assertEqual(self.archive.list()["activeId"], "chat")

    def test_revision_validation_import_reset_and_legacy_missing_revision(self):
        value = fixture()
        for bad in (-1, True, "1", 0.5):
            value["revision"] = bad
            with self.assertRaisesRegex(ValueError, "revision"):
                self.archive.save(value)
        value["revision"] = 88
        raw_source = copy.deepcopy(value["source"])
        self.archive.import_sessions([value])
        self.assertEqual(self.archive.list()["sessions"][0]["revision"], 1)
        self.assertEqual(self.archive.list()["sessions"][0]["source"], raw_source)
        # Simulate an archive created by the module before revision support.
        with self.archive._connect() as db:
            legacy = fixture("legacy")
            self.archive._put(db, legacy, json.dumps(legacy, ensure_ascii=False))
        legacy["title"] = "Upgrade legacy"
        self.assertEqual(self.archive.save(legacy)["revision"], 1)


if __name__ == "__main__":
    unittest.main()
