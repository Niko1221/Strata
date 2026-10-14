"""The chat sidebar: the conversations kept in this browser, the open one on the screen.
The feature is opt-in; with the switch off the page is the page from the last release.

The checks read the page and the script as text, like test_monitor.py's page check: a function body where the
check is about what a function does, and a string where the check is about a storage key or a message the user
sees. What the storage path does is run against app.js in test_chat_sidebar.js, with a stub DOM and a stub
IndexedDB; that one is skipped on a machine without node.
"""
import json
import re
import shutil
import subprocess
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class ChatSidebar(unittest.TestCase):
    def setUp(self):
        web = Path(__file__).parent / "web"
        self.html = (web / "index.html").read_text(encoding="utf-8")
        self.js = (web / "app.js").read_text(encoding="utf-8")

    def body(self, name):
        """the source of a function, so a check is about what it does, not about one line of it"""
        found = re.search(rf"function {name}\(.*?\n}}", self.js, re.S)
        self.assertTrue(found, f"{name} is not in app.js")
        return found.group(0)

    def test_the_page_has_the_sidebar_and_its_switch(self):
        for el in ("sidebar", "sidebar-toggle", "sidebar-search", "sidebar-list", "sidebar-foot",
                   "history-toggle"):
            self.assertIn(f'id="{el}"', self.html)
            self.assertIn(f'"{el}"', self.js)
        self.assertIn('id="sidebar" data-open="true" aria-label="Chats" hidden', self.html)   # hidden until the switch is on
        self.assertIn('[hidden] { display: none !important; }', (Path(__file__).parent / "web" / "app.css").read_text(encoding="utf-8"))

    def test_it_is_opt_in_and_off_is_the_page_from_the_last_release(self):
        self.assertIn('store.get("history", false)', self.js)             # off by default
        self.assertIn("if (!historyOn) { store.set(\"chat\", cleanMessages(messages)); return; }", self.js)
        self.assertIn("const backup = messages;", self.js)                # off: "New chat" clears, with an Undo
        # the storage helper is as it was: the list did not add a method to it
        helper = re.search(r"const store = \{.*?\n\};", self.js, re.S).group(0)
        self.assertEqual(sorted(re.findall(r"^\s{2}(\w+)\(", helper, re.M)), ["get", "set"])

    def test_the_list_stays_in_the_browser(self):
        self.assertIn('indexedDB.open("strata", 1)', self.js)             # a chat can be megabytes; localStorage is ~5 MB
        self.assertIn('createObjectStore("chats", {keyPath: "id"})', self.js)
        self.assertIn("function putChat(c)", self.js)                     # one record per chat, not the whole list
        self.assertIn("function dropChat(c)", self.js)
        self.assertNotIn('fetch("chats"', self.js)                        # nothing about it goes to the server
        self.assertIn("function chatTitle(c)", self.js)                   # the name: what the user called it, else the first question

    def test_the_read_takes_the_result_from_the_request(self):
        # oncomplete is the transaction's event, and the store has no result: reading there gives nothing
        read = self.body("allChats")
        self.assertIn("req.onsuccess", read)
        self.assertIn("req.transaction.onabort", read)                 # an aborted read fires abort, not error
        self.assertNotIn("store.result", read)
        self.assertNotIn(".oncomplete", read)

    def test_the_opened_database_is_only_used_when_the_switch_is_on(self):
        open_db = self.body("openDb")
        self.assertIn("catch (e) { resolve(null); return; }", open_db)    # private mode: memory only
        self.assertIn("r.onblocked = () => resolve(null)", open_db)       # another tab holds an older version
        # a database already at version 1 does not run onupgradeneeded, so the store may not be there
        self.assertIn('objectStoreNames.contains("chats") ? r.result : null', open_db)
        self.assertIn("openHistory", self.body("setHistory"))
        off_branch = self.body("setHistory").split("if (!on)")[1].split("return;")[0]
        self.assertNotIn("openHistory", off_branch)                                   # off: the store is never opened

    def test_a_write_the_browser_refuses_still_shows_the_change(self):
        for name in ("putChat", "dropChat"):
            self.assertIn("renderSidebar()", self.body(name))            # memory only: the list has to update
        # a request that fails because the storage is full aborts the transaction, so it fires abort, not error
        quota = self.body("onQuota")
        self.assertIn("t.onerror = warn; t.onabort = warn;", quota)
        self.assertIn('t.error.name === "QuotaExceededError"', quota)    # the message is about a full storage
        for name in ("writeChat", "dropRecord"):
            self.assertIn("onQuota(t)", self.body(name))

    def test_the_list_is_rendered_once_per_change(self):
        # the record helpers write without rendering, so a caller that renders does not render twice
        self.assertIn("writeChat", self.body("showChat"))
        self.assertNotIn("putChat", self.body("showChat"))
        self.assertIn("dropRecord", self.body("deleteChat"))
        self.assertNotIn("dropChat", self.body("deleteChat"))

    def test_a_new_chat_stays_empty_after_a_reload(self):
        # the bug reported on #1787: the single chat from before the switch must not be copied into an empty chat
        history = self.body("openHistory")
        self.assertIn("chats = db ? await allChats() : []", history)
        self.assertIn("const old = fromScreen ? messages : store.get(\"chat\", [])", history)

    def test_turning_the_switch_off_keeps_the_chat_on_the_screen(self):
        # the single chat is written when the switch is flipped, otherwise a reload shows the chat from before it
        off_branch = self.body("setHistory").split("if (!on)")[1].split("return;")[0]
        self.assertIn('store.set("chat", cleanMessages(messages))', off_branch)
        self.assertIn("renderChat()", off_branch)

    def test_pictures_and_files_are_kept_as_names_only(self):
        clean = re.search(r"const cleanMessages = .*?;\n", self.js, re.S).group(0)
        self.assertIn("map((i) => ({name: i.name}))", clean)
        self.assertIn("map((f) => ({name: f.name}))", clean)
        self.assertNotIn("url", clean)                                    # a data URL would fill the browser's storage

    def test_the_search_and_the_day_groups(self):
        render = self.body("renderSidebar")
        self.assertIn("chatTitle(c).toLowerCase().includes(q)", render)
        self.assertIn("c.messages.some((m) => (m.text || \"\").toLowerCase().includes(q))", render)
        self.assertIn("function groupName(t)", self.js)
        for group in ("Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"):
            self.assertIn(f'"{group}"', self.js)

    def test_the_open_chat_is_what_a_request_is_built_from(self):
        self.assertIn("messages = chat.messages", self.js)                # the list and the screen share the array
        self.assertIn("function apiMessages()", self.js)
        self.assertIn("function showChat(c)", self.js)


class NothingGoesToTheServer(unittest.TestCase):
    """The list is the browser's; the server has no endpoint for it and never sees a chat title."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.service = Service(MockEngine(self.tok, "ok"), self.tok,
                               ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.service, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_the_page_and_its_script_are_served(self):
        code, body = self.get("/web/app.js")
        self.assertEqual(code, 200)
        self.assertIn("indexedDB.open(\"strata\", 1)", body)              # the script that keeps the list

    def test_there_is_no_chat_history_endpoint(self):
        for path in ("/chats", "/api/chats", "/history", "/api/history"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)
        # the server answers, so those 404s are the routes that do not exist and not a broken page
        self.assertEqual(self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})[0], 200)


class TheStoragePath(unittest.TestCase):
    """The storage path, run against app.js with a stub DOM and a stub IndexedDB: the read, the write,
    private mode, a full storage, a store that is not there, and a reload. It is skipped on a machine
    without node, so the suite still runs there."""

    def test_the_storage_path(self):
        if not shutil.which("node"):
            self.skipTest("node is not installed")
        script = Path(__file__).parent / "test_chat_sidebar.js"
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
        if result.returncode:
            self.fail((result.stderr or result.stdout).strip())


if __name__ == "__main__":
    unittest.main()
