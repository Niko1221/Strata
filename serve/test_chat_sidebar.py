"""The chat sidebar: the conversations kept in this browser, the open one on the screen.
The feature is opt-in; with the switch off the page is the page from the last release."""
import unittest
from pathlib import Path


class ChatSidebar(unittest.TestCase):
    def setUp(self):
        web = Path(__file__).parent / "web"
        self.html = (web / "index.html").read_text(encoding="utf-8")
        self.js = (web / "app.js").read_text(encoding="utf-8")

    def test_the_page_has_the_sidebar_and_its_switch(self):
        for el in ("sidebar", "sidebar-toggle", "sidebar-search", "sidebar-list", "sidebar-foot",
                   "history-toggle"):
            self.assertIn(f'id="{el}"', self.html)
            self.assertIn(f'"{el}"', self.js)

    def test_it_is_opt_in_and_off_is_the_page_from_the_last_release(self):
        self.assertIn('store.get("history", false)', self.js)             # off by default
        self.assertIn('id="sidebar" data-open="true" aria-label="Chats" hidden', self.html)
        self.assertIn('if (!historyOn) { store.set("chat", cleanMessages(messages)); return; }', self.js)
        self.assertIn("const backup = messages;", self.js)                # off: "New chat" clears, with an Undo

    def test_the_list_stays_in_the_browser(self):
        self.assertIn('indexedDB.open("strata", 1)', self.js)             # a chat can be megabytes; localStorage is ~5 MB
        self.assertIn('createObjectStore("chats", {keyPath: "id"})', self.js)
        self.assertIn("function putChat(c)", self.js)                     # one record per chat, not the whole list
        self.assertIn("function dropChat(c)", self.js)
        self.assertIn('r.onblocked = () => resolve(null)', self.js)       # another tab holds an older version
        self.assertIn('objectStoreNames.contains("chats") ? r.result : null', self.js)   # a db at version 1 does not run onupgradeneeded
        self.assertIn("t.onerror = warn; t.onabort = warn;", self.js)     # a full storage aborts, it does not fire error
        self.assertIn("if (!db) { renderSidebar(); return; }", self.js)   # memory only: the change still has to show
        self.assertNotIn('fetch("chats"', self.js)                        # nothing about it goes to the server
        self.assertNotIn("setChecked", self.js)                           # the storage helper is as it was
        self.assertIn("function chatTitle(c)", self.js)                   # the name: what the user called it, else the first question

    def test_turning_the_switch_off_keeps_the_chat_on_the_screen(self):
        # the single chat is written when the switch is flipped, otherwise a reload shows the chat from before it
        self.assertIn('if (!on) { messages = chat ? chat.messages : messages; store.set("chat", cleanMessages(messages)); renderChat(); return; }', self.js)

    def test_a_new_chat_stays_empty_after_a_reload(self):
        # the bug reported on #1787: the single chat from before the switch must not be copied into an empty chat
        self.assertIn("chats = db ? await allChats() : []", self.js)
        self.assertIn('const old = fromScreen ? messages : store.get("chat", [])', self.js)

    def test_the_open_chat_is_what_a_request_is_built_from(self):
        self.assertIn("messages = chat.messages", self.js)                # the list and the screen share the array
        self.assertIn("function apiMessages()", self.js)
        self.assertIn("function showChat(c)", self.js)


if __name__ == "__main__":
    unittest.main()
