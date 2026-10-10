"""The chat sidebar: the conversations kept in this browser (localStorage), the open one on the screen.
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

    def test_the_chat_before_the_sidebar_becomes_the_first_entry(self):
        self.assertIn('store.get("chats", null)', self.js)
        self.assertIn('store.get("chat", [])', self.js)                   # read once: a browser that only had that key

    def test_the_list_stays_in_the_browser(self):
        self.assertIn('store.setChecked("chats"', self.js)
        self.assertNotIn('fetch("chats"', self.js)                        # nothing about it goes to the server
        self.assertIn("function chatTitle(c)", self.js)                   # the name: what the user called it, else the first question

    def test_the_open_chat_is_what_a_request_is_built_from(self):
        self.assertIn("messages = chat.messages", self.js)                # the list and the screen share the array
        self.assertIn("function apiMessages()", self.js)
        self.assertIn("function showChat(c)", self.js)


if __name__ == "__main__":
    unittest.main()
