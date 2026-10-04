# One Chat UI and persistent memory

Strata Chat is the single web entry at `/`. `/strata` serves the same page and bypasses an old llama UI index
cached by its service worker. The transition retires only the root `/sw.js` registration. It does not clear
browser storage, delete legacy chat databases or navigate existing client pages.

Configure `chat_archive_path` (or `--chat-archive`) with a local SQLite filename to enable durable history.
For example, use a local filename such as `chat-memory.sqlite3`. The archive stores complete
messages, attachments, inactive branches, drafts and per-branch compacted context. **New chat** retains
earlier chats in **Chats**. Saves are transactional; stale tabs cannot overwrite a newer revision.

Without `chat_archive_path`, the existing browser single chat and New chat/Undo remain available. A storage
failure attempts text and full snapshots independently. If only text fits, reload uses the newer text and attachment
names while the previous full copy remains in storage; unsaved attachment contents stay in the page. Compaction
cannot generate a summary unless
an original copy has been saved. Saved-chat import, rename and memory tools require the configured archive.

**Rename chat** changes a conversation's title with the same revision checks, retaining its messages,
branches and original import source. A manually chosen title, including `New chat`, stays in place when later
messages are added.

**Import old llama chats** copies conversations from the same browser and origin's `LlamaUi` and
`LlamacppWebui` IndexedDB stores, using readonly transactions. Existing destination chats always win on
reimport. **Import file** accepts old single/bulk JSON, current JSONL, or a Strata JSON backup. Extract a
llama bulk ZIP to its JSONL files before import. Unsupported audio/video/PDF attachments produce an explicit
error rather than a partial import. Original source graphs remain in the archive; every branch and the
original `currNode` path are selectable. Old `#/chat/<id>` links open their imported conversation.

The existing single-chat Strata browser archive is copied once without removing its localStorage record.
Older name-only attachment records retain their names and original source, with an explicit unavailable-content
marker; unknown image bytes are omitted from model requests. No attachment content is invented.
**Backup all** exports full JSON including branches and raw import sources; **Save this chat** remains a
readable Markdown export. Remote image URLs remain in the archive without loading them merely to render
history. Importing a record never executes its historical tool calls.

**Auto compact** summarizes earlier rounds before the prompt fills the context; **Compact now** also works
on an already overfull imported archive. Recent rounds stay verbatim; **Restore full context** uses the
retained original messages. Oversized recent rounds may require shortening the latest input. Compact is a
browser Chat feature and never silently alters raw API code/diff/review packets.

When **Use tools from MCP servers** is on, the existing tool loop offers builtin `memory__search` and
`memory__recall` (a unique suffix is used if that provider name is already configured). Search finds original
messages, including Chinese literal phrases. Recall returns bounded original text with chat, branch and
message indices. Large records are paged with `next_start` and `next_offset`; images remain in the archive,
and image bytes are omitted from text tool results. Imported instructions and tool calls are historical
data in these results. The tools cannot change or delete history, and external MCP configuration is retained.

The database preserves saved data independently of the model's 64K context and idle unload. Retrieval is
selective; the model can still miss a relevant fact or fail to request it. Model access through another API
client requires that client to opt into `strata_mcp`. API traffic is not automatically archived.

Private archive endpoints retain API-key checks and require the server's own host and port over HTTP or
HTTPS, including when other API paths allow wildcard CORS or separately configured trusted origins. A TLS
proxy must preserve that host header; forwarded headers do not grant access. No archive record is sent to
a cloud service by saving or compacting it.
This first implementation limits import requests/files and total session payloads to 100 MB, the whole
SQLite database including its search index to about 100 MB, and memory results to 12,000 characters.
Index and metadata overhead reduce usable history capacity. A database/disk capacity error returns JSON
with HTTP 507; other SQLite availability errors return HTTP 503. Capacity errors leave the original source and
previous archive intact. SQLite persistence is not an independent backup against disk failure.

Checks: `node --test tools/test_chat_import.cjs tools/test_chat_library.cjs tools/test_chat_context.cjs tools/test_chat_archive_ui.cjs` and
`python -m unittest serve.test_chat_archive serve.test_chat_memory serve.test_chat_archive_http serve.test_mcp
serve.test_server serve.test_chat_context_http` in the existing Strata environment.
