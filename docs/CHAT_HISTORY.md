# Chat history and conversation compaction

The browser Chat page saves conversations on the Strata server's PC. Open the history panel to find a saved chat,
search its messages, or start a new conversation. Starting a new chat keeps the previous one. Choosing a saved
title opens that conversation so you can continue it.

These features apply to Strata's Chat page. Other API clients, including coding agents, manage their own
conversation history; their requests do not automatically become saved chats in this page.

## Storage and loading

History is enabled by default. The server opens its database on the first authorized history request, without
loading model weights. Python's SQLite library must include FTS5 and its trigram tokenizer (SQLite 3.34 or newer).
Use the Python environment installed by Strata setup. Data lives under the project folder:

```text
Strata-data/chat-history/history.sqlite3
Strata-data/chat-history/backups/
```

Messages and conversation state use gzip-compressed SQLite records. A separate text search index retains bounded
excerpts. This compression reduces repeated text but does not encrypt it. Compression depends on the content;
for a small history, database and index overhead can exceed the size of the original text.

The page loads the conversation catalog in batches of 50, then only the selected chat's recent 40 messages. Pages
are bounded to about 4 MiB of message JSON; an individual larger message is returned on its own. The older-messages
control loads earlier messages when needed. Titles and catalog metadata do not require decompressing every chat.
Only changed messages and state are saved. Original messages, reasoning, tool results, supported attachments,
drafts and compaction summaries remain on disk.

Browsers connected to the same Strata server share the saved chat database. Browser storage still keeps display
settings, the last-opened chat and a temporary recovery copy of changes that have not reached the server. Treat
the database, backups and exported chats as private data. They are ignored by Git.

## Migrating browser history

When older Strata browser history is detected, the page sends it to the disk store. The server first writes a
compressed backup of the source data, verifies its checksum, checks imported message counts and reads back the
selected chat. The browser retires its old copy only after that verification succeeds.

Migration does not intentionally discard the old data when a request fails. Keep the original browser profile
until migration has completed, and keep a separate export before clearing browser storage or replacing a project.

## Saving and recovery

The page saves changed records as you work, including drafts. If it closes before a save completes, it retains a
temporary browser recovery copy and tries again when reopened. Save receipts make retrying the same write safe
when a response was lost.

Each saved conversation has a revision. If another window has changed the same chat, recovery saves the unsent
copy as a separate conversation rather than overwriting the newer version. A failed save leaves the local draft
available for retry. Do not clear browser storage while unsaved changes remain.

## Export and import

The history panel's export control downloads all saved chats as `strata-history.json.gz`. Import accepts a Strata
JSON or JSON.GZ history backup and adds its conversations without replacing existing chats. Re-importing the same
backup does not create another copy. Saving the current chat as Markdown remains a separate control.

An import must be at most 64 MiB both before and after decompression. Exported archives can grow beyond that
limit, so keep importable backups at an appropriate size. Larger single conversations may also need to be split
before they can be saved. The original data stays unchanged when an import fails validation.

## Continuing a long conversation

Gzip storage and conversation compaction solve different problems. Gzip compresses disk records without changing
their contents. Compaction asks the selected local model to summarize older text so later requests fit its context
window, while preserving the original messages on disk.

The Chat page shows its input token count and a manual compaction control. Before a text request approaches the
context limit, it can summarize older turns, retain recent turns where they fit, and reserve room for the answer.
Long archived chats are summarized in bounded batches. You can inspect the summary used for subsequent requests.
Stopping compaction does not replace the stored conversation or commit a partial summary.

Summaries can omit facts, names and other details. Review the summary when accuracy matters and reopen the original
messages when needed. Automatic text compaction cannot faithfully summarize image contents; conversations with
images keep their existing image generation path. Additional providers use an estimated token count, so configure
their context limit conservatively.

## Optional history recall

Enable the past-chat recall option to search saved messages for words related to the current question. Recall can
use older parts of the current conversation and other saved chats. It adds at most four excerpts, totaling at most
6,000 characters, to the request. Source controls open the original referenced message on demand.

Recall is a keyword search, not a guarantee that the model remembers everything. It can miss relevant messages or
retrieve unrelated ones. Disabling it prevents these excerpts from being added; the original chats stay saved.

## Authentication and developer checks

History routes use the server's existing API authentication and checks for requests from its own page. Responses
have `Cache-Control: no-store`. Keep the server on loopback for personal use. A shared server shares this chat store;
this feature does not provide separate accounts or separate histories for different users.

The storage, migration, recovery and compaction tests use temporary fixture data and do not need a GPU:

```text
python -m unittest serve.test_chat_history serve.test_compaction
node --test serve/web/test_history.cjs serve/web/test_disk_history.cjs serve/web/test_context.cjs
```
