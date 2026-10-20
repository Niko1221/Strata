# Models on another Strata computer

In **Settings → Models**, choose **Manage models** beside a saved custom API connection. The window names the destination computer throughout. Models download there and keep downloading when you close the window. Your Mac keeps its own model library.

The destination must run a Strata desktop host with remote model management support. On that computer:

1. Open **Serving**, set an API key, and turn on **Allow remote model management**.
2. Share the API through your existing local-network or tunnel setup.
3. On the controlling Mac, save that API address and key as a custom model connection. An empty connection can be saved before any models are loaded.

This option trusts clients holding that key to download and cancel model downloads, and browse/import GGUF files on the destination. It is off by default. Turning it off removes the management routes while leaving the usual inference API available. Older Strata hosts and other compatible inference servers show an update or unsupported message.

## Download

Choose a model from the destination's catalog and review its download size, free disk space and picture-reading option. Confirm **Download on…** to start. The window shows progress and lets you stop or retry; completed files are retained.

Preparation can unload the destination's native models and update its matching engine. It waits for conversations, goals, requests and model operations to finish. A preview expires after five minutes; changed model requirements or disk space require a new review. Downloaded models are not automatically loaded.

## Import existing files

Choose **Import GGUF**, then browse folders on the destination or enter a file path there. Strata checks the GGUF metadata and every shard before you confirm. Importing adds a reference to those existing files; it does not upload or copy files from the controlling Mac. The files must remain in place.

After downloading or importing, load the model on the destination to make it available through its inference API. Then fetch that connection's available models in **Settings → Models** to add it to chat.

## iPhone

The paired iPhone's **Models → Download models** section can choose the paired Mac or one of its saved Strata connections. It uses the same catalog, preview and progress controls. The Mac keeps API credentials. A phone can stop downloads it started; other downloads can be managed from the Mac. File imports are available in the Mac app.
