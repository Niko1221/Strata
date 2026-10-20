# Models on another Strata computer

In **Settings → Models**, choose **Manage models** beside a saved custom API connection. The window names the destination computer throughout. Models download there and keep downloading when you close the window. Your Mac keeps its own model library.

Choose a management method when editing each custom connection. The API address above it continues to handle chat. SSH usernames, hosts and ports belong to each connection; Strata supplies no fixed login.

## Terminal servers over SSH

In **Settings → Models → Edit**, set **Model management → Terminal server over SSH**. Enter the server host or an alias from this Mac’s SSH configuration. Username and port are optional; leave them blank to use your SSH configuration. Set **Strata source folder** only if automatic detection cannot find the installed Terminal source. It must contain `setup.py`; use an absolute path or a path starting with `~/`. Save, then choose **Manage models**.

Strata uses this Mac’s system SSH client, keys and agent. First connect to the server in Terminal and verify its host key. Password prompts are unavailable inside Strata; configure key authentication or an SSH agent. The server needs Python 3.10 or newer on its SSH command path. Inference API keys are not used for SSH. Removing SSH settings returns the connection to API management.

Opening the manager installs a versioned Python helper in the server account’s private data folder. Downloads use an isolated library there. They continue when you close the window or disconnect. Progress and Stop apply to the selected server and job. Only files from the installed Strata catalog with pinned checksums can be downloaded. No engine, serving process, active profile or existing weights are replaced.

An SSH download means the files were downloaded and verified. A GGUF import saves a reference after inspecting its metadata and complete shards, including split files whose later shards omit model metadata. Neither operation loads a model or proves engine compatibility. Choose **Prepare & serve** to continue from the Mac. Existing serving models keep running during file management. Windows SSH servers are not supported by this helper.

## Prepare and switch over SSH

In **Manage models → Prepare & serve**, choose downloaded or imported model files, an existing Strata engine configuration and a context length. **Review preparation** shows the destination and disk requirement. Confirm **Prepare on…** to start; progress continues if you close the window. **Stop preparation** removes unfinished private preparation copies and preserves the original model files.

Preparation supports recognized complete Strata GGUF variants on Linux NVIDIA servers. New managed profiles are text-only. Other imports remain file references with an explanation. It uses the server's installed Strata Python environment and engine; missing dependencies or unsupported engines are reported before serving. It does not install packages or replace an existing engine. A private source copy from this Mac app supplies the preparation and serving controls. Pinned tokenizer and preparation tools may be downloaded. Inputs and derived assets are copied or reflinked into a separate private profile; disk previews allow space for full copies. Existing Strata and Maya configurations, weights and APIs are retained.

When preparation finishes, choose **Review start API**. The first start uses a separate authenticated API address and port on that server. It starts with the model unloaded. Choose **Save managed connection** to add it to the Mac's chat picker and **Serving → Destination**, where you can review loading it. Credentials stay on the Mac and server. Loading checks free memory on the selected NVIDIA GPUs and refuses if it cannot measure enough space; it never unloads Maya or another service to make room.

Choose **Review switch** beside another prepared profile to replace the model configuration of this managed API. The endpoint and key stay the same. The server must atomically claim an idle instance before switching; active or queued requests prevent it. The new model starts unloaded. If startup fails, Strata attempts to restore the previous profile and reports whether that succeeded. Save the managed connection again to update its available chat model. User-edited connections are not overwritten.

Only APIs created by this workflow can be switched. Existing independently started Strata servers keep running; their configured model can still be loaded or unloaded through **Serving**. These preparation/switch controls are available only in the controlling Mac app, through the selected SSH connection. They are not exposed to the iPhone, inference-sharing listeners or holders of an inference API key.

## Desktop computers through their API

The destination must run a Strata desktop host with remote model management support. On that computer:

1. Open **Serving**, set an API key, and turn on **Allow remote model management**.
2. Share the API through your existing local-network or tunnel setup.
3. On the controlling Mac, save that API address and key as a custom model connection. An empty connection can be saved before any models are loaded.

This option trusts clients holding that key to download and cancel model downloads, and browse/import GGUF files on the destination. It is off by default. Turning it off removes the management routes while leaving the usual inference API available. Older Strata hosts and other compatible inference servers show an update or unsupported message.

## Download through the desktop API

Choose a model from the destination's catalog and review its download size, free disk space and picture-reading option. Confirm **Download on…** to start. The window shows progress and lets you stop or retry; completed files are retained.

Preparation can unload the destination's native models and update its matching engine. It waits for conversations, goals, requests and model operations to finish. A preview expires after five minutes; changed model requirements or disk space require a new review. Downloaded models are not automatically loaded.

## Import existing files through the desktop API

Choose **Import GGUF**, then browse folders on the destination or enter a file path there. Strata checks the GGUF metadata and every shard before you confirm. Importing adds a reference to those existing files; it does not upload or copy files from the controlling Mac. The files must remain in place.

After downloading or importing, load the model on the destination to make it available through its inference API. Then fetch that connection's available models in **Settings → Models** to add it to chat.

## Serving a Terminal model from the Mac

Open **Serving** and choose a saved connection in **Destination**. **This Mac** remains the default. The remote panel reads the server's current model and loaded state through its saved API address and key; no remote desktop is needed.

Choose **Review load** or **Review unload**, check the computer and model, then confirm. These actions apply only to the model already configured on that API listener. Existing server load hooks can run. Unloading frees that model's memory while leaving its API available; a later inference request may load it again. The server must report a reliable idle admission count before either control is available. Busy servers wait; older servers without that information show status with an explanation and disabled controls.

SSH model downloads and imports manage files. Use **Manage models → Prepare & serve** to prepare them and switch a separate managed API from the Mac. Keep separate connections for separate services, such as Maya and Strata on different ports. Loading another large model also needs enough free memory on that computer.

These serving controls are available in the Mac app. They are not shared through its inference API or the paired iPhone bridge. API keys stay in the Mac's saved connections.

## iPhone

The paired iPhone's **Models → Download models** section can choose the paired Mac or one of its saved Strata connections. It uses the same catalog, preview and progress controls. The Mac keeps API credentials and SSH configuration. Set up SSH for a Terminal server on the Mac first; the phone uses that saved target without receiving SSH details. A phone can stop downloads it started; other downloads can be managed from the Mac. File imports are available in the Mac app.
