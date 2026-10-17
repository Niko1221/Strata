# Enable web search and image tools in Strata

This guide accompanies PR #2473. The chat model decides when to call configured MCP tools; `image_results` only enables raster transport, previews and downloads. It does not install search, ComfyUI, a diffusion model, or a GPU handoff controller. An image encoder is needed for the model to inspect pictures; user-visible previews also work without one.

## 1. Keep the working model configuration

Back up the JSON used by your existing run script. Merge the fragments below into that file; keep its `exe`, `args`, tokenizer, model paths and memory settings. Restart the Python server after changing MCP configuration. Refresh the chat page afterwards.

```json
{
  "mcp": {
    "image_results": true,
    "timeout_s": 1800,
    "max_result_chars": 20000,
    "max_rounds": 8,
    "start_timeout_s": 120
  }
}
```

Use JSON booleans, not the strings `"true"` or `"false"`. `image_results` defaults to false. Increasing a timeout does not make an oversized model fit in VRAM.

## 2. Add web search

One optional provider is [nickclyde/duckduckgo-mcp-server](https://github.com/nickclyde/duckduckgo-mcp-server). Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Its browser extra supports the provider's current search fallback:

```json
{
  "mcp_servers": {
    "web": {
      "command": "uvx",
      "args": ["--with", "duckduckgo-mcp-server[browser]", "duckduckgo-mcp-server"]
    }
  }
}
```

Use an absolute executable path if Strata cannot find `uvx`. First startup downloads dependencies. Open **Monitor → MCP servers** and confirm the provider is connected. Ask: “Use web search to find the latest Strata release, read its release page and cite the URL.” Search results and fetched pages are external content, not instructions to trust automatically. This provider's installation and tool behavior are documented in its linked README; Strata does not guarantee search-provider availability.

## 3. Connect a local image backend

First install ComfyUI, the desired model and required nodes, and make the workflow succeed directly in ComfyUI. For Qwen Image, use a workflow matching that model's loader, text encoder and VAE; an SDXL checkpoint workflow is not interchangeable.

The first-party [Comfy-Org/comfy-mcp](https://github.com/Comfy-Org/comfy-mcp) is one optional bridge. Its documented installation is:

```sh
python -m pip install comfy-mcp "comfy-cli>=1.14.0"
```

Create a separate environment first, for example `python -m venv C:/AI/mcp-tools`, then run the installation using `C:/AI/mcp-tools/Scripts/python.exe -m pip`. The Windows fragment below assumes its scripts are under `C:/AI/mcp-tools/Scripts`; substitute your actual paths (on Linux, use the environment's `bin` directory).

```json
{
  "mcp_servers": {
    "comfy": {
      "command": "C:/AI/mcp-tools/Scripts/comfy-mcp.exe",
      "env": {
        "COMFY_BIN": "C:/AI/mcp-tools/Scripts/comfy.exe"
      }
    }
  }
}
```

Follow that project's setup for the existing ComfyUI workspace. Ask the model to call `server_info`, run a known working workflow with `run_workflow`, then collect its results with `fetch_outputs`. For this example, collect into a directory you control, such as `C:/AI/generated`. Use local workflows without partner/cloud nodes. This bridge collects output files; a file path alone is not an MCP image block.

## 4. Return generated files as actual image blocks

For inline previews, the MCP result must contain `{"type":"image","mimeType":"image/png","data":"<base64 bytes>"}`. A text URL, local path or `resource_link` does not activate this PR's preview path. Do not put base64 in a text block.

If your backend already returns MCP image blocks, skip this step. Otherwise, save this small standard-library example as `mcp-preview.py`. It only reads raster files beneath the output directory passed on its command line. It is a tutorial adapter, not a built-in Strata tool or a generation backend.

```python
import base64
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve(strict=True)
MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
LIMIT = 16 * 1024 * 1024

def tool_result(params):
    try:
        if params.get("name") != "read_generated_image":
            raise ValueError("Unknown tool")
        name = params.get("arguments", {}).get("filename")
        if not isinstance(name, str) or not name:
            raise ValueError("Provide a filename relative to the output directory")
        path = (ROOT / name).resolve(strict=True)
        path.relative_to(ROOT)
        if path.suffix.lower() not in MIMES:
            raise ValueError("Only PNG, JPEG and WebP are supported")
        with path.open("rb") as stream:
            data = stream.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("Image exceeds 16 MiB")
        return {"content": [
            {"type": "text", "text": "Generated image: " + path.name},
            {"type": "image", "mimeType": MIMES[path.suffix.lower()],
             "data": base64.b64encode(data).decode("ascii")}
        ]}
    except Exception as exc:
        return {"isError": True, "content": [{"type": "text", "text": str(exc)}]}

for line in sys.stdin:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(message, dict) or "id" not in message:
        continue
    method = message.get("method")
    params = message.get("params") or {}
    reply = {"jsonrpc": "2.0", "id": message["id"]}
    if method == "initialize":
        reply["result"] = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "generated-image-preview", "version": "1"}}
    elif method == "tools/list":
        reply["result"] = {"tools": [{"name": "read_generated_image",
            "description": "Return a generated output picture for preview and visual inspection",
            "inputSchema": {"type": "object", "properties": {"filename": {"type": "string"}},
                            "required": ["filename"]}}]}
    elif method == "tools/call":
        reply["result"] = tool_result(params)
    elif method == "ping":
        reply["result"] = {}
    else:
        reply["error"] = {"code": -32601, "message": "Method not found"}
    print(json.dumps(reply), flush=True)
```

Register it alongside the other providers (combine entries into one `mcp_servers` object; do not duplicate the JSON key):

```json
{
  "mcp_servers": {
    "preview": {
      "command": "C:/AI/mcp-tools/Scripts/python.exe",
      "args": ["C:/AI/mcp-preview.py", "C:/AI/generated"]
    }
  }
}
```

Create the output directory first. Ask: “Call preview__read_generated_image for example.png.” The tool card should show the picture and **Download image**. Refresh the page and check it returns. Strata caches up to four PNG/JPEG/WebP images per result, using small references in chat history. Missing cache files become text notes on later turns. See [MCP_IMAGES.md](MCP_IMAGES.md) for limits and authentication.

## 5. Share a small GPU safely

The example MCP registrations do **not** coordinate VRAM. Use separate GPUs/machines, or enough memory for both models. On a small shared GPU, do not start an image workflow while Qwen still occupies the card.

For manual use, finish the chat turn, unload Strata, generate directly in ComfyUI, release ComfyUI's idle models, and reload Strata before returning to chat:

```sh
curl -X POST http://127.0.0.1:8085/unload -H "Content-Type: application/json" -d "{}"
# Generate in ComfyUI and wait for its running AND pending queues to become empty.
curl -X POST http://127.0.0.1:8188/free -H "Content-Type: application/json" -d '{"unload_models":true,"free_memory":true}'
curl -X POST http://127.0.0.1:8085/load -H "Content-Type: application/json" -d "{}"
```

These are shell examples; in Windows PowerShell use `Invoke-RestMethod` with the same URLs, JSON body and content type. Add your server's authorization header if configured. Check each response and the actual free VRAM. Do not send another chat request during this manual generation window: that request can reload Qwen. Once Qwen is back, use the preview tool to read the saved file.

For automatic generation *inside* a native Strata chat on the same small GPU, a dedicated generation bridge must own the handoff: serialize GPU use, unload Strata during the tool call, generate, free the image backend, restore Strata even on failure/cancellation, then return the image block. This PR supplies image transport and rendering; it does not ship that lifecycle controller. Increasing `min_free_vram_mib` and using a strict `before_load` guard can prevent unsafe reloads; thresholds depend on the model and machine.

## 6. Verify and troubleshoot

- **No tools:** check Monitor, executable paths and the server's stderr. Starting a stdio MCP program in a terminal can look idle because it waits for protocol input.
- **Search works but images do not:** the search provider and image backend are separate programs. `image_results` is not an installer.
- **A file is saved but no preview:** call the preview adapter, or make the generation bridge return an MCP image block. A browser on a phone cannot use a PC's `localhost` file URL.
- **Preview exists but the model cannot describe it:** configure the vision encoder. Rendering a picture does not give a text-only engine vision.
- **503 on model reload:** free VRAM/RAM or wait for the other GPU job. Preserve the guard rather than bypassing it.
- **Long wait:** backend startup, generation and Qwen reload are separate stages. Keep the chat connection open. This synchronous integration may return the picture only after model restoration.

Validation for this guide covers Strata's configuration parser, the standard-library preview adapter and mock transport. The third-party provider installations above follow their linked project documentation; they were not installed or benchmarked as part of this documentation change.
