# STRATA-MAP: instructions for agents

This folder is a map of the upstream Strata engine source, pinned to commit **`296b59d2`**.
Use it to find code fast instead of re-reading the whole repo, and keep it accurate when you change it.

**The Strata source is read-only for you.** Never edit, build, or run it while working on the map. Read it only.

---

## 1. What is in this folder

| File | What it is | Who it is for |
|---|---|---|
| `index.tsv` | Full lookup table, 843 rows, every platform | Agents (grep it first) |
| `index-amd.tsv` | AMD-only subset (HIP on Linux, RDNA3/RDNA4, gfx906 notes), 726 rows | Agents working on AMD |
| `README.md` | Plain-English overview with diagrams, all platforms | Humans |
| `README-amd.md` | Same overview trimmed to AMD | Humans |
| `<area>.md` | One detail doc per area (list in section 2) | Both |
| `view-all.html`, `view-amd.html` | Self-contained browser viewers with the docs embedded | Humans |
| `HUMANS.txt` | Short plain-text pointer for people who open the folder | Humans |
| `AGENTS.md` | This file | Agents |

Agents should read the `.tsv` and `.md` files directly. The HTML viewers are generated copies (section 7).

---

## 2. Areas

Each TSV row's `area` column is the name of a detail doc without `.md`.

| Area doc | Covers |
|---|---|
| `engine-core.md` | Core engine types, buffers, graphs, device setup |
| `generate-program.md` | The main program, `src/program/generate.cpp` |
| `prefill.md` | Prompt prefill path |
| `moe-experts.md` | MoE experts, expert cache and streaming |
| `gpu-cpu-kernels.md` | GPU and CPU kernels |
| `hip-amd-tools.md` | AMD/HIP tools such as the hipBLASLt tuner |
| `speculative-decoding.md` | MTP, suffix drafter and draft policy |
| `server.md` | HTTP server and serve scripts |
| `model-spec-ngram-platform.md` | Model spec, n-gram lookup, platform layer |
| `plan-vision-imagegen.md` | VRAM planner, vision encoder, image generation |
| `qwen35-engine.md` | Separate Qwen3.5 engine |
| `build-system.md` | CMake and build options |
| `ci-and-repo-root.md` | CI and root-level scripts |
| `tests-bench-docs.md` | Tests, benchmarks, upstream docs |
| `bench-results.md` | Published benchmark numbers |
| `tools-and-scripts.md` | Helper tools and scripts |
| `artifacts-platform-misc.md` | Artifacts, packaging, misc platform code |
| `intel-sycl-backend.md` | Intel SYCL backend (not in the AMD subset) |
| `apple-metal-backend.md` | Apple Metal backend (not in the AMD subset) |

---

## 3. Looking things up

```bash
rg -i 'STRATA_HIPBLASLT' index.tsv                # find a symbol or env var
awk -F'\t' '$1=="moe-experts"' index.tsv           # every row in one area
awk -F'\t' '$2=="env"' index-amd.tsv               # every env var that matters on AMD
```

Then open the cited file in the Strata checkout at the cited line.
If the code there no longer matches the row, the map is stale: fix it (section 5) or report it. Do not trust a stale row.

---

## 4. Formats you must keep

### TSV rows

Exactly **5 tab-separated columns**, with this header line first:

```
area	kind	name	file:line	how
```

| Column | Rule |
|---|---|
| `area` | Must equal an existing area doc name without `.md` |
| `kind` | One of `tool`, `flag`, `env`, `const`, `constant`, `function`, `call`, `lambda`, `macro`, `kernel`, `behavior`, `gotcha`, `doc` |
| `name` | The symbol, flag, env var or short label |
| `file:line` | Repo-relative path plus `:N` or `:N-M`, e.g. `src/prefill/prefill.cpp:2556` |
| `how` | One short phrase: default, meaning, or effect |

No tabs inside a field, no duplicate rows, no blank lines.

### Line references in the docs

- Always write the **full repo-relative path**: `src/core/expert_source.cpp:1173`, never `expert_source.cpp:1173` and never a bare `:1173`.
- Ranges use a hyphen: `src/spec/draft_policy.cpp:60-81`.
- Separate several refs in one table cell with `; `.
- Mermaid diagrams use the same full repo-relative paths (`src/prefill/prefill.cpp:2556`); wrap long labels with `<br/>`.
- Every number must exist in the file: the line (or range end) must be at or below the file's line count.

### Markdown style

- Lists of things with the same fields (flags, env vars, constants, files) go in a **table**, not prose.
- One fact per bullet. Split long bullets.
- Start detail bullets with a bold label: `**Where:**`, `**Default:**`, `**Why:**`.
- Bold real warnings and non-obvious defaults; do not bold whole sentences.
- Write for any reader. No first person, no "TODO/RESOLVED/Unverified" notes, no private machine names or local paths.
- Leave a blank line before and after every table, list, code block and heading.

### Mermaid diagrams

Every diagram starts with a YAML frontmatter config, not a `%%{init}%%` line (newer Mermaid ignores that).
Copy the config block from an existing diagram in `README.md` exactly, so all diagrams keep the same dark theme and font.
Use `<br/>` to wrap any label longer than about 40 characters.

---

## 5. How to update the map

Do these steps in order. Do not skip the checks.

### 5.1 Before you edit

1. Back up the folder: `tar czf ../STRATA-MAP-bak-$(date +%Y%m%d-%H%M).tgz .`
2. Make sure nobody else is editing the same file. **One writer per file.**
3. Never use `sed` or blind search-and-replace. Edit with a small script that does an exact `str.replace` and asserts the match count is 1:

```python
p = "moe-experts.md"
s = open(p).read()
old = "src/core/expert_source.cpp:1172"
new = "src/core/expert_source.cpp:1173"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
```

### 5.2 Fixing or adding a single fact

1. Read the code in the Strata checkout and find the exact line.
2. Update the detail doc (`<area>.md`) using the formats in section 4.
3. Update or add the matching row in `index.tsv`.
4. If the row matters on AMD (HIP, RDNA, gfx targets, shared engine code), make the same change in `index-amd.tsv`.
5. If the fact appears in `README.md`, also update `README-amd.md` (it is a trimmed copy).
6. Run the checks in 5.4, then rebuild the viewers (section 7).

### 5.3 Moving the map to a newer Strata commit

1. In the Strata checkout, list what changed since the pinned commit:

   ```bash
   git diff --stat 296b59d2 <new-commit>
   git diff 296b59d2 <new-commit> -- <file>        # per file
   ```

2. For each changed file, find every map reference to it:

   ```bash
   rg -n 'src/program/generate.cpp:' *.md *.tsv
   ```

3. Re-locate each referenced line in the new version (search for the symbol, not the old number), and update the number.
4. Add rows for new flags, env vars and kernels; delete rows for removed ones (and their doc lines).
5. Renamed or deleted files: replace or remove every reference to the old path.
6. Replace the pinned commit everywhere it appears (`rg -n 296b59d2`).
7. Run the checks in 5.4 and rebuild the viewers.

### 5.4 Checks to run after every change

Run from this folder, with `STRATA` set to the Strata checkout:

```python
import re, glob, os
STRATA = os.environ["STRATA"]
bad = []
for f in glob.glob("*.md") + glob.glob("*.tsv"):
    t = open(f).read()
    for m in re.finditer(r'((?:[\w.\-]+/)+[\w.\-]+\.\w+):(\d+)(?:-(\d+))?', t):
        p = os.path.join(STRATA, m.group(1))
        if not os.path.isfile(p):
            continue
        n = sum(1 for _ in open(p, "rb"))
        if int(m.group(3) or m.group(2)) > n:
            bad.append((f, m.group(0), n))
    for m in re.finditer(r'(?<=[\s(\[,;`]):\d{2,5}\b', t):
        bad.append((f, "bare ref " + m.group(0), 0))
for b in bad:
    print(*b)
print("problems:", len(bad))
```

Also check:

- Every TSV has exactly 5 fields per line: `awk -F'\t' 'NF!=5' index.tsv index-amd.tsv` prints nothing.
- Every `area` value has a doc: `cut -f1 index.tsv | sort -u` matches the list in section 2 (plus `area`).
- No leftover notes: `rg -n 'RESOLVED|Unverified|TODO|my range' *.md` prints nothing.
- Each changed fact gets a second, independent read of the code by someone other than the writer.

---

## 6. Known open items

These cannot be settled by reading code; they need a real run:

- Whether hipBLASLt accepts the T=8192 tuning ids at T=16384 (`hip-amd-tools.md`).
- Whether other qwen35moe GGUF files load (`qwen35-engine.md`).

---

## 7. Rebuilding the HTML viewers

The viewers embed every doc as a JSON object that starts with `const D=`. After editing any `.md` or `.tsv`, refresh both viewers:

```python
import json, os
for v in ["view-amd.html", "view-all.html"]:
    s = open(v).read()
    i = s.index("const D=") + len("const D=")
    D, end = json.JSONDecoder().raw_decode(s, i)
    for name in D:
        if os.path.isfile(name):
            D[name] = open(name).read()
    out = json.dumps(D, ensure_ascii=False).replace("</", "<\\/")
    open(v, "w").write(s[:i] + out + s[end:])
```

To add a new doc to a viewer, add its file name as a new key in `D` (key order is the sidebar order), then run the script.
Open both viewers in a browser and check that every diagram renders and no table runs off the page.
