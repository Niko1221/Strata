# dev/

Erweiterungspunkte von Strata Dev (siehe [docs/DEV_BUILD.md](../docs/DEV_BUILD.md)). Live liest diesen Ordner nie.

| Ordner | Was hinein kommt | Ab Phase |
| --- | --- | --- |
| `extensions/ui/` | UI-Extensions (Slots, Überschreiben vorhandener Elemente); Beispiel `info-button/` | 2 |
| `extensions/backend/` | Host-Hooks (`before_request`, `after_response`, `on_token`, `on_model_load`, `on_tool_call`) und Routen unter `/dev/...`; Beispiele `example.mjs`, `_example_python.py` | 2 |
| `claude-commands/` | Vorlagen für Slash-Commands von Claude Code (`dev-debug.md`); einmal nach `.claude/commands/` kopieren | 2 |
| `skills/<name>/SKILL.md` | Skills im Claude-Skill-Format, neben denen der App | 3 |
| `models/*.json` | Overlays auf die Run-Configs `strata-<model>.json` | 3 |

Dateien und Ordner, deren Name mit `_` beginnt, werden nicht geladen (zum Ausprobieren umbenennen).
