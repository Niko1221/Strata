# dev/

Erweiterungspunkte von Strata Dev (siehe [docs/DEV_BUILD.md](../docs/DEV_BUILD.md)). Live liest diesen Ordner nie.

| Ordner | Was hinein kommt | Ab Phase |
| --- | --- | --- |
| `extensions/ui/` | UI-Extensions (Slots, Überschreiben vorhandener Elemente) | 2 |
| `extensions/backend/` | Host-Hooks (`before_request`, `after_response`, ...) und Routen unter `/dev/...` | 2 |
| `skills/<name>/SKILL.md` | Skills im Claude-Skill-Format, neben denen der App | 3 |
| `models/*.json` | Overlays auf die Run-Configs `strata-<model>.json` | 3 |

Die Ordner sind noch leer; sie stehen hier, damit die nächsten Phasen einen festen Platz haben.
