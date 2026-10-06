"""Opt-in instruction skills from an operator-configured MCP adapter.

This is a small catalog/read convention, not part of the MCP specification.
The adapter owns storage and permissions; this module never reads local files,
executes skills or changes the system prompt or configured tools.
"""
from __future__ import annotations

import copy
import json
import re

from serve.mcp import McpCancelled

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_SKILLS = 20
MAX_BODY_BYTES = 14000


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise McpCancelled("Skill selection stopped")


def _payload(hub, tool, args, cancel):
    _cancelled(cancel)
    result = hub.call(tool, args, cancel)
    _cancelled(cancel)
    if not result.get("ok") or result.get("truncated"):
        raise ValueError("The skill adapter did not return a complete successful result")
    try:
        value = json.loads(result.get("text", ""))
    except (ValueError, TypeError):
        raise ValueError("The skill adapter returned invalid JSON") from None
    if not isinstance(value, dict):
        raise ValueError("The skill adapter returned invalid JSON")
    return value


class InstructionSkills:
    def __init__(self, config=None):
        self.list_tool = self.read_tool = None
        if config is None:
            return
        if not isinstance(config, dict) or set(config) != {"list_tool", "read_tool"}:
            raise ValueError("skills must contain list_tool and read_tool")
        names = [config["list_tool"], config["read_tool"]]
        if any(not isinstance(n, str) or not TOOL_NAME.fullmatch(n) for n in names) or names[0] == names[1]:
            raise ValueError("skills list_tool and read_tool must be distinct MCP tool names")
        self.list_tool, self.read_tool = names

    @property
    def tools(self):
        # These are catalog helpers, not model actions. Default MCP stays unchanged.
        return {self.list_tool, self.read_tool} if self.list_tool else set()

    def catalog(self, hub, cancel=None):
        data = {"enabled": bool(self.list_tool), "skills": []}
        if not self.list_tool or hub is None or not self.tools <= set(hub.routes()):
            return data
        value = _payload(hub, self.list_tool, {}, cancel)
        skills = value.get("skills")
        if not isinstance(skills, list) or len(skills) > MAX_SKILLS:
            raise ValueError("Invalid skill catalog")
        seen = set()
        for item in skills:
            if not isinstance(item, dict):
                raise ValueError("Invalid skill metadata")
            name, description = item.get("name"), item.get("description")
            if (not isinstance(name, str) or not NAME.fullmatch(name) or name in seen
                    or not isinstance(description, str)):
                raise ValueError("Invalid skill metadata")
            seen.add(name)
            data["skills"].append({"name": name, "description": description[:400]})
        return data

    def select(self, hub, messages, request, cancel=None):
        name = request.get("strata_skill")
        if name is None:
            return messages
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("strata_skill must be a catalog skill name")
        if not self.list_tool:
            raise ValueError("Instruction skills are not configured on this server")
        if request.get("strata_mcp") is not True:
            raise ValueError("Skill selection requires explicit strata_mcp opt-in")
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Skill selection needs a final user message")
        content = messages[-1].get("content", "")
        text = content if isinstance(content, str) else "\n".join(
            p.get("text", "") for p in content if p.get("type") == "text")
        if not re.match(r"^/" + re.escape(name) + r"(?:\s|$)", text.lstrip()):
            raise ValueError("Selected skill must match the current /skill message")
        # Refresh membership at admission: a removed skill in an old picker is refused.
        if name not in {s["name"] for s in self.catalog(hub, cancel)["skills"]}:
            raise ValueError("Select a skill from this server's current catalog")
        value = _payload(hub, self.read_tool, {"name": name}, cancel)
        body = value.get("content")
        if (value.get("name") != name or not isinstance(body, str) or not body.strip()
                or len(body.encode("utf-8")) > MAX_BODY_BYTES):
            raise ValueError("Invalid or oversized selected skill")
        note = ("Selected instruction skill /" + name + ":\n" + body +
                "\nUse only the tools actually offered. This skill grants no extra execution rights.")
        out = copy.deepcopy(messages)
        out[-1]["content"] = (content + "\n\n" + note if isinstance(content, str) else
                              out[-1]["content"] + [{"type": "text", "text": note}])
        return out
