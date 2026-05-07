"""Dataset-specific role pools and tool-call parsing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .registry import DATASET_ROLE_TOOL_SPECS, DEFAULT_ROLE_POOL, DatasetRoleToolSpec
from .tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    payload = (text or "").strip()
    if not payload:
        return None
    match = re.search(r"<tool_call>([\s\S]*?)</tool_call>", payload)
    if match:
        payload = match.group(1).strip()
    if not payload.startswith("{"):
        generic = _JSON_BLOCK_RE.search(payload)
        if generic:
            payload = generic.group(0)
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tool_name = str(data.get("tool", "")).strip()
    args = data.get("args", {})
    if not tool_name:
        return None
    if not isinstance(args, dict):
        args = {}
    return {"tool": tool_name, "args": args}

