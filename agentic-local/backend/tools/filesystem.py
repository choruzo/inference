from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from backend.config import MAX_FILE_READ_CHARS, WORKSPACE_ROOT
from backend.tools.base import Tool, ToolRegistry


def _ensure_workspace() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_path(raw_path: str | None) -> Path:
    _ensure_workspace()
    raw_path = raw_path or "."
    candidate = (WORKSPACE_ROOT / raw_path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("Path escapes the agent workspace") from exc
    return candidate


def _relative(path: Path) -> str:
    return str(path.relative_to(WORKSPACE_ROOT))


def list_dir(args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(args.get("path"))
    if not path.exists():
        raise FileNotFoundError(_relative(path))
    if not path.is_dir():
        raise NotADirectoryError(_relative(path))

    entries = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entries.append(
            {
                "name": child.name,
                "path": _relative(child),
                "type": "directory" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return {"path": _relative(path), "entries": entries}


def glob_files(args: dict[str, Any]) -> dict[str, Any]:
    pattern = str(args.get("pattern") or "**/*")
    include_dirs = bool(args.get("include_dirs", False))
    limit = int(args.get("limit") or 200)
    _ensure_workspace()

    matches: list[str] = []
    for path in WORKSPACE_ROOT.glob(pattern):
        if path == WORKSPACE_ROOT:
            continue
        if path.is_dir() and not include_dirs:
            continue
        matches.append(_relative(path))
        if len(matches) >= limit:
            break
    return {"pattern": pattern, "matches": matches, "truncated": len(matches) >= limit}


def find_text(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "")
    if not query:
        raise ValueError("query is required")
    pattern = str(args.get("pattern") or "**/*")
    limit = int(args.get("limit") or 100)
    case_sensitive = bool(args.get("case_sensitive", False))
    needle = query if case_sensitive else query.lower()

    matches = []
    for path in WORKSPACE_ROOT.glob(pattern):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines, start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                matches.append({"path": _relative(path), "line": index, "text": line[:500]})
                if len(matches) >= limit:
                    return {"query": query, "matches": matches, "truncated": True}
    return {"query": query, "matches": matches, "truncated": False}


def read_file(args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(str(args.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(_relative(path))
    text = path.read_text(encoding="utf-8")
    truncated = len(text) > MAX_FILE_READ_CHARS
    return {"path": _relative(path), "content": text[:MAX_FILE_READ_CHARS], "truncated": truncated}


def write_file(args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(str(args.get("path") or ""))
    content = str(args.get("content") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": _relative(path), "bytes": path.stat().st_size}


def edit_file(args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(str(args.get("path") or ""))
    old = args.get("old")
    new = args.get("new")
    replace_all = bool(args.get("replace_all", False))
    if old is None or new is None:
        raise ValueError("old and new are required")
    if not path.is_file():
        raise FileNotFoundError(_relative(path))

    text = path.read_text(encoding="utf-8")
    count = text.count(str(old))
    if count == 0:
        raise ValueError("old text was not found")
    if count > 1 and not replace_all:
        raise ValueError(f"old text occurs {count} times; set replace_all=true or provide a more specific old value")

    updated = text.replace(str(old), str(new)) if replace_all else text.replace(str(old), str(new), 1)
    path.write_text(updated, encoding="utf-8")
    return {"path": _relative(path), "replacements": count if replace_all else 1}


def file_info(args: dict[str, Any]) -> dict[str, Any]:
    path = _safe_path(str(args.get("path") or "."))
    if not path.exists():
        raise FileNotFoundError(_relative(path))
    stat = path.stat()
    return {
        "path": _relative(path),
        "type": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


registry = ToolRegistry()
registry.register(
    Tool(
        name="list_dir",
        description="List files and directories inside the agent workspace.",
        parameters={"path": "relative directory path, defaults to ."},
        handler=list_dir,
    )
)
registry.register(
    Tool(
        name="glob",
        description="Find files by glob pattern inside the agent workspace.",
        parameters={"pattern": "glob such as **/*.py", "include_dirs": "boolean", "limit": "max matches"},
        handler=glob_files,
    )
)
registry.register(
    Tool(
        name="find",
        description="Search text in UTF-8 files inside the agent workspace.",
        parameters={"query": "text to search", "pattern": "file glob", "case_sensitive": "boolean", "limit": "max matches"},
        handler=find_text,
    )
)
registry.register(
    Tool(
        name="read_file",
        description="Read a UTF-8 text file from the agent workspace.",
        parameters={"path": "relative file path"},
        handler=read_file,
    )
)
registry.register(
    Tool(
        name="write_file",
        description="Create or replace a UTF-8 text file inside the agent workspace.",
        parameters={"path": "relative file path", "content": "complete file content"},
        handler=write_file,
    )
)
registry.register(
    Tool(
        name="edit_file",
        description="Replace text in a UTF-8 file inside the agent workspace.",
        parameters={"path": "relative file path", "old": "exact text to replace", "new": "replacement text", "replace_all": "boolean"},
        handler=edit_file,
    )
)
registry.register(
    Tool(
        name="file_info",
        description="Return type, size, and modified timestamp for a workspace path.",
        parameters={"path": "relative path"},
        handler=file_info,
    )
)
