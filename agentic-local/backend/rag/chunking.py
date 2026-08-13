from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    content: str
    start_line: int
    end_line: int
    section: str
    token_count: int
    content_hash: str
    symbol: str = ""


# 3 chars/token, not the ~4 of typical English prose: dense multilingual/legal text
# (accents, punctuation-heavy numbering like "n.o 300/2008") tokenizes closer to 3, and
# this estimate sizes chunks against RAG_CHUNK_TOKENS to fit the embedding model's
# ctx-size (512 for bge-small). Underestimating here overflows that limit at embed time.
_CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _window(lines: list[str], start: int, end: int, section: str, target: int, overlap: int, symbol: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = start
    max_chars = max(100, target * _CHARS_PER_TOKEN)
    overlap_chars = max(0, overlap * _CHARS_PER_TOKEN)
    while cursor <= end:
        size = 0
        stop = cursor
        while stop <= end and (size + len(lines[stop - 1]) + 1 <= max_chars or stop == cursor):
            size += len(lines[stop - 1]) + 1
            stop += 1
        text = "\n".join(lines[cursor - 1 : stop - 1]).strip()
        if text:
            chunks.append(Chunk(text, cursor, stop - 1, section, estimate_tokens(text), hashlib.sha256(text.encode()).hexdigest(), symbol))
        if stop > end:
            break
        rewind = 0
        next_cursor = stop
        while next_cursor > cursor and rewind < overlap_chars:
            next_cursor -= 1
            rewind += len(lines[next_cursor - 1]) + 1
        cursor = max(cursor + 1, next_cursor)
    return chunks


def _markdown(lines: list[str], target: int, overlap: int) -> list[Chunk]:
    headings: list[str] = []
    blocks: list[tuple[int, int, str]] = []
    block_start = 1
    section = ""
    for number, line in enumerate(lines, 1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        if number > block_start:
            blocks.append((block_start, number - 1, section))
        level = len(match.group(1))
        headings[level - 1 :] = [match.group(2)]
        section = " > ".join(headings)
        block_start = number
    if block_start <= len(lines):
        blocks.append((block_start, len(lines), section))
    return [chunk for start, end, name in blocks for chunk in _window(lines, start, end, name, target, overlap)]


def _python(lines: list[str], target: int, overlap: int) -> list[Chunk]:
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return _window(lines, 1, len(lines), "", target, overlap)
    spans: list[tuple[int, int, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno), node.name))
    if not spans:
        return _window(lines, 1, len(lines), "", target, overlap)
    chunks: list[Chunk] = []
    cursor = 1
    for start, end, symbol in spans:
        if cursor < start:
            chunks.extend(_window(lines, cursor, start - 1, "module", target, overlap))
        chunks.extend(_window(lines, start, end, symbol, target, overlap, symbol))
        cursor = end + 1
    if cursor <= len(lines):
        chunks.extend(_window(lines, cursor, len(lines), "module", target, overlap))
    return chunks


def _javascript(lines: list[str], target: int, overlap: int) -> list[Chunk]:
    starts: list[tuple[int, str]] = []
    pattern = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([\w$]+)|^\s*(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*=.*=>")
    for number, line in enumerate(lines, 1):
        match = pattern.search(line)
        if match:
            starts.append((number, match.group(1) or match.group(2)))
    if not starts:
        return _window(lines, 1, len(lines), "", target, overlap)
    spans = [(start, starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines), symbol) for index, (start, symbol) in enumerate(starts)]
    chunks: list[Chunk] = []
    cursor = 1
    for start, end, symbol in spans:
        if cursor < start:
            chunks.extend(_window(lines, cursor, start - 1, "module", target, overlap))
        chunks.extend(_window(lines, start, end, symbol, target, overlap, symbol))
        cursor = end + 1
    return chunks


def chunk_text(path: Path, text: str, target_tokens: int = 450, overlap_tokens: int = 80) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    suffix = path.suffix.lower()
    if suffix in {".md", ".mdx"}:
        return _markdown(lines, target_tokens, overlap_tokens)
    if suffix == ".py":
        return _python(lines, target_tokens, overlap_tokens)
    if suffix in {".js", ".ts", ".tsx"}:
        return _javascript(lines, target_tokens, overlap_tokens)
    return _window(lines, 1, len(lines), "", target_tokens, overlap_tokens)
