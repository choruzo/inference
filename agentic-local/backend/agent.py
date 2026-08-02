from __future__ import annotations

import os

import json
import re
from typing import Any

import httpx

from backend.config import LLM_BASE_URL, LLM_MODEL, MAX_AGENT_STEPS, MAX_RESPONSE_TOKENS, MAX_TOOL_OUTPUT_CHARS, REQUEST_TIMEOUT, WORKSPACE_ROOT
from backend.tools import registry


SYSTEM_PROMPT = """You are a local agent running in a constrained workspace.
You can answer normally or request exactly one tool call.

Available tools:
{tools}

Tool call format:
{{"tool": "tool_name", "args": {{...}}}}

Final answer format:
{{"final": "answer for the user"}}

Rules:
- Use tools when you need workspace information or need to edit files.
- If the user asks about files, directories, workspace contents, search, reading, or editing, you must call a tool first.
- Never describe workspace contents from memory or guesswork.
- Only operate inside the workspace.
- Do not invent file contents; inspect first when changing existing files.
- Return only JSON for tool calls and final answers.
"""

AGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_step",
        "strict": True,
        "schema": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": ["list_dir", "glob", "find", "read_file", "write_file", "edit_file", "file_info"],
                        },
                        "args": {"type": "object"},
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"final": {"type": "string"}},
                    "required": ["final"],
                    "additionalProperties": False,
                },
            ]
        },
    },
}

TOOL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_step",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": ["list_dir", "glob", "find", "read_file", "write_file", "edit_file", "file_info"],
                },
                "args": {"type": "object"},
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        },
    },
}


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    cleaned = _strip_thinking(stripped)
    candidates = [stripped, cleaned]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    candidates.extend(block.strip() for block in fenced)
    for source in (stripped, cleaned):
        brace_match = re.search(r"\{.*\}", source, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _truncate(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        return text[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
    return text


def _needs_workspace_tool(message: str) -> bool:
    text = message.lower()
    keywords = [
        "workspace",
        "archivo",
        "archivos",
        "directorio",
        "directorios",
        "carpeta",
        "carpetas",
        "lista",
        "listar",
        "busca",
        "buscar",
        "find",
        "glob",
        "lee",
        "leer",
        "edita",
        "editar",
        "write",
        "edit",
        "read",
    ]
    return any(keyword in text for keyword in keywords)


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(args)
    if tool_name == "list_dir":
        path = normalized.get("path")
        if not path:
            normalized["path"] = "."
        elif isinstance(path, str) and (path.startswith("/") or "workspace" in path.lower()):
            normalized["path"] = "."
    return normalized


class LocalAgent:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def close(self) -> None:
        await self.client.aclose()

    async def chat(self, user_message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        direct = _extract_json(user_message)
        if direct and "tool" in direct:
            tool_name = str(direct.get("tool"))
            args = direct.get("args") if isinstance(direct.get("args"), dict) else {}
            try:
                result = registry.run(tool_name, args)
                return {
                    "answer": _truncate({"ok": True, "tool": tool_name, "result": result}),
                    "trace": [{"step": 0, "tool": tool_name, "args": args, "observation": result}],
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "direct_tool",
                }
            except Exception as exc:
                return {
                    "answer": _truncate({"ok": False, "tool": tool_name, "error": f"{type(exc).__name__}: {exc}"}),
                    "trace": [{"step": 0, "tool": tool_name, "args": args, "error": str(exc)}],
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "direct_tool_error",
                }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools=registry.describe_for_prompt())},
        ]
        for item in history or []:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": user_message})

        trace: list[dict[str, Any]] = []
        force_tool = _needs_workspace_tool(user_message)
        for step in range(1, MAX_AGENT_STEPS + 1):
            try:
                content = await self._complete(messages, force_tool=force_tool and step == 1)
            except httpx.TimeoutException:
                return {
                    "answer": "El modelo local ha agotado el tiempo de respuesta. Baja MAX_RESPONSE_TOKENS o habilita GPU en Docker para hacerlo mas rapido.",
                    "trace": trace,
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "llm_timeout",
                }
            except httpx.HTTPError as exc:
                return {
                    "answer": f"Error llamando al servidor LLM: {type(exc).__name__}: {exc}",
                    "trace": trace,
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "llm_error",
                }
            parsed = _extract_json(content)
            trace.append({"step": step, "model": content})

            if not parsed:
                return {
                    "answer": _strip_thinking(content),
                    "trace": trace,
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "non_json_model_output",
                }

            if "final" in parsed:
                return {
                    "answer": str(parsed["final"]),
                    "trace": trace,
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "final",
                }

            tool_name = parsed.get("tool")
            args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
            args = _normalize_tool_args(str(tool_name), args)
            if not tool_name:
                return {
                    "answer": json.dumps(parsed, ensure_ascii=False),
                    "trace": trace,
                    "workspace": str(WORKSPACE_ROOT),
                    "stopped": "unknown_model_intent",
                }

            try:
                result = registry.run(str(tool_name), args)
                observation = {"ok": True, "tool": tool_name, "result": result}
            except Exception as exc:
                observation = {"ok": False, "tool": tool_name, "error": f"{type(exc).__name__}: {exc}"}

            trace.append({"step": step, "tool": tool_name, "args": args, "observation": observation})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Tool observation:\n" + _truncate(observation)})

        return {
            "answer": "He alcanzado el limite de pasos del agente sin una respuesta final.",
            "trace": trace,
            "workspace": str(WORKSPACE_ROOT),
            "stopped": "max_steps",
        }

    async def _complete(self, messages: list[dict[str, str]], force_tool: bool = False) -> str:
        response = await self.client.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "response_format": TOOL_RESPONSE_FORMAT if force_tool else AGENT_RESPONSE_FORMAT,
                "temperature": 0.0,
                "top_k": 50,
                "repeat_penalty": 1.05,
                "max_tokens": MAX_RESPONSE_TOKENS,
                "stop": ["<|im_end|>"],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
