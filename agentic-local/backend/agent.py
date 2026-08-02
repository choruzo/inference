from __future__ import annotations

import os

import json
import re
from typing import Any

import httpx

from backend.config import LLM_BASE_URL, LLM_MODEL, MAX_AGENT_STEPS, MAX_RESPONSE_TOKENS, MAX_TOOL_OUTPUT_CHARS, RAG_RERANK_TOP_K, REQUEST_TIMEOUT, WORKSPACE_ROOT
from backend.contracts import RetrievalChunk, RetrievalResult
from backend.modes import ChatModes
from backend.rag.citations import build_citations, context_with_citations, validate_citations
from backend.rag.rerank import rerank
from backend.rag.search import hybrid_search
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
                            "enum": ["list_dir", "glob", "find", "read_file", "write_file", "edit_file", "file_info", "rag_status", "rag_reindex", "rag_search"],
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
                    "enum": ["list_dir", "glob", "find", "read_file", "write_file", "edit_file", "file_info", "rag_status", "rag_reindex", "rag_search"],
                },
                "args": {"type": "object"},
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        },
    },
}

RERANK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_ranking",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"ranked_chunk_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["ranked_chunk_ids"],
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
    return _split_thinking(text)[1]


def _thinking(text: str) -> str | None:
    return _split_thinking(text)[0]


def _split_thinking(text: str) -> tuple[str | None, str]:
    if "<think>" not in text:
        return None, text.strip()
    prefix, body = text.split("<think>", 1)
    if "</think>" in body:
        reasoning, answer = body.split("</think>", 1)
        return reasoning.strip(), (prefix + answer).strip()
    return body.strip(), prefix.strip()


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


def _paragraphs_are_cited(answer: str) -> bool:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    return bool(paragraphs) and all(re.search(r"\[\d+\]", paragraph) or paragraph.lower().startswith("inferencia:") for paragraph in paragraphs)


class LocalAgent:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    async def close(self) -> None:
        await self.client.aclose()

    async def chat(self, user_message: str, history: list[dict[str, str]] | None = None, modes: ChatModes | None = None) -> dict[str, Any]:
        if modes is not None:
            modes = modes.effective()
            if modes.rag:
                return await self._rag_chat(user_message, history, modes)
            if not modes.chat:
                return self._response("Activa Chat o RAG para responder.", [], "no_active_mode")
            return await self._plain_chat(user_message, history, modes)
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

    def _response(self, answer: str, trace: list[dict[str, Any]], stopped: str, **extra: Any) -> dict[str, Any]:
        return {"answer": answer, "citations": [], "reasoning": None, "trace": trace, "retrieval": None, "workspace": str(WORKSPACE_ROOT), "stopped": stopped} | extra

    async def _plain_chat(self, user_message: str, history: list[dict[str, str]] | None, modes: ChatModes) -> dict[str, Any]:
        messages = [{"role": "system", "content": "Responde en el idioma del usuario. No uses herramientas ni afirmes haber inspeccionado archivos."}]
        messages.extend(item for item in (history or []) if item.get("role") in {"user", "assistant"} and item.get("content"))
        messages.append({"role": "user", "content": user_message})
        try:
            content = await self._complete_text(messages)
        except httpx.TimeoutException:
            return self._response("El modelo local ha agotado el tiempo de respuesta.", [], "llm_timeout")
        except httpx.HTTPError as exc:
            return self._response(f"Error llamando al servidor LLM: {type(exc).__name__}: {exc}", [], "llm_error")
        parsed = _extract_json(content)
        answer = str(parsed["final"]) if parsed and "final" in parsed else _strip_thinking(content)
        trace = [{"type": "model", "event": "completion", "data": {"raw": content}}] if modes.reasoning_panel else []
        return self._response(answer, trace, "final", reasoning=_thinking(content) if modes.reasoning_panel else None)

    async def _rag_chat(self, user_message: str, history: list[dict[str, str]] | None, modes: ChatModes) -> dict[str, Any]:
        candidates, search_trace = hybrid_search(user_message, top_k=max(RAG_RERANK_TOP_K * 2, 8))
        trace: list[dict[str, Any]] = [{"type": "rag", "event": "retrieval", "data": search_trace}]
        if not candidates:
            return self._response("No lo encuentro en la documentacion indexada.", trace, "no_evidence", retrieval=RetrievalResult(query=user_message).model_dump())

        async def rank_completion(prompt: str) -> str:
            raw = await self._complete_text(
                [{"role": "system", "content": "Eres un reranker. Devuelve exclusivamente JSON valido."}, {"role": "user", "content": prompt}],
                max_tokens=2048,
                response_format=RERANK_RESPONSE_FORMAT,
            )
            return json.dumps(_extract_json(raw) or {})

        ranked, rank_trace = await rerank(user_message, candidates, rank_completion, top_k=RAG_RERANK_TOP_K)
        trace.append({"type": "rerank", "event": "ranking", "data": rank_trace})
        context = context_with_citations(ranked)
        prompt = (
            "Responde exclusivamente con evidencia de las fuentes. Cita cada parrafo factual con [1], [2], etc. "
            "Separa claramente cualquier inferencia. Si la evidencia no basta, di exactamente: No lo encuentro en la documentacion indexada.\n\n"
            "Despues de la respuesta, escribe END_OF_ANSWER en una linea separada.\n\n"
            f"Pregunta: {user_message}\n\n{context}"
        )
        messages = [{"role": "system", "content": "Eres un asistente RAG riguroso. Responde en el idioma de la pregunta y no inventes datos."}]
        messages.extend(item for item in (history or [])[-6:] if item.get("role") in {"user", "assistant"} and item.get("content"))
        messages.append({"role": "user", "content": prompt})
        try:
            raw = await self._complete_text(messages, max_tokens=max(1024, MAX_RESPONSE_TOKENS * 2))
            parsed = _extract_json(raw)
            answer = str(parsed["final"]) if parsed and "final" in parsed else _strip_thinking(raw)
            reasoning = _thinking(raw) if modes.reasoning_panel else None
            if "END_OF_ANSWER" not in answer:
                retry_prompt = (
                    "Responde con una sola frase basada en FUENTE 1 e incluye la cita [1]. Despues escribe END_OF_ANSWER "
                    "en una linea separada. No describas tu proceso.\n\n"
                    f"Pregunta: {user_message}\n\n{context_with_citations(ranked[:1])}"
                )
                raw = await self._complete_text(
                    [{"role": "system", "content": "Da una respuesta factual, breve y citada."}, {"role": "user", "content": retry_prompt}],
                    max_tokens=max(2048, MAX_RESPONSE_TOKENS * 4),
                )
                answer = _strip_thinking(raw)
                reasoning = _thinking(raw) if modes.reasoning_panel else None
            if "END_OF_ANSWER" not in answer:
                return self._response("La generacion RAG no termino correctamente.", trace, "generation_incomplete")
            answer = answer.split("END_OF_ANSWER", 1)[0].strip()
        except httpx.TimeoutException:
            return self._response("El modelo local ha agotado el tiempo de respuesta.", trace, "llm_timeout")
        except httpx.HTTPError as exc:
            return self._response(f"Error llamando al servidor LLM: {type(exc).__name__}: {exc}", trace, "llm_error")
        citations = build_citations(ranked)
        if not answer or not _paragraphs_are_cited(answer):
            answer = "No lo encuentro en la documentacion indexada."
            citations = []
        else:
            referenced = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
            citations = [citation for citation in citations if citation.id in referenced]
            if not citations or not validate_citations(citations):
                return self._response("No lo encuentro en la documentacion indexada.", trace, "invalid_citations")
        retrieval_chunks = [RetrievalChunk(chunk_id=str(item["id"]), path=str(item["path"]), title=str(item.get("title", "")), section=str(item.get("section", "")), start_line=int(item["start_line"]), end_line=int(item["end_line"]), content=str(item["content"]), score=float(item["score"]), reasons=list(item.get("reasons", []))) for item in ranked]
        retrieval = RetrievalResult(query=user_message, chunks=retrieval_chunks, reranked=[str(item["id"]) for item in ranked]).model_dump()
        return self._response(answer, trace if modes.reasoning_panel else [], "final", citations=[item.model_dump() for item in citations], reasoning=reasoning, retrieval=retrieval)

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

    async def _complete_text(self, messages: list[dict[str, str]], max_tokens: int = MAX_RESPONSE_TOKENS, response_format: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {"model": LLM_MODEL, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stop": ["<|im_end|>"]}
        if response_format is not None:
            payload["response_format"] = response_format
        response = await self.client.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
