from __future__ import annotations

import os

import json
import re
from typing import Any

import httpx

from backend.config import LLM_BASE_URL, LLM_MODEL, MAX_AGENT_STEPS, MAX_CHAT_HISTORY_MESSAGES, MAX_RESPONSE_TOKENS, MAX_STRUCTURED_TOKENS, MAX_TOOL_OUTPUT_CHARS, RAG_RERANK_TOP_K, REQUEST_TIMEOUT, RESPONSE_TIMEOUT, WORKSPACE_ROOT
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


def _rag_answer_response_format(source_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rag_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "paragraphs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "source_ids": {
                                    "type": "array",
                                    "items": {"type": "integer", "enum": list(range(1, source_count + 1))},
                                    "uniqueItems": True,
                                },
                            },
                            "required": ["text", "source_ids"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["paragraphs"],
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


def _has_substantive_text(text: str) -> bool:
    without_markers = re.sub(r"\[\d+\]|END_OF_ANSWER", " ", text)
    words = re.findall(r"[^\W\d_]+", without_markers, flags=re.UNICODE)
    return bool(words) and sum(len(word) for word in words) >= 4


def _paragraphs_are_cited(answer: str) -> bool:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    return bool(paragraphs) and all(
        _has_substantive_text(paragraph)
        and (re.search(r"\[\d+\]", paragraph) or paragraph.lower().startswith("inferencia:"))
        for paragraph in paragraphs
    )


def _parse_rag_answer(raw: str, source_count: int) -> tuple[str, set[int]]:
    parsed = _extract_json(raw)
    paragraphs = parsed.get("paragraphs") if parsed else None
    if not isinstance(paragraphs, list):
        return "", set()

    rendered: list[str] = []
    referenced: set[int] = set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            continue
        text = re.sub(r"\s*\[\d+\]", "", str(paragraph.get("text", ""))).strip()
        raw_ids = paragraph.get("source_ids")
        if not isinstance(raw_ids, list):
            continue
        source_ids = sorted({value for value in raw_ids if isinstance(value, int) and 1 <= value <= source_count})
        if not source_ids or not _has_substantive_text(text):
            continue
        references = "".join(f"[{value}]" for value in source_ids)
        rendered.append(f"{text} {references}")
        referenced.update(source_ids)
    return "\n\n".join(rendered), referenced


class LocalAgent:
    def __init__(self, rag_store=None, embedding_client=None) -> None:
        self.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        self.rag_store = rag_store
        self.embedding_client = embedding_client

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
        for item in (history or [])[-MAX_CHAT_HISTORY_MESSAGES:]:
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
        messages.extend(item for item in (history or [])[-MAX_CHAT_HISTORY_MESSAGES:] if item.get("role") in {"user", "assistant"} and item.get("content"))
        messages.append({"role": "user", "content": user_message})
        raw_outputs: list[str] = []
        try:
            content = await self._complete_text(messages, max_tokens=MAX_RESPONSE_TOKENS)
            raw_outputs.append(content)
            parsed = _extract_json(content)
            answer = str(parsed["final"]).strip() if parsed and "final" in parsed else _strip_thinking(content)
            if not answer:
                retry_messages = messages + [{"role": "user", "content": "Responde ahora de forma directa. Devuelve el campo final y no dejes el razonamiento sin cerrar."}]
                content = await self._complete_text(retry_messages, max_tokens=MAX_RESPONSE_TOKENS)
                raw_outputs.append(content)
                parsed = _extract_json(content)
                answer = str(parsed["final"]).strip() if parsed and "final" in parsed else _strip_thinking(content)
        except httpx.TimeoutException:
            return self._response("El modelo local ha agotado el tiempo de respuesta.", [], "llm_timeout")
        except httpx.HTTPError as exc:
            return self._response(f"Error llamando al servidor LLM: {type(exc).__name__}: {exc}", [], "llm_error")
        trace = [{"type": "model", "event": "completion", "data": {"attempts": raw_outputs}}] if modes.reasoning_panel else []
        reasoning = "\n\n--- reintento ---\n\n".join(filter(None, (_thinking(value) for value in raw_outputs))) or None
        if not answer:
            return self._response("El modelo no produjo una respuesta final completa.", trace, "generation_incomplete", reasoning=reasoning if modes.reasoning_panel else None)
        return self._response(answer, trace, "final", reasoning=reasoning if modes.reasoning_panel else None)

    async def _rag_chat(self, user_message: str, history: list[dict[str, str]] | None, modes: ChatModes) -> dict[str, Any]:
        candidates, search_trace = hybrid_search(user_message, store=self.rag_store, client=self.embedding_client, top_k=max(RAG_RERANK_TOP_K * 2, 8))
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

        ranked, rank_trace = await rerank(user_message, candidates, rank_completion, store=self.rag_store, top_k=RAG_RERANK_TOP_K)
        trace.append({"type": "rerank", "event": "ranking", "data": rank_trace})
        context = context_with_citations(ranked)
        prompt = (
            "Usa exclusivamente las FUENTES para responder. Devuelve el JSON solicitado: paragraphs contiene objetos "
            "con un texto factual y los source_ids que lo respaldan. En text responde directamente a la pregunta, "
            "resume las medidas concretas y no menciones las fuentes ni el proceso. No escribas citas dentro de text. "
            "Si las fuentes no permiten responder, devuelve paragraphs vacio.\n\n"
            f"Pregunta: {user_message}\n\n{context}"
        )
        messages = [{"role": "system", "content": "Eres un asistente RAG riguroso. Redacta respuestas breves en el idioma de la pregunta y no inventes datos."}]
        messages.extend(item for item in (history or [])[-MAX_CHAT_HISTORY_MESSAGES:] if item.get("role") in {"user", "assistant"} and item.get("content"))
        messages.append({"role": "user", "content": prompt})
        raw_outputs: list[str] = []
        try:
            raw = await self._complete_text(
                messages,
                max_tokens=MAX_STRUCTURED_TOKENS,
                response_format=_rag_answer_response_format(len(ranked)),
            )
            raw_outputs.append(raw)
            answer, referenced = _parse_rag_answer(raw, len(ranked))
            if not answer:
                retry_prompt = (
                    "Devuelve un parrafo factual breve en el JSON solicitado, usando solo la fuente. "
                    "Pon 1 en source_ids y no incluyas razonamiento. Si no responde la pregunta, devuelve paragraphs vacio.\n\n"
                    f"Pregunta: {user_message}\n\n{context_with_citations(ranked[:1])}"
                )
                raw = await self._complete_text(
                    [{"role": "system", "content": "Extrae una respuesta breve de la fuente y devuelve solo JSON."}, {"role": "user", "content": retry_prompt}],
                    max_tokens=MAX_STRUCTURED_TOKENS,
                    response_format=_rag_answer_response_format(1),
                )
                raw_outputs.append(raw)
                answer, referenced = _parse_rag_answer(raw, 1)
        except httpx.TimeoutException:
            return self._response("El modelo local ha agotado el tiempo de respuesta.", trace, "llm_timeout")
        except httpx.HTTPError as exc:
            return self._response(f"Error llamando al servidor LLM: {type(exc).__name__}: {exc}", trace, "llm_error")
        if modes.reasoning_panel:
            trace.append({"type": "model", "event": "rag_completion", "data": {"attempts": raw_outputs}})
        if not answer:
            return self._response("El modelo no produjo una respuesta RAG completa.", trace, "generation_incomplete")
        citations = build_citations(ranked)
        citations = [citation for citation in citations if citation.id in referenced]
        if not citations or not validate_citations(citations, self.rag_store) or not _paragraphs_are_cited(answer):
            return self._response("El modelo produjo una respuesta RAG sin respaldo valido.", trace, "invalid_citations")
        retrieval_chunks = [RetrievalChunk(chunk_id=str(item["id"]), path=str(item["path"]), title=str(item.get("title", "")), section=str(item.get("section", "")), start_line=int(item["start_line"]), end_line=int(item["end_line"]), content=str(item["content"]), score=float(item["score"]), reasons=list(item.get("reasons", []))) for item in ranked]
        retrieval = RetrievalResult(query=user_message, chunks=retrieval_chunks, reranked=[str(item["id"]) for item in ranked]).model_dump()
        return self._response(answer, trace if modes.reasoning_panel else [], "final", citations=[item.model_dump() for item in citations], retrieval=retrieval)

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
                "max_tokens": MAX_STRUCTURED_TOKENS,
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
            timeout=None if RESPONSE_TIMEOUT <= 0 else RESPONSE_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
