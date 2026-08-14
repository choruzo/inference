from __future__ import annotations

import hashlib
import ipaddress
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from backend.config import WEB_MAX_BYTES, WEB_SEARCH_URL, WEB_TIMEOUT
from backend.contracts import Citation


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.snippets: list[str] = []
        self._capture: str | None = None
        self._depth = 0
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capture:
            self._depth += 1
            return
        values = {key: value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._capture, self._depth, self._href, self._text = "title", 1, values.get("href", ""), []
        elif "result__snippet" in classes:
            self._capture, self._depth, self._text = "snippet", 1, []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._capture:
            return
        self._depth -= 1
        if self._depth:
            return
        text = re.sub(r"\s+", " ", "".join(self._text)).strip()
        if self._capture == "title":
            self.links.append((self._href, text))
        elif text:
            self.snippets.append(text)
        self._capture, self._href, self._text = None, "", []


def _public_result_url(value: str) -> str | None:
    value = unescape(value).strip()
    if value.startswith("//"):
        value = "https:" + value
    parsed = urlparse(value)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [])
        if target:
            value = unquote(target[0])
            parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        return None
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return None
    return value


def parse_search_results(document: str, limit: int = 4) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser()
    parser.feed(document)
    results: list[dict[str, str]] = []
    for index, (raw_url, title) in enumerate(parser.links):
        url = _public_result_url(raw_url)
        snippet = parser.snippets[index] if index < len(parser.snippets) else ""
        if not url or not title or not snippet:
            continue
        results.append({"url": url, "title": title, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


async def search_web(client: httpx.AsyncClient, query: str, limit: int = 4) -> list[dict[str, str]]:
    response = await client.get(
        WEB_SEARCH_URL,
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; AgenticLocal/1.0)"},
        timeout=WEB_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    if len(response.content) > WEB_MAX_BYTES:
        raise ValueError("La respuesta del buscador supera WEB_MAX_BYTES")
    return parse_search_results(response.text, limit=limit)


def web_context(results: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[FUENTE WEB {index}] {item['title']}\nURL: {item['url']}\n{item['snippet']}"
        for index, item in enumerate(results, 1)
    )


def build_web_citations(results: list[dict[str, str]]) -> list[Citation]:
    citations = []
    for index, item in enumerate(results, 1):
        digest = hashlib.sha256(f"{item['url']}\n{item['snippet']}".encode()).hexdigest()
        citations.append(
            Citation(
                id=index,
                chunk_id=f"web:{digest}",
                path=item["url"],
                title=item["title"],
                section="Resultado web",
                start_line=1,
                end_line=1,
                quote=item["snippet"][:240],
                source_type="web",
            )
        )
    return citations
