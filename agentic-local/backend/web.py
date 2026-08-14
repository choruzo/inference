from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
from datetime import date, timedelta
from html import unescape
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, parse_qsl, urlencode, unquote, urljoin, urlparse, urlunparse

import httpx

from backend.config import (
    SEARXNG_URL,
    TAVILY_API_KEY,
    TAVILY_URL,
    WEB_FETCH_MAX_CHARS,
    WEB_FETCH_RESULTS,
    WEB_MAX_BYTES,
    WEB_MAX_REDIRECTS,
    WEB_RETRY_ATTEMPTS,
    WEB_RETRY_BACKOFF,
    WEB_SEARCH_FALLBACK,
    WEB_SEARCH_LIMIT,
    WEB_SEARCH_MIN_RESULTS,
    WEB_SEARCH_PROVIDER,
    WEB_TIMEOUT,
    WEB_USER_AGENT,
)
from backend.contracts import Citation


WebResult = dict[str, Any]
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class WebProviderError(RuntimeError):
    """A configured search provider could not serve a request."""


class UnsafeUrlError(ValueError):
    """A URL is not safe for direct server-side fetching."""


class WebSearchResults(list[WebResult]):
    def __init__(self, values: list[WebResult], trace: dict[str, Any] | None = None) -> None:
        super().__init__(values)
        self.trace = trace or {}


class _DuckDuckGoParser(HTMLParser):
    """Compatibility parser for old fixtures; DuckDuckGo is no longer a provider."""

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


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.published_at: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag == "meta" and (
            values.get("property", "").lower() == "article:published_time"
            or values.get("name", "").lower() == "date"
        ):
            self.published_at = _clean_text(values.get("content")) or self.published_at
        elif tag == "time" and values.get("datetime") and not self.published_at:
            self.published_at = _clean_text(values["datetime"])
        if tag in {"script", "style", "noscript", "svg", "template"}:
            self.hidden_depth += 1
        elif tag == "title" and not self.hidden_depth:
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "template"} and self.hidden_depth:
            self.hidden_depth -= 1
        elif tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.hidden_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        self.parts.append(data)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", unescape(str(value or ""))).strip()


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


def canonicalize_url(value: str) -> str | None:
    safe = _public_result_url(value)
    if not safe:
        return None
    parsed = urlparse(safe)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
        netloc = f"{netloc}:{port}"
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        ),
        doseq=True,
    )
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


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


def normalize_search_results(values: Any, provider: str, limit: int) -> list[WebResult]:
    if not isinstance(values, list):
        raise WebProviderError(f"{provider} devolvio un campo results invalido")
    normalized: list[WebResult] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        url = canonicalize_url(str(value.get("url", "")))
        title = _clean_text(value.get("title"))
        snippet = _clean_text(value.get("content") or value.get("snippet"))
        if not url or not title or not snippet:
            continue
        raw_score = value.get("score")
        try:
            score = float(raw_score) if raw_score is not None else max(0.0, 1.0 - index / max(len(values), 1))
        except (TypeError, ValueError):
            score = 0.0
        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "published_at": value.get("published_at") or value.get("publishedDate") or value.get("published_date"),
                "score": score,
                "provider": provider,
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


def deduplicate_results(values: list[WebResult], limit: int) -> list[WebResult]:
    results: list[WebResult] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonicalize_url(str(value.get("url", "")))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        results.append(dict(value) | {"url": canonical})
        if len(results) >= limit:
            break
    return results


def _recency_range(recency_days: int | None, provider: str) -> str | None:
    if not recency_days:
        return None
    if provider == "searxng":
        return "day" if recency_days <= 1 else "month" if recency_days <= 31 else "year"
    return "day" if recency_days <= 1 else "week" if recency_days <= 7 else "month" if recency_days <= 31 else "year"


async def _request_with_backoff(request: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
    response: httpx.Response | None = None
    for attempt in range(max(1, WEB_RETRY_ATTEMPTS)):
        response = await request()
        if response.status_code != 429 or attempt + 1 >= max(1, WEB_RETRY_ATTEMPTS):
            return response
        retry_after = response.headers.get("Retry-After", "")
        try:
            delay = min(float(retry_after), 5.0)
        except ValueError:
            delay = WEB_RETRY_BACKOFF * (2**attempt)
        await asyncio.sleep(max(0.0, delay))
    assert response is not None
    return response


async def search_searxng(
    client: httpx.AsyncClient, query: str, limit: int, recency_days: int | None = None
) -> tuple[list[WebResult], dict[str, Any]]:
    params: dict[str, Any] = {"q": query, "format": "json", "safesearch": 1}
    time_range = _recency_range(recency_days, "searxng")
    if time_range:
        params["time_range"] = time_range
    response = await _request_with_backoff(
        lambda: client.get(
            f"{SEARXNG_URL.rstrip('/')}/search",
            params=params,
            headers={"User-Agent": WEB_USER_AGENT, "Accept": "application/json"},
            timeout=WEB_TIMEOUT,
            follow_redirects=False,
        )
    )
    response.raise_for_status()
    if len(response.content) > WEB_MAX_BYTES:
        raise WebProviderError("La respuesta de SearXNG supera WEB_MAX_BYTES")
    data = response.json()
    return normalize_search_results(data.get("results"), "searxng", limit), {
        "provider": "searxng",
        "status": response.status_code,
    }


async def search_tavily(
    client: httpx.AsyncClient, query: str, limit: int, recency_days: int | None = None
) -> tuple[list[WebResult], dict[str, Any]]:
    if not TAVILY_API_KEY:
        raise WebProviderError("TAVILY_API_KEY no esta configurado")
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": "basic",
        "max_results": limit,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    if recency_days:
        payload["start_date"] = (date.today() - timedelta(days=recency_days)).isoformat()
    response = await _request_with_backoff(
        lambda: client.post(
            TAVILY_URL,
            json=payload,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "User-Agent": WEB_USER_AGENT},
            timeout=WEB_TIMEOUT,
            follow_redirects=False,
        )
    )
    response.raise_for_status()
    if len(response.content) > WEB_MAX_BYTES:
        raise WebProviderError("La respuesta de Tavily supera WEB_MAX_BYTES")
    data = response.json()
    return normalize_search_results(data.get("results"), "tavily", limit), {
        "provider": "tavily",
        "status": response.status_code,
        "request_id": data.get("request_id"),
        "usage": data.get("usage"),
    }


async def _provider_search(
    provider: str, client: httpx.AsyncClient, query: str, limit: int, recency_days: int | None
) -> tuple[list[WebResult], dict[str, Any]]:
    if provider == "searxng":
        return await search_searxng(client, query, limit, recency_days)
    if provider == "tavily":
        return await search_tavily(client, query, limit, recency_days)
    raise WebProviderError(f"Proveedor web no soportado: {provider or '(vacio)'}")


async def search_web(
    client: httpx.AsyncClient, query: str, limit: int | None = None, recency_days: int | None = None
) -> WebSearchResults:
    query = query.strip()
    if not query:
        raise ValueError("La consulta web no puede estar vacia")
    if recency_days is not None and recency_days < 1:
        raise ValueError("recency_days debe ser positivo")
    requested_limit = min(20, max(1, limit or WEB_SEARCH_LIMIT))
    minimum = min(requested_limit, max(1, WEB_SEARCH_MIN_RESULTS))
    trace: dict[str, Any] = {"query": query, "provider": WEB_SEARCH_PROVIDER, "attempts": []}
    primary: list[WebResult] = []
    primary_error: Exception | None = None
    try:
        primary, metadata = await _provider_search(WEB_SEARCH_PROVIDER, client, query, requested_limit, recency_days)
        trace["attempts"].append(metadata | {"results": len(primary)})
    except (httpx.HTTPError, ValueError, WebProviderError) as exc:
        primary_error = exc
        trace["attempts"].append({"provider": WEB_SEARCH_PROVIDER, "error": f"{type(exc).__name__}: {exc}"})

    combined = list(primary)
    fallback = WEB_SEARCH_FALLBACK
    should_fallback = fallback and fallback != WEB_SEARCH_PROVIDER and len(primary) < minimum
    if should_fallback:
        trace["fallback_reason"] = "primary_error" if primary_error else "insufficient_results"
        try:
            secondary, metadata = await _provider_search(fallback, client, query, requested_limit, recency_days)
            trace["attempts"].append(metadata | {"results": len(secondary)})
            combined.extend(secondary)
        except (httpx.HTTPError, ValueError, WebProviderError) as exc:
            trace["attempts"].append({"provider": fallback, "error": f"{type(exc).__name__}: {exc}"})
            if not combined:
                raise WebProviderError(f"No hay proveedor web disponible: {exc}") from exc
    elif primary_error and not combined:
        raise WebProviderError(f"Fallo el proveedor web {WEB_SEARCH_PROVIDER}: {primary_error}") from primary_error

    results = deduplicate_results(combined, requested_limit)
    trace["results"] = len(results)
    trace["providers_used"] = list(dict.fromkeys(str(item.get("provider", "")) for item in results))
    return WebSearchResults(results, trace)


async def _resolve_public_host(hostname: str, port: int) -> list[str]:
    try:
        records = await asyncio.to_thread(socket.getaddrinfo, hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"No se puede resolver el host: {hostname}") from exc
    addresses = sorted({record[4][0].split("%", 1)[0] for record in records})
    if not addresses:
        raise UnsafeUrlError(f"El host no tiene direcciones: {hostname}")
    for value in addresses:
        if not ipaddress.ip_address(value).is_global:
            raise UnsafeUrlError(f"Destino local o privado bloqueado: {hostname}")
    return addresses


async def validate_fetch_url(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("Solo se permiten URLs HTTP/HTTPS absolutas")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("No se permiten credenciales en la URL")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".local"):
        raise UnsafeUrlError("Destino local bloqueado")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return await _resolve_public_host(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    if not address.is_global:
        raise UnsafeUrlError("Destino local, privado, loopback o link-local bloqueado")
    return [str(address)]


def _extract_html(document: str) -> tuple[str, str, str | None, str]:
    method = "html_parser"
    title = ""
    published_at: str | None = None
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(document, "html.parser")
        title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
        for selector in (
            ('meta[property="article:published_time"]', "content"),
            ('meta[name="date"]', "content"),
            ("time[datetime]", "datetime"),
        ):
            node = soup.select_one(selector[0])
            if node and node.get(selector[1]):
                published_at = _clean_text(node.get(selector[1]))
                break
        for node in soup(["script", "style", "noscript", "svg", "template"]):
            node.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        content = root.get_text("\n", strip=True)
        method = "beautifulsoup"
    except ImportError:
        parser = _VisibleTextParser()
        parser.feed(document)
        title = _clean_text(" ".join(parser.title_parts))
        content = "\n".join(parser.parts)
        published_at = parser.published_at

    try:
        import trafilatura

        extracted = trafilatura.extract(
            document,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            favor_precision=True,
        )
        if extracted and len(extracted.strip()) >= 80:
            content = extracted
            method = "trafilatura"
    except ImportError:
        pass
    return title, re.sub(r"\n{3,}", "\n\n", content).strip(), published_at, method


async def web_fetch(client: httpx.AsyncClient, url: str, max_chars: int = WEB_FETCH_MAX_CHARS) -> WebResult:
    current = url
    redirects = 0
    while True:
        resolved_addresses = await validate_fetch_url(current)
        response = await _request_with_backoff(
            lambda: client.get(
                current,
                headers={"User-Agent": WEB_USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9"},
                timeout=WEB_TIMEOUT,
                follow_redirects=False,
            )
        )
        if response.status_code in _REDIRECT_STATUSES:
            if redirects >= WEB_MAX_REDIRECTS:
                raise UnsafeUrlError("Demasiados redirects")
            location = response.headers.get("Location")
            if not location:
                raise WebProviderError("Redirect sin cabecera Location")
            current = urljoin(current, location)
            redirects += 1
            continue
        response.raise_for_status()
        if len(response.content) > WEB_MAX_BYTES:
            raise WebProviderError("La pagina supera WEB_MAX_BYTES")
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type and content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise WebProviderError(f"Tipo de contenido no permitido: {content_type}")
        if content_type == "text/plain":
            title, content, published_at, extraction = "", response.text.strip(), None, "plain_text"
        else:
            title, content, published_at, extraction = _extract_html(response.text)
        if not content:
            raise WebProviderError("No se pudo extraer contenido principal")
        return {
            "url": current,
            "requested_url": url,
            "title": title,
            "content": content[: min(max(1, max_chars), WEB_FETCH_MAX_CHARS)],
            "published_at": published_at,
            "content_type": content_type or "text/html",
            "bytes": len(response.content),
            "redirects": redirects,
            "resolved_addresses": resolved_addresses,
            "extraction": extraction,
            "provider": "direct",
        }


async def fetch_web_results(
    client: httpx.AsyncClient, results: list[WebResult], limit: int = WEB_FETCH_RESULTS
) -> tuple[list[WebResult], list[dict[str, Any]]]:
    enriched = [dict(item) for item in results]
    trace: list[dict[str, Any]] = []
    for item in enriched[: max(0, limit)]:
        # Provider metadata marks real search results. This also keeps callers that
        # inject already-complete evidence from unexpectedly making network calls.
        if not item.get("provider"):
            continue
        try:
            fetched = await web_fetch(client, str(item["url"]))
            item["url"] = fetched["url"]
            item["content"] = fetched["content"]
            item["title"] = fetched["title"] or item["title"]
            item["published_at"] = fetched["published_at"] or item.get("published_at")
            trace.append(
                {
                    "url": item["url"],
                    "status": "ok",
                    "bytes": fetched["bytes"],
                    "redirects": fetched["redirects"],
                    "extraction": fetched["extraction"],
                }
            )
        except (httpx.HTTPError, ValueError, WebProviderError) as exc:
            trace.append({"url": item.get("url"), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return deduplicate_results(enriched, len(enriched)), trace


def web_context(results: list[WebResult]) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        metadata = [f"URL: {item['url']}", f"Proveedor: {item.get('provider', 'web')}"]
        if item.get("published_at"):
            metadata.append(f"Fecha: {item['published_at']}")
        evidence = str(item.get("content") or item.get("snippet") or "")
        blocks.append(f"[FUENTE WEB {index}] {item['title']}\n" + "\n".join(metadata) + f"\n{evidence}")
    return "\n\n".join(blocks)


def build_web_citations(results: list[WebResult]) -> list[Citation]:
    citations = []
    for index, item in enumerate(results, 1):
        evidence = str(item.get("content") or item.get("snippet") or "")
        digest = hashlib.sha256(f"{item['url']}\n{evidence}".encode()).hexdigest()
        section = "Resultado web"
        if item.get("published_at"):
            section += f" · {item['published_at']}"
        citations.append(
            Citation(
                id=index,
                chunk_id=f"web:{digest}",
                path=item["url"],
                title=item["title"],
                section=section,
                start_line=1,
                end_line=1,
                quote=evidence[:240],
                source_type="web",
                provider=str(item.get("provider") or "web"),
            )
        )
    return citations
