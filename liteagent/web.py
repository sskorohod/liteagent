"""Web utilities — HTTP client, content extraction, search providers, crawl, caching.

Provides industrial-grade internet access tools for LiteAgent:
- web_fetch: fetch URL and extract readable content as markdown/text
- web_search: multi-provider search (Brave, DuckDuckGo, Tavily, SearXNG, Perplexity)
- web_crawl: depth-limited multi-page crawl
- web_extract: structured data extraction (links, images, headings, tables, metadata)

All external dependencies are optional and imported lazily inside functions.
Falls back gracefully when packages are not installed.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urljoin, quote

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FetchResult:
    url: str
    title: str = ""
    content: str = ""
    content_type: str = ""
    status_code: int = 0
    extractor: str = ""
    error: str = ""
    cached: bool = False
    content_length: int = 0
    extracted_length: int = 0


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    provider: str = ""
    cached: bool = False
    error: str = ""


@dataclass
class CrawlResult:
    url: str
    title: str = ""
    content: str = ""
    links: list[str] = field(default_factory=list)
    depth: int = 0
    error: str = ""


@dataclass
class ExtractResult:
    url: str
    title: str = ""
    description: str = ""
    og_tags: dict = field(default_factory=dict)
    headings: list[str] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    error: str = ""


# ══════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════

class WebCache:
    """In-memory TTL cache for web results."""

    def __init__(self, default_ttl: int = 300, max_entries: int = 200):
        self._store: dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        expires_at, value = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None):
        if len(self._store) >= self._max_entries:
            self._evict()
        self._store[key] = (time.time() + (ttl or self._default_ttl), value)

    def _evict(self):
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        if len(self._store) >= self._max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ══════════════════════════════════════════════════════════════════════
# Security
# ══════════════════════════════════════════════════════════════════════

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Invisible unicode used for prompt injection
_INVISIBLE_CHARS_RE = re.compile(
    r'[\u200b-\u200f\u2028-\u202f\u2060-\u2064\u206a-\u206f'
    r'\ufeff\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\uffa0]+'
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def is_ssrf_target(url: str) -> bool:
    """Check if URL targets a private/reserved IP address."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return True
    # Check literal IPs
    try:
        ip = ipaddress.ip_address(hostname)
        return any(ip in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        pass
    # Resolve DNS and check
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if any(ip in net for net in _PRIVATE_NETWORKS):
                return True
    except socket.gaierror:
        return True  # Can't resolve → block
    return False


def strip_invisible_unicode(text: str) -> str:
    """Remove invisible Unicode characters that could be used for prompt injection."""
    return _INVISIBLE_CHARS_RE.sub('', text)


def _check_domain_policy(url: str, blocked: list[str] | None = None,
                         allowed: list[str] | None = None) -> str | None:
    """Return error message if domain is blocked, or None if OK."""
    hostname = urlparse(url).hostname or ""
    if allowed:
        if not any(hostname == d or hostname.endswith("." + d) for d in allowed):
            return f"Domain '{hostname}' not in allowlist"
    if blocked:
        if any(hostname == d or hostname.endswith("." + d) for d in blocked):
            return f"Domain '{hostname}' is blocked"
    return None


# ══════════════════════════════════════════════════════════════════════
# HTTP client (async with fallback)
# ══════════════════════════════════════════════════════════════════════

async def _async_get(url: str, *,
                     headers: dict | None = None,
                     timeout: int = 15,
                     max_size: int = 5_000_000,
                     user_agent: str = "",
                     max_redirects: int = 5) -> tuple[bytes, int, dict[str, str]]:
    """Async HTTP GET. Uses httpx if available, falls back to urllib.

    Returns (body_bytes, status_code, response_headers).
    """
    ua = user_agent or _DEFAULT_USER_AGENT
    hdrs = {"User-Agent": ua, **(headers or {})}

    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            max_redirects=max_redirects,
        ) as client:
            resp = await client.get(url, headers=hdrs)
            body = resp.content
            if len(body) > max_size:
                body = body[:max_size]
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return body, resp.status_code, resp_headers
    except ImportError:
        pass

    # Fallback: urllib (sync, run in executor)
    import urllib.request
    loop = asyncio.get_event_loop()

    def _sync_get():
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(max_size + 1)
            if len(data) > max_size:
                data = data[:max_size]
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            return data, resp.status, resp_headers

    return await loop.run_in_executor(None, _sync_get)


async def _async_post_json(url: str, payload: dict, *,
                           headers: dict | None = None,
                           timeout: int = 30) -> tuple[dict, int]:
    """Async HTTP POST with JSON body. Returns (json_response, status_code)."""
    hdrs = {"Content-Type": "application/json", "User-Agent": "LiteAgent/1.0",
            **(headers or {})}
    body_bytes = json.dumps(payload).encode()

    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(url, content=body_bytes, headers=hdrs)
            return resp.json(), resp.status_code
    except ImportError:
        pass

    # Fallback: urllib
    import urllib.request
    loop = asyncio.get_event_loop()

    def _sync_post():
        req = urllib.request.Request(url, data=body_bytes, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), resp.status

    return await loop.run_in_executor(None, _sync_post)


# ══════════════════════════════════════════════════════════════════════
# Content extraction strategies (fallback chain)
# ══════════════════════════════════════════════════════════════════════

def _extract_trafilatura(html: str, url: str) -> str | None:
    """Strategy 1: trafilatura — best for articles and news pages."""
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        result = trafilatura.extract(
            html, url=url,
            include_links=True,
            include_tables=True,
            include_comments=False,
            output_format="txt",
            favor_recall=True,
        )
        return result if result and len(result.strip()) > 50 else None
    except Exception as e:
        logger.debug("trafilatura extraction failed: %s", e)
        return None


def _extract_readability(html: str, url: str) -> str | None:
    """Strategy 2: readability-lxml — Mozilla Readability algorithm."""
    try:
        from readability import Document
    except ImportError:
        return None
    try:
        doc = Document(html, url=url)
        title = doc.short_title() or ""
        content_html = doc.summary()
        # Convert readability HTML to text
        text = _html_to_text(content_html)
        if title and text and not text.startswith(title):
            text = f"{title}\n\n{text}"
        return text if text and len(text.strip()) > 50 else None
    except Exception as e:
        logger.debug("readability extraction failed: %s", e)
        return None


def _extract_beautifulsoup(html: str, url: str) -> str | None:
    """Strategy 3: BeautifulSoup — structured HTML-to-markdown."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script, style, nav, footer, aside
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside",
                                   "noscript", "iframe", "svg"]):
            tag.decompose()
        lines = []
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            lines.append(f"# {title_tag.string.strip()}\n")
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                                    "p", "li", "td", "th", "blockquote", "pre"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue
            tag = elem.name
            if tag.startswith("h"):
                level = int(tag[1])
                lines.append(f"{'#' * level} {text}\n")
            elif tag == "li":
                lines.append(f"- {text}")
            elif tag == "blockquote":
                lines.append(f"> {text}")
            elif tag == "pre":
                lines.append(f"```\n{text}\n```")
            else:
                lines.append(text)
        result = "\n".join(lines)
        return result if result and len(result.strip()) > 50 else None
    except Exception as e:
        logger.debug("BeautifulSoup extraction failed: %s", e)
        return None


def _extract_raw_text(html: str) -> str:
    """Strategy 4: regex-based tag stripping (last resort)."""
    # Remove script/style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Convert headings
    text = re.sub(r'<h([1-6])[^>]*>(.*?)</h\1>', lambda m: '#' * int(m.group(1)) + ' ' + m.group(2) + '\n', text, flags=re.IGNORECASE)
    # Convert links
    text = re.sub(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r'[\2](\1)', text, flags=re.IGNORECASE)
    # Convert line breaks
    text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    # Convert block elements to newlines
    text = re.sub(r'</(p|div|section|article|li|tr|blockquote)>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = _decode_entities(text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _decode_entities(text: str) -> str:
    """Decode common HTML entities."""
    import html
    return html.unescape(text)


def _html_to_text(html_content: str) -> str:
    """Convert HTML to readable text (used by readability output)."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        return _extract_raw_text(html_content)


async def _extract_firecrawl(url: str, api_key: str,
                              base_url: str = "https://api.firecrawl.dev") -> str | None:
    """Optional: Firecrawl API for JavaScript-heavy pages."""
    try:
        data, status = await _async_post_json(
            f"{base_url}/v1/scrape",
            payload={"url": url, "formats": ["markdown"]},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if status == 200 and data.get("success"):
            return data.get("data", {}).get("markdown", "")
        return None
    except Exception as e:
        logger.debug("Firecrawl extraction failed: %s", e)
        return None


def _get_title_from_html(html: str) -> str:
    """Extract <title> from HTML."""
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return _decode_entities(m.group(1).strip())
    return ""


# ══════════════════════════════════════════════════════════════════════
# Public API: web_fetch
# ══════════════════════════════════════════════════════════════════════

async def web_fetch(url: str, *, config: dict | None = None,
                    cache: WebCache | None = None) -> FetchResult:
    """Fetch URL and extract readable content with fallback chain."""
    cfg = config or {}
    fetch_cfg = cfg.get("fetch", {})
    security_cfg = cfg.get("security", {})

    # SSRF check
    if security_cfg.get("ssrf_protection", True) and is_ssrf_target(url):
        return FetchResult(url=url, error="URL blocked (SSRF protection) — cannot access private/reserved IP addresses")

    # Domain policy check
    domain_err = _check_domain_policy(
        url,
        blocked=security_cfg.get("blocked_domains"),
        allowed=security_cfg.get("allowed_domains"),
    )
    if domain_err:
        return FetchResult(url=url, error=domain_err)

    # Cache check
    cache_key = f"fetch:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            cached.cached = True
            return cached

    # Fetch
    timeout = cfg.get("timeout", 15)
    max_size = cfg.get("max_content_size", 5_000_000)
    user_agent = cfg.get("user_agent", "")

    try:
        body, status, headers = await _async_get(
            url, timeout=timeout, max_size=max_size, user_agent=user_agent)
    except Exception as e:
        return FetchResult(url=url, error=f"Request failed: {e}")

    if status >= 400:
        return FetchResult(url=url, status_code=status,
                           error=f"HTTP {status} for {url}")

    content_type = headers.get("content-type", "")

    # Detect encoding
    encoding = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip()
        encoding = charset

    try:
        html = body.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        html = body.decode("utf-8", errors="replace")

    title = _get_title_from_html(html)

    # Extraction chain
    strategies = fetch_cfg.get("strategies",
                                ["trafilatura", "readability", "beautifulsoup", "raw"])
    content = None
    extractor = ""

    for strategy in strategies:
        if strategy == "trafilatura":
            content = _extract_trafilatura(html, url)
            if content:
                extractor = "trafilatura"
                break
        elif strategy == "readability":
            content = _extract_readability(html, url)
            if content:
                extractor = "readability"
                break
        elif strategy == "beautifulsoup":
            content = _extract_beautifulsoup(html, url)
            if content:
                extractor = "beautifulsoup"
                break
        elif strategy == "raw":
            content = _extract_raw_text(html)
            extractor = "raw"
            break

    # Firecrawl fallback
    if not content:
        fc_cfg = fetch_cfg.get("firecrawl", {})
        if fc_cfg.get("enabled"):
            fc_key = os.environ.get(fc_cfg.get("api_key_env", "FIRECRAWL_API_KEY"), "")
            if fc_key:
                content = await _extract_firecrawl(
                    url, fc_key, base_url=fc_cfg.get("base_url", "https://api.firecrawl.dev"))
                if content:
                    extractor = "firecrawl"

    if not content:
        content = ""
        extractor = "none"

    # Security: strip invisible unicode
    if security_cfg.get("strip_invisible_unicode", True):
        content = strip_invisible_unicode(content)

    # Truncate
    max_extract = cfg.get("max_extract_length", 50000)
    if len(content) > max_extract:
        content = content[:max_extract]

    result = FetchResult(
        url=url, title=title, content=content,
        content_type=content_type, status_code=status,
        extractor=extractor, content_length=len(html),
        extracted_length=len(content),
    )

    if cache:
        cache_ttl = cfg.get("cache", {}).get("ttl", 300)
        cache.set(cache_key, result, ttl=cache_ttl)

    return result


# ══════════════════════════════════════════════════════════════════════
# Search providers
# ══════════════════════════════════════════════════════════════════════

async def _search_brave(query: str, count: int, api_key: str, *,
                        language: str = "", country: str = "",
                        freshness: str = "") -> list[SearchResult]:
    """Brave Search API."""
    params = f"q={quote(query)}&count={count}"
    if language:
        params += f"&search_lang={language}"
    if country:
        params += f"&country={country}"
    _freshness_map = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
    if freshness:
        params += f"&freshness={_freshness_map.get(freshness, freshness)}"

    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    body, status, _ = await _async_get(
        url, headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=10)

    if status != 200:
        raise RuntimeError(f"Brave search HTTP {status}")

    data = json.loads(body)
    results = []
    for r in data.get("web", {}).get("results", [])[:count]:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("description", ""),
            source="brave",
        ))
    return results


async def _search_duckduckgo(query: str, count: int, *,
                              language: str = "",
                              region: str = "") -> list[SearchResult]:
    """DuckDuckGo via duckduckgo-search package (no API key needed)."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise RuntimeError("duckduckgo-search package not installed")

    loop = asyncio.get_event_loop()

    def _sync_search():
        ddgs = DDGS()
        kwargs: dict[str, Any] = {"max_results": count}
        if region:
            kwargs["region"] = region
        return list(ddgs.text(query, **kwargs))

    raw = await loop.run_in_executor(None, _sync_search)
    results = []
    for r in raw[:count]:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("href", r.get("link", "")),
            snippet=r.get("body", r.get("snippet", "")),
            source="duckduckgo",
        ))
    return results


async def _search_tavily(query: str, count: int, api_key: str, *,
                          language: str = "") -> list[SearchResult]:
    """Tavily AI Search API."""
    payload: dict[str, Any] = {
        "query": query,
        "max_results": count,
        "search_depth": "basic",
    }
    if language:
        payload["include_domains"] = []  # Tavily doesn't have language param directly

    data, status = await _async_post_json(
        "https://api.tavily.com/search",
        payload=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if status != 200:
        raise RuntimeError(f"Tavily search HTTP {status}")

    results = []
    for r in data.get("results", [])[:count]:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            source="tavily",
        ))
    return results


async def _search_searxng(query: str, count: int, base_url: str, *,
                           language: str = "") -> list[SearchResult]:
    """SearXNG self-hosted search instance."""
    params = f"q={quote(query)}&format=json&pageno=1"
    if language:
        params += f"&language={language}"

    url = f"{base_url.rstrip('/')}/search?{params}"
    body, status, _ = await _async_get(url, timeout=10)

    if status != 200:
        raise RuntimeError(f"SearXNG search HTTP {status}")

    data = json.loads(body)
    results = []
    for r in data.get("results", [])[:count]:
        results.append(SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            source="searxng",
        ))
    return results


async def _search_perplexity(query: str, count: int, api_key: str) -> list[SearchResult]:
    """Perplexity AI Search via OpenAI-compatible API."""
    data, status = await _async_post_json(
        "https://api.perplexity.ai/chat/completions",
        payload={
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if status != 200:
        raise RuntimeError(f"Perplexity search HTTP {status}")

    # Extract answer + citations
    answer = ""
    citations = []
    if data.get("choices"):
        answer = data["choices"][0].get("message", {}).get("content", "")
    citations = data.get("citations", [])

    results = []
    # Add citations as search results
    for i, url in enumerate(citations[:count]):
        results.append(SearchResult(
            title=f"Source {i + 1}",
            url=url,
            snippet="",
            source="perplexity",
        ))
    # If we got an answer, prepend it as the first result
    if answer:
        results.insert(0, SearchResult(
            title="Perplexity AI Answer",
            url="",
            snippet=answer[:2000],
            source="perplexity",
        ))
    return results[:count]


# Provider registry
SEARCH_PROVIDERS: dict[str, dict] = {
    "brave": {"fn": _search_brave, "key_env": "BRAVE_SEARCH_API_KEY", "needs_key": True},
    "duckduckgo": {"fn": _search_duckduckgo, "key_env": None, "needs_key": False},
    "tavily": {"fn": _search_tavily, "key_env": "TAVILY_API_KEY", "needs_key": True},
    "searxng": {"fn": _search_searxng, "key_env": None, "needs_key": False},
    "perplexity": {"fn": _search_perplexity, "key_env": "PERPLEXITY_API_KEY", "needs_key": True},
}


def _detect_available_providers(config: dict) -> list[str]:
    """Detect which search providers are available based on API keys and config."""
    search_cfg = config.get("search", {})
    configured = search_cfg.get("providers")
    if configured:
        return configured

    available = []
    for name, info in SEARCH_PROVIDERS.items():
        if info["needs_key"]:
            key_env = info["key_env"]
            provider_cfg = search_cfg.get(name, {})
            api_key = provider_cfg.get("api_key") or os.environ.get(key_env or "", "")
            if api_key:
                available.append(name)
        else:
            if name == "searxng":
                if search_cfg.get("searxng", {}).get("base_url"):
                    available.append(name)
            else:
                available.append(name)
    return available or ["duckduckgo"]


# ══════════════════════════════════════════════════════════════════════
# Public API: web_search
# ══════════════════════════════════════════════════════════════════════

async def web_search(query: str, *, config: dict | None = None,
                     cache: WebCache | None = None,
                     provider: str = "",
                     count: int = 5,
                     language: str = "",
                     country: str = "",
                     freshness: str = "") -> SearchResponse:
    """Search the web using configured providers with fallback chain."""
    cfg = config or {}
    search_cfg = cfg.get("search", {})
    count = max(1, min(count, search_cfg.get("max_count", 20)))

    # Cache check
    cache_key = f"search:{hashlib.sha256(f'{provider}:{query}:{language}:{freshness}:{count}'.encode()).hexdigest()[:16]}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            cached.cached = True
            return cached

    # Determine provider order
    if provider:
        providers = [provider]
    else:
        providers = _detect_available_providers(cfg)

    fallback = search_cfg.get("fallback", True)
    last_error = ""

    for prov_name in providers:
        prov_info = SEARCH_PROVIDERS.get(prov_name)
        if not prov_info:
            last_error = f"Unknown provider: {prov_name}"
            continue

        # Get API key
        api_key = ""
        if prov_info["needs_key"]:
            provider_cfg = search_cfg.get(prov_name, {})
            api_key = provider_cfg.get("api_key") or os.environ.get(prov_info["key_env"] or "", "")
            if not api_key:
                last_error = f"{prov_name}: no API key"
                if fallback:
                    continue
                return SearchResponse(query=query, provider=prov_name,
                                      error=f"No API key for {prov_name}")

        try:
            fn = prov_info["fn"]
            kwargs: dict[str, Any] = {"query": query, "count": count}
            if prov_info["needs_key"]:
                kwargs["api_key"] = api_key
            if prov_name == "brave":
                kwargs["language"] = language
                kwargs["country"] = country
                kwargs["freshness"] = freshness
            elif prov_name == "duckduckgo":
                kwargs["language"] = language
                kwargs["region"] = country
            elif prov_name == "tavily":
                kwargs["language"] = language
            elif prov_name == "searxng":
                kwargs["base_url"] = search_cfg.get("searxng", {}).get(
                    "base_url", "http://localhost:8888")
                kwargs["language"] = language

            results = await fn(**kwargs)
            resp = SearchResponse(query=query, results=results, provider=prov_name)

            if cache:
                cache_ttl = cfg.get("cache", {}).get("ttl", 600)
                cache.set(cache_key, resp, ttl=cache_ttl)

            return resp

        except Exception as e:
            last_error = f"{prov_name}: {e}"
            logger.warning("Search provider %s failed: %s", prov_name, e)
            if not fallback:
                return SearchResponse(query=query, provider=prov_name,
                                      error=str(e))
            continue

    return SearchResponse(query=query, error=f"All search providers failed. Last: {last_error}")


# ══════════════════════════════════════════════════════════════════════
# Public API: web_crawl
# ══════════════════════════════════════════════════════════════════════

async def web_crawl(url: str, *, config: dict | None = None,
                    cache: WebCache | None = None,
                    max_depth: int = 2,
                    max_pages: int = 10) -> list[CrawlResult]:
    """Crawl pages from a URL with depth limiting."""
    cfg = config or {}
    crawl_cfg = cfg.get("crawl", {})
    max_depth = min(max_depth, crawl_cfg.get("max_depth", 3))
    max_pages = min(max_pages, crawl_cfg.get("max_pages", 20))
    rate_limit = crawl_cfg.get("rate_limit_ms", 1000) / 1000.0
    respect_robots = crawl_cfg.get("respect_robots_txt", True)

    base_parsed = urlparse(url)
    base_domain = base_parsed.netloc

    # Robots.txt check
    disallowed_paths: list[str] = []
    if respect_robots:
        disallowed_paths = await _parse_robots_txt(f"{base_parsed.scheme}://{base_domain}")

    visited: set[str] = set()
    results: list[CrawlResult] = []
    queue: list[tuple[str, int]] = [(url, 0)]

    while queue and len(results) < max_pages:
        current_url, depth = queue.pop(0)

        # Normalize URL
        current_url = current_url.split("#")[0].rstrip("/")
        if current_url in visited:
            continue
        visited.add(current_url)

        # Robots.txt check
        current_path = urlparse(current_url).path
        if any(current_path.startswith(dp) for dp in disallowed_paths):
            continue

        # Fetch page
        fetch_result = await web_fetch(current_url, config=cfg, cache=cache)
        if fetch_result.error:
            results.append(CrawlResult(url=current_url, depth=depth,
                                       error=fetch_result.error))
            continue

        # Extract links from the page for further crawling
        links = []
        if depth < max_depth:
            links = await _extract_links(current_url, cfg)

        # Filter to same domain only
        same_domain_links = []
        for link in links:
            link_parsed = urlparse(link)
            if link_parsed.netloc == base_domain and link.split("#")[0].rstrip("/") not in visited:
                same_domain_links.append(link)

        results.append(CrawlResult(
            url=current_url,
            title=fetch_result.title,
            content=fetch_result.content,
            links=same_domain_links[:20],
            depth=depth,
        ))

        # Add links to queue
        for link in same_domain_links:
            if len(queue) + len(results) < max_pages * 3:  # Don't over-queue
                queue.append((link, depth + 1))

        # Rate limiting
        if rate_limit > 0 and queue:
            await asyncio.sleep(rate_limit)

    return results


async def _parse_robots_txt(base_url: str) -> list[str]:
    """Parse robots.txt and return disallowed paths for *."""
    try:
        body, status, _ = await _async_get(f"{base_url}/robots.txt", timeout=5)
        if status != 200:
            return []
        text = body.decode("utf-8", errors="replace")
        disallowed = []
        in_user_agent_all = False
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                in_user_agent_all = (agent == "*")
            elif in_user_agent_all and line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    disallowed.append(path)
        return disallowed
    except Exception:
        return []


async def _extract_links(url: str, config: dict) -> list[str]:
    """Extract all links from a page."""
    try:
        body, status, _ = await _async_get(url, timeout=10,
                                            max_size=config.get("max_content_size", 5_000_000))
        if status >= 400:
            return []
        html = body.decode("utf-8", errors="replace")

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full_url = urljoin(url, href)
                if full_url.startswith(("http://", "https://")):
                    links.append(full_url)
            return links
        except ImportError:
            # Fallback: regex
            raw_links = re.findall(r'href=["\']([^"\']+)["\']', html)
            return [urljoin(url, href) for href in raw_links
                    if urljoin(url, href).startswith(("http://", "https://"))]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════
# Public API: web_extract
# ══════════════════════════════════════════════════════════════════════

async def web_extract(url: str, *, config: dict | None = None,
                      selectors: dict | None = None) -> ExtractResult:
    """Extract structured data from a URL."""
    cfg = config or {}
    security_cfg = cfg.get("security", {})

    # SSRF check
    if security_cfg.get("ssrf_protection", True) and is_ssrf_target(url):
        return ExtractResult(url=url, error="URL blocked (SSRF protection)")

    # Domain policy
    domain_err = _check_domain_policy(
        url,
        blocked=security_cfg.get("blocked_domains"),
        allowed=security_cfg.get("allowed_domains"),
    )
    if domain_err:
        return ExtractResult(url=url, error=domain_err)

    try:
        body, status, headers = await _async_get(url, timeout=cfg.get("timeout", 15))
    except Exception as e:
        return ExtractResult(url=url, error=f"Request failed: {e}")

    if status >= 400:
        return ExtractResult(url=url, error=f"HTTP {status}")

    html = body.decode("utf-8", errors="replace")

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ExtractResult(url=url, error="beautifulsoup4 not installed — required for web_extract")

    soup = BeautifulSoup(html, "html.parser")

    # Apply CSS selector if provided
    if selectors and selectors.get("css"):
        target = soup.select_one(selectors["css"])
        if target:
            soup = BeautifulSoup(str(target), "html.parser")

    result = ExtractResult(url=url)

    # Title
    title_tag = soup.find("title")
    if title_tag:
        result.title = title_tag.get_text(strip=True)

    # Meta description
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag:
        result.description = desc_tag.get("content", "")

    # OpenGraph tags
    for og_tag in soup.find_all("meta", attrs={"property": re.compile(r'^og:')}):
        prop = og_tag.get("property", "")
        content = og_tag.get("content", "")
        if prop and content:
            result.og_tags[prop] = content

    # Headings
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = h.get_text(strip=True)
        if text:
            level = h.name
            result.headings.append(f"{level}: {text}")

    # Links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        full_url = urljoin(url, href)
        if full_url.startswith(("http://", "https://")):
            result.links.append({"text": text, "url": full_url})

    # Images
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        if src:
            full_src = urljoin(url, src)
            result.images.append({"src": full_src, "alt": alt})

    # Tables
    for table in soup.find_all("table"):
        table_data = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if cells:
                table_data.append(cells)
        if table_data:
            result.tables.append(table_data)

    return result
