"""Tests for web utilities — fetch, search, crawl, extract, cache, security."""

import asyncio
import json
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from liteagent.web import (
    WebCache,
    FetchResult, SearchResult, SearchResponse, CrawlResult, ExtractResult,
    is_ssrf_target, strip_invisible_unicode, _check_domain_policy,
    _extract_raw_text, _get_title_from_html, _decode_entities,
    web_fetch, web_search, web_crawl, web_extract,
    _search_brave, _search_duckduckgo, _search_tavily, _search_searxng,
    _search_perplexity, _search_grok, _search_gemini, _search_kimi,
    _detect_available_providers, _resolve_search_key, SEARCH_PROVIDERS,
    _extract_trafilatura, _extract_readability, _extract_beautifulsoup,
    _async_get,
    sanitize_html_for_llm, _should_remove_element, check_html_depth,
    wrap_untrusted_content, MAX_HTML_NESTING_DEPTH,
)


# ══════════════════════════════════════════════════════════════════════
# WebCache tests
# ══════════════════════════════════════════════════════════════════════

class TestWebCache:
    def test_set_and_get(self):
        cache = WebCache(default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_ttl_expiry(self):
        cache = WebCache(default_ttl=1)
        cache.set("key1", "value1", ttl=0)
        # TTL=0 means it expires immediately at time.time()+0
        # Since the check is time.time() > expires_at, it might still be valid
        # Let's use a very small TTL and mock time
        cache2 = WebCache(default_ttl=60)
        cache2.set("key1", "value1")
        # Manually expire
        cache2._store["key1"] = (time.time() - 1, "value1")
        assert cache2.get("key1") is None

    def test_max_entries_eviction(self):
        cache = WebCache(default_ttl=60, max_entries=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict oldest
        assert cache.size <= 3

    def test_clear(self):
        cache = WebCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_size_property(self):
        cache = WebCache()
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1

    def test_get_nonexistent_key(self):
        cache = WebCache()
        assert cache.get("nonexistent") is None

    def test_overwrite_existing_key(self):
        cache = WebCache()
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_custom_ttl_per_entry(self):
        cache = WebCache(default_ttl=3600)
        cache.set("short", "val", ttl=1)
        cache._store["short"] = (time.time() - 1, "val")
        assert cache.get("short") is None
        cache.set("long", "val", ttl=3600)
        assert cache.get("long") == "val"

    def test_evict_expired_first(self):
        cache = WebCache(default_ttl=60, max_entries=2)
        cache.set("a", 1)
        cache._store["a"] = (time.time() - 1, 1)  # Expired
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict expired "a", not "b"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_empty_cache_operations(self):
        cache = WebCache()
        assert cache.get("x") is None
        cache.clear()
        assert cache.size == 0


# ══════════════════════════════════════════════════════════════════════
# Security tests
# ══════════════════════════════════════════════════════════════════════

class TestSecurity:
    def test_ssrf_blocks_localhost(self):
        assert is_ssrf_target("http://127.0.0.1/admin") is True

    def test_ssrf_blocks_private_10(self):
        assert is_ssrf_target("http://10.0.0.1/api") is True

    def test_ssrf_blocks_private_172(self):
        assert is_ssrf_target("http://172.16.0.1/api") is True

    def test_ssrf_blocks_private_192(self):
        assert is_ssrf_target("http://192.168.1.1/api") is True

    def test_ssrf_blocks_zero(self):
        assert is_ssrf_target("http://0.0.0.0/") is True

    @patch("socket.getaddrinfo", return_value=[
        (2, 1, 6, '', ('93.184.216.34', 0))
    ])
    def test_ssrf_allows_public_ip(self, mock_dns):
        assert is_ssrf_target("http://example.com/page") is False

    def test_ssrf_blocks_ipv6_loopback(self):
        assert is_ssrf_target("http://[::1]/admin") is True

    def test_ssrf_no_hostname(self):
        assert is_ssrf_target("not-a-url") is True

    def test_invisible_unicode_stripping(self):
        text = "Hello\u200bWorld\u200c\u200d\ufeffInvisible"
        clean = strip_invisible_unicode(text)
        assert clean == "HelloWorldInvisible"

    def test_strip_normal_text_unchanged(self):
        text = "Normal text with spaces and punctuation!"
        assert strip_invisible_unicode(text) == text

    def test_domain_blocking(self):
        err = _check_domain_policy("http://evil.com/page", blocked=["evil.com"])
        assert err is not None
        assert "blocked" in err

    def test_domain_blocking_subdomain(self):
        err = _check_domain_policy("http://sub.evil.com/page", blocked=["evil.com"])
        assert err is not None

    def test_domain_allowlist(self):
        err = _check_domain_policy("http://unknown.com/page", allowed=["safe.com"])
        assert err is not None
        assert "allowlist" in err

    def test_domain_allowlist_match(self):
        err = _check_domain_policy("http://safe.com/page", allowed=["safe.com"])
        assert err is None

    def test_domain_no_policy(self):
        err = _check_domain_policy("http://anything.com/page")
        assert err is None


# ══════════════════════════════════════════════════════════════════════
# Content extraction tests
# ══════════════════════════════════════════════════════════════════════

class TestContentExtraction:
    def test_raw_text_strips_scripts(self):
        html = "<html><script>alert('xss')</script><p>Hello World</p></html>"
        result = _extract_raw_text(html)
        assert "alert" not in result
        assert "Hello World" in result

    def test_raw_text_strips_styles(self):
        html = "<style>.hidden{display:none}</style><p>Content</p>"
        result = _extract_raw_text(html)
        assert ".hidden" not in result
        assert "Content" in result

    def test_raw_text_converts_headings(self):
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        result = _extract_raw_text(html)
        assert "# Title" in result
        assert "## Subtitle" in result

    def test_raw_text_converts_links(self):
        html = '<a href="https://example.com">Link</a>'
        result = _extract_raw_text(html)
        assert "[Link](https://example.com)" in result

    def test_raw_text_empty_html(self):
        assert _extract_raw_text("") == ""

    def test_get_title_from_html(self):
        html = "<html><head><title>My Page</title></head></html>"
        assert _get_title_from_html(html) == "My Page"

    def test_get_title_missing(self):
        html = "<html><head></head></html>"
        assert _get_title_from_html(html) == ""

    def test_decode_entities(self):
        assert _decode_entities("&amp; &lt; &gt; &quot;") == '& < > "'
        assert _decode_entities("&#39;") == "'"

    def test_trafilatura_import_error(self):
        with patch.dict("sys.modules", {"trafilatura": None}):
            result = _extract_trafilatura("<p>text</p>", "http://example.com")
            assert result is None

    def test_readability_import_error(self):
        with patch.dict("sys.modules", {"readability": None}):
            result = _extract_readability("<p>text</p>", "http://example.com")
            assert result is None

    def test_beautifulsoup_import_error(self):
        with patch.dict("sys.modules", {"bs4": None}):
            result = _extract_beautifulsoup("<p>text</p>", "http://example.com")
            assert result is None

    def test_beautifulsoup_extracts_content(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = """<html><head><title>Test Page</title></head>
        <body><h1>Main Title</h1><p>Paragraph one.</p>
        <p>Paragraph two with enough text to pass the 50-char threshold for the extractor check.</p>
        </body></html>"""
        result = _extract_beautifulsoup(html, "http://example.com")
        assert result is not None
        assert "Main Title" in result

    def test_beautifulsoup_removes_scripts(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = """<html><body>
        <script>evil();</script>
        <nav>Nav bar</nav>
        <p>Real content that should be extracted because it is long enough.</p>
        </body></html>"""
        result = _extract_beautifulsoup(html, "http://example.com")
        if result:
            assert "evil" not in result


# ══════════════════════════════════════════════════════════════════════
# web_fetch tests
# ══════════════════════════════════════════════════════════════════════

class TestWebFetch:
    async def test_fetch_ssrf_blocked(self):
        result = await web_fetch("http://127.0.0.1/admin")
        assert result.error
        assert "SSRF" in result.error

    async def test_fetch_ssrf_disabled(self):
        """When SSRF protection is disabled, private IPs are allowed."""
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (b"<p>test</p>", 200, {"content-type": "text/html"})
            result = await web_fetch("http://127.0.0.1/admin",
                                     config={"security": {"ssrf_protection": False}})
            assert not result.error or "SSRF" not in result.error

    async def test_fetch_domain_blocked(self):
        result = await web_fetch("http://evil.com/page",
                                 config={"security": {"blocked_domains": ["evil.com"]}})
        assert result.error
        assert "blocked" in result.error

    async def test_fetch_success(self):
        html = b"<html><head><title>Test</title></head><body><p>Hello world content here with enough text to pass threshold.</p></body></html>"
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_fetch("http://example.com/page")
                assert result.status_code == 200
                assert result.title == "Test"

    async def test_fetch_http_error(self):
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (b"Not Found", 404, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_fetch("http://example.com/missing")
                assert result.error
                assert "404" in result.error

    async def test_fetch_timeout(self):
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = TimeoutError("timed out")
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_fetch("http://slow.example.com/page")
                assert result.error
                assert "failed" in result.error.lower()

    async def test_fetch_with_cache(self):
        cache = WebCache(default_ttl=60)
        html = b"<html><head><title>Cached</title></head><body><p>Body text content that is long enough for extraction.</p></body></html>"
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                r1 = await web_fetch("http://example.com/cached", cache=cache)
                assert not r1.cached
                r2 = await web_fetch("http://example.com/cached", cache=cache)
                assert r2.cached
                assert mock_get.call_count == 1  # Only one actual request

    async def test_fetch_strips_invisible_unicode(self):
        html = b"<p>Hello\xc2\xa0World\xe2\x80\x8bInvisible</p>"  # \u00a0 (nbsp) + \u200b (zwsp)
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_fetch("http://example.com/unicode")
                assert "\u200b" not in result.content

    async def test_fetch_content_truncation(self):
        long_content = "x" * 100000
        html = f"<p>{long_content}</p>".encode()
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_fetch("http://example.com/big",
                                         config={"max_extract_length": 1000})
                assert len(result.content) <= 1000

    async def test_fetch_firecrawl_fallback(self):
        """When all extractors return None and firecrawl is enabled."""
        html = b"<html></html>"  # Minimal content — extractors will return None
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                with patch("liteagent.web._extract_firecrawl", new_callable=AsyncMock) as mock_fc:
                    mock_fc.return_value = "Firecrawl extracted content that is long enough."
                    result = await web_fetch(
                        "http://example.com/js-heavy",
                        config={"fetch": {"firecrawl": {"enabled": True, "api_key_env": "FC_KEY"}},
                                "security": {"ssrf_protection": False}})


# ══════════════════════════════════════════════════════════════════════
# web_search tests
# ══════════════════════════════════════════════════════════════════════

class TestWebSearch:
    async def test_search_brave(self):
        brave_resp = json.dumps({
            "web": {"results": [
                {"title": "Result 1", "url": "http://r1.com", "description": "Desc 1"},
                {"title": "Result 2", "url": "http://r2.com", "description": "Desc 2"},
            ]}
        }).encode()
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (brave_resp, 200, {})
            results = await _search_brave("test query", 5, "fake-key")
            assert len(results) == 2
            assert results[0].title == "Result 1"
            assert results[0].source == "brave"

    async def test_search_brave_error(self):
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (b"error", 401, {})
            with pytest.raises(RuntimeError, match="HTTP 401"):
                await _search_brave("test", 5, "bad-key")

    async def test_search_duckduckgo_import_error(self):
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            with pytest.raises(RuntimeError, match="not installed"):
                await _search_duckduckgo("test", 5)

    async def test_search_duckduckgo(self):
        mock_ddgs = MagicMock()
        mock_ddgs.return_value.text.return_value = [
            {"title": "DDG Result", "href": "http://ddg.com", "body": "DDG desc"},
        ]
        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs
        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            results = await _search_duckduckgo("test", 5)
            assert len(results) == 1
            assert results[0].source == "duckduckgo"

    async def test_search_tavily(self):
        tavily_resp = {
            "results": [
                {"title": "Tavily Result", "url": "http://tav.com", "content": "Tavily desc"},
            ]
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (tavily_resp, 200)
            results = await _search_tavily("test", 5, "fake-key")
            assert len(results) == 1
            assert results[0].source == "tavily"

    async def test_search_searxng(self):
        searxng_resp = json.dumps({
            "results": [
                {"title": "SearX Result", "url": "http://sx.com", "content": "SearX desc"},
            ]
        }).encode()
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (searxng_resp, 200, {})
            results = await _search_searxng("test", 5, "http://localhost:8888")
            assert len(results) == 1
            assert results[0].source == "searxng"

    async def test_search_perplexity(self):
        pplx_resp = {
            "choices": [{"message": {"content": "AI answer"}}],
            "citations": ["http://cite1.com", "http://cite2.com"],
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (pplx_resp, 200)
            results = await _search_perplexity("test", 5, "fake-key")
            assert len(results) >= 1
            assert any(r.source == "perplexity" for r in results)

    async def test_search_provider_fallback(self):
        """First provider fails, falls back to second."""
        config = {
            "search": {
                "providers": ["brave", "duckduckgo"],
                "fallback": True,
            }
        }
        mock_ddgs = MagicMock()
        mock_ddgs.return_value.text.return_value = [
            {"title": "Fallback", "href": "http://fb.com", "body": "Fallback result"},
        ]
        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs
        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            # Brave fails (no key)
            resp = await web_search("test query", config=config)
            assert resp.provider == "duckduckgo"
            assert len(resp.results) == 1

    async def test_search_all_fail(self):
        config = {"search": {"providers": ["brave"], "fallback": False}}
        resp = await web_search("test", config=config)
        assert resp.error  # No API key for brave

    async def test_search_caching(self):
        cache = WebCache(default_ttl=60)
        config = {"search": {"providers": ["brave"]}}
        brave_resp = json.dumps({
            "web": {"results": [
                {"title": "Cached", "url": "http://c.com", "description": "Desc"},
            ]}
        }).encode()
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (brave_resp, 200, {})
            with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "key"}):
                r1 = await web_search("cached query", config=config, cache=cache)
                assert not r1.cached
                r2 = await web_search("cached query", config=config, cache=cache)
                assert r2.cached
                assert mock_get.call_count == 1

    async def test_search_count_capped(self):
        config = {"search": {"providers": ["brave"], "max_count": 10}}
        brave_resp = json.dumps({"web": {"results": []}}).encode()
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (brave_resp, 200, {})
            with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "key"}):
                resp = await web_search("test", config=config, count=100)
                # Count should be capped to max_count
                call_url = mock_get.call_args[0][0]
                assert "count=10" in call_url

    async def test_search_empty_query(self):
        resp = await web_search("", config={"search": {"providers": ["brave"]}})
        # Should still work or return error, not crash
        assert isinstance(resp, SearchResponse)

    async def test_detect_providers_with_env(self):
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "key123"}):
            providers = _detect_available_providers({})
            assert "brave" in providers

    async def test_detect_providers_default_duckduckgo(self):
        with patch.dict("os.environ", {}, clear=True):
            providers = _detect_available_providers({})
            assert "duckduckgo" in providers


# ══════════════════════════════════════════════════════════════════════
# web_crawl tests
# ══════════════════════════════════════════════════════════════════════

class TestWebCrawl:
    async def test_crawl_single_page(self):
        with patch("liteagent.web.web_fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = FetchResult(
                url="http://example.com",
                title="Home", content="Hello world",
                extractor="raw", status_code=200)
            with patch("liteagent.web._extract_links", new_callable=AsyncMock) as mock_links:
                mock_links.return_value = []
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_depth=0, max_pages=1)
                    assert len(results) == 1
                    assert results[0].title == "Home"

    async def test_crawl_depth_limit(self):
        call_count = 0
        async def mock_fetch_fn(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return FetchResult(url=url, title=f"Page {call_count}",
                              content="Content", extractor="raw", status_code=200)

        async def mock_links_fn(url, config):
            if "page2" not in url:
                return ["http://example.com/page2"]
            return ["http://example.com/page3"]

        with patch("liteagent.web.web_fetch", side_effect=mock_fetch_fn):
            with patch("liteagent.web._extract_links", side_effect=mock_links_fn):
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_depth=1, max_pages=10,
                                             config={"crawl": {"rate_limit_ms": 0}})
                    depths = {r.depth for r in results}
                    assert max(depths) <= 1

    async def test_crawl_max_pages_limit(self):
        async def mock_fetch_fn(url, **kwargs):
            return FetchResult(url=url, title="Page",
                              content="Content", extractor="raw", status_code=200)

        async def mock_links_fn(url, config):
            return [f"http://example.com/page{i}" for i in range(50)]

        with patch("liteagent.web.web_fetch", side_effect=mock_fetch_fn):
            with patch("liteagent.web._extract_links", side_effect=mock_links_fn):
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_pages=3,
                                             config={"crawl": {"rate_limit_ms": 0}})
                    assert len(results) <= 3

    async def test_crawl_same_domain_only(self):
        async def mock_fetch_fn(url, **kwargs):
            return FetchResult(url=url, title="Page",
                              content="Content", extractor="raw", status_code=200)

        async def mock_links_fn(url, config):
            return ["http://example.com/internal", "http://other.com/external"]

        with patch("liteagent.web.web_fetch", side_effect=mock_fetch_fn):
            with patch("liteagent.web._extract_links", side_effect=mock_links_fn):
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_depth=1, max_pages=10,
                                             config={"crawl": {"rate_limit_ms": 0}})
                    for r in results:
                        assert "other.com" not in r.url

    async def test_crawl_handles_errors(self):
        call_count = 0
        async def mock_fetch_fn(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return FetchResult(url=url, error="HTTP 500")
            return FetchResult(url=url, title="OK",
                              content="Content", extractor="raw", status_code=200)

        with patch("liteagent.web.web_fetch", side_effect=mock_fetch_fn):
            with patch("liteagent.web._extract_links", new_callable=AsyncMock) as mock_links:
                mock_links.return_value = ["http://example.com/page2"]
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_depth=1, max_pages=5,
                                             config={"crawl": {"rate_limit_ms": 0}})
                    errors = [r for r in results if r.error]
                    assert len(errors) >= 1

    async def test_crawl_deduplicates(self):
        async def mock_fetch_fn(url, **kwargs):
            return FetchResult(url=url, title="Page",
                              content="Content", extractor="raw", status_code=200)

        async def mock_links_fn(url, config):
            # Returns same URL multiple times
            return ["http://example.com/page1", "http://example.com/page1"]

        with patch("liteagent.web.web_fetch", side_effect=mock_fetch_fn):
            with patch("liteagent.web._extract_links", side_effect=mock_links_fn):
                with patch("liteagent.web._parse_robots_txt", new_callable=AsyncMock) as mock_robots:
                    mock_robots.return_value = []
                    results = await web_crawl("http://example.com",
                                             max_depth=1, max_pages=10,
                                             config={"crawl": {"rate_limit_ms": 0}})
                    urls = [r.url for r in results]
                    assert len(urls) == len(set(urls))


# ══════════════════════════════════════════════════════════════════════
# web_extract tests
# ══════════════════════════════════════════════════════════════════════

class TestWebExtract:
    async def test_extract_ssrf_blocked(self):
        result = await web_extract("http://127.0.0.1/admin")
        assert result.error
        assert "SSRF" in result.error

    async def test_extract_requires_beautifulsoup(self):
        html = b"<html><body><p>test</p></body></html>"
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                with patch.dict("sys.modules", {"bs4": None}):
                    result = await web_extract("http://example.com")
                    assert result.error
                    assert "beautifulsoup4" in result.error

    async def test_extract_metadata(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = b"""<html><head>
        <title>Test Page</title>
        <meta name="description" content="A test page">
        <meta property="og:title" content="OG Title">
        <meta property="og:image" content="http://img.com/og.jpg">
        </head><body></body></html>"""
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {"content-type": "text/html"})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com")
                assert result.title == "Test Page"
                assert result.description == "A test page"
                assert "og:title" in result.og_tags

    async def test_extract_links_and_images(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = b"""<html><body>
        <a href="/about">About</a>
        <a href="http://external.com">External</a>
        <img src="/img.jpg" alt="Photo">
        </body></html>"""
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com")
                assert len(result.links) >= 2
                assert len(result.images) >= 1

    async def test_extract_headings(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = b"<html><body><h1>Title</h1><h2>Subtitle</h2><h3>Section</h3></body></html>"
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com")
                assert len(result.headings) == 3
                assert "h1: Title" in result.headings

    async def test_extract_tables(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = b"""<html><body><table>
        <tr><th>Name</th><th>Value</th></tr>
        <tr><td>A</td><td>1</td></tr>
        </table></body></html>"""
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com")
                assert len(result.tables) == 1
                assert result.tables[0][0] == ["Name", "Value"]

    async def test_extract_with_css_selector(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")
        html = b"""<html><body>
        <div class="header"><h1>Header</h1></div>
        <div class="content"><h2>Content Title</h2><p>Text</p></div>
        </body></html>"""
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (html, 200, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com",
                                           selectors={"css": ".content"})
                assert any("Content Title" in h for h in result.headings)
                # Should NOT contain the header h1
                assert not any("Header" in h for h in result.headings)

    async def test_extract_http_error(self):
        with patch("liteagent.web._async_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (b"", 403, {})
            with patch("liteagent.web.is_ssrf_target", return_value=False):
                result = await web_extract("http://example.com")
                assert result.error
                assert "403" in result.error


# ══════════════════════════════════════════════════════════════════════
# HTTP client tests
# ══════════════════════════════════════════════════════════════════════

class TestAsyncGet:
    async def test_fallback_to_urllib(self):
        """When httpx is not available, falls back to urllib."""
        with patch.dict("sys.modules", {"httpx": None}):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = b"response data"
                mock_resp.status = 200
                mock_resp.headers = MagicMock()
                mock_resp.headers.items.return_value = [("Content-Type", "text/html")]
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp
                body, status, headers = await _async_get("http://example.com")
                assert body == b"response data"
                assert status == 200


# ══════════════════════════════════════════════════════════════════════
# Tool registration tests
# ══════════════════════════════════════════════════════════════════════

class TestToolRegistration:
    def test_web_tools_registered(self):
        """Web tools are registered when agent initializes."""
        from liteagent.tools import ToolRegistry
        from unittest.mock import patch as _patch

        with _patch("liteagent.agent.create_provider") as mock_prov, \
             _patch("liteagent.agent.MemorySystem") as mock_mem, \
             _patch("liteagent.agent.get_soul_prompt", return_value="test"), \
             _patch("liteagent.agent.load_plugins", return_value=[]), \
             _patch("liteagent.agent.CircuitBreaker"), \
             _patch("liteagent.agent.SkillRegistry"):
            mock_prov.return_value = MagicMock()
            mock_mem.return_value = MagicMock()
            mock_mem.return_value.db = MagicMock()

            from liteagent.agent import LiteAgent
            agent = LiteAgent({"agent": {"provider": "anthropic"}})
            assert "web_fetch" in agent.tools._tools
            assert "web_search" in agent.tools._tools
            assert "web_crawl" in agent.tools._tools
            assert "web_extract" in agent.tools._tools

    def test_web_tools_disabled(self):
        """Web tools are NOT registered when web.enabled=false."""
        from unittest.mock import patch as _patch

        with _patch("liteagent.agent.create_provider") as mock_prov, \
             _patch("liteagent.agent.MemorySystem") as mock_mem, \
             _patch("liteagent.agent.get_soul_prompt", return_value="test"), \
             _patch("liteagent.agent.load_plugins", return_value=[]), \
             _patch("liteagent.agent.CircuitBreaker"), \
             _patch("liteagent.agent.SkillRegistry"):
            mock_prov.return_value = MagicMock()
            mock_mem.return_value = MagicMock()
            mock_mem.return_value.db = MagicMock()

            from liteagent.agent import LiteAgent
            agent = LiteAgent({"agent": {"provider": "anthropic"},
                               "web": {"enabled": False}})
            assert "web_fetch" not in agent.tools._tools


# ══════════════════════════════════════════════════════════════════════
# Data class tests
# ══════════════════════════════════════════════════════════════════════

class TestDataClasses:
    def test_fetch_result_defaults(self):
        r = FetchResult(url="http://example.com")
        assert r.url == "http://example.com"
        assert r.content == ""
        assert r.error == ""
        assert r.cached is False

    def test_search_result(self):
        r = SearchResult(title="Test", url="http://test.com", snippet="desc", source="brave")
        assert r.title == "Test"
        assert r.source == "brave"

    def test_search_response(self):
        r = SearchResponse(query="test", results=[
            SearchResult(title="R1", url="http://r1.com"),
        ], provider="brave")
        assert len(r.results) == 1
        assert r.provider == "brave"

    def test_crawl_result(self):
        r = CrawlResult(url="http://example.com", depth=1, content="text")
        assert r.depth == 1

    def test_extract_result(self):
        r = ExtractResult(url="http://example.com")
        assert r.headings == []
        assert r.links == []
        assert r.tables == []
        assert r.og_tags == {}


# ══════════════════════════════════════════════════════════════════════
# HTML sanitization tests (Phase 1: enhanced security)
# ══════════════════════════════════════════════════════════════════════

_bs4_available = False
try:
    import bs4
    _bs4_available = True
except ImportError:
    pass

bs4_required = pytest.mark.skipif(not _bs4_available, reason="beautifulsoup4 not installed")


@bs4_required
class TestSanitizeHtml:
    def test_removes_display_none(self):
        html = '<div><p style="display:none">hidden injection</p><p>visible</p></div>'
        result = sanitize_html_for_llm(html)
        assert "hidden injection" not in result
        assert "visible" in result

    def test_removes_visibility_hidden(self):
        html = '<div><span style="visibility: hidden">secret</span><span>public</span></div>'
        result = sanitize_html_for_llm(html)
        assert "secret" not in result
        assert "public" in result

    def test_removes_aria_hidden(self):
        html = '<div><span aria-hidden="true">screen reader only</span><p>content</p></div>'
        result = sanitize_html_for_llm(html)
        assert "screen reader only" not in result
        assert "content" in result

    def test_removes_html_hidden_attribute(self):
        html = '<div><p hidden>hidden paragraph</p><p>visible paragraph</p></div>'
        result = sanitize_html_for_llm(html)
        assert "hidden paragraph" not in result
        assert "visible paragraph" in result

    def test_removes_hidden_input(self):
        html = '<form><input type="hidden" value="inject_me"><input type="text" value="normal"></form>'
        result = sanitize_html_for_llm(html)
        assert "inject_me" not in result
        assert "normal" in result

    def test_removes_sr_only_class(self):
        html = '<div><span class="sr-only">screen reader text</span><p>normal</p></div>'
        result = sanitize_html_for_llm(html)
        assert "screen reader text" not in result
        assert "normal" in result

    def test_removes_d_none_class(self):
        html = '<div class="d-none">bootstrap hidden</div><div>visible</div>'
        result = sanitize_html_for_llm(html)
        assert "bootstrap hidden" not in result
        assert "visible" in result

    def test_removes_opacity_zero(self):
        html = '<div><p style="opacity: 0">transparent</p><p>opaque</p></div>'
        result = sanitize_html_for_llm(html)
        assert "transparent" not in result
        assert "opaque" in result

    def test_removes_offscreen_positioning(self):
        html = '<div><span style="left: -9999px">offscreen</span><span>onscreen</span></div>'
        result = sanitize_html_for_llm(html)
        assert "offscreen" not in result
        assert "onscreen" in result

    def test_removes_text_indent_offscreen(self):
        html = '<div><p style="text-indent: -99999px">indent hidden</p><p>visible</p></div>'
        result = sanitize_html_for_llm(html)
        assert "indent hidden" not in result
        assert "visible" in result

    def test_removes_clip_path_inset(self):
        html = '<div><span style="clip-path: inset(100%)">clipped</span><span>shown</span></div>'
        result = sanitize_html_for_llm(html)
        assert "clipped" not in result
        assert "shown" in result

    def test_removes_transform_scale_zero(self):
        html = '<div><p style="transform: scale(0)">scaled</p><p>normal</p></div>'
        result = sanitize_html_for_llm(html)
        assert "scaled" not in result
        assert "normal" in result

    def test_removes_noscript_and_template(self):
        html = '<div><template>template content</template><noscript>noscript</noscript><p>real</p></div>'
        result = sanitize_html_for_llm(html)
        assert "template content" not in result
        assert "noscript" not in result
        assert "real" in result

    def test_removes_html_comments(self):
        html = '<div><!-- hidden comment with instructions --><p>visible</p></div>'
        result = sanitize_html_for_llm(html)
        assert "hidden comment" not in result
        assert "visible" in result

    def test_preserves_normal_content(self):
        html = '<div><h1>Title</h1><p>Normal paragraph with content.</p></div>'
        result = sanitize_html_for_llm(html)
        assert "Title" in result
        assert "Normal paragraph" in result

    def test_font_size_zero(self):
        html = '<div><span style="font-size: 0px">tiny</span><span>normal</span></div>'
        result = sanitize_html_for_llm(html)
        assert "tiny" not in result
        assert "normal" in result

    def test_multiple_hidden_elements(self):
        html = '''<div>
            <p style="display:none">hidden1</p>
            <p aria-hidden="true">hidden2</p>
            <p class="visually-hidden">hidden3</p>
            <p>visible content</p>
        </div>'''
        result = sanitize_html_for_llm(html)
        assert "hidden1" not in result
        assert "hidden2" not in result
        assert "hidden3" not in result
        assert "visible content" in result


class TestSanitizeHtmlFallback:
    def test_fallback_without_beautifulsoup(self):
        html = '<div><p style="display:none">hidden</p><p>visible</p></div>'
        with patch.dict("sys.modules", {"bs4": None}):
            result = sanitize_html_for_llm(html)
            # Without bs4, returns original HTML unchanged
            assert result == html


@bs4_required
class TestShouldRemoveElement:
    def _make_elem(self, tag="div", style=None, classes=None, attrs=None):
        """Create a BeautifulSoup element for testing."""
        from bs4 import BeautifulSoup
        html_parts = [f"<{tag}"]
        if style:
            html_parts.append(f' style="{style}"')
        if classes:
            html_parts.append(f' class="{" ".join(classes)}"')
        if attrs:
            for k, v in attrs.items():
                html_parts.append(f' {k}="{v}"')
        html_parts.append(f">content</{tag}>")
        soup = BeautifulSoup("".join(html_parts), "html.parser")
        return soup.find(tag)

    def test_template_tag(self):
        elem = self._make_elem("template")
        assert _should_remove_element(elem) is True

    def test_noscript_tag(self):
        elem = self._make_elem("noscript")
        assert _should_remove_element(elem) is True

    def test_aria_hidden(self):
        elem = self._make_elem(attrs={"aria-hidden": "true"})
        assert _should_remove_element(elem) is True

    def test_hidden_attribute(self):
        elem = self._make_elem(attrs={"hidden": ""})
        assert _should_remove_element(elem) is True

    def test_hidden_input(self):
        elem = self._make_elem("input", attrs={"type": "hidden"})
        assert _should_remove_element(elem) is True

    def test_sr_only_class(self):
        elem = self._make_elem(classes=["sr-only"])
        assert _should_remove_element(elem) is True

    def test_normal_div_not_removed(self):
        elem = self._make_elem()
        assert _should_remove_element(elem) is False

    def test_normal_paragraph(self):
        elem = self._make_elem("p", style="color: red; font-weight: bold")
        assert _should_remove_element(elem) is False


class TestCheckHtmlDepth:
    def test_shallow_html(self):
        html = "<div><p><span>text</span></p></div>"
        assert check_html_depth(html) is True

    def test_deeply_nested_html(self):
        html = "<div>" * 600 + "content" + "</div>" * 600
        assert check_html_depth(html) is False

    def test_exactly_at_limit(self):
        # At exactly the limit, depth == max_depth, check is `> max_depth` so it passes
        html = "<div>" * MAX_HTML_NESTING_DEPTH + "ok" + "</div>" * MAX_HTML_NESTING_DEPTH
        assert check_html_depth(html) is True  # depth == limit is OK

    def test_just_under_limit(self):
        depth = MAX_HTML_NESTING_DEPTH - 1
        html = "<div>" * depth + "ok" + "</div>" * depth
        assert check_html_depth(html) is True

    def test_empty_html(self):
        assert check_html_depth("") is True

    def test_flat_html(self):
        html = "<p>one</p><p>two</p><p>three</p>"
        assert check_html_depth(html) is True

    def test_custom_max_depth(self):
        html = "<div><div><div><div>deep</div></div></div></div>"
        assert check_html_depth(html, max_depth=3) is False
        assert check_html_depth(html, max_depth=10) is True

    def test_self_closing_tags_ignored(self):
        html = "<div><br/><img/><p>text</p></div>"
        assert check_html_depth(html) is True

    def test_comments_ignored(self):
        html = "<div><!-- comment --><p>text</p></div>"
        assert check_html_depth(html) is True


class TestWrapUntrustedContent:
    def test_basic_wrapping(self):
        content = "Hello world"
        result = wrap_untrusted_content(content, "https://example.com")
        assert '<web_content source="https://example.com" trust="untrusted">' in result
        assert "Hello world" in result
        assert "</web_content>" in result

    def test_escapes_quotes_in_url(self):
        content = "text"
        result = wrap_untrusted_content(content, 'https://example.com/page?a="b"')
        assert '&quot;' in result
        assert "text" in result

    def test_preserves_content(self):
        content = "Line 1\nLine 2\n# Heading\n- List item"
        result = wrap_untrusted_content(content, "https://example.com")
        assert "Line 1\nLine 2" in result
        assert "# Heading" in result

    def test_wrapping_structure(self):
        result = wrap_untrusted_content("content", "https://x.com")
        lines = result.strip().split("\n")
        assert lines[0].startswith("<web_content")
        assert lines[-1] == "</web_content>"


# ══════════════════════════════════════════════════════════════════════
# New search providers (Phase 2: Grok, Gemini, Kimi)
# ══════════════════════════════════════════════════════════════════════

class TestSearchGrok:
    async def test_grok_search_success(self):
        grok_resp = {
            "output": [
                {
                    "type": "web_search_call",
                    "results": [
                        {"title": "Result 1", "url": "https://r1.com", "snippet": "Desc 1"},
                        {"title": "Result 2", "url": "https://r2.com", "snippet": "Desc 2"},
                    ],
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "AI summary of results"},
                    ],
                },
            ]
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (grok_resp, 200)
            results = await _search_grok("test query", 5, "fake-xai-key")
            assert len(results) >= 2
            # First result should be AI answer
            assert results[0].title == "Grok AI Answer"
            assert results[0].source == "grok"
            # Actual search results
            search_results = [r for r in results if r.url]
            assert len(search_results) == 2
            assert search_results[0].source == "grok"

    async def test_grok_search_http_error(self):
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ({}, 429)
            with pytest.raises(RuntimeError, match="HTTP 429"):
                await _search_grok("test", 5, "key")

    async def test_grok_search_no_results(self):
        grok_resp = {"output": []}
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (grok_resp, 200)
            results = await _search_grok("obscure query", 5, "key")
            assert results == []

    async def test_grok_search_count_limit(self):
        grok_resp = {
            "output": [
                {
                    "type": "web_search_call",
                    "results": [
                        {"title": f"R{i}", "url": f"https://r{i}.com", "snippet": f"D{i}"}
                        for i in range(10)
                    ],
                },
            ]
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (grok_resp, 200)
            results = await _search_grok("test", 3, "key")
            assert len(results) <= 3


class TestSearchGemini:
    async def test_gemini_search_success(self):
        gemini_resp = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "AI grounded answer"}],
                },
                "groundingMetadata": {
                    "groundingChunks": [
                        {"web": {"uri": "https://site1.com", "title": "Site 1"}},
                        {"web": {"uri": "https://site2.com", "title": "Site 2"}},
                    ],
                },
            }]
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (gemini_resp, 200)
            results = await _search_gemini("test query", 5, "fake-google-key")
            assert len(results) >= 2
            # First result should be AI answer
            assert results[0].title == "Gemini AI Answer"
            assert results[0].source == "gemini"
            # Grounding results
            urls = [r.url for r in results if r.url]
            assert "https://site1.com" in urls

    async def test_gemini_search_http_error(self):
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ({}, 403)
            with pytest.raises(RuntimeError, match="HTTP 403"):
                await _search_gemini("test", 5, "bad-key")

    async def test_gemini_search_no_candidates(self):
        gemini_resp = {"candidates": []}
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (gemini_resp, 200)
            results = await _search_gemini("test", 5, "key")
            assert results == []

    async def test_gemini_search_no_grounding(self):
        gemini_resp = {
            "candidates": [{
                "content": {"parts": [{"text": "Answer without sources"}]},
                # No groundingMetadata
            }]
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (gemini_resp, 200)
            results = await _search_gemini("test", 5, "key")
            assert len(results) == 1
            assert results[0].title == "Gemini AI Answer"


class TestSearchKimi:
    async def test_kimi_search_success(self):
        kimi_resp = {
            "choices": [{
                "message": {"content": "Kimi AI answer to your query"},
            }],
            "search_results": [
                {"title": "Kimi Result 1", "url": "https://kr1.com", "content": "KR1 desc"},
                {"title": "Kimi Result 2", "url": "https://kr2.com", "content": "KR2 desc"},
            ],
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (kimi_resp, 200)
            results = await _search_kimi("test query", 5, "fake-moonshot-key")
            assert len(results) >= 2
            # First should be AI answer
            assert results[0].title == "Kimi AI Answer"
            assert results[0].source == "kimi"
            # Search results
            search_results = [r for r in results if r.url]
            assert len(search_results) == 2

    async def test_kimi_search_http_error(self):
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ({}, 500)
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await _search_kimi("test", 5, "key")

    async def test_kimi_search_no_results(self):
        kimi_resp = {"choices": [], "search_results": []}
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (kimi_resp, 200)
            results = await _search_kimi("test", 5, "key")
            assert results == []

    async def test_kimi_search_count_limit(self):
        kimi_resp = {
            "choices": [{"message": {"content": "answer"}}],
            "search_results": [
                {"title": f"KR{i}", "url": f"https://kr{i}.com", "content": f"desc{i}"}
                for i in range(10)
            ],
        }
        with patch("liteagent.web._async_post_json", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (kimi_resp, 200)
            results = await _search_kimi("test", 3, "key")
            assert len(results) <= 3


class TestNewSearchProvidersInRegistry:
    def test_grok_in_registry(self):
        assert "grok" in SEARCH_PROVIDERS
        assert SEARCH_PROVIDERS["grok"]["key_env"] == "XAI_API_KEY"
        assert SEARCH_PROVIDERS["grok"]["needs_key"] is True

    def test_gemini_in_registry(self):
        assert "gemini" in SEARCH_PROVIDERS
        assert SEARCH_PROVIDERS["gemini"]["key_env"] == "GOOGLE_API_KEY"
        assert SEARCH_PROVIDERS["gemini"]["needs_key"] is True

    def test_kimi_in_registry(self):
        assert "kimi" in SEARCH_PROVIDERS
        assert SEARCH_PROVIDERS["kimi"]["key_env"] == "MOONSHOT_API_KEY"
        assert SEARCH_PROVIDERS["kimi"]["needs_key"] is True

    def test_all_providers_have_fn(self):
        for name, info in SEARCH_PROVIDERS.items():
            assert callable(info["fn"]), f"Provider {name} missing callable fn"

    async def test_detect_grok_with_env(self):
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            providers = _detect_available_providers({})
            assert "grok" in providers

    async def test_detect_gemini_with_env(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            providers = _detect_available_providers({})
            assert "gemini" in providers

    async def test_detect_kimi_with_env(self):
        with patch.dict("os.environ", {"MOONSHOT_API_KEY": "test-key"}):
            providers = _detect_available_providers({})
            assert "kimi" in providers


# ══════════════════════════════════════════════════════════════════════
# _resolve_search_key tests — keys.json integration
# ══════════════════════════════════════════════════════════════════════

class TestResolveSearchKey:
    """Test _resolve_search_key resolves keys from multiple sources."""

    def test_inline_config_has_priority(self):
        """Inline config api_key takes priority over everything."""
        info = {"key_env": "XAI_API_KEY", "needs_key": True}
        search_cfg = {"grok": {"api_key": "inline-key-123"}}
        with patch.dict("os.environ", {"XAI_API_KEY": "env-key"}):
            result = _resolve_search_key("grok", info, search_cfg)
        assert result == "inline-key-123"

    def test_env_var_fallback(self):
        """Env var is used when inline config is missing."""
        info = {"key_env": "XAI_API_KEY", "needs_key": True}
        with patch.dict("os.environ", {"XAI_API_KEY": "env-key-456"}):
            result = _resolve_search_key("grok", info, {})
        assert result == "env-key-456"

    def test_keys_json_fallback(self):
        """keys.json is used when both inline and env are missing."""
        info = {"key_env": "XAI_API_KEY", "needs_key": True}
        with patch.dict("os.environ", {}, clear=True):
            with patch("liteagent.config.get_api_key", return_value="keys-json-789"):
                result = _resolve_search_key("grok", info, {})
        assert result == "keys-json-789"

    def test_returns_empty_when_no_key(self):
        """Returns empty string when no key found anywhere."""
        info = {"key_env": "NONEXISTENT_KEY_12345", "needs_key": True}
        with patch.dict("os.environ", {}, clear=True):
            with patch("liteagent.config.get_api_key", return_value=None):
                result = _resolve_search_key("nonexistent", info, {})
        assert result == ""

    def test_detect_providers_uses_keys_json(self):
        """_detect_available_providers finds providers via keys.json."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("liteagent.config.get_api_key") as mock_gpk:
                mock_gpk.side_effect = lambda name: "key-from-json" if name == "grok" else None
                providers = _detect_available_providers({})
                assert "grok" in providers

    def test_detect_providers_gemini_via_keys_json(self):
        """_detect_available_providers finds gemini via keys.json."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("liteagent.config.get_api_key") as mock_gpk:
                mock_gpk.side_effect = lambda name: "gem-key" if name == "gemini" else None
                providers = _detect_available_providers({})
                assert "gemini" in providers
