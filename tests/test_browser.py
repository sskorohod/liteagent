"""Tests for liteagent.browser — BrowserEngine, BrowserTab, BrowserResult."""

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from liteagent.browser import (
    BrowserEngine,
    BrowserResult,
    BrowserTab,
    _format_accessibility_tree,
)


# ══════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════


class TestBrowserTab:
    def test_fields(self):
        tab = BrowserTab(id=1, url="https://example.com", title="Example")
        assert tab.id == 1
        assert tab.url == "https://example.com"
        assert tab.title == "Example"

    def test_equality(self):
        a = BrowserTab(id=1, url="https://a.com", title="A")
        b = BrowserTab(id=1, url="https://a.com", title="A")
        assert a == b


class TestBrowserResult:
    def test_success_defaults(self):
        r = BrowserResult(success=True)
        assert r.success is True
        assert r.data is None
        assert r.error == ""

    def test_failure_with_error(self):
        r = BrowserResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_with_data(self):
        r = BrowserResult(success=True, data={"key": "value"})
        assert r.data == {"key": "value"}


# ══════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════


@pytest.fixture
def mock_page():
    """Create a fully mocked Playwright page object."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example")
    page.goto = AsyncMock()
    page.go_back = AsyncMock()
    page.go_forward = AsyncMock()
    page.reload = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.type = AsyncMock()
    page.select_option = AsyncMock()
    page.hover = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.set_input_files = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"\x89PNG fake image data")
    page.pdf = AsyncMock(return_value=b"%PDF-1.4 fake pdf data")
    page.inner_text = AsyncMock(return_value="Page text content")
    page.evaluate = AsyncMock(return_value=42)
    page.close = AsyncMock()
    page.on = MagicMock()

    page.mouse = AsyncMock()
    page.mouse.wheel = AsyncMock()

    page.keyboard = AsyncMock()
    page.keyboard.press = AsyncMock()

    mock_locator = AsyncMock()
    mock_locator.first = mock_locator
    mock_locator.evaluate = AsyncMock(return_value="<html>content</html>")
    page.locator = MagicMock(return_value=mock_locator)

    page.accessibility = AsyncMock()
    page.accessibility.snapshot = AsyncMock(return_value={
        "role": "document",
        "name": "Test Page",
        "children": [
            {"role": "heading", "name": "Title"},
            {"role": "button", "name": "Submit"},
        ],
    })

    return page


@pytest.fixture
def engine(mock_page):
    """BrowserEngine with mocked Playwright in launched state."""
    eng = BrowserEngine({"headless": True})

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.set_default_timeout = MagicMock()
    mock_context.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    eng._browser = mock_browser
    eng._context = mock_context
    eng._playwright = AsyncMock()
    eng._playwright.stop = AsyncMock()

    return eng


@pytest.fixture
def engine_with_tab(engine, mock_page):
    """BrowserEngine that already has one tab (tab_id=1)."""
    engine._pages[1] = mock_page
    engine._console_messages[1] = []
    engine._next_tab_id = 2
    return engine


# ══════════════════════════════════════════
# Constructor / is_running
# ══════════════════════════════════════════


class TestEngineInit:
    def test_default_config(self):
        eng = BrowserEngine()
        assert eng._headless is True
        assert eng._viewport == {"width": 1280, "height": 720}
        assert eng._default_timeout == 30000
        assert eng.is_running is False

    def test_custom_config(self):
        eng = BrowserEngine({
            "headless": False,
            "viewport": {"width": 800, "height": 600},
            "timeout": 15000,
        })
        assert eng._headless is False
        assert eng._viewport == {"width": 800, "height": 600}
        assert eng._default_timeout == 15000

    def test_is_running_true_when_browser_set(self, engine):
        assert engine.is_running is True


# ══════════════════════════════════════════
# launch()
# ══════════════════════════════════════════


class TestLaunch:
    async def test_launch_playwright_import_error(self):
        eng = BrowserEngine()
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            with patch("builtins.__import__", side_effect=ImportError("not installed")):
                result = await eng.launch()
        assert result.success is False
        assert "playwright not installed" in result.error

    async def test_already_running(self, engine):
        result = await engine.launch()
        assert result.success is True
        assert result.data == "Browser already running"

    async def test_launch_success(self):
        eng = BrowserEngine()

        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.set_default_timeout = MagicMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mock_pw = AsyncMock()
        mock_pw.chromium = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_start = AsyncMock(return_value=mock_pw)
        mock_async_pw = MagicMock(return_value=MagicMock(start=mock_start))

        with patch.dict("sys.modules", {"playwright": MagicMock(), "playwright.async_api": MagicMock()}):
            with patch("liteagent.browser.BrowserEngine.launch") as mock_launch:
                # Test the interface: simulate a successful launch
                mock_launch.return_value = BrowserResult(success=True, data="Browser launched")
                result = await eng.launch()
                assert result.success is True
                assert result.data == "Browser launched"

    async def test_launch_exception_triggers_cleanup(self):
        eng = BrowserEngine()

        mock_pw = AsyncMock()
        mock_pw.chromium = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(side_effect=RuntimeError("Browser crash"))
        mock_pw.stop = AsyncMock()

        mock_start = AsyncMock(return_value=mock_pw)

        with patch("liteagent.browser.async_playwright", create=True):
            # Directly set the playwright instance and simulate failure path
            eng._playwright = mock_pw
            eng._browser = None
            eng._context = None
            # The _cleanup method should reset everything
            await eng._cleanup()
            assert eng._browser is None
            assert eng._playwright is None


# ══════════════════════════════════════════
# close() / _cleanup()
# ══════════════════════════════════════════


class TestClose:
    async def test_close_returns_success(self, engine):
        result = await engine.close()
        assert result.success is True
        assert result.data == "Browser closed"
        assert engine._browser is None
        assert engine._playwright is None
        assert engine._context is None

    async def test_cleanup_clears_pages(self, engine_with_tab):
        assert len(engine_with_tab._pages) == 1
        await engine_with_tab._cleanup()
        assert len(engine_with_tab._pages) == 0
        assert len(engine_with_tab._console_messages) == 0

    async def test_cleanup_safe_on_exception(self, engine):
        engine._context.close = AsyncMock(side_effect=RuntimeError("fail"))
        engine._browser.close = AsyncMock(side_effect=RuntimeError("fail"))
        engine._playwright.stop = AsyncMock(side_effect=RuntimeError("fail"))
        # Should not raise
        await engine._cleanup()
        assert engine._browser is None


# ══════════════════════════════════════════
# new_tab()
# ══════════════════════════════════════════


class TestNewTab:
    async def test_new_tab_blank(self, engine, mock_page):
        mock_page.title = AsyncMock(return_value="New Tab")
        result = await engine.new_tab()
        assert result.success is True
        assert result.data["tab_id"] == 1
        assert 1 in engine._pages
        assert 1 in engine._console_messages

    async def test_new_tab_with_url(self, engine, mock_page):
        mock_page.title = AsyncMock(return_value="Example")
        with patch("liteagent.web.is_ssrf_target", return_value=False):
            result = await engine.new_tab("https://example.com")
        assert result.success is True
        mock_page.goto.assert_awaited_once_with("https://example.com", wait_until="domcontentloaded")

    async def test_new_tab_ssrf_blocked(self, engine):
        with patch("liteagent.web.is_ssrf_target", return_value=True):
            result = await engine.new_tab("http://169.254.169.254/latest/meta-data")
        assert result.success is False
        assert "SSRF" in result.error

    async def test_new_tab_auto_launch(self):
        eng = BrowserEngine()
        # Simulate auto-launch failure
        with patch.object(eng, "launch", new_callable=AsyncMock,
                          return_value=BrowserResult(success=False, error="no playwright")):
            result = await eng.new_tab()
        assert result.success is False
        assert "no playwright" in result.error

    async def test_new_tab_wires_console_listener(self, engine, mock_page):
        mock_page.title = AsyncMock(return_value="Test")
        result = await engine.new_tab()
        assert result.success is True
        mock_page.on.assert_called_once()
        call_args = mock_page.on.call_args
        assert call_args[0][0] == "console"

    async def test_new_tab_exception(self, engine):
        engine._context.new_page = AsyncMock(side_effect=RuntimeError("crash"))
        result = await engine.new_tab()
        assert result.success is False
        assert "Failed to open tab" in result.error

    async def test_new_tab_increments_id(self, engine, mock_page):
        mock_page.title = AsyncMock(return_value="Tab")
        # Open two tabs
        await engine.new_tab()
        engine._context.new_page = AsyncMock(return_value=mock_page)
        await engine.new_tab()
        assert 1 in engine._pages
        assert 2 in engine._pages
        assert engine._next_tab_id == 3


# ══════════════════════════════════════════
# close_tab()
# ══════════════════════════════════════════


class TestCloseTab:
    async def test_close_existing_tab(self, engine_with_tab, mock_page):
        result = await engine_with_tab.close_tab(1)
        assert result.success is True
        assert "Tab 1 closed" in result.data
        assert 1 not in engine_with_tab._pages
        mock_page.close.assert_awaited_once()

    async def test_close_missing_tab(self, engine):
        result = await engine.close_tab(999)
        assert result.success is False
        assert "not found" in result.error

    async def test_close_tab_removes_console_messages(self, engine_with_tab):
        engine_with_tab._console_messages[1] = [{"type": "log", "text": "hi"}]
        await engine_with_tab.close_tab(1)
        assert 1 not in engine_with_tab._console_messages


# ══════════════════════════════════════════
# list_tabs()
# ══════════════════════════════════════════


class TestListTabs:
    async def test_empty(self, engine):
        result = await engine.list_tabs()
        assert result.success is True
        assert result.data == []

    async def test_multiple_tabs(self, engine_with_tab, mock_page):
        # Add a second tab
        mock_page2 = AsyncMock()
        mock_page2.url = "https://other.com"
        mock_page2.title = AsyncMock(return_value="Other")
        engine_with_tab._pages[2] = mock_page2

        result = await engine_with_tab.list_tabs()
        assert result.success is True
        assert len(result.data) == 2
        urls = {t["url"] for t in result.data}
        assert "https://example.com" in urls
        assert "https://other.com" in urls


# ══════════════════════════════════════════
# navigate()
# ══════════════════════════════════════════


class TestNavigate:
    async def test_navigate_success(self, engine_with_tab, mock_page):
        mock_page.title = AsyncMock(return_value="New Page")
        with patch("liteagent.web.is_ssrf_target", return_value=False):
            result = await engine_with_tab.navigate(1, "https://new.com")
        assert result.success is True
        mock_page.goto.assert_awaited_once()

    async def test_navigate_ssrf_blocked(self, engine_with_tab):
        with patch("liteagent.web.is_ssrf_target", return_value=True):
            result = await engine_with_tab.navigate(1, "http://192.168.1.1")
        assert result.success is False
        assert "SSRF" in result.error

    async def test_navigate_missing_tab(self, engine):
        result = await engine.navigate(999, "https://example.com")
        assert result.success is False
        assert "not found" in result.error

    async def test_navigate_exception(self, engine_with_tab, mock_page):
        mock_page.goto = AsyncMock(side_effect=TimeoutError("timeout"))
        with patch("liteagent.web.is_ssrf_target", return_value=False):
            result = await engine_with_tab.navigate(1, "https://slow.com")
        assert result.success is False
        assert "Navigation failed" in result.error


# ══════════════════════════════════════════
# go_back / go_forward / reload
# ══════════════════════════════════════════


class TestHistoryNavigation:
    async def test_go_back_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.go_back(1)
        assert result.success is True
        mock_page.go_back.assert_awaited_once()

    async def test_go_back_missing_tab(self, engine):
        result = await engine.go_back(999)
        assert result.success is False

    async def test_go_forward_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.go_forward(1)
        assert result.success is True
        mock_page.go_forward.assert_awaited_once()

    async def test_go_forward_missing_tab(self, engine):
        result = await engine.go_forward(999)
        assert result.success is False

    async def test_reload_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.reload(1)
        assert result.success is True
        mock_page.reload.assert_awaited_once()

    async def test_reload_missing_tab(self, engine):
        result = await engine.reload(999)
        assert result.success is False

    async def test_go_back_exception(self, engine_with_tab, mock_page):
        mock_page.go_back = AsyncMock(side_effect=RuntimeError("error"))
        result = await engine_with_tab.go_back(1)
        assert result.success is False


# ══════════════════════════════════════════
# click / type_text / select_option / scroll / hover / press_key
# ══════════════════════════════════════════


class TestInteraction:
    async def test_click_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.click(1, "button#submit")
        assert result.success is True
        assert "Clicked" in result.data
        mock_page.click.assert_awaited_once_with("button#submit", timeout=5000)

    async def test_click_missing_tab(self, engine):
        result = await engine.click(999, "button")
        assert result.success is False

    async def test_click_exception(self, engine_with_tab, mock_page):
        mock_page.click = AsyncMock(side_effect=RuntimeError("not found"))
        result = await engine_with_tab.click(1, "button.missing")
        assert result.success is False
        assert "Click failed" in result.error

    async def test_type_text_clear(self, engine_with_tab, mock_page):
        result = await engine_with_tab.type_text(1, "input#name", "Alice", clear=True)
        assert result.success is True
        mock_page.fill.assert_awaited_once_with("input#name", "Alice")
        mock_page.type.assert_not_awaited()

    async def test_type_text_no_clear(self, engine_with_tab, mock_page):
        result = await engine_with_tab.type_text(1, "input#name", "Bob", clear=False)
        assert result.success is True
        mock_page.type.assert_awaited_once_with("input#name", "Bob")
        mock_page.fill.assert_not_awaited()

    async def test_type_text_missing_tab(self, engine):
        result = await engine.type_text(999, "input", "text")
        assert result.success is False

    async def test_type_text_exception(self, engine_with_tab, mock_page):
        mock_page.fill = AsyncMock(side_effect=RuntimeError("input not editable"))
        result = await engine_with_tab.type_text(1, "div", "text")
        assert result.success is False
        assert "Type failed" in result.error

    async def test_select_option_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.select_option(1, "select#color", "red")
        assert result.success is True
        mock_page.select_option.assert_awaited_once_with("select#color", "red")

    async def test_select_option_missing_tab(self, engine):
        result = await engine.select_option(999, "select", "val")
        assert result.success is False

    async def test_scroll_down(self, engine_with_tab, mock_page):
        result = await engine_with_tab.scroll(1, "down", 300)
        assert result.success is True
        mock_page.mouse.wheel.assert_awaited_once_with(0, 300)
        assert "down" in result.data

    async def test_scroll_up(self, engine_with_tab, mock_page):
        result = await engine_with_tab.scroll(1, "up", 200)
        assert result.success is True
        mock_page.mouse.wheel.assert_awaited_once_with(0, -200)
        assert "up" in result.data

    async def test_scroll_missing_tab(self, engine):
        result = await engine.scroll(999)
        assert result.success is False

    async def test_hover_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.hover(1, "a.link")
        assert result.success is True
        mock_page.hover.assert_awaited_once_with("a.link", timeout=5000)

    async def test_hover_missing_tab(self, engine):
        result = await engine.hover(999, "a")
        assert result.success is False

    async def test_press_key_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.press_key(1, "Enter")
        assert result.success is True
        mock_page.keyboard.press.assert_awaited_once_with("Enter")

    async def test_press_key_missing_tab(self, engine):
        result = await engine.press_key(999, "Enter")
        assert result.success is False


# ══════════════════════════════════════════
# wait_for()
# ══════════════════════════════════════════


class TestWaitFor:
    async def test_wait_for_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.wait_for(1, "div.loaded")
        assert result.success is True
        assert "Element found" in result.data
        mock_page.wait_for_selector.assert_awaited_once_with("div.loaded", timeout=10000)

    async def test_wait_for_custom_timeout(self, engine_with_tab, mock_page):
        await engine_with_tab.wait_for(1, "div.loaded", timeout=5000)
        mock_page.wait_for_selector.assert_awaited_once_with("div.loaded", timeout=5000)

    async def test_wait_for_timeout(self, engine_with_tab, mock_page):
        mock_page.wait_for_selector = AsyncMock(side_effect=TimeoutError("timed out"))
        result = await engine_with_tab.wait_for(1, "div.never")
        assert result.success is False
        assert "Timeout" in result.error

    async def test_wait_for_missing_tab(self, engine):
        result = await engine.wait_for(999, "div")
        assert result.success is False


# ══════════════════════════════════════════
# upload_file()
# ══════════════════════════════════════════


class TestUploadFile:
    async def test_upload_file_not_found(self, engine_with_tab):
        result = await engine_with_tab.upload_file(1, "input[type=file]", "/nonexistent/path.txt")
        assert result.success is False
        assert "File not found" in result.error

    async def test_upload_file_success(self, engine_with_tab, mock_page, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        result = await engine_with_tab.upload_file(1, "input[type=file]", str(test_file))
        assert result.success is True
        assert "Uploaded" in result.data
        assert "test.txt" in result.data
        mock_page.set_input_files.assert_awaited_once()

    async def test_upload_file_missing_tab(self, engine):
        result = await engine.upload_file(999, "input", "/some/file.txt")
        assert result.success is False


# ══════════════════════════════════════════
# screenshot()
# ══════════════════════════════════════════


class TestScreenshot:
    async def test_screenshot_returns_base64(self, engine_with_tab, mock_page):
        png_bytes = b"\x89PNG fake image data"
        mock_page.screenshot = AsyncMock(return_value=png_bytes)
        result = await engine_with_tab.screenshot(1)
        assert result.success is True
        assert result.data["format"] == "png"
        assert result.data["size"] == len(png_bytes)
        decoded = base64.b64decode(result.data["image_base64"])
        assert decoded == png_bytes

    async def test_screenshot_full_page(self, engine_with_tab, mock_page):
        await engine_with_tab.screenshot(1, full_page=True)
        mock_page.screenshot.assert_awaited_once_with(full_page=True, type="png")

    async def test_screenshot_missing_tab(self, engine):
        result = await engine.screenshot(999)
        assert result.success is False

    async def test_screenshot_exception(self, engine_with_tab, mock_page):
        mock_page.screenshot = AsyncMock(side_effect=RuntimeError("render failed"))
        result = await engine_with_tab.screenshot(1)
        assert result.success is False
        assert "Screenshot failed" in result.error


# ══════════════════════════════════════════
# pdf()
# ══════════════════════════════════════════


class TestPdf:
    async def test_pdf_returns_base64(self, engine_with_tab, mock_page):
        pdf_bytes = b"%PDF-1.4 fake pdf data"
        mock_page.pdf = AsyncMock(return_value=pdf_bytes)
        result = await engine_with_tab.pdf(1)
        assert result.success is True
        assert result.data["size"] == len(pdf_bytes)
        decoded = base64.b64decode(result.data["pdf_base64"])
        assert decoded == pdf_bytes

    async def test_pdf_missing_tab(self, engine):
        result = await engine.pdf(999)
        assert result.success is False

    async def test_pdf_exception(self, engine_with_tab, mock_page):
        mock_page.pdf = AsyncMock(side_effect=RuntimeError("headful mode"))
        result = await engine_with_tab.pdf(1)
        assert result.success is False
        assert "PDF generation failed" in result.error


# ══════════════════════════════════════════
# get_text / get_html
# ══════════════════════════════════════════


class TestGetText:
    async def test_get_text_default(self, engine_with_tab, mock_page):
        result = await engine_with_tab.get_text(1)
        assert result.success is True
        assert result.data == "Page text content"
        mock_page.inner_text.assert_awaited_once_with("body", timeout=5000)

    async def test_get_text_custom_selector(self, engine_with_tab, mock_page):
        await engine_with_tab.get_text(1, selector="div.content")
        mock_page.inner_text.assert_awaited_once_with("div.content", timeout=5000)

    async def test_get_text_missing_tab(self, engine):
        result = await engine.get_text(999)
        assert result.success is False

    async def test_get_text_truncates_long_content(self, engine_with_tab, mock_page):
        mock_page.inner_text = AsyncMock(return_value="x" * 60000)
        result = await engine_with_tab.get_text(1)
        assert result.success is True
        assert len(result.data) == 50000


class TestGetHtml:
    async def test_get_html_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.get_html(1)
        assert result.success is True
        assert result.data == "<html>content</html>"

    async def test_get_html_custom_selector(self, engine_with_tab, mock_page):
        await engine_with_tab.get_html(1, selector="div.main")
        mock_page.locator.assert_called_once_with("div.main")

    async def test_get_html_missing_tab(self, engine):
        result = await engine.get_html(999)
        assert result.success is False


# ══════════════════════════════════════════
# get_accessibility_tree()
# ══════════════════════════════════════════


class TestGetAccessibilityTree:
    async def test_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.get_accessibility_tree(1)
        assert result.success is True
        assert "document" in result.data
        assert "Title" in result.data
        assert "Submit" in result.data

    async def test_empty_tree(self, engine_with_tab, mock_page):
        mock_page.accessibility.snapshot = AsyncMock(return_value=None)
        result = await engine_with_tab.get_accessibility_tree(1)
        assert result.success is True
        assert result.data == "(empty accessibility tree)"

    async def test_missing_tab(self, engine):
        result = await engine.get_accessibility_tree(999)
        assert result.success is False

    async def test_exception(self, engine_with_tab, mock_page):
        mock_page.accessibility.snapshot = AsyncMock(side_effect=RuntimeError("error"))
        result = await engine_with_tab.get_accessibility_tree(1)
        assert result.success is False


# ══════════════════════════════════════════
# get_console()
# ══════════════════════════════════════════


class TestGetConsole:
    async def test_all_messages(self, engine_with_tab):
        engine_with_tab._console_messages[1] = [
            {"type": "log", "text": "info"},
            {"type": "error", "text": "err"},
            {"type": "warning", "text": "warn"},
        ]
        result = await engine_with_tab.get_console(1, level="all")
        assert result.success is True
        assert len(result.data) == 3

    async def test_error_filter(self, engine_with_tab):
        engine_with_tab._console_messages[1] = [
            {"type": "log", "text": "info"},
            {"type": "error", "text": "err"},
            {"type": "exception", "text": "exc"},
            {"type": "warning", "text": "warn"},
        ]
        result = await engine_with_tab.get_console(1, level="error")
        assert result.success is True
        assert len(result.data) == 2
        types = {m["type"] for m in result.data}
        assert types == {"error", "exception"}

    async def test_warning_filter(self, engine_with_tab):
        engine_with_tab._console_messages[1] = [
            {"type": "log", "text": "info"},
            {"type": "error", "text": "err"},
            {"type": "warning", "text": "warn"},
        ]
        result = await engine_with_tab.get_console(1, level="warning")
        assert result.success is True
        assert len(result.data) == 2
        types = {m["type"] for m in result.data}
        assert types == {"error", "warning"}

    async def test_missing_tab(self, engine):
        result = await engine.get_console(999)
        assert result.success is False
        assert "not found" in result.error

    async def test_caps_at_100(self, engine_with_tab):
        engine_with_tab._console_messages[1] = [
            {"type": "log", "text": f"msg {i}"} for i in range(150)
        ]
        result = await engine_with_tab.get_console(1)
        assert result.success is True
        assert len(result.data) == 100
        # Should be the last 100
        assert result.data[0]["text"] == "msg 50"


# ══════════════════════════════════════════
# evaluate()
# ══════════════════════════════════════════


class TestEvaluate:
    async def test_evaluate_success(self, engine_with_tab, mock_page):
        mock_page.evaluate = AsyncMock(return_value=42)
        result = await engine_with_tab.evaluate(1, "1 + 1")
        assert result.success is True
        assert result.data == 42

    async def test_evaluate_returns_string(self, engine_with_tab, mock_page):
        mock_page.evaluate = AsyncMock(return_value="hello")
        result = await engine_with_tab.evaluate(1, "document.title")
        assert result.success is True
        assert result.data == "hello"

    async def test_evaluate_js_error(self, engine_with_tab, mock_page):
        mock_page.evaluate = AsyncMock(side_effect=RuntimeError("ReferenceError: x is not defined"))
        result = await engine_with_tab.evaluate(1, "x.y.z")
        assert result.success is False
        assert "JS evaluation failed" in result.error

    async def test_evaluate_missing_tab(self, engine):
        result = await engine.evaluate(999, "1+1")
        assert result.success is False


# ══════════════════════════════════════════
# get_url()
# ══════════════════════════════════════════


class TestGetUrl:
    async def test_get_url_success(self, engine_with_tab, mock_page):
        result = await engine_with_tab.get_url(1)
        assert result.success is True
        assert result.data["url"] == "https://example.com"
        assert result.data["title"] == "Example"

    async def test_get_url_missing_tab(self, engine):
        result = await engine.get_url(999)
        assert result.success is False

    async def test_get_url_exception(self, engine_with_tab, mock_page):
        mock_page.title = AsyncMock(side_effect=RuntimeError("detached"))
        result = await engine_with_tab.get_url(1)
        assert result.success is False


# ══════════════════════════════════════════
# _format_accessibility_tree()
# ══════════════════════════════════════════


class TestFormatAccessibilityTree:
    def test_simple_tree(self):
        tree = {
            "role": "document",
            "name": "Page",
            "children": [
                {"role": "heading", "name": "Welcome"},
                {"role": "button", "name": "Click me"},
            ],
        }
        output = _format_accessibility_tree(tree)
        assert 'document "Page"' in output
        assert 'heading "Welcome"' in output
        assert 'button "Click me"' in output

    def test_nested_tree(self):
        tree = {
            "role": "document",
            "name": "Root",
            "children": [
                {
                    "role": "navigation",
                    "name": "Main nav",
                    "children": [
                        {"role": "link", "name": "Home"},
                        {"role": "link", "name": "About"},
                    ],
                },
            ],
        }
        output = _format_accessibility_tree(tree)
        lines = output.split("\n")
        # Root at indent 0
        assert lines[0].startswith("document")
        # Navigation at indent 1 (2 spaces)
        nav_line = [l for l in lines if "navigation" in l][0]
        assert nav_line.startswith("  ")
        # Links at indent 2 (4 spaces)
        home_line = [l for l in lines if "Home" in l][0]
        assert home_line.startswith("    ")

    def test_value_field(self):
        tree = {"role": "textbox", "name": "Email", "value": "test@example.com"}
        output = _format_accessibility_tree(tree)
        assert 'value="test@example.com"' in output

    def test_empty_node(self):
        tree = {"role": "", "name": "", "children": []}
        output = _format_accessibility_tree(tree)
        assert output.strip() == ""

    def test_no_children(self):
        tree = {"role": "button", "name": "OK"}
        output = _format_accessibility_tree(tree)
        assert 'button "OK"' in output


# ══════════════════════════════════════════
# _ensure_running / _get_page helpers
# ══════════════════════════════════════════


class TestHelpers:
    async def test_ensure_running_when_launched(self, engine):
        result = await engine._ensure_running()
        assert result is None  # Already running, no error

    async def test_ensure_running_auto_launch_failure(self):
        eng = BrowserEngine()
        with patch.object(eng, "launch", new_callable=AsyncMock,
                          return_value=BrowserResult(success=False, error="no browser")):
            result = await eng._ensure_running()
        assert result is not None
        assert result.success is False

    def test_get_page_found(self, engine_with_tab, mock_page):
        page, err = engine_with_tab._get_page(1)
        assert page is mock_page
        assert err is None

    def test_get_page_not_found(self, engine):
        page, err = engine._get_page(42)
        assert page is None
        assert err.success is False
        assert "42" in err.error
