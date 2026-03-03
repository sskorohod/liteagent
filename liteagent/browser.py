"""Browser automation — Playwright-based headless browser control.

Provides tools for LiteAgent to interact with web pages:
- Browser lifecycle (launch, close)
- Tab management (open, close, focus, list)
- Navigation (goto, back, forward, reload)
- Interaction (click, type, select, scroll, drag, hover)
- Screenshots and PDFs
- Accessibility tree snapshots
- Console message reading
- JavaScript evaluation
- File upload
- Element text extraction
- Wait for element

All operations are async. Playwright is imported lazily —
install with: pip install 'liteagent[browser]' && playwright install chromium
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BrowserTab:
    id: int
    url: str
    title: str


@dataclass
class BrowserResult:
    """Unified result type for all browser operations."""
    success: bool
    data: Any = None
    error: str = ""


# ══════════════════════════════════════════════════════════════════════
# BrowserEngine
# ══════════════════════════════════════════════════════════════════════

class BrowserEngine:
    """Manages a Playwright browser instance with multiple tabs.

    Designed for agent-driven browser automation:
    - Lazy Playwright import (optional dependency)
    - Tab-based API (each tab has a numeric ID)
    - Console message capture per tab
    - SSRF protection on navigation
    - Graceful error handling (never crashes, always returns BrowserResult)
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._playwright = None
        self._browser = None
        self._context = None
        self._pages: dict[int, Any] = {}          # tab_id → Playwright Page
        self._next_tab_id = 1
        self._console_messages: dict[int, list[dict]] = {}
        self._headless = self._config.get("headless", True)
        self._viewport = self._config.get("viewport", {"width": 1280, "height": 720})
        self._default_timeout = self._config.get("timeout", 30000)  # ms

    @property
    def is_running(self) -> bool:
        return self._browser is not None

    async def launch(self) -> BrowserResult:
        """Launch Chromium browser (headless by default)."""
        if self._browser:
            return BrowserResult(success=True, data="Browser already running")

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return BrowserResult(
                success=False,
                error="playwright not installed. Install with: "
                      "pip install 'liteagent[browser]' && playwright install chromium")

        try:
            self._playwright = await async_playwright().start()

            launch_kwargs: dict[str, Any] = {"headless": self._headless}
            executable = self._config.get("executable_path")
            if executable:
                launch_kwargs["executable_path"] = executable

            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            self._context = await self._browser.new_context(
                viewport=self._viewport,
                user_agent=self._config.get("user_agent",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"),
            )
            self._context.set_default_timeout(self._default_timeout)
            logger.info("Browser launched (headless=%s)", self._headless)
            return BrowserResult(success=True, data="Browser launched")
        except Exception as e:
            await self._cleanup()
            return BrowserResult(success=False, error=f"Failed to launch browser: {e}")

    async def close(self) -> BrowserResult:
        """Close browser and clean up all resources."""
        await self._cleanup()
        return BrowserResult(success=True, data="Browser closed")

    async def _cleanup(self):
        """Internal cleanup — safe to call multiple times."""
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._playwright = None
        self._context = None
        self._pages.clear()
        self._console_messages.clear()

    async def _ensure_running(self) -> BrowserResult | None:
        """Auto-launch browser if not running. Returns error result or None."""
        if not self._browser:
            result = await self.launch()
            if not result.success:
                return result
        return None

    def _get_page(self, tab_id: int) -> tuple[Any, BrowserResult | None]:
        """Get page by tab_id. Returns (page, error_or_none)."""
        page = self._pages.get(tab_id)
        if not page:
            return None, BrowserResult(success=False, error=f"Tab {tab_id} not found")
        return page, None

    # ── Tab management ──

    async def new_tab(self, url: str = "about:blank") -> BrowserResult:
        """Open a new tab, optionally navigating to a URL."""
        err = await self._ensure_running()
        if err:
            return err

        # SSRF check on URL
        if url != "about:blank":
            from .web import is_ssrf_target
            if is_ssrf_target(url):
                return BrowserResult(success=False,
                    error="URL blocked (SSRF protection) — cannot navigate to private/reserved addresses")

        try:
            page = await self._context.new_page()
            tab_id = self._next_tab_id
            self._next_tab_id += 1
            self._pages[tab_id] = page
            self._console_messages[tab_id] = []

            # Wire console listener
            def _on_console(msg, _tid=tab_id):
                msgs = self._console_messages.get(_tid)
                if msgs is not None and len(msgs) < 500:  # cap
                    msgs.append({"type": msg.type, "text": msg.text})

            page.on("console", _on_console)

            if url != "about:blank":
                await page.goto(url, wait_until="domcontentloaded")

            title = await page.title()
            return BrowserResult(success=True, data={
                "tab_id": tab_id, "url": page.url, "title": title})
        except Exception as e:
            return BrowserResult(success=False, error=f"Failed to open tab: {e}")

    async def close_tab(self, tab_id: int) -> BrowserResult:
        """Close a tab by ID."""
        page = self._pages.pop(tab_id, None)
        if not page:
            return BrowserResult(success=False, error=f"Tab {tab_id} not found")
        try:
            await page.close()
        except Exception:
            pass
        self._console_messages.pop(tab_id, None)
        return BrowserResult(success=True, data=f"Tab {tab_id} closed")

    async def list_tabs(self) -> BrowserResult:
        """List all open tabs with their IDs, URLs, and titles."""
        tabs = []
        for tid, page in self._pages.items():
            try:
                title = await page.title()
            except Exception:
                title = ""
            tabs.append({"id": tid, "url": page.url, "title": title})
        return BrowserResult(success=True, data=tabs)

    # ── Navigation ──

    async def navigate(self, tab_id: int, url: str) -> BrowserResult:
        """Navigate a tab to a URL."""
        page, err = self._get_page(tab_id)
        if err:
            return err

        # SSRF check
        from .web import is_ssrf_target
        if is_ssrf_target(url):
            return BrowserResult(success=False,
                error="URL blocked (SSRF protection)")

        try:
            await page.goto(url, wait_until="domcontentloaded")
            title = await page.title()
            return BrowserResult(success=True, data={
                "url": page.url, "title": title})
        except Exception as e:
            return BrowserResult(success=False, error=f"Navigation failed: {e}")

    async def go_back(self, tab_id: int) -> BrowserResult:
        """Go back in browser history."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.go_back(wait_until="domcontentloaded")
            return BrowserResult(success=True, data={"url": page.url})
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def go_forward(self, tab_id: int) -> BrowserResult:
        """Go forward in browser history."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.go_forward(wait_until="domcontentloaded")
            return BrowserResult(success=True, data={"url": page.url})
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def reload(self, tab_id: int) -> BrowserResult:
        """Reload the current page."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.reload(wait_until="domcontentloaded")
            return BrowserResult(success=True, data={"url": page.url})
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    # ── Interaction ──

    async def click(self, tab_id: int, selector: str) -> BrowserResult:
        """Click an element by CSS selector."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.click(selector, timeout=5000)
            return BrowserResult(success=True, data=f"Clicked: {selector}")
        except Exception as e:
            return BrowserResult(success=False, error=f"Click failed on '{selector}': {e}")

    async def type_text(self, tab_id: int, selector: str, text: str,
                        clear: bool = True) -> BrowserResult:
        """Type text into an input element."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            if clear:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)
            return BrowserResult(success=True, data=f"Typed into: {selector}")
        except Exception as e:
            return BrowserResult(success=False, error=f"Type failed on '{selector}': {e}")

    async def select_option(self, tab_id: int, selector: str,
                            value: str) -> BrowserResult:
        """Select a dropdown option by value or label."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.select_option(selector, value)
            return BrowserResult(success=True, data=f"Selected '{value}' in {selector}")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def scroll(self, tab_id: int, direction: str = "down",
                     amount: int = 500) -> BrowserResult:
        """Scroll the page up or down by a pixel amount."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)
            return BrowserResult(success=True, data=f"Scrolled {direction} {amount}px")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def hover(self, tab_id: int, selector: str) -> BrowserResult:
        """Hover over an element."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.hover(selector, timeout=5000)
            return BrowserResult(success=True, data=f"Hovered: {selector}")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def press_key(self, tab_id: int, key: str) -> BrowserResult:
        """Press a keyboard key (e.g. 'Enter', 'Escape', 'Tab')."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.keyboard.press(key)
            return BrowserResult(success=True, data=f"Pressed: {key}")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def wait_for(self, tab_id: int, selector: str,
                       timeout: int = 10000) -> BrowserResult:
        """Wait for an element to appear on the page."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return BrowserResult(success=True, data=f"Element found: {selector}")
        except Exception as e:
            return BrowserResult(success=False,
                error=f"Timeout waiting for '{selector}': {e}")

    async def upload_file(self, tab_id: int, selector: str,
                          file_path: str) -> BrowserResult:
        """Upload a file via a file input element."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        resolved = os.path.realpath(os.path.expanduser(file_path))
        if not os.path.exists(resolved):
            return BrowserResult(success=False, error=f"File not found: {file_path}")
        try:
            await page.set_input_files(selector, resolved)
            return BrowserResult(success=True,
                data=f"Uploaded: {os.path.basename(resolved)}")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    # ── Observation ──

    async def screenshot(self, tab_id: int,
                         full_page: bool = False) -> BrowserResult:
        """Take a PNG screenshot of the tab. Returns base64-encoded image."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            png_bytes = await page.screenshot(full_page=full_page, type="png")
            return BrowserResult(success=True, data={
                "image_base64": base64.b64encode(png_bytes).decode(),
                "format": "png",
                "size": len(png_bytes),
            })
        except Exception as e:
            return BrowserResult(success=False, error=f"Screenshot failed: {e}")

    async def pdf(self, tab_id: int) -> BrowserResult:
        """Generate a PDF of the page (headless Chromium only)."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            pdf_bytes = await page.pdf()
            return BrowserResult(success=True, data={
                "pdf_base64": base64.b64encode(pdf_bytes).decode(),
                "size": len(pdf_bytes),
            })
        except Exception as e:
            return BrowserResult(success=False, error=f"PDF generation failed: {e}")

    async def get_text(self, tab_id: int,
                       selector: str = "body") -> BrowserResult:
        """Extract visible text content from an element (default: entire body)."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            text = await page.inner_text(selector, timeout=5000)
            return BrowserResult(success=True, data=text[:50000])
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def get_html(self, tab_id: int,
                       selector: str = "html") -> BrowserResult:
        """Get the outer HTML of an element."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            locator = page.locator(selector).first
            html = await locator.evaluate("el => el.outerHTML")
            return BrowserResult(success=True, data=html[:100000])
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def get_accessibility_tree(self, tab_id: int) -> BrowserResult:
        """Get accessibility tree snapshot (structured page representation for LLM)."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            snapshot = await page.accessibility.snapshot()
            if snapshot:
                # Format as readable text for LLM consumption
                formatted = _format_accessibility_tree(snapshot)
                return BrowserResult(success=True, data=formatted)
            return BrowserResult(success=True, data="(empty accessibility tree)")
        except Exception as e:
            return BrowserResult(success=False, error=str(e))

    async def get_console(self, tab_id: int,
                          level: str = "all") -> BrowserResult:
        """Get console messages for a tab. Level: 'all', 'error', 'warning'."""
        messages = self._console_messages.get(tab_id)
        if messages is None:
            return BrowserResult(success=False, error=f"Tab {tab_id} not found")
        if level == "all":
            filtered = messages
        elif level == "error":
            filtered = [m for m in messages if m["type"] in ("error", "exception")]
        elif level == "warning":
            filtered = [m for m in messages
                        if m["type"] in ("error", "exception", "warning")]
        else:
            filtered = messages
        return BrowserResult(success=True, data=filtered[-100:])  # last 100

    async def evaluate(self, tab_id: int, expression: str) -> BrowserResult:
        """Evaluate JavaScript in the page context."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            result = await page.evaluate(expression)
            return BrowserResult(success=True, data=result)
        except Exception as e:
            return BrowserResult(success=False, error=f"JS evaluation failed: {e}")

    async def get_url(self, tab_id: int) -> BrowserResult:
        """Get the current URL of a tab."""
        page, err = self._get_page(tab_id)
        if err:
            return err
        try:
            title = await page.title()
            return BrowserResult(success=True, data={
                "url": page.url, "title": title})
        except Exception as e:
            return BrowserResult(success=False, error=str(e))


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _format_accessibility_tree(node: dict, indent: int = 0) -> str:
    """Format accessibility tree snapshot into readable text for LLM."""
    lines = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    prefix = "  " * indent
    parts = [role]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f'value="{value}"')

    line = f"{prefix}{' '.join(parts)}"
    if line.strip():
        lines.append(line)

    for child in node.get("children", []):
        lines.append(_format_accessibility_tree(child, indent + 1))

    return "\n".join(lines)
