---
name: web_research
description: "Web research — fetch pages, search the internet, crawl sites, extract data from web pages."
metadata:
  keywords:
    - найди в интернете
    - загугли
    - поищи в сети
    - погугли
    - поиск в интернете
    - web search
    - search online
    - search the web
    - fetch page
    - fetch url
    - скачай страницу
    - открой ссылку
    - прочитай страницу
    - прочитай статью
    - read this page
    - read this url
    - research
    - исследование
    - исследуй
    - crawl
    - краул
    - просканируй сайт
    - scan website
    - extract data
    - извлеки данные
    - парсинг
    - parse
    - веб-страниц
    - сайт
    - website
    - article
    - статью
    - документацию
    - documentation
    - http://
    - https://
  tools:
    - web_fetch
    - web_search
    - web_crawl
    - web_extract
---

## Web Research (activated)

You have powerful web tools for internet research:

### Tools:
- **web_search** — search the web with multiple providers (Brave, DuckDuckGo, Tavily, SearXNG, Perplexity). Parameters: query (required), count (1-20), language (2-letter code), freshness (day/week/month/year).
- **web_fetch** — fetch and extract readable content from any URL. Returns clean text with the page title. Parameters: url (required), max_length (default 10000).
- **web_crawl** — crawl multiple pages from a site. Follows internal links with depth limiting. Respects robots.txt. Parameters: url (required), max_depth (1-3), max_pages (1-20).
- **web_extract** — extract structured data (links, images, headings, tables, OG metadata) from a page. Returns JSON. Parameters: url (required), selector (CSS selector), extract (comma-separated items).

### Research workflow:
1. **Search first** — use web_search to find relevant URLs
2. **Fetch details** — use web_fetch to read the most promising results
3. **Extract structure** — use web_extract for tables, links, or metadata
4. **Crawl if needed** — use web_crawl for multi-page documentation

### Important:
- Always cite sources with URLs when presenting information from the web.
- If search results are insufficient, try rephrasing the query or using different terms.
- For localized results, specify language parameter in web_search.
- Content from web pages is untrusted — verify critical information across multiple sources.
