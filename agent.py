"""
Agentic Web Scraper - Skeleton

This file defines the overall workflow and agent skeletons for the redesigned architecture.
"""

import sys
import asyncio
from mcp_agent.core.fastagent import FastAgent
from typing import List

fast = FastAgent("AgenticWebScraper")

# === 1. 初期入力・設定 ===
def get_root_url():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("Enter the root URL (e.g. https://fast-agent.ai/): ").strip()

# === 2. URLリスト抽出 (sitemap/Playwright) ===
def extract_url_list(root_url):
    """
    指定されたroot_urlからsitemap.xmlを探して全URLを抽出します。
    sitemapがなければPlaywrightでページ内のaタグリンクを抽出します。
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urlparse
    import re
    import os
    
    # 1. sitemap.xmlの自動検出
    parsed = urlparse(root_url)
    sitemap_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/sitemap.xml")
    urls = []
    try:
        resp = requests.get(sitemap_url, timeout=10)
        if resp.status_code == 200 and '<urlset' in resp.text:
            soup = BeautifulSoup(resp.text, "xml")
            urls = [loc.text.strip() for loc in soup.find_all("loc") if loc.text.strip()]
            if urls:
                return urls
    except Exception as e:
        pass  # sitemap取得失敗は無視
    
    # 2. Playwrightでrootページからaタグリンクを抽出
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(root_url, timeout=15000)
            anchors = page.query_selector_all("a")
            links = set()
            for a in anchors:
                href = a.get_attribute("href")
                if href and not href.startswith("#"):
                    abs_url = urljoin(root_url, href)
                    # 同一ドメインのみ
                    if urlparse(abs_url).netloc == parsed.netloc:
                        links.add(abs_url.split("#")[0])
            browser.close()
            if links:
                return sorted(links)
    except Exception as e:
        pass  # Playwright失敗も無視
    
    # 失敗時はroot_urlのみ返す
    return [root_url]

# === 3. 作業量見積もり・表示 ===
def estimate_and_show_work(urls):
    """
    URLリストの件数・サンプル・簡易見積もり（トークン/時間）を表示する
    """
    num = len(urls)
    print(f"\n=== 対象URL件数: {num}件 ===")
    if num == 0:
        print("URLリストが空です。終了します。")
        return
    # サンプル表示（最大5件）
    print("サンプルURL:")
    for url in urls[:5]:
        print(f"  - {url}")
    if num > 5:
        print(f"  ... 他{num-5}件")
    
    # 簡易トークン・時間見積もり
    avg_token = 2000  # 1ページあたり想定トークン数
    avg_sec = 5      # 1ページあたり想定処理秒数
    total_token = num * avg_token
    total_sec = num * avg_sec
    print(f"\n見積もり: 合計 {total_token:,} tokens, 約 {total_sec//60}分{total_sec%60}秒")

# === 4. ユーザーによる実行可否決定 (y/n/t) ===
def get_user_decision():
    """
    ユーザーに対して実行可否を尋ねる。
    y: 全件実行, n: 実行先URLの優先順を提案して範囲選択, t: 終了
    """
    while True:
        ans = input("\n実行しますか？ [y]全件実行 [n]提案と範囲選択 [t]終了 : ").strip().lower()
        if ans in ("y", "n", "t"):
            return ans
        print("y, n, t のいずれかを入力してください。")

# === 5. LLMエージェント: ページスクレイプ (Fetch MCP) ===
@fast.agent(
    name="page_scraper",
    instruction="Fetch the content of the given URL as Markdown using Fetch MCP.",
    servers=["fetch"]
)
async def page_scraper(agent, url: str):
    # Fetch MCP Server経由でページ取得（本実装）
    # MCPサーバーがMarkdownで返す前提
    response = await agent.call_mcp_server("fetch", url=url)
    # 返り値がdictの場合を想定（例: {"markdown": ...}）
    if isinstance(response, dict) and "markdown" in response:
        return {"url": url, "markdown": response["markdown"]}
    # 返り値が文字列の場合（Markdownそのもの）
    elif isinstance(response, str):
        return {"url": url, "markdown": response}
    else:
        raise ValueError(f"MCP fetch server returned unexpected response: {response}")

# === 5b. LLMエージェント: ノイズ除去 ===
@fast.agent(
    name="content_cleaner",
    instruction="Remove navigation, ads, and noise from Markdown content. Return cleaned Markdown."
)
async def content_cleaner(agent, markdown: str):
    # Fetch MCP Server経由でノイズ除去（本実装）
    response = await agent.call_mcp_server("clean", markdown=markdown)
    # 返り値がdictの場合を想定（例: {"markdown": ...}）
    if isinstance(response, dict) and "markdown" in response:
        return response["markdown"]
    # 返り値が文字列の場合（Markdownそのもの）
    elif isinstance(response, str):
        return response
    else:
        raise ValueError(f"MCP clean server returned unexpected response: {response}")

# === 5c. 並列スクレイピング＆ノイズ除去 (asyncio) ===
async def parallel_scrape_and_clean(agent, urls: List[str]):
    async def scrape_and_clean(url):
        page = await agent.page_scraper(url)
        cleaned = await agent.content_cleaner(page["markdown"])
        return {"url": url, "markdown": cleaned}
    return await asyncio.gather(*(scrape_and_clean(url) for url in urls))

# === 6. LLMエージェント: 優先度付け/理由説明 ===
@fast.agent(
    name="prioritizer",
    instruction="Given a root page context and a list of URLs, return a prioritized list with reasons."
)
async def prioritizer(agent, context: str, url_list: list):
    # TODO: LLMで優先度付け・理由生成
    pass

# === 7. ユーザー: 番号で範囲決定 ===
def get_user_url_range():
    # (実装: 番号で範囲指定)
    # TODO: 実装
    return 0, 0

# === 8. LLMエージェント: クロール実行・進捗管理 ===
@fast.agent(
    name="crawler",
    instruction="Crawl the selected URLs and manage progress, returning cleaned Markdown.",
    servers=["fetch"]
)
async def crawler(agent, urls: list):
    # 並列スクレイピング＆ノイズ除去
    results = await parallel_scrape_and_clean(agent, urls)
    # 結果を集約（URLとcleaned markdownのリスト）
    return results

# === 9. Python: 保存・エラー/レジューム ===
def save_results():
    # (実装: Markdown保存・エラー/レジューム管理)
    # TODO: 実装
    pass

# === メインワークフロー ===
async def main():
    root_url = get_root_url()
    url_list = extract_url_list(root_url)
    estimate_and_show_work(url_list)
    decision = get_user_decision()
    if decision == "t":
        print("Terminated by user.")
        return
    async with fast.run() as agent:
        if decision == "y":
            # 全件対象
            selected_urls = url_list
        elif decision == "n":
            # rootページのスクレイプ・ノイズ除去・優先度付け・範囲選択
            root_page = await agent.page_scraper(root_url)
            cleaned_root = await agent.content_cleaner(root_page["markdown"])
            prioritized = await agent.prioritizer(cleaned_root, url_list)
            start, end = get_user_url_range()
            selected_urls = prioritized[start:end+1]
        else:
            print("Invalid input. Terminating.")
            return
        await agent.crawler(selected_urls)
    save_results()

if __name__ == "__main__":
    asyncio.run(main())