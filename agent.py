#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import aiohttp # Use aiohttp for async requests
from bs4 import BeautifulSoup
import logging
import sys # Restore import
import os
import argparse # Restore import
import hashlib # Restore import
import time # Restore import
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from urllib.parse import urljoin, urlparse # Added urlparse
from mcp_agent.core.fastagent import FastAgent # Restore import

# --- Constants ---
DEFAULT_MAX_URLS = 100
DEFAULT_OUTPUT_FILE = "site_crawl_result.md"
DEFAULT_RESUME_FILE_PREFIX = "last_processed_index_"
MAX_FETCH_LENGTH = 500000
MAX_CRAWL_DEPTH = 3
SAVE_INTERVAL = 10
LOG_FILE = "logs/scraper.log"
RESUME_STATE_FILE = "crawler_resume_state_{hash}.json"

# --- Initialize FastAgent Globally ---
fast = FastAgent("Agentic-web-scraper", config_path="fastagent.config.yaml")

# --- Logging Setup ---
logger = logging.getLogger(__name__) # Define logger globally

def setup_logging():
    """Configures logging to file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "scraper.log"

    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

# --- Utility Functions ---
def is_valid_url(url: str) -> bool:
    """Checks if the URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except ValueError:
        return False

# --- URL Manager Class ---
class URLManager:
    """Manages URLs to be crawled, tracking progress and state."""
    def __init__(self, max_urls: int, resume_enabled: bool, root_url: str):
        self.max_urls = max_urls
        self.resume_enabled = resume_enabled
        self.url_hash = hashlib.sha256(root_url.encode('utf-8')).hexdigest()[:10]
        self.state_file = Path(RESUME_STATE_FILE.format(hash=self.url_hash))

        self.queue: asyncio.Queue = asyncio.Queue()
        self.seen_urls: set = set()
        self.processed_urls: set = set()
        self.failed_urls: set = set()
        self.processed_count = 0
        self.start_time = time.time()

        if self.resume_enabled and self.state_file.exists():
            self.load_state()
        else:
            logger.info("Starting a new crawl or resume disabled.")

    def add_urls(self, urls_data: List[Dict[str, Any]]):
        """Adds new URLs to the queue if not seen and within depth/limit."""
        added_count = 0
        skipped_depth = 0
        skipped_seen = 0
        skipped_limit = 0
        for data in urls_data:
            url = data.get('url')
            depth = data.get('depth', 0)

            if not url or not is_valid_url(url):
                logger.debug(f"Skipping invalid URL: {url}")
                continue

            if depth > MAX_CRAWL_DEPTH:
                logger.debug(f"Skipping URL due to depth limit ({depth} > {MAX_CRAWL_DEPTH}): {url}")
                skipped_depth += 1
                continue

            if url in self.seen_urls:
                logger.debug(f"Skipping already seen URL: {url}")
                skipped_seen += 1
                continue

            if self.processed_count + self.queue.qsize() >= self.max_urls:
                 logger.debug(f"Skipping URL due to max_urls limit reached ({self.max_urls}): {url}")
                 skipped_limit += 1
                 continue # Stop adding if queue + processed >= max_urls

            self.seen_urls.add(url)
            # Use asyncio.create_task for putting into queue from potentially sync context if needed
            # but add_urls is likely called from async main, so await is fine.
            try:
                asyncio.create_task(self.queue.put(data))
                added_count += 1
            except Exception as e:
                 logger.exception(f"Error adding URL {url} to queue: {e}")

        if added_count > 0:
            logger.info(f"Added {added_count} new URLs to queue. Current queue size: {self.queue.qsize()}")
        if skipped_depth > 0:
            logger.debug(f"Skipped {skipped_depth} URLs due to depth limit.")
        if skipped_seen > 0:
            logger.debug(f"Skipped {skipped_seen} already seen URLs.")
        if skipped_limit > 0:
            logger.debug(f"Skipped {skipped_limit} URLs due to max_urls limit.")


    async def get_next_url(self) -> Optional[Dict[str, Any]]:
        """Gets the next URL from the queue, skipping processed/failed."""
        if self.processed_count >= self.max_urls:
            logger.info(f"Reached max URL limit ({self.max_urls}). Stopping crawl.")
            # Ensure queue is emptied to terminate consuming loops gracefully
            while not self.queue.empty():
                try:
                    await self.queue.get()
                    self.queue.task_done()
                except asyncio.CancelledError:
                    logger.warning("Task cancelled during queue emptying.")
                    break
                except Exception as e:
                    logger.exception(f"Error emptying queue: {e}")
            return None

        try:
            # Get item with a timeout to prevent indefinite blocking if queue logic has issues
            url_data = await asyncio.wait_for(self.queue.get(), timeout=5.0)
            url = url_data['url']

            # Double-check if processed/failed while in queue (less likely with proper marking)
            if url in self.processed_urls or url in self.failed_urls:
                logger.debug(f"Skipping already processed/failed URL from queue: {url}")
                self.queue.task_done()
                return await self.get_next_url() # Recursively get the next valid one
            else:
                return url_data
        except asyncio.TimeoutError:
            logger.debug("Timeout waiting for URL from queue. Queue might be empty or stuck.")
            return None
        except asyncio.QueueEmpty:
            logger.info("URL queue is empty.")
            return None
        except Exception as e:
            logger.exception(f"Error getting URL from queue: {e}")
            return None

    def mark_as_processed(self, url: str):
        """Marks a URL as successfully processed."""
        if url not in self.processed_urls:
             self.processed_urls.add(url)
             self.processed_count += 1
             if url in self.failed_urls: # Remove from failed if processed successfully later?
                 self.failed_urls.remove(url)
             try:
                 self.queue.task_done()
             except ValueError:
                 logger.warning(f"task_done() called too many times for processed URL: {url}")
             logger.debug(f"Marked as processed: {url}. Count: {self.processed_count}/{self.max_urls}")
        else:
             logger.debug(f"URL already marked as processed: {url}")

    def mark_as_failed(self, url: str):
        """Marks a URL as failed."""
        if url not in self.failed_urls:
            self.failed_urls.add(url)
            # Failed URLs *do not* count towards the processed_count limit here
            # self.processed_count += 1 # Decide if failures count to max_urls
            try:
                self.queue.task_done()
            except ValueError:
                 logger.warning(f"task_done() called too many times for failed URL: {url}")
            logger.warning(f"Marked as failed: {url}")
        else:
            logger.debug(f"URL already marked as failed: {url}")

    def get_queue_size(self) -> int:
        return self.queue.qsize()

    def is_done(self) -> bool:
        # Done if queue is empty AND (all seen are processed/failed OR max limit reached)
        all_accounted_for = self.seen_urls.issubset(self.processed_urls.union(self.failed_urls))
        limit_reached = self.processed_count >= self.max_urls
        queue_empty = self.queue.empty()
        # logger.debug(f"is_done check: queue_empty={queue_empty}, limit_reached={limit_reached}, all_accounted_for={all_accounted_for}")
        # Need to be careful with async queue checks, qsize might be more reliable than empty() in loops
        return queue_empty and (limit_reached or all_accounted_for)

    def save_state(self):
        """Saves the current state of the crawler."""
        # Drain the queue safely to save its current content
        queue_items = []
        temp_queue = asyncio.Queue()
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait() # Use non-blocking get
                queue_items.append(item)
                # We don't put it back in the original queue here
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.exception(f"Error draining queue for saving state: {e}")

        state = {
            "seen_urls": list(self.seen_urls),
            "processed_urls": list(self.processed_urls),
            "failed_urls": list(self.failed_urls),
            "processed_count": self.processed_count,
            "queue": queue_items # Save the drained items
        }
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Crawler state saved to {self.state_file}")
        except Exception as e:
            logger.exception(f"Failed to save crawler state: {e}")
        finally:
             # Repopulate the original queue (important!)
             for item in queue_items:
                 asyncio.create_task(self.queue.put(item))
             logger.debug(f"Repopulated queue with {len(queue_items)} items after saving state.")


    def load_state(self):
        """Loads the crawler state from the file."""
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.seen_urls = set(state.get("seen_urls", []))
            self.processed_urls = set(state.get("processed_urls", []))
            self.failed_urls = set(state.get("failed_urls", []))
            self.processed_count = state.get("processed_count", 0)

            # Repopulate queue only with items not already processed/failed
            queue_items = state.get("queue", [])
            self.queue = asyncio.Queue() # Reset queue before loading
            loaded_count = 0
            for item in queue_items:
                url = item.get('url')
                if url and url not in self.processed_urls and url not in self.failed_urls:
                     asyncio.create_task(self.queue.put(item))
                     loaded_count += 1

            logger.info(f"Crawler state loaded from {self.state_file}")
            logger.info(f"Resuming with: {len(self.processed_urls)} processed, {len(self.failed_urls)} failed, {loaded_count} in queue.")
        except FileNotFoundError:
            logger.warning(f"Resume state file not found: {self.state_file}. Starting fresh.")
            self.reset_state()
        except Exception as e:
            logger.exception(f"Failed to load crawler state from {self.state_file}. Starting fresh.")
            self.reset_state()

    def reset_state(self):
         """Resets the manager's state."""
         self.seen_urls = set()
         self.processed_urls = set()
         self.failed_urls = set()
         self.processed_count = 0
         self.queue = asyncio.Queue()

# --- Error Handling & Progress Saving ---
def log_url_error(url_manager: URLManager, url: str, error: str):
    """Logs an error for a specific URL and marks it as failed."""
    logger.error(f"Processing failed for {url}: {error}")
    if url_manager:
        url_manager.mark_as_failed(url)
    else:
        logger.error("URLManager instance not provided to log_url_error")

# === Coder comment ===
# URL取得関数
#    - コマンドライン引数からURL取得
#    - バリデーションとエラーメッセージ
# === === ===
def get_root_url():
    message = None
    for i, arg in enumerate(sys.argv):
        if arg == "--message":
            if i + 1 < len(sys.argv):
                message = sys.argv[i + 1]
                break

    if message:
        root_url = message
        print(f"[DEBUG] Using URL from --message option: {root_url}")
        return root_url
    else:
        root_url = input("Enter the root URL (e.g. https://fast-agent.ai/): ").strip()
        if not root_url.startswith("http"):
            print("Please provide a valid URL (e.g. https://example.com/)")
            return None
        return root_url

# === Coder comment ===
# レジュームインデックス取得関数
#    - レジュームファイルから最後のインデックスを読み込み
# === === ===
def get_resume_index(resume_file):
    try:
        with open(resume_file, "r") as f:
            start_index = int(f.read().strip())
            print(f"[INFO] Resuming from index {start_index + 1}")
            return start_index
    except FileNotFoundError:
        return 0

# === Coder comment ===
# URL処理関数
#    - URLの取得、処理、保存を一連の流れで実装
# === === ===
async def process_url(agent, url, url_manager, index, total_urls):
    print(f"[DEBUG process_url] type(url_manager): {type(url_manager)}")
    print(f"[INFO] Processing URL {index}/{total_urls}: {url}")
    
    next_url = url_manager['get_next_url']()
    if not next_url:
        print("[WARNING process_url] No next URL from url_manager")
        return
        
    markdown = await agent.content_fetcher_agent(url=next_url)

    if markdown and not markdown.startswith("[ERROR]"):
        cleaned = await agent.content_cleaner_agent(markdown=markdown)
    
    url_manager['mark_as_processed'](next_url)
    print(f"[PROGRESS] {index}/{total_urls} URLs processed ({(index/total_urls)*100:.2f}%)")
    
    save_crawled_content(cleaned, next_url)

# === Coder comment ===
# プログレス保存関数
#    - 現在の進行状態をファイルに保存
# === === ===
async def save_progress(url_manager: URLManager, content_list: List[str], output_filename: str):
    """Saves the collected content and the crawler state."""
    logger = logging.getLogger(__name__) # Ensure logger is available
    # Debounce or rate-limit saving if called very frequently
    current_time = time.time()
    # Use a simple approach for debounce check, avoiding complex hasattr state
    if not hasattr(save_progress, 'last_save_time'):
        save_progress.last_save_time = 0 # Initialize if not present

    if current_time - save_progress.last_save_time < 5:
        logger.debug("Skipping save progress due to rate limit.")
        return
    save_progress.last_save_time = current_time

    logger.info(f"Saving progress... {url_manager.processed_count} URLs processed.")
    # Save content
    try:
        # Join content carefully, ensuring structure
        full_content = "".join(content_list).strip() # Content already includes headers/newlines

        with open(output_filename, "w", encoding="utf-8") as f:
             # Add a main header if the file is potentially empty or just resuming
             if not full_content.startswith("# Web Scraping Results"):
                 f.write("# Web Scraping Results\n")
             # Ensure there's a newline after header if content exists
             if not full_content.startswith("# Web Scraping Results\n\n") and full_content:
                  f.write("\n")
             f.write(full_content)
             f.write("\n") # Ensure trailing newline
        logger.debug(f"Content saved to {output_filename}")
    except Exception as e:
        logger.exception(f"Error saving content to {output_filename}: {e}")

    # Save state for resume
    if url_manager and url_manager.resume_enabled:
        url_manager.save_state()

# === Coder comment ===
# エラー処理関数
#    - エラー時の処理とログ記録
# === === ===
def handle_error(agent, url, url_manager, error):
    print(f"[DEBUG handle_error] type(url_manager): {type(url_manager)}")
    print(f"[ERROR] Failed to process {url}: {error}")
    url_manager['mark_as_failed'](url)
    save_crawled_content("", url, str(error))

# === Coder comment ===
# 1. エージェント指向のWebクローリング実装
# 2. URL収集、スクリーニング、内容抽出、ノイズ除去の各機能を分離
# 3. プログレス管理とエラー処理を実装
# === === ===



async def url_collector_agent(root_url: str) -> List[str]:
    """Fetches and parses sitemap.xml from the root URL.
    
    Args:
        root_url: The base URL of the website.
    
    Returns:
        A list of unique URLs found in the sitemap(s).
    """
    sitemap_urls_to_check = set()
    # Assume sitemap is at /sitemap.xml, handle robots.txt later if needed
    sitemap_url = urljoin(root_url, "/sitemap.xml")
    sitemap_urls_to_check.add(sitemap_url)
    processed_sitemaps = set()
    all_found_urls = set()
    
    # Use aiohttp for async HTTP requests
    async with aiohttp.ClientSession() as session:
        while sitemap_urls_to_check:
            current_sitemap_url = sitemap_urls_to_check.pop()
            if current_sitemap_url in processed_sitemaps:
                continue
            
            logger.info(f"Fetching sitemap: {current_sitemap_url}")
            processed_sitemaps.add(current_sitemap_url)
            
            try:
                async with session.get(current_sitemap_url, timeout=30) as response:
                    response.raise_for_status() # Raise error for bad responses (4xx or 5xx)
                    content = await response.text()
                    soup = BeautifulSoup(content, 'xml')
                    
                    # Check for sitemap index file
                    sitemap_tags = soup.find_all('sitemap')
                    if sitemap_tags:
                        for tag in sitemap_tags:
                            loc = tag.find('loc')
                            if loc and loc.text:
                                nested_sitemap_url = loc.text.strip()
                                if nested_sitemap_url not in processed_sitemaps:
                                    sitemap_urls_to_check.add(nested_sitemap_url)
                        continue # Move to the next sitemap URL
                    
                    # If not an index, find URLs within this sitemap
                    url_tags = soup.find_all('url')
                    for tag in url_tags:
                        loc = tag.find('loc')
                        if loc and loc.text:
                            found_url = loc.text.strip()
                            # Basic validation: ensure it's an HTTP/HTTPS URL
                            if found_url.startswith(("http://", "https://")):
                                all_found_urls.add(found_url)
                    
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching sitemap {current_sitemap_url}: {e}")
            except Exception as e:
                logger.error(f"Error parsing sitemap {current_sitemap_url}: {e}")
                
    logger.info(f"Found {len(all_found_urls)} unique URLs in sitemaps.")
    return list(all_found_urls)

@fast.agent(
    "link_extractor_agent",
    instruction="Given the raw HTML of a web page, extract all unique internal links (absolute URLs within the same domain) that are likely to be important for crawling (e.g., documentation, guides, API, tutorials, etc). Return a Python list of URLs as strings. Do not include navigation, footer, or external links."
)
async def link_extractor_agent(agent, html: str, base_url: str):
    """HTMLから内部リンクを抽出
    
    1. HTMLからリンクを抽出
    2. 外部リンクを除外
    3. 重複を削除
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        
        if href.startswith("http"):
            if base_url in href:
                links.add(href)
            continue
            
        try:
            from urllib.parse import urljoin
            abs_url = urljoin(base_url, href)
            links.add(abs_url)
        except:
            continue
            
    return sorted(list(links))

@fast.agent(
    "metrics_generator_agent",
    instruction="Given the full content of a website's root page (Markdown or HTML), extract a list of important page types, categories, or features that should be prioritized for crawling (e.g., API reference, documentation, tutorials, release notes, etc). Return a Python list of keywords or patterns that can be used to evaluate the importance of URLs in the sitemap."
)
async def metrics_generator_agent(agent, root_content: str):
    """ルートページから重要なキーワードを抽出
    
    1. ページタイトルの解析
    2. ヘッダー（h1-h6）の解析
    3. ナビゲーションメニューの解析
    4. メタタグの解析
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(root_content, "html.parser")
    
    keywords = set()
    
    title = soup.title.string if soup.title else ""
    keywords.update(title.lower().split())
    
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if header.string:
            keywords.update(header.string.lower().split())
    
    nav = soup.find("nav")
    if nav:
        for a in nav.find_all("a", href=True):
            if a.string:
                keywords.update(a.string.lower().split())
    
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and "content" in meta.attrs:
        keywords.update(meta["content"].lower().split())
    
    common_keywords = {
        "api", "documentation", "docs", "tutorial", "guide", "reference",
        "faq", "examples", "how-to", "usage", "installation"
    }
    keywords.update(common_keywords)
    
    return sorted(list(keywords))

@fast.agent(
    "url_evaluator_agent",
    instruction="Evaluate URLs based on metrics and assign priority scores.",
    human_input=False
)
async def url_evaluator_agent(agent, urls: list, metrics: list):
    """URLの評価を行い、優先度スコアを付与する"""
    try:
        evaluated = []
        for url in urls:
            score = "HIGH"
            evaluated.append({
                "url": url,
                "score": score
            })
        return evaluated
    except Exception as e:
        print(f"[ERROR] URL評価に失敗しました: {e}")
        return []

@fast.agent(
    "content_fetcher_agent",
    instruction="Fetch the content of the given URL as Markdown using Fetch MCP.",
    servers=["fetch"]
)
async def content_fetcher_agent(agent, url: str):
    """Fetches web page content using Fetch MCP.
 
    Args:
        agent: The FastAgent instance.
        url: The URL to fetch.
 
    Returns:
        The fetched content (Markdown) or an error message.
    """
    logger.debug(f"content_fetcher_agent received URL: {url}")
 
    if not url:
        logger.error("content_fetcher_agent received an empty URL.")
        return "[ERROR] Invalid URL"
 
    try:
        # Check MCP structure (simplified)
        if hasattr(agent, 'mcp') and agent.mcp and hasattr(agent.mcp, 'fetch') and agent.mcp.fetch and hasattr(agent.mcp.fetch, 'fetch_markdown'):
            logger.debug("Attempting to fetch Markdown...")
            content_result = await agent.mcp.fetch.fetch_markdown(url=url)
            logger.debug(f"Markdown fetch result type: {type(content_result)}")
            # Log first 500 chars if it's a string
            if isinstance(content_result, str):
                 logger.debug(f"Markdown fetch result (first 500 chars): {content_result[:500]}")
            else:
                logger.debug(f"Markdown fetch result: {content_result}")
            return content_result
        else:
            logger.error("MCP fetch server or 'fetch_markdown' method not available on agent.")
            # Enhanced debugging for missing MCP parts
            mcp_present = hasattr(agent, 'mcp') and agent.mcp is not None
            fetch_present = mcp_present and hasattr(agent.mcp, 'fetch') and agent.mcp.fetch is not None
            fetch_md_present = fetch_present and hasattr(agent.mcp.fetch, 'fetch_markdown')
            logger.debug(f"MCP Checks: mcp={mcp_present}, fetch={fetch_present}, fetch_markdown={fetch_md_present}")
            return "[ERROR] MCP fetch server misconfiguration or unavailable."
 
    except Exception as e:
        logger.exception(f"Unexpected error in content_fetcher_agent for {url}: {e}")
        return f"[ERROR] Exception during fetch: {e}"

@fast.agent(
    "content_cleaner_agent",
    instruction="Given the Markdown content of a web page, remove navigation, footer, sidebar, ads, duplicated explanations, and error messages. Do NOT summarize or rewrite the content, just remove obvious noise and keep the main text as Markdown. Return only the cleaned Markdown."
)
async def content_cleaner_agent(agent, markdown: str):
    """コンテンツのノイズを除去
    
    1. ナビゲーションの除去
    2. フッターの除去
    3. サイドバーの除去
    4. 広告の除去
    5. 重複説明の削除
    6. エラーメッセージの削除
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(markdown, "html.parser")
    
    for nav in soup.find_all(["nav", "header"]):
        nav.decompose()
    
    for footer in soup.find_all("footer"):
        footer.decompose()
    
    for aside in soup.find_all("aside"):
        aside.decompose()
    
    for ad in soup.find_all(class_=lambda x: x and "ad" in x.lower()):
        ad.decompose()
    
    seen = set()
    for p in soup.find_all("p"):
        text = p.get_text().strip().lower()
        if text in seen:
            p.decompose()
        else:
            seen.add(text)
    
    for error in soup.find_all(class_=lambda x: x and any(w in x.lower() for w in ["error", "warning", "alert"])):
        error.decompose()
    
    cleaned = "".join(str(tag) for tag in soup.find_all(text=True))
    return cleaned

def calculate_scale(urls: list) -> dict:
    """Calculate the scale of website scraping based on URL count."""
    try:
        if not isinstance(urls, list):
            urls = []
            print("[WARNING] Input is not a list of URLs. Using empty list.")
        
        url_count = len(urls)
        
        processing_time_per_url = 5
        tokens_per_url = 1000
        
        estimated_time = url_count * processing_time_per_url
        estimated_tokens = url_count * tokens_per_url
        
        return {
            'url_count': url_count,
            'estimated_time': estimated_time,
            'estimated_tokens': estimated_tokens
        }
    except Exception as e:
        print(f"[ERROR] Scale calculation failed: {e}")
        return {
            'url_count': 0,
            'estimated_time': 0,
            'estimated_tokens': 0
        }

@fast.agent(
    "user_decision_agent",
    instruction="Display scale estimates to the user and get their choice. Provide options: [y]es to proceed, [n]o for URL screening, [e]xit to terminate. Return user's choice.",
    human_input=True
)
async def user_decision_agent(agent, estimates):
    """ユーザー意思決定Agent
    
    1. 規模情報を表示
    2. ユーザーからの選択を受け取る
    """
    print("\n=== Website Scraping Scale Estimates ===")
    print(f"URL数: {estimates.get('url_count', 'N/A')}件")
    print(f"推定処理時間: 約{estimates.get('estimated_time', 0) // 60}分 {estimates.get('estimated_time', 0) % 60}秒")
    print(f"推定トークン消費: 約{estimates.get('estimated_tokens', 0):,}トークン")
    
    while True:
        print("\n選択肢:")
        print("[y] すべてのURLを処理")
        print("[n] URLをスクリーニング")
        print("[e] 処理を中止")
        
        choice = input("\n選択してください（y/n/e）: ").lower()
        if choice in ["y", "n", "e"]:
            return choice
        print("\n無効な選択です。y, n, e のいずれかを入力してください。")

def url_screening_agent(urls: list):
    """URLスクリーニング関数
    
    Args:
        urls: URLのリスト
        
    Returns:
        スクリーニング後のURLのリスト
    """
    def evaluate_url(url: str) -> float:
        """URLの重要度を評価する
        
        Args:
            url: 評価対象のURL
            
        Returns:
            0.0から1.0の間のスコア
        """
        try:
            path = url.split('/')[-1]
            
            important_keywords = ['documentation', 'docs', 'api', 'tutorial', 'guide', 'examples']
            
            score = 0.0
            
            for keyword in important_keywords:
                if keyword in path.lower():
                    score += 0.2
            
            if path == '' or path == 'index.html':
                score += 0.3
                
            score = min(score, 1.0)
            
            print(f"[DEBUG] URL evaluation: {url} -> Score: {score}")
            return score
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate URL {url}: {e}")
            return 0.5

    try:
        if not urls:
            print("[WARNING] No URLs to screen")
            return []
            
        filtered_urls = []
        for url in urls:
            score = evaluate_url(url)
            print(f"[DEBUG] URL evaluation: {url} -> Score: {score}")
            if score > 0.5:
                filtered_urls.append(url)
        
        print(f"[INFO] URLスクリーニング後: {len(filtered_urls)}件のURLが残りました")
        return filtered_urls
        
    except Exception as e:
        print(f"[ERROR] URLスクリーニングに失敗しました: {e}")
        return urls

@fast.agent(
    "url_manager_agent",
    instruction="Manage URL processing queue and states. Track progress and handle failures. Save state to JSON.",
    use_history=False
)
async def url_manager_agent(agent, urls: list, url_hash: str):
    """URL管理Agent
    
    1. URLの状態管理
    2. プライオリティキューの管理
    3. 状態のJSON保存
    """
    state_file = f"crawl_state_{url_hash}.json"
    
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            url_states = state_data["url_states"]
            processed_count = state_data["processed_count"]
    except (FileNotFoundError, json.JSONDecodeError):
        url_states = {url: "pending" for url in urls}
        processed_count = 0
    
    priority_queue = deque(
        [url for url, state in url_states.items() if state == "pending"]
    )
    
    def get_next_url():
        if priority_queue:
            return priority_queue.popleft()
        return None
    
    def mark_as_processed(url: str):
        url_states[url] = "processed"
        nonlocal processed_count
        processed_count += 1
        
        state_data = {
            "url_states": url_states,
            "processed_count": processed_count
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def mark_as_failed(url: str):
        url_states[url] = "failed"
        
        state_data = {
            "url_states": url_states,
            "processed_count": processed_count
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def get_remaining_urls():
        return [url for url, state in url_states.items() if state == "pending"]
    
    return {
        "get_next_url": get_next_url,
        "mark_as_processed": mark_as_processed,
        "mark_as_failed": mark_as_failed,
        "get_remaining_urls": get_remaining_urls
    }

async def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Agentic Web Scraper')
    parser.add_argument('--message', type=str, required=True, help='The root URL to start scraping (e.g., https://fast-agent.ai/)')
    parser.add_argument('--max_urls', type=int, default=DEFAULT_MAX_URLS, help='Maximum number of URLs to process')
    parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE, help='Output file name for crawled content')
    parser.add_argument('--resume', action='store_true', help='Resume from the last processed URL')
    args = parser.parse_args()

    root_url = args.message
    max_urls = args.max_urls
    output_filename = args.output_file
    resume_enabled = args.resume

    if not is_valid_url(root_url):
        logger.error(f"Invalid root URL provided: {root_url}")
        print(f"Error: Invalid root URL provided: {root_url}. Please include http:// or https://")
        sys.exit(1)

    logger.info(f"Starting crawl for root URL: {root_url}")
    logger.info(f"Maximum URLs to process: {max_urls}")
    logger.info(f"Output file: {output_filename}")
    logger.info(f"Resume enabled: {resume_enabled}")

    url_manager = URLManager(max_urls=max_urls, resume_enabled=resume_enabled, root_url=root_url)

    processed_content = []

    async with fast.run() as agent:
        logger.info("FastAgent context entered.")
        logger.debug(f"Agent object in context: {type(agent)}")
        logger.debug(f"Agent has mcp attribute: {'mcp' in dir(agent)}")
        if 'mcp' in dir(agent):
            logger.debug(f"Agent.mcp has fetch attribute: {'fetch' in dir(agent.mcp)}")
            if 'fetch' in dir(agent.mcp):
                logger.debug(f"Agent.mcp.fetch has fetch_markdown attribute: {'fetch_markdown' in dir(agent.mcp.fetch)}")

        # --- Initial URL Collection --- #
        logger.info("Collecting initial URLs from sitemap...")
        try:
            logger.info("Calling url_collector_agent...")
            # Pass primary input as positional argument
            initial_urls_result = await url_collector_agent(root_url)
            logger.info("url_collector_agent call finished.")
            logger.debug(f"url_collector_agent returned: {initial_urls_result}")
            # Assuming run returns the direct result of the agent function
            initial_urls = initial_urls_result
 
            if initial_urls:
                url_manager.add_urls([{"url": u, "score": "HIGH", "depth": 0} for u in initial_urls])
            else:
                logger.warning("URL Collector returned no initial URLs. Adding root URL.")
                url_manager.add_urls([{"url": root_url, "score": "HIGH", "depth": 0}])
        except Exception as e:
            logger.exception(f"Error during url_collector_agent execution: {e}")
            initial_urls = [] # Ensure initial_urls is defined even on error
            logger.warning("Proceeding without initial URLs due to collection error.")
            url_manager.add_urls([{"url": root_url, "score": "HIGH", "depth": 0}]) # Add root as fallback

        processed_urls_count = 0
        if resume_enabled and os.path.exists(output_filename):
            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    processed_urls_count = content.count('\n# URL:')
                    processed_content.append(content)
                logger.info(f"Loaded {processed_urls_count} entries from existing output file {output_filename}")
            except Exception as e:
                logger.exception(f"Error reading resume file {output_filename}. Starting fresh.")
                processed_urls_count = 0
                processed_content = []

        logger.debug(f"URL Manager queue size before loop: {url_manager.get_queue_size()}")
        # --- Main Loop --- #
        while url_data := await url_manager.get_next_url():
            url = url_data['url']
            depth = url_data.get('depth', 0)
            logger.info(f"Processing URL {url_manager.processed_count}/{url_manager.get_queue_size() + url_manager.processed_count}: {url} (Depth: {depth})")

            try:
                logger.debug(f"Fetching content for {url}")
                markdown_content_result = await agent.content_fetcher_agent(url)

                if isinstance(markdown_content_result, str) and markdown_content_result.startswith("[ERROR]"):
                    logger.error(f"content_fetcher_agent returned an error string: {markdown_content_result}")
                    log_url_error(url_manager, url, markdown_content_result)
                    continue
                markdown_content = markdown_content_result # Assign if not an error
 
                if markdown_content:
                    cleaned_content_result = await agent.content_cleaner_agent(markdown_content)

                    if isinstance(cleaned_content_result, str) and cleaned_content_result.startswith("[ERROR]"):
                        logger.error(f"content_cleaner_agent returned an error string: {cleaned_content_result}")
                        log_url_error(url_manager, url, cleaned_content_result)
                        continue
                    cleaned_content = cleaned_content_result

                    if cleaned_content:
                        content_entry = f"\n# URL: {url}\n# Depth: {depth}\n\n{cleaned_content}\n"
                        processed_content.append(content_entry)
                        logger.info(f"Successfully fetched and cleaned content for {url}")
                    else:
                        logger.warning(f"Content cleaner returned empty for {url}")
                        log_url_error(url_manager, url, "Cleaning returned empty")
                else:
                    logger.error(f"Content fetcher failed for {url}: {markdown_content}")
                    log_url_error(url_manager, url, f"Fetching failed: {markdown_content}")
            except Exception as e:
                logger.exception(f"Error processing URL {url}: {e}")
                log_url_error(url_manager, url, str(e))

            if url_manager.processed_count % SAVE_INTERVAL == 0 or url_manager.is_done():
                 await save_progress(url_manager, processed_content, output_filename)

    logger.info("FastAgent context exited.")

    await save_progress(url_manager, processed_content, output_filename)

    logger.info("Crawling finished.")
    logger.info(f"Total URLs visited: {url_manager.processed_count}")
    logger.info(f"Results saved to {output_filename}")

if __name__ == "__main__":
    asyncio.run(main())