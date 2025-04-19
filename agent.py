import asyncio
import requests
import xml.etree.ElementTree as ET
import sys
import hashlib
import json
from pathlib import Path
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.direct_decorators import agent
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from collections import deque
import lxml

fast = FastAgent("SiteMapCrawler")

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
    
    # URL管理Agentから次のURLを取得
    next_url = url_manager['get_next_url']()
    if not next_url:
        print("[WARNING process_url] No next URL from url_manager")
        return
        
    # コンテンツ取得と処理
    markdown = await agent.content_fetcher_agent(next_url)
    cleaned = await agent.content_cleaner_agent(markdown)
    
    # URL管理Agentに処理完了を通知
    url_manager['mark_as_processed'](next_url)
    print(f"[PROGRESS] {index}/{total_urls} URLs processed ({(index/total_urls)*100:.2f}%)")
    
    # コンテンツを保存
    save_crawled_content(cleaned, url)

# === Coder comment ===
# プログレス保存関数
#    - 現在の進行状態をファイルに保存
# === === ===
def save_progress(resume_file, index):
    with open(resume_file, "w") as f:
        f.write(str(index))

# === Coder comment ===
# エラー処理関数
#    - エラー時の処理とログ記録
# === === ===
def handle_error(agent, url, url_manager, error):
    print(f"[DEBUG handle_error] type(url_manager): {type(url_manager)}")
    print(f"[ERROR] Failed to process {url}: {error}")
    # URL管理Agentに処理失敗を通知
    url_manager['mark_as_failed'](url)
    # エラー情報を保存
    save_crawled_content("", url, str(error))

# === Coder comment ===
# 1. エージェント指向のWebクローリング実装
# 2. URL収集、スクリーニング、内容抽出、ノイズ除去の各機能を分離
# 3. プログレス管理とエラー処理を実装
# === === ===



def url_collector_agent(root_url: str) -> list:
    """URL収集関数
    
    Args:
        root_url: ルートURL
        
    Returns:
        URLのリスト
    """
    try:
        print(f"[DEBUG] Starting URL collection for: {root_url}")
        
        # サイトマップからURLを収集
        sitemap_url = root_url.rstrip('/') + '/sitemap.xml'
        print(f"[DEBUG] Trying to fetch sitemap from: {sitemap_url}")
        
        try:
            response = requests.get(sitemap_url, timeout=10)
            print(f"[DEBUG] Sitemap response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    soup = BeautifulSoup(response.content, 'xml')
                    print(f"[DEBUG] Successfully parsed XML")
                    
                    # URLを抽出
                    url_elements = soup.find_all('url')
                    print(f"[DEBUG] Found {len(url_elements)} URL elements")
                    
                    urls = []
                    for url_elem in url_elements:
                        loc = url_elem.find('loc')
                        if loc and loc.text:
                            urls.append(loc.text.strip())
                    print(f"[INFO] Collected {len(urls)} URLs from sitemap")
                    return urls
                except Exception as parse_error:
                    print(f"[ERROR] Failed to parse XML: {parse_error}")
                    print(f"[DEBUG] Response content: {response.content[:500]}...")
                    return []
            else:
                print(f"[WARNING] Sitemap request failed with status: {response.status_code}")
                print(f"[DEBUG] Response headers: {dict(response.headers)}")
                return []
        except requests.RequestException as e:
            print(f"[WARNING] Failed to fetch sitemap: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return []
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in URL collection: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        print(f"[DEBUG] Exception args: {e.args}")
        return []

@fast.agent(
    "url_collector_agent",
    instruction="Collect URLs from sitemap.xml using requests and BeautifulSoup. Return a list of unique URLs.",
    servers=[]
)
async def url_collector_agent(agent, root_url: str):
    """URL収集Agent
    
    Args:
        agent: FastAgent instance
        root_url: ルートURL
        
    Returns:
        URLのリスト
    """
    try:
        print(f"[DEBUG] Starting URL collection for: {root_url}")
        
        # サイトマップからURLを収集
        sitemap_url = root_url.rstrip('/') + '/sitemap.xml'
        print(f"[DEBUG] Trying to fetch sitemap from: {sitemap_url}")
        
        try:
            response = requests.get(sitemap_url, timeout=10)
            print(f"[DEBUG] Sitemap response status: {response.status_code}")
            print(f"[DEBUG] Sitemap headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    soup = BeautifulSoup(response.content, 'xml')
                    print(f"[DEBUG] Successfully parsed XML")
                    print(f"[DEBUG] XML content length: {len(response.content)} bytes")
                    
                    # URLを抽出
                    url_elements = soup.find_all('url')
                    print(f"[DEBUG] Found {len(url_elements)} URL elements")
                    
                    urls = []
                    for url_elem in url_elements:
                        loc = url_elem.find('loc')
                        if loc and loc.text:
                            url = loc.text.strip()
                            print(f"[DEBUG] Found URL: {url}")
                            urls.append(url)
                    print(f"[INFO] Collected {len(urls)} URLs from sitemap")
                    return urls
                except Exception as parse_error:
                    print(f"[ERROR] Failed to parse XML: {parse_error}")
                    print(f"[DEBUG] Response content: {response.content[:500]}...")
                    return []
            else:
                print(f"[WARNING] Sitemap request failed with status: {response.status_code}")
                print(f"[DEBUG] Response headers: {dict(response.headers)}")
                return []
        except requests.RequestException as e:
            print(f"[WARNING] Failed to fetch sitemap: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            return []
            
    except Exception as e:
        print(f"[ERROR] Unexpected error in URL collection: {e}")
        print(f"[DEBUG] Exception type: {type(e)}")
        print(f"[DEBUG] Exception args: {e.args}")
        return []

def collect_urls(root_url):
    try:
        # サイトマップの取得を試みる
        sitemap_url = root_url.rstrip('/') + '/sitemap.xml'
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            urls = [url.loc.text for url in soup.find_all('url')]
            print(f"[INFO] サイトマップから{len(urls)}件のURLを取得しました")
            return urls
        else:
            print(f"[WARNING] サイトマップの取得に失敗しました (status: {response.status_code})")
            return []
        normalized_urls = []
        for url in urls:
            if url and not url.startswith('#'):
                if not url.startswith('http'):
                    url = urljoin(root_url, url)
                normalized_urls.append(url)
        
        unique_urls = list(set(normalized_urls))
        print(f"[INFO] 合計{len(unique_urls)}件のユニークなURLを取得しました")
        return unique_urls
        
    except Exception as e:
        print(f"[ERROR] URL収集に失敗しました: {e}")
        return []

@fast.agent(
    "link_extractor_agent",
    instruction="Given the raw HTML of a web page, extract all unique internal links (absolute URLs within the same domain) that are likely to be important for crawling (e.g., documentation, guides, API, tutorials, etc). Return a Python list of URLs as strings. Do not include navigation, footer, or external links."
)
    # [Reviewer comment] link_extractor_agent: 内部リンク抽出ロジックが実装され、外部リンク除外や重複排除も考慮されています。
async def link_extractor_agent(agent, html: str, base_url: str):
    """HTMLから内部リンクを抽出
    
    1. HTMLからリンクを抽出
    2. 外部リンクを除外
    3. 重複を削除
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    
    # すべてのリンクを取得
    for a in soup.find_all("a", href=True):
        href = a["href"]
        
        # 絶対パスのURLはそのまま追加
        if href.startswith("http"):
            if base_url in href:
                links.add(href)
            continue
            
        # 相対パスのURLを絶対パスに変換
        try:
            from urllib.parse import urljoin
            abs_url = urljoin(base_url, href)
            links.add(abs_url)
        except:
            continue
            
    # リンクのリスト化とソート
    return sorted(list(links))

@fast.agent(
    "metrics_generator_agent",
    instruction="Given the full content of a website's root page (Markdown or HTML), extract a list of important page types, categories, or features that should be prioritized for crawling (e.g., API reference, documentation, tutorials, release notes, etc). Return a Python list of keywords or patterns that can be used to evaluate the importance of URLs in the sitemap."
)
    # [Reviewer comment] metrics_generator_agent: ルートページからタイトル・ヘッダー・ナビゲーション・メタタグ等を解析し、重要キーワードを抽出するロジックが実装されています。
async def metrics_generator_agent(agent, root_content: str):
    """ルートページから重要なキーワードを抽出
    
    1. ページタイトルの解析
    2. ヘッダー（h1-h6）の解析
    3. ナビゲーションメニューの解析
    4. メタタグの解析
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(root_content, "html.parser")
    
    # 重要なキーワードのリスト
    keywords = set()
    
    # ページタイトルからキーワードを抽出
    title = soup.title.string if soup.title else ""
    keywords.update(title.lower().split())
    
    # ヘッダーからキーワードを抽出
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if header.string:
            keywords.update(header.string.lower().split())
    
    # ナビゲーションメニューからキーワードを抽出
    nav = soup.find("nav")
    if nav:
        for a in nav.find_all("a", href=True):
            if a.string:
                keywords.update(a.string.lower().split())
    
    # メタタグからキーワードを抽出
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and "content" in meta.attrs:
        keywords.update(meta["content"].lower().split())
    
    # 一般的な重要ページのキーワードを追加
    common_keywords = {
        "api", "documentation", "docs", "tutorial", "guide", "reference",
        "faq", "examples", "how-to", "usage", "installation"
    }
    keywords.update(common_keywords)
    
    # キーワードのリスト化とソート
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
            score = "HIGH"  # デフォルトは高優先度
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
    instruction="Fetch the content of the given URL as Markdown using Fetch MCP. If raw_html is True, fetch the raw HTML instead.",
    servers=["fetch"]
)
async def content_fetcher_agent(agent, url: str, raw_html: bool = False):
    try:
        if raw_html:
            content = await agent.mcp.fetch.fetch_markdown(url=url, raw=True, max_length=100000)
        else:
            content = await agent.mcp.fetch.fetch_markdown(url=url, max_length=100000)
        if isinstance(content, str):
            return content
        else:
            return "[ERROR] fetch_markdown did not return string"
    except Exception as e:
        return f"[ERROR] Exception: {e}"

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
    
    # HTMLをBeautifulSoupでパース
    soup = BeautifulSoup(markdown, "html.parser")
    
    # ナビゲーションの除去
    for nav in soup.find_all(["nav", "header"]):
        nav.decompose()
    
    # フッターの除去
    for footer in soup.find_all("footer"):
        footer.decompose()
    
    # サイドバーの除去
    for aside in soup.find_all("aside"):
        aside.decompose()
    
    # 広告の除去
    for ad in soup.find_all(class_=lambda x: x and "ad" in x.lower()):
        ad.decompose()
    
    # 重複説明の削除
    seen = set()
    for p in soup.find_all("p"):
        text = p.get_text().strip().lower()
        if text in seen:
            p.decompose()
        else:
            seen.add(text)
    
    # エラーメッセージの削除
    for error in soup.find_all(class_=lambda x: x and any(w in x.lower() for w in ["error", "warning", "alert"])):
        error.decompose()
    
    # Markdown形式に変換
    cleaned = "".join(str(tag) for tag in soup.find_all(text=True))
    return cleaned

def calculate_scale(urls: list) -> dict:
    """Calculate the scale of website scraping based on URL count."""
    try:
        if not isinstance(urls, list):
            urls = []
            print("[WARNING] Input is not a list of URLs. Using empty list.")
        
        url_count = len(urls)
        
        # Constants for estimation
        processing_time_per_url = 5  # seconds per URL
        tokens_per_url = 1000  # tokens per URL
        
        # Calculate estimates
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
            # URLのパスを取得
            path = url.split('/')[-1]
            
            # 重要なキーワードのリスト
            important_keywords = ['documentation', 'docs', 'api', 'tutorial', 'guide', 'examples']
            
            # スコア計算
            score = 0.0
            
            # キーワードマッチング
            for keyword in important_keywords:
                if keyword in path.lower():
                    score += 0.2
            
            # ルートページのスコア
            if path == '' or path == 'index.html':
                score += 0.3
                
            # スコアを0.0から1.0の範囲に正規化
            score = min(score, 1.0)
            
            print(f"[DEBUG] URL evaluation: {url} -> Score: {score}")
            return score
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate URL {url}: {e}")
            return 0.5  # エラー時は中立的なスコアを返す

    try:
        if not urls:
            print("[WARNING] No URLs to screen")
            return []
            
        # URLを評価
        filtered_urls = []
        for url in urls:
            score = evaluate_url(url)
            print(f"[DEBUG] URL evaluation: {url} -> Score: {score}")
            if score > 0.5:  # スコアが0.5以上のURLのみを保持
                filtered_urls.append(url)
        
        print(f"[INFO] URLスクリーニング後: {len(filtered_urls)}件のURLが残りました")
        return filtered_urls
        
    except Exception as e:
        print(f"[ERROR] URLスクリーニングに失敗しました: {e}")
        return urls  # エラー時は元のURLリストを返す

@fast.agent(
    "url_manager_agent",
    instruction="Manage URL processing queue and states. Track progress and handle failures. Save state to JSON.",
    use_history=False
)
    # [Reviewer comment] url_manager_agent: 状態管理・JSON保存の実装により、進捗の永続化・レジューム性が大幅に向上しました。
async def url_manager_agent(agent, urls: list, url_hash: str):
    """URL管理Agent
    
    1. URLの状態管理
    2. プライオリティキューの管理
    3. 状態のJSON保存
    """
    state_file = f"crawl_state_{url_hash}.json"
    
    # 既存の状態ファイルを読み込む
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            url_states = state_data["url_states"]
            processed_count = state_data["processed_count"]
    except (FileNotFoundError, json.JSONDecodeError):
        url_states = {url: "pending" for url in urls}
        processed_count = 0
    
    # プライオリティキューの初期化
    priority_queue = deque(
        [url for url, state in url_states.items() if state == "pending"]
    )
    
    def get_next_url():
        """次の処理対象のURLを取得"""
        if priority_queue:
            return priority_queue.popleft()
        return None
    
    def mark_as_processed(url: str):
        """URLを処理済みとしてマーク"""
        url_states[url] = "processed"
        nonlocal processed_count
        processed_count += 1
        
        # 状態を保存
        state_data = {
            "url_states": url_states,
            "processed_count": processed_count
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def mark_as_failed(url: str):
        """URLを処理失敗としてマーク"""
        url_states[url] = "failed"
        
        # 状態を保存
        state_data = {
            "url_states": url_states,
            "processed_count": processed_count
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
    
    def get_remaining_urls():
        """未処理のURLを取得"""
        return [url for url, state in url_states.items() if state == "pending"]
    
    return {
        "get_next_url": get_next_url,
        "mark_as_processed": mark_as_processed,
        "mark_as_failed": mark_as_failed,
        "get_remaining_urls": get_remaining_urls
    }

    # === Coder comment ===
# === Coder comment ===
# 1. URL入力処理
#    - コマンドライン引数からURL取得
#    - バリデーションとエラーメッセージ
# === === ===
async def main():
    """メイン関数"""
    # URL取得
    root_url = get_root_url()
    if not root_url:
        print("[ERROR] Root URL not found")
        return
    
    print(f"[INFO] Starting with root URL: {root_url}")
    
    # URLの収集
    urls = await url_collector_agent(None, root_url)
    if not urls:
        print("[WARNING] No URLs found")
        return
    
    # URLをリストに変換（リストでない場合は）
    if not isinstance(urls, list):
        print(f"[DEBUG] Converting input to list... type: {type(urls)}")
        urls = [urls] if isinstance(urls, str) else []
        print(f"[INFO] Converted URLs: {urls}")
    
    # 規模の見積もり
    estimates = calculate_scale(urls)
    print(f"[INFO] 規模の見積もり:")
    print(f"  - URL数: {estimates['url_count']}件")
    print(f"  - 推定処理時間: {estimates['estimated_time']}秒")
    print(f"  - 推定トークン数: {estimates['estimated_tokens']}トークン")
    
    # ユーザーの判断を求める
    user_decision = await user_decision_agent(None, estimates)
    if user_decision != 'y':
        print("[INFO] 処理を中止します")
        return
    
    # URL管理の初期化
    url_manager = {
        'urls': urls,
        'processed': [],
        'failed': [],
        'current_index': 0
    }
    
    # URLの処理ループ
    while True:
        # URL管理の型チェック
        if not isinstance(url_manager, dict):
            print(f"[ERROR] url_manager is not a dictionary: {type(url_manager)}")
            break
            
        # 次のURLを取得
        current_index = url_manager.get('current_index', 0)
        if current_index >= len(url_manager['urls']):
            print("[INFO] All URLs processed")
            break
            
        url = url_manager['urls'][current_index]
        print(f"[DEBUG] Processing URL: {url}")
        
        try:
            # コンテンツ取得
            content = await content_fetcher_agent(None, url)
            
            # コンテンツクリーニング
            cleaned_content = await content_cleaner_agent(None, content)
            
            # メトリクス生成
            metrics = await metrics_generator_agent(None, cleaned_content)
            
            # 処理済みURLを追加
            url_manager['processed'].append({
                'url': url,
                'metrics': metrics,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process URL {url}: {e}")
            url_manager['failed'].append({
                'url': url,
                'error': str(e),
                'status': 'failed'
            })
        
        # URL管理の更新
        url_manager['current_index'] = current_index + 1
        
    # 最終的な統計情報の表示
    print(f"[INFO] Processing completed:")
    print(f"  - Total URLs: {len(url_manager['urls'])}")
    print(f"  - Successfully processed: {len(url_manager['processed'])}")
    print(f"  - Failed: {len(url_manager['failed'])}")
    
    return url_manager

async def extract_links_from_html_async(url):
    from playwright.async_api import async_playwright
    
    # Playwrightの初期化
    async with async_playwright() as pw:
        # ブラウザの起動
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        
        try:
            # URLにアクセス
            await page.goto(url)
            # ネットワークアイドル状態になるまで待機
            await page.wait_for_load_state("networkidle")
            
            # リンクを取得
            links = await page.evaluate('''() => {
                const internalLinks = new Set();
                const base = document.baseURI;
                const links = document.getElementsByTagName('a');
                
                for (const link of links) {
                    const href = link.getAttribute('href');
                    if (href && !href.startsWith('#') && !href.startsWith('mailto:') && !href.startsWith('tel:')) {
                        try {
                            const url = new URL(href, base);
                            // 同一ドメインの内部リンクのみを追加
                            if (url.hostname === new URL(base).hostname) {
                                internalLinks.add(url.href);
                            }
                        } catch (e) {
                            // 無効なURLはスキップ
                            continue;
                        }
                    }
                }
                return Array.from(internalLinks);
            }''')
            
            return links
        finally:
            # ブラウザの終了
            await browser.close()

def load_crawl_targets(url_hash: str):
    """クロール対象のURLとメトリクスをJSONから読み込み
    
    1. JSONファイルからデータを読み込み
    2. 状態情報の復元
    """
    crawl_file = f"crawl_targets_{url_hash}.json"
    try:
        with open(crawl_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Return a dictionary
            return {
                "urls": data.get("urls", []),  # Use .get for safety
                "metrics": data.get("metrics", []) # Use .get for safety
            }
    except (FileNotFoundError, json.JSONDecodeError):
        # Return an empty dictionary structure if file not found or invalid JSON
        return {"urls": [], "metrics": []}

def save_crawl_targets(url_hash: str, urls: list, metrics: list):
    """クロール対象のURLとメトリクスをJSON形式で保存
    
    1. URLリストとメトリクスの保存
    2. 状態情報（待機中、処理中、完了）の保存
    """
    crawl_file = f"crawl_targets_{url_hash}.json"
    
    # 既存のファイルを読み込む
    try:
        with open(crawl_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {
            "urls": {},
            "metrics": {},
            "state": "pending"
        }
    
    # URLとメトリクスの更新
    for url in urls:
        data["urls"][url] = {
            "status": "pending",
            "priority": 0,
            "last_attempt": None,
            "error_count": 0
        }
    
    data["metrics"] = metrics
    
    # JSONファイルに保存
    with open(crawl_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Saved crawl targets to {crawl_file}")

def save_crawled_content(content: str, url: str, error: str = None, filename="site_crawl_result.md"):
    """コンテンツを保存
    
    1. 成功時: コンテンツを追記
    2. エラー時: エラーログを追記
    """
    # ファイルが存在しない場合は作成
    if not Path(filename).exists():
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Web Scraping Results\n\n")
    
    # コンテンツの追記
    with open(filename, "a", encoding="utf-8") as f:
        if error:
            f.write(f"## Error processing {url}\n\n")
            f.write(f"Error: {error}\n\n")
        else:
            f.write(f"# {url}\n\n")
            f.write(content)
            f.write("\n\n")
    
    print(f"[INFO] Saved content for {url} to {filename}")

if __name__ == "__main__":
    asyncio.run(main())