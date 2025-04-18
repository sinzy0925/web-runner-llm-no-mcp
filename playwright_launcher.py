# --- ファイル: playwright_launcher.py (起動/終了分離・完全コード) ---
import asyncio
import logging
import os
import time
import traceback
import warnings
from playwright.async_api import (
    async_playwright, Playwright, # ★ Playwright をインポート
    Page, Browser, BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
from playwright_stealth import stealth_async
from typing import List, Tuple, Dict, Any, Optional

# configモジュールをインポートしてタイムアウト値などを参照
try:
    import config
except ImportError:
    # config.py がない場合のデフォルト値を設定
    class ConfigMock:
        DEFAULT_ACTION_TIMEOUT = 10000 # 例: 10秒
        # 他に必要な設定があればここに追加
    config = ConfigMock()
    logging.warning("config.py が見つかりません。デフォルト値を使用します。")


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport") # Playwrightの既知の警告を抑制

# --- ステルスモードエラー検出用定数 (バッチ処理では直接使わないが、残しておく) ---
STEALTH_ERROR_MESSAGE = "大変申し訳ありませんが、ページを正しく表示できませんでした。\n推奨されているブラウザであるかをご確認の上、時間をおいて再度お試しください。"
ERROR_MESSAGE_SELECTOR = "body"


# --- Playwrightとブラウザの起動関数 ---
async def launch_browser(
    headless_mode: bool = False,
    slow_motion: int = 0 # デフォルトのSlowMoは0に変更
) -> Tuple[Playwright, Browser]:
    """Playwrightの起動とブラウザの起動を行う"""
    playwright: Optional[Playwright] = None # 初期化
    browser: Optional[Browser] = None # 初期化
    try:
        playwright = await async_playwright().start()
        logger.info(f"ブラウザ起動 (Chromium, Headless: {headless_mode}, SlowMo: {slow_motion}ms)...")
        browser = await playwright.chromium.launch(
            headless=headless_mode,
            slow_mo=slow_motion,
            # 必要に応じて他の起動オプションを追加
            # args=["--disable-blink-features=AutomationControlled"]
        )
        logger.info("ブラウザの起動に成功しました。")
        # playwright と browser の両方を返す
        # playwright オブジェクトは最後に stop() するために必要
        return playwright, browser
    except Exception as e:
        logger.critical(f"ブラウザの起動に失敗しました: {e}", exc_info=True)
        # 起動失敗時も、可能な範囲でクリーンアップを試みる
        if browser and browser.is_connected(): await browser.close()
        if playwright: await playwright.stop()
        raise # エラーを再送出して呼び出し元に知らせる

# --- コンテキストとページの作成関数 ---
async def create_context_and_page(
    browser: Browser,
    default_timeout: Optional[int] = None, # オプショナルに変更
    apply_stealth: bool = True
) -> Tuple[BrowserContext, Page]:
    """新しいブラウザコンテキストとページを作成し、設定を適用する"""
    context: Optional[BrowserContext] = None # 初期化
    try:
        logger.info("新しいブラウザコンテキストを作成します...")
        # 一般的なユーザーエージェントや設定を適用
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='ja-JP',
            timezone_id='Asia/Tokyo',
            java_script_enabled=True,
            # accept_downloads=True, # 必要に応じてPDFダウンロードなどを有効化
            extra_http_headers={'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7'}
            # 必要に応じて他のコンテキストオプションを追加
        )
        # デフォルトタイムアウトを設定 (引数 > config.py > フォールバック値)
        effective_default_timeout = default_timeout if default_timeout is not None \
                                     else getattr(config, 'DEFAULT_ACTION_TIMEOUT', 10000)
        context.set_default_timeout(effective_default_timeout)
        logger.info(f"コンテキストのデフォルトタイムアウトを {effective_default_timeout}ms に設定しました。")

        # ステルスモードの適用
        if apply_stealth:
            logger.info("Applying stealth mode to the context...")
            try:
                await stealth_async(context)
                logger.info("Stealth mode applied successfully.")
            except Exception as stealth_err:
                # ステルスモード適用失敗は警告に留める
                logger.warning(f"Failed to apply stealth mode: {stealth_err}", exc_info=True)
        else:
            logger.info("Stealth mode is disabled for this context.")

        # 新しいページを作成
        page = await context.new_page()
        logger.info("新しいページを作成しました。")
        return context, page
    except Exception as e:
        logger.error(f"コンテキストまたはページの作成中にエラーが発生しました: {e}", exc_info=True)
        # 作成途中のコンテキストがあれば閉じる
        if context and not context.is_closed():
             try: await context.close()
             except Exception: pass
        raise # エラーを再送出

# --- 終了関数 ---
async def close_browser(
    playwright: Optional[Playwright] = None,
    browser: Optional[Browser] = None,
    context: Optional[BrowserContext] = None
):
    """ブラウザコンテキスト、ブラウザ、Playwrightインスタンスを安全に閉じる"""
    logger.info("クリーンアップ処理を開始します...")
    # コンテキストを閉じる (存在し、まだ開いている場合)
    # is_closed() は同期メソッドなので注意 (Playwright 1.33時点)
    # -> 非同期で安全にチェックする方法がないため、try-exceptで囲むのが一般的
    if context:
        try:
            await context.close()
            logger.info("ブラウザコンテキストを閉じました。")
        except Exception as e:
             # 既に閉じられている場合などのエラーは警告レベル
             logger.warning(f"ブラウザコンテキストのクローズ中にエラー (無視): {e}")

    # ブラウザを閉じる (存在し、接続されている場合)
    if browser and browser.is_connected():
        try:
            await browser.close()
            logger.info("ブラウザを閉じました。")
        except Exception as e:
            # ブラウザクローズエラーは少し重要度が高い可能性
            logger.error(f"ブラウザのクローズ中にエラーが発生しました: {e}", exc_info=True)

    # Playwrightインスタンスを停止 (存在する場合)
    if playwright:
        try:
            await playwright.stop()
            logger.info("Playwright を停止しました。")
        except Exception as e:
            logger.error(f"Playwright の停止中にエラーが発生しました: {e}", exc_info=True)

    # イベントループのクリーンアップを促すための短い待機 (オプション)
    try:
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning(f"クリーンアップ後の待機中にエラーが発生しました: {e}")

    logger.info("--- Playwright クリーンアップ終了 ---")


# --- 以前の run_playwright_automation_async 関数 ---
# この関数はアプローチ2では直接使用しないため、削除またはコメントアウトします。
# 参考のために残しておく場合はコメントアウトしてください。
"""
async def run_playwright_automation_async(
        target_url: str,
        actions: List[Dict[str, Any]],
        headless_mode: bool = False,
        slow_motion: int = 100,
        default_timeout: int = config.DEFAULT_ACTION_TIMEOUT
    ) -> Tuple[bool, List[Dict[str, Any]]]:
    # (旧関数の実装 ... )
    pass
"""

# --- 単体テスト用コード (オプション) ---
async def _test_launcher():
    """ランチャー関数の簡単なテスト"""
    logging.basicConfig(level=logging.INFO) # テスト用にロギング設定
    pw = None; br = None; ctx = None; pg = None
    try:
        print("--- ランチャーテスト開始 ---")
        pw, br = await launch_browser(headless_mode=False, slow_motion=500)
        ctx, pg = await create_context_and_page(br)
        print(f"ページタイトル (初期): {await pg.title()}")
        await pg.goto("https://example.com")
        print(f"ページタイトル (遷移後): {await pg.title()}")
        print("--- ランチャーテスト成功 ---")
    except Exception as e:
        print(f"--- ランチャーテスト失敗 ---")
        traceback.print_exc()
    finally:
        # 必ずクリーンアップを実行
        await close_browser(pw, br, ctx)
        print("--- クリーンアップ完了 ---")

if __name__ == "__main__":
    # python playwright_launcher.py と実行するとテストが動く
    asyncio.run(_test_launcher())