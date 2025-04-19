# --- ファイル: playwright_actions.py (press_key, Tabキーサポート, select_option改善 含む完全版) ---
import asyncio
import logging
import os
import time
import pprint
import traceback
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Optional, Union, Dict, Any, Set

from playwright.async_api import (
    Page,
    Frame,
    Locator,
    FrameLocator,
    BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
    APIRequestContext,
    expect
)

# --- 必要なモジュール ---
# config と utils は playwright_actions.py 自身は直接参照しないことが多いが、
# 呼び出し元 (batch_runner など) が渡す値のデフォルト値などで間接的に関連する可能性がある
try:
    import config
except ImportError:
    # config.py がない場合のデフォルト値 (必要に応じて)
    class ConfigMock:
        DEFAULT_ACTION_TIMEOUT = 10000
        IFRAME_LOCATOR_TIMEOUT = 5000
        PDF_DOWNLOAD_TIMEOUT = 60000
        NEW_PAGE_EVENT_TIMEOUT = 4000
        DYNAMIC_SEARCH_MAX_DEPTH = 2
        DEFAULT_SCREENSHOT_DIR = 'screenshots'
        # 他にもあれば追加
    config = ConfigMock()
    logging.warning("config.py が見つかりません。デフォルト値を使用します。")

try:
    import utils
except ImportError:
    # utils がなくても動作はするが、PDF抽出などは機能しない
    utils = None
    logging.warning("utils.py が見つかりません。PDF抽出などの機能は利用できません。")

# playwright_finders と playwright_helper_funcs は必須
try:
    from playwright_finders import find_element_dynamically, find_all_elements_dynamically
    from playwright_helper_funcs import get_page_inner_text
except ImportError as e:
     logging.critical(f"playwright_actions の依存モジュールが見つかりません: {e.name}")
     raise # 起動時にエラーにする

logger = logging.getLogger(__name__)

# --- メールアドレス抽出用ヘルパー関数 ---
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
async def _extract_emails_from_page_async(context: BrowserContext, url: str, timeout: int) -> List[str]:
    """指定されたURLにアクセスし、メールアドレスを抽出する"""
    page = None; emails_found: Set[str] = set(); start_time = time.monotonic()
    # タイムアウト設定 (アクションタイムアウトの80%か15秒の大きい方)
    page_access_timeout = max(int(timeout * 0.8), 15000)
    logger.info(f"URLからメール抽出開始: {url} (タイムアウト: {page_access_timeout}ms)")
    try:
        page = await context.new_page()
        nav_timeout = max(int(page_access_timeout * 0.9), 10000) # ナビゲーションタイムアウト
        await page.goto(url, wait_until="load", timeout=nav_timeout)

        # テキスト取得タイムアウト
        remaining_time_for_text = page_access_timeout - (time.monotonic() - start_time) * 1000
        if remaining_time_for_text <= 1000: raise PlaywrightTimeoutError("Not enough time for text extraction")
        text_timeout = int(remaining_time_for_text)

        # ページ全体のテキストから抽出
        try:
            page_text = await page.locator(':root').inner_text(timeout=text_timeout)
            if page_text:
                found_in_text = EMAIL_REGEX.findall(page_text)
                if found_in_text: emails_found.update(found_in_text)
        except Exception as text_err:
             logger.warning(f"ページテキスト取得中にエラー ({url}): {text_err}")

        # mailtoリンクから抽出
        try:
            mailto_links = await page.locator("a[href^='mailto:']").all()
            if mailto_links:
                for link in mailto_links:
                    try:
                        href = await link.get_attribute('href', timeout=500) # mailto取得は短めタイムアウト
                        if href and href.startswith('mailto:'):
                            email_part = href[len('mailto:'):].split('?')[0] # ?以降のパラメータを除去
                            if email_part and EMAIL_REGEX.match(email_part): emails_found.add(email_part)
                    except Exception: pass # 個々のリンクエラーは無視
        except Exception as mailto_err:
             logger.warning(f"mailtoリンク取得中にエラー ({url}): {mailto_err}")


        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(f"メール抽出完了 ({url}). ユニーク候補: {len(emails_found)} ({elapsed:.0f}ms)")

        # (オプション) 抽出したメールをファイルに追記 (utils依存)
        if emails_found and utils and hasattr(utils, 'write_emails_to_file'): # 関数存在チェック
            unique_emails_list = list(emails_found)
            try:
                await asyncio.to_thread(utils.write_emails_to_file, unique_emails_list, "extracted_mails.txt")
            except Exception as mfwe: logger.error(f"Mail file write error: {mfwe}")

        return list(emails_found)
    except Exception as e:
        logger.warning(f"メール抽出処理全体でエラー ({url}): {type(e).__name__} - {e}", exc_info=False)
        logger.debug(f"Traceback:", exc_info=True)
        return [] # エラー時は空リスト
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception: pass


# --- Locator試行ヘルパー関数 ---
async def try_locators_sequentially(
    target_scope: Union[Page, FrameLocator],
    hints: List[Dict[str, Any]],
    action_name: str,
    overall_timeout: int
) -> Tuple[Optional[Locator], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    LLMが生成した複数のヒントを優先度順に試し、要素を見つける。
    戻り値: (見つかったLocator, 使用したヒント, 試行ログ)
    """
    start_time = time.monotonic()
    successful_locator: Optional[Locator] = None
    successful_hint: Optional[Dict[str, Any]] = None
    attempt_logs: List[Dict[str, Any]] = []
    # ヒントがない場合はすぐにリターン
    if not hints or not isinstance(hints, list):
        logger.warning(f"[{action_name}] No valid hints provided.")
        return None, None, []

    hint_count = len(hints)
    # 個々のヒント検証タイムアウト (短めにする)
    base_validation_timeout = max(500, min(2000, overall_timeout // max(1, hint_count)))
    logger.info(f"[{action_name}] Trying {hint_count} locator hints (overall timeout: {overall_timeout}ms, validation_timeout per hint: {base_validation_timeout}ms)...")

    # 確信度でソート (high -> medium -> low)
    hints.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("confidence", "low"), 2))

    for i, hint in enumerate(hints):
        elapsed_ms = (time.monotonic() - start_time) * 1000
        if elapsed_ms >= overall_timeout:
            logger.warning(f"[{action_name}] Overall timeout reached during hint processing ({elapsed_ms:.0f}ms).")
            break
        remaining_ms = overall_timeout - elapsed_ms
        # 検証タイムアウトは残り時間と基本値の小さい方、最低100msは確保
        current_validation_timeout = min(base_validation_timeout, max(100, int(remaining_ms * 0.8)))

        hint_type = hint.get("type")
        hint_value = hint.get("value")
        hint_name = hint.get("name") # role_and_text 用
        hint_level = hint.get("level") # role=heading 用
        hint_common_selector = hint.get("common_selector") # nth_child 用
        hint_index = hint.get("index") # nth_child 用
        hint_confidence = hint.get("confidence", "low")

        locator_description = f"Hint {i+1}/{hint_count}: type={hint_type}, confidence={hint_confidence}"
        value_str = str(hint_value)
        value_log = f"{value_str[:50]}{'...' if len(value_str) > 50 else ''}"
        logging.info(f"  Trying: {locator_description} (value: {value_log})")

        attempt_log = {"hint": hint, "status": "Skipped", "reason": "No locator generated"}
        locator_to_try: Optional[Locator] = None

        try:
            options = {}
            # --- 各ヒントタイプに基づいてLocatorを生成 ---
            if hint_type == "role_and_text":
                 actual_role = hint_value if isinstance(hint_value, str) else hint.get("role") # roleはvalueかroleキー
                 if not isinstance(actual_role, str) or not actual_role:
                     raise ValueError("Role missing/invalid in 'role_and_text' hint")
                 if isinstance(hint_name, str):
                     # 正規表現サポート (簡易)
                     if hint_name.startswith('/') and hint_name.endswith('/') and len(hint_name) > 2:
                          try:
                              pattern = re.compile(hint_name[1:-1]) # フラグなし
                              options["name"] = pattern
                          except re.error as re_err:
                              options["name"] = hint_name # エラー時は完全一致として扱う
                              logging.warning(f"Invalid regex in hint, using as exact text: {re_err}")
                     else:
                         options["name"] = hint_name
                 if actual_role == "heading" and isinstance(hint_level, int):
                     options["level"] = hint_level
                 locator_to_try = target_scope.get_by_role(actual_role, **options)
            elif hint_type == "test_id" and isinstance(hint_value, str):
                locator_to_try = target_scope.get_by_test_id(hint_value)
            elif hint_type == "text_exact" and isinstance(hint_value, str):
                locator_to_try = target_scope.get_by_text(hint_value, exact=True)
            elif hint_type == "placeholder" and isinstance(hint_value, str):
                locator_to_try = target_scope.get_by_placeholder(hint_value)
            elif hint_type == "aria_label" and isinstance(hint_value, str):
                locator_to_try = target_scope.get_by_label(hint_value, exact=True) # get_by_label が推奨
            elif hint_type == "css_selector_candidate" and isinstance(hint_value, str):
                locator_to_try = target_scope.locator(hint_value)
            elif hint_type == "nth_child" and isinstance(hint_common_selector, str) and isinstance(hint_index, int) and hint_index >= 0:
                locator_to_try = target_scope.locator(hint_common_selector).nth(hint_index)
            else:
                # 未知または無効なヒントタイプ
                attempt_log["reason"] = f"Invalid/unknown hint type or value: {hint}"
                logging.warning(f"    -> {attempt_log['reason']}")

            # --- Locatorが生成されたか確認し、有効性をチェック ---
            if locator_to_try:
                attempt_log["reason"] = "Checking validity..."
                try:
                    # 実際に試行したセレクター表現を記録 (エラーになる場合もあるのでtry-except)
                    attempt_log["final_selector_attempted"] = repr(locator_to_try)
                except Exception:
                    attempt_log["final_selector_attempted"] = "[repr failed]"

                # 要素数をカウント
                count = await locator_to_try.count()
                attempt_log["count"] = count
                logging.debug(f"    -> Count: {count}")

                if count == 1: # 要素が1つだけ見つかった場合
                    is_visible = False
                    try:
                        # 要素が表示状態になるまで待機 (短いタイムアウトで)
                        await locator_to_try.wait_for(state='visible', timeout=current_validation_timeout)
                        is_visible = True
                        logging.debug(f"    -> Visible.")
                    except PlaywrightTimeoutError:
                        logging.debug(f"    -> Not visible within timeout.")
                    except Exception as vis_err:
                        # 他のエラー（要素がデタッチされたなど）も考慮
                        logging.warning(f"    -> Visibility check error: {vis_err}")

                    attempt_log["is_visible"] = is_visible
                    if is_visible: # 見えていれば成功
                        successful_locator = locator_to_try
                        successful_hint = hint
                        attempt_log["status"] = "Success"
                        attempt_log["reason"] = "Found unique and visible element."
                        logging.info(f"    -> Success!")
                        attempt_logs.append(attempt_log)
                        break # 成功したのでループを抜ける
                    else: # 見つかったが見えていない
                        attempt_log["status"] = "Fail"
                        attempt_log["reason"] = "Element found but not visible."
                        logging.info(f"    -> Fail: Not visible.")
                elif count > 1: # 複数見つかった場合
                    attempt_log["status"] = "Fail"
                    attempt_log["reason"] = f"Multiple elements found ({count}). Hint is not specific enough."
                    logging.info(f"    -> Fail: Multiple elements.")
                else: # count == 0、要素が見つからない場合
                    attempt_log["status"] = "Fail"
                    attempt_log["reason"] = "Element not found."
                    logging.info(f"    -> Fail: Not found.")
            else: # locator_to_try が None の場合 (ヒントが無効だった場合)
                attempt_log["status"] = "Fail"
                attempt_log["reason"] = "Could not generate locator from hint."

        except ValueError as ve: # ロール指定ミスなど
            attempt_log["status"] = "Error"
            attempt_log["reason"] = f"ValueError during locator generation: {ve}"
            logging.warning(f"    -> ValueError: {ve}")
        except PlaywrightError as pe: # Playwright固有のエラー
            attempt_log["status"] = "Error"
            attempt_log["reason"] = f"PlaywrightError during locator check: {pe}"
            logging.warning(f"    -> PlaywrightError: {pe}")
        except Exception as e: # その他の予期せぬエラー
            attempt_log["status"] = "Error"
            attempt_log["reason"] = f"Unexpected Error: {e}"
            logging.error(f"    -> Unexpected error during hint processing: {e}", exc_info=True)

        attempt_logs.append(attempt_log) # 各試行のログを追加

    if successful_locator:
        logger.info(f"[{action_name}] Located element successfully using hint: {successful_hint}")
    else:
        logger.warning(f"[{action_name}] Failed to locate element using any of the provided hints.")

    return successful_locator, successful_hint, attempt_logs


# --- Playwright アクション実行コア ---
async def execute_actions_async(
    page: Page,
    actions: List[Dict[str, Any]],
    api_request_context: APIRequestContext,
    default_timeout: int
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Playwright アクションを非同期で実行し、結果と最終URLを返す。
    iframe探索やLLMヒントを動的に利用する。
    """
    results: List[Dict[str, Any]] = []
    # 操作対象のスコープ (ページまたはフレーム)
    current_target: Union[Page, FrameLocator] = page
    root_page: Page = page # メインのページオブジェクトは保持
    current_context: BrowserContext = root_page.context
    iframe_stack: List[Union[Page, FrameLocator]] = [] # iframe移動履歴
    final_url: str = page.url # 最後に確認されたURL

    for i, step_data in enumerate(actions):
        step_num = i + 1
        # アクション情報を取得 (小文字に変換)
        action = step_data.get("action", "").lower()
        # --- 各パラメータを取得 ---
        selector = step_data.get("selector") # フォールバック用CSSセレクター
        target_hints = step_data.get("target_hints") # LLMからのヒントリスト
        iframe_selector_input = step_data.get("iframe_selector") # iframe切り替え用
        value = step_data.get("value") # input, sleep, press_key など
        attribute_name = step_data.get("attribute_name") # get_attribute, get_all_attributes 用
        option_type = step_data.get("option_type") # select_option 用
        option_value = step_data.get("option_value") # select_option 用
        key_to_press = step_data.get("value") if action == "press_key" else None # press_key 用キー名
        press_count = int(step_data.get("count", 1)) if action == "press_key" else 1 # press_key 用回数
        # アクション固有タイムアウト > デフォルトタイムアウト
        action_wait_time = step_data.get("wait_time_ms", default_timeout)

        # --- ログ出力 ---
        logger.info(f"--- ステップ {step_num}/{len(actions)}: Action='{action}' ---")
        step_info = { # ログ表示用の情報をまとめる
            "selector": selector,
            "target_hints_count": len(target_hints) if isinstance(target_hints, list) else None,
            "iframe(指定)": iframe_selector_input,
            "value": value,
            "attribute_name": attribute_name,
            "option_type": option_type,
            "option_value": option_value,
        }
        if action == "press_key": # press_key 固有情報
             step_info["key_to_press"] = key_to_press
             step_info["press_count"] = press_count
        # None の値を除外し、長い文字列は省略して表示
        step_info_str = ", ".join([
            f"{k}='{str(v)[:50]}...'" if isinstance(v, str) and len(v) > 50 else f"{k}='{v}'"
            for k, v in step_info.items() if v is not None
        ])
        logger.info(f"詳細: {step_info_str} (timeout: {action_wait_time}ms)")
        if isinstance(target_hints, list): # ヒントが多い場合はデバッグログに詳細表示
            logger.debug(f"Target Hints:\n{pprint.pformat(target_hints)}")

        # --- 結果記録用のベース辞書 ---
        step_result_base = {"step": step_num, "action": action}
        if step_data.get("memo"): # メモがあれば記録
            step_result_base["memo"] = step_data["memo"]

        # --- メインの処理ブロック ---
        try:
            # ルートページが閉じられていないか確認
            if root_page.is_closed():
                raise PlaywrightError(f"Root page closed unexpectedly before step {step_num}.")

            # 現在のURLを更新 (ページ遷移に備える)
            final_url = root_page.url
            current_base_url = final_url # 相対URL解決用

            # --- Iframe/Parent Frame 切替 ---
            if action == "switch_to_iframe":
                if not iframe_selector_input:
                    raise ValueError("Action 'switch_to_iframe' requires 'iframe_selector'.")
                logger.info(f"Switching to iframe using selector: '{iframe_selector_input}'")
                # 現在のスコープから指定のiframeを探す
                target_frame_locator = current_target.frame_locator(iframe_selector_input)
                try:
                    # iframeが実際に存在し、アタッチされるまで待機
                    await target_frame_locator.locator(':root').wait_for(state='attached', timeout=action_wait_time)
                except PlaywrightTimeoutError as e:
                    raise PlaywrightTimeoutError(f"Iframe '{iframe_selector_input}' not found or timed out ({action_wait_time}ms).") from e

                # 現在のスコープをスタックに保存 (後で戻れるように)
                # 同じスコープを何度も積まないようにチェック
                if id(current_target) not in [id(s) for s in iframe_stack]:
                    iframe_stack.append(current_target)
                # 操作対象を新しいフレームに更新
                current_target = target_frame_locator
                logger.info(f"Successfully switched to FrameLocator: {iframe_selector_input}")
                results.append({**step_result_base, "status": "success", "selector": iframe_selector_input})
                final_url = root_page.url # URLは通常変わらないはずだが念のため更新
                continue # 次のステップへ

            elif action == "switch_to_parent_frame":
                status="warning"; message=None
                if not iframe_stack: # スタックが空 = トップレベル
                    logger.warning("Already at top-level frame or iframe stack is empty.")
                    message="Already at top-level or stack empty."
                    # 念のため、current_targetがFrameLocatorならルートページに戻す
                    if isinstance(current_target, FrameLocator):
                         current_target = root_page
                         logger.info("Current target was FrameLocator, reset to root Page.")
                else: # スタックに親があれば戻る
                    parent_target = iframe_stack.pop()
                    current_target = parent_target
                    logger.info(f"Switched back to parent target: {type(current_target).__name__}")
                    status="success"
                results.append({**step_result_base, "status": status, "message": message})
                final_url = root_page.url # URL更新
                continue # 次のステップへ

            # --- ページ全体操作 ---
            if action in ["wait_page_load", "sleep", "scroll_page_to_bottom"]:
                 if action == "wait_page_load":
                     logger.info(f"Waiting for page load state 'load' (timeout: {action_wait_time}ms)...")
                     await root_page.wait_for_load_state("load", timeout=action_wait_time)
                     logger.info("Page load complete.")
                 elif action == "sleep":
                     try:
                         seconds = float(value) if value is not None else 1.0 # デフォルト1秒
                         if seconds < 0: raise ValueError("Sleep duration cannot be negative.")
                     except (TypeError, ValueError): raise ValueError("Invalid 'value' for sleep action. Must be a non-negative number (seconds).")
                     logger.info(f"Sleeping for {seconds:.2f} seconds...")
                     await asyncio.sleep(seconds)
                     results.append({**step_result_base, "status": "success", "duration_sec": seconds})
                     final_url = root_page.url # sleep後もURL更新
                     continue # 次のステップへ
                 elif action == "scroll_page_to_bottom":
                     logger.info("Scrolling page to bottom...")
                     await root_page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                     await asyncio.sleep(0.5) # スクロール後のレンダリング待ち
                     logger.info("Scrolled to page bottom.")
                 results.append({**step_result_base, "status": "success"})
                 final_url = root_page.url # スクロール後もURL更新
                 continue # 次のステップへ

            # --- 要素操作準備と要素特定ロジック ---
            target_element: Optional[Locator] = None
            found_elements_list: List[Tuple[Locator, Union[Page, FrameLocator]]] = []
            found_scope: Optional[Union[Page, FrameLocator]] = None
            successful_hint_info: Optional[Dict[str, Any]] = None
            locator_attempt_logs: List[Dict[str, Any]] = []

            # アクションタイプに応じて必要な要素数を判断
            single_element_actions = ["click", "input", "hover", "get_inner_text", "get_text_content", "get_inner_html", "get_attribute", "wait_visible", "select_option", "scroll_to_element", "press_key"]
            multiple_elements_actions = ["get_all_attributes", "get_all_text_contents"]
            is_screenshot_action = action == "screenshot"
            needs_single_element = action in single_element_actions or (is_screenshot_action and (target_hints or selector))
            needs_multiple_elements = action in multiple_elements_actions

            # --- 単一要素の特定 ---
            if needs_single_element:
                element_located_by = None
                # press_key は要素がなくても実行できる場合がある (ページ全体へのキー入力)
                allow_no_element = action == "press_key"

                # 1. target_hints を試す
                if target_hints and isinstance(target_hints, list) and target_hints:
                    logger.info(f"Locating element based on {len(target_hints)} target_hints...")
                    target_element, successful_hint_info, locator_attempt_logs = await try_locators_sequentially(
                        current_target, target_hints, action, action_wait_time
                    )
                    step_result_base["locator_attempts"] = locator_attempt_logs # 試行ログを結果に含める
                    if target_element:
                        logger.info(f"Element located using target_hints: {successful_hint_info}")
                        step_result_base["locator_hint_used"] = successful_hint_info
                        found_scope = current_target # ヒントで見つかった場合は現在のスコープ
                        element_located_by = 'hints'
                    else:
                        logger.warning("Failed to locate element using target_hints.")

                # 2. ヒントで失敗 or ヒントなし and セレクターがある場合、フォールバック探索
                if not target_element and selector:
                    fallback_reason = "target_hints not provided" if not target_hints else "target_hints failed"
                    logger.warning(f"{fallback_reason}, falling back to dynamic search using selector: '{selector}'")
                    # アクションに応じて必要な要素の状態を決定
                    # press_key は可視の方が安全だが、必須ではないかもしれない
                    required_state = 'visible' if action not in ['get_attribute', 'get_all_attributes', 'get_all_text_contents'] else 'attached'
                    target_element, found_scope = await find_element_dynamically(
                        current_target, selector,
                        max_depth=config.DYNAMIC_SEARCH_MAX_DEPTH,
                        timeout=action_wait_time,
                        target_state=required_state
                    )
                    if target_element and found_scope:
                        logger.info(f"Element located using fallback selector '{selector}'.")
                        step_result_base["selector_used_as_fallback"] = selector
                        element_located_by = 'fallback_selector'
                    else:
                        logger.warning(f"Failed to locate element using fallback selector '{selector}' as well.")

                # 3. 最終的な要素特定成否チェック (press_key 以外)
                if not target_element and not allow_no_element:
                    # どちらの方法でも見つからなかった場合のエラー
                    if not target_hints and not selector:
                        # そもそも特定情報がない場合
                         raise ValueError(f"Action '{action}' requires 'target_hints' or 'selector' to locate the element.")
                    # 情報はあったが見つからなかった場合
                    error_msg = f"Failed to locate required single element for action '{action}'."
                    tried_hints = step_result_base.get("locator_attempts") is not None
                    tried_fallback = step_result_base.get("selector_used_as_fallback") is not None
                    original_selector = step_data.get("selector")
                    if tried_hints and tried_fallback: error_msg += f" Tried target_hints and fallback selector '{step_result_base['selector_used_as_fallback']}'."
                    elif tried_hints and not original_selector: error_msg += " Tried target_hints, but no fallback selector was provided."
                    elif tried_hints: error_msg += f" Tried target_hints and fallback selector '{original_selector}' (failed)."
                    elif original_selector: error_msg += f" Tried selector '{original_selector}'."
                    raise PlaywrightError(error_msg)
                elif not target_element and allow_no_element:
                     logger.info(f"Action '{action}' allows execution without a specific target element.")

            # --- 複数要素の特定 ---
            elif needs_multiple_elements:
                if not selector:
                    raise ValueError(f"Action '{action}' requires a 'selector' to find multiple elements.")
                logger.info(f"Locating multiple elements using selector '{selector}' with dynamic search...")
                # find_all_elements_dynamically は見つからなくてもエラーにはならない (空リストを返す)
                found_elements_list = await find_all_elements_dynamically(
                    current_target, selector,
                    max_depth=config.DYNAMIC_SEARCH_MAX_DEPTH,
                    timeout=action_wait_time
                )
                if not found_elements_list:
                    logger.warning(f"No elements found matching selector '{selector}' within dynamic search.")
                step_result_base["selector"] = selector # 使用したセレクターを記録

            # --- スコープ更新処理 ---
            # 単一要素が見つかり、それが現在のスコープと異なるiframe内だった場合
            if needs_single_element and found_scope and id(found_scope) != id(current_target):
                logger.info(f"Updating operating scope from {type(current_target).__name__} to {type(found_scope).__name__}")
                if id(current_target) not in [id(s) for s in iframe_stack]:
                    iframe_stack.append(current_target) # 元のスコープをスタックへ
                current_target = found_scope # 新しいスコープに切り替え
            # 複数要素の場合は current_target は変更しない（個々の要素のスコープは found_elements_list 内に保持）
            logger.info(f"Final operating scope for this step: {type(current_target).__name__}")

            # --- 各アクション実行 ---
            action_result_details = {} # このステップの詳細結果用
            # どの方法で要素が見つかったかを記録
            if step_result_base.get("locator_hint_used"):
                action_result_details["locator_hint_used"] = step_result_base["locator_hint_used"]
                action_result_details["locator_method"] = "hints"
            elif step_result_base.get("selector_used_as_fallback"):
                action_result_details["selector"] = step_result_base["selector_used_as_fallback"]
                action_result_details["locator_method"] = "fallback_selector"
            elif step_result_base.get("selector"): # 複数要素の場合など
                action_result_details["selector"] = step_result_base["selector"]
                action_result_details["locator_method"] = "selector"

            # --- click ---
            if action == "click":
                if not target_element: raise PlaywrightError("Click failed: Target element was not located.")
                logger.info("Clicking element...")
                context_for_click = root_page.context # 新しいページが開くか監視
                new_page: Optional[Page] = None
                try:
                    # 指定時間内に新しいページが開くのを待機しながらクリック
                    async with context_for_click.expect_page(timeout=config.NEW_PAGE_EVENT_TIMEOUT) as new_page_info:
                        await target_element.click(timeout=action_wait_time)
                    # 新しいページが開いた場合
                    new_page = await new_page_info.value
                    new_page_url = new_page.url
                    logger.info(f"New page opened: {new_page_url}")
                    try: # 新しいページのロード完了も待つ
                        await new_page.wait_for_load_state("load", timeout=action_wait_time)
                    except PlaywrightTimeoutError: logger.warning(f"New page load timed out.")
                    # 操作対象を新しいページに切り替える
                    root_page, current_target, current_context, iframe_stack = new_page, new_page, new_page.context, []
                    action_result_details.update({"new_page_opened": True, "new_page_url": new_page_url})
                except PlaywrightTimeoutError:
                    # 新しいページが開かなかった場合
                    logger.info(f"Click completed, but no new page opened within {config.NEW_PAGE_EVENT_TIMEOUT}ms.")
                    action_result_details["new_page_opened"] = False
                except Exception as click_err:
                    logger.error(f"Click action failed: {click_err}", exc_info=True)
                    raise # エラーを再送出
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- input ---
            elif action == "input":
                 if not target_element: raise PlaywrightError("Input failed: Target element was not located.")
                 if value is None: raise ValueError("Action 'input' requires a 'value'.")
                 input_value_str = str(value)
                 logger.info(f"Inputting text into element: '{input_value_str[:50]}{'...' if len(input_value_str) > 50 else ''}'")
                 try:
                     # 入力前に要素が表示されているか確認し、クリックしてフォーカスを確実に当てる
                     await target_element.wait_for(state='visible', timeout=max(1000, action_wait_time // 3))
                     logger.debug("Clicking element to ensure focus before input.")
                     await target_element.click(timeout=max(1000, action_wait_time // 3))
                     # fill() で入力実行
                     await target_element.fill(input_value_str, timeout=action_wait_time)
                     await asyncio.sleep(0.1) # 入力後の短い待機
                     logger.info("Input successful.")
                     action_result_details["value"] = value # 結果には元の値を記録
                     results.append({**step_result_base, "status": "success", **action_result_details})
                 except Exception as e:
                     logger.error(f"Input action failed: {e}", exc_info=True)
                     raise PlaywrightError(f"Failed to input value: {e}") from e

            # --- hover ---
            elif action == "hover":
                 if not target_element: raise PlaywrightError("Hover failed: Target element was not located.")
                 logger.info("Hovering over element...")
                 await target_element.hover(timeout=action_wait_time)
                 logger.info("Hover successful.")
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_inner_text ---
            elif action == "get_inner_text":
                 if not target_element: raise PlaywrightError("Get innerText failed: Target element was not located.")
                 logger.info("Getting innerText from element...")
                 text = await target_element.inner_text(timeout=action_wait_time)
                 text = text.strip() if text else ""
                 logger.info(f"Got innerText: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                 action_result_details["text"] = text
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_text_content ---
            elif action == "get_text_content":
                 if not target_element: raise PlaywrightError("Get textContent failed: Target element was not located.")
                 logger.info("Getting textContent from element...")
                 text = await target_element.text_content(timeout=action_wait_time)
                 text = text.strip() if text else "" # 前後の空白を除去
                 logger.info(f"Got textContent: '{text[:100]}{'...' if len(text) > 100 else ''}'")
                 action_result_details["text"] = text
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_inner_html ---
            elif action == "get_inner_html":
                 if not target_element: raise PlaywrightError("Get innerHTML failed: Target element was not located.")
                 logger.info("Getting innerHTML from element...")
                 html_content = await target_element.inner_html(timeout=action_wait_time)
                 logger.info(f"Got innerHTML (first 500 chars): {html_content[:500]}...")
                 action_result_details["html"] = html_content
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_attribute ---
            elif action == "get_attribute":
                if not target_element: raise PlaywrightError("Get attribute failed: Target element was not located.")
                if not attribute_name: raise ValueError("Action 'get_attribute' requires 'attribute_name'.")
                logger.info(f"Getting attribute '{attribute_name}' from element...")
                attr_value = await target_element.get_attribute(attribute_name, timeout=action_wait_time)
                pdf_text_content = None
                processed_value = attr_value # 変換後の値用
                scraped_content = None

                # --- 特殊な属性名の処理 ---
                if attribute_name.lower() == 'href' and attr_value is not None:
                     try:
                         absolute_url = urljoin(current_base_url, attr_value) # 絶対URLに変換
                         processed_value = absolute_url
                     except Exception as url_e:
                         logger.error(f"Failed to resolve href URL '{attr_value}' relative to '{current_base_url}': {url_e}")
                         processed_value = f"Error resolving URL: {url_e}"
                elif attribute_name.lower() == 'pdf' and attr_value is not None:
                     try:
                         absolute_url = urljoin(current_base_url, attr_value)
                         processed_value = absolute_url
                         if absolute_url.lower().endswith('.pdf'):
                             if not utils: raise RuntimeError("utils module (for PDF processing) is not available.")
                             pdf_bytes = await utils.download_pdf_async(api_request_context, absolute_url)
                             if pdf_bytes:
                                 # PDF抽出は同期処理なので別スレッドで実行
                                 pdf_text_content = await asyncio.to_thread(utils.extract_text_from_pdf_sync, pdf_bytes)
                                 if pdf_text_content is None: pdf_text_content = "Error: PDF text extraction failed (returned None)."
                             else: pdf_text_content = "Error: PDF download failed (returned no data)."
                         else: pdf_text_content = "Warning: Link does not end with .pdf"
                     except Exception as pdf_e:
                         logger.error(f"PDF processing error for URL '{attr_value}': {pdf_e}", exc_info=True)
                         pdf_text_content = f"Error processing PDF: {pdf_e}"
                elif attribute_name.lower() == 'content' and attr_value is not None:
                     try:
                         absolute_url = urljoin(current_base_url, attr_value)
                         processed_value = absolute_url
                         parsed_url = urlparse(absolute_url)
                         # HTTP/HTTPS かつ PDF でない場合にコンテンツ取得
                         if parsed_url.scheme in ['http', 'https'] and not absolute_url.lower().endswith('.pdf'):
                              success_content, content_or_error = await get_page_inner_text(current_context, absolute_url, action_wait_time)
                              scraped_content = content_or_error # 成功時はテキスト、失敗時はエラーメッセージ
                         elif absolute_url.lower().endswith('.pdf'):
                             scraped_content = "Warning: Link points to a PDF. Use attribute_name='pdf' to extract text."
                         else:
                             scraped_content = f"Warning: Cannot fetch content. Invalid URL scheme or not HTTP/S: {absolute_url}"
                     except Exception as content_e:
                         logger.error(f"Content scraping error for URL '{attr_value}': {content_e}", exc_info=True)
                         scraped_content = f"Error scraping content: {content_e}"

                logger.info(f"Got attribute '{attribute_name}': '{str(processed_value)[:100]}{'...' if isinstance(processed_value, str) and len(processed_value) > 100 else ''}'")
                action_result_details.update({"attribute": attribute_name, "value": processed_value})
                if pdf_text_content is not None:
                    action_result_details["pdf_text"] = pdf_text_content
                if scraped_content is not None:
                    action_result_details["scraped_text"] = scraped_content # スクレイピング結果も追加
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_all_attributes ---
            elif action == "get_all_attributes":
                if not selector: raise ValueError("Action 'get_all_attributes' requires 'selector'.")
                if not attribute_name: raise ValueError("Action 'get_all_attributes' requires 'attribute_name'.")

                if not found_elements_list:
                    logger.warning(f"No elements found for selector '{selector}'. Skipping attribute extraction.")
                    action_result_details["results_count"] = 0
                else:
                    num_found = len(found_elements_list)
                    logger.info(f"Getting attribute '{attribute_name}' from {num_found} elements found by '{selector}'...")

                    # --- 結果リスト初期化 ---
                    url_list, pdf_list, content_list, mail_list, generic_list = [],[],[],[],[]
                    processed_domains = set() # mail モード用

                    # --- href, pdf, content, mail の特殊処理 ---
                    if attribute_name.lower() in ['href', 'pdf', 'content', 'mail']:
                        CONCURRENT_LIMIT=5 # 同時処理数制限
                        semaphore=asyncio.Semaphore(CONCURRENT_LIMIT)

                        # 単一要素の処理関数 (非同期)
                        async def process_single_element_for_href_related(
                            locator: Locator, index: int, base_url: str, attr_mode: str, sem: asyncio.Semaphore
                        ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
                            """hrefを取得し、モードに応じてPDF/コンテンツ/メールを処理"""
                            original_href: Optional[str] = None
                            absolute_url: Optional[str] = None
                            pdf_text: Optional[str] = None
                            scraped_text: Optional[str] = None
                            emails_from_page: Optional[List[str]] = None
                            async with sem: # セマフォで同時実行数を制御
                                try:
                                    # href属性取得 (個々の要素のタイムアウトは短めに)
                                    href_timeout = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                                    original_href = await locator.get_attribute("href", timeout=href_timeout)
                                    if original_href is None: # hrefがなければ処理終了
                                        return None, None, None, None

                                    # 絶対URLに変換
                                    try:
                                        absolute_url = urljoin(base_url, original_href)
                                        parsed_url = urlparse(absolute_url)
                                    except Exception:
                                        return f"Error converting URL: {original_href}", None, None, None

                                    # 有効なHTTP/S URLかチェック
                                    if parsed_url.scheme not in ['http', 'https']:
                                        return absolute_url, None, None, None # スキームが無効でもURL自体は返す

                                    # --- モードに応じた処理 ---
                                    if attr_mode == 'pdf' and absolute_url.lower().endswith('.pdf'):
                                        if not utils: raise RuntimeError("utils module (for PDF processing) is not available.")
                                        pdf_bytes = await utils.download_pdf_async(api_request_context, absolute_url)
                                        if pdf_bytes:
                                            pdf_text = await asyncio.to_thread(utils.extract_text_from_pdf_sync, pdf_bytes)
                                            if pdf_text is None: pdf_text = "Error: PDF text extraction failed."
                                        else: pdf_text = "Error: PDF download failed."
                                    elif attr_mode == 'content' and not absolute_url.lower().endswith('.pdf'):
                                        success_content, content_or_error = await get_page_inner_text(current_context, absolute_url, action_wait_time)
                                        scraped_text = content_or_error
                                    elif attr_mode == 'mail':
                                        emails_from_page = await _extract_emails_from_page_async(current_context, absolute_url, action_wait_time)

                                    return absolute_url, pdf_text, scraped_text, emails_from_page

                                except (PlaywrightError, Exception) as e:
                                    # 個々の要素処理エラー
                                    error_msg = f"Error processing element {index}: {type(e).__name__}"
                                    logger.warning(error_msg, exc_info=False)
                                    # エラーが発生しても、取得できたURLは返し、関連テキストはエラーメッセージとする
                                    err_detail = f"Error: {type(e).__name__}"
                                    return absolute_url or f"Error getting href for element {index}", \
                                           err_detail if attr_mode=='pdf' else None, \
                                           err_detail if attr_mode=='content' else None, \
                                           [err_detail] if attr_mode=='mail' else None
                        # --- 非同期タスク作成と実行 ---
                        tasks = [
                            process_single_element_for_href_related(loc, idx, current_base_url, attribute_name.lower(), semaphore)
                            for idx, (loc, _) in enumerate(found_elements_list)
                        ]
                        results_tuples = await asyncio.gather(*tasks)

                        # --- 結果の集約 ---
                        flat_mails=[] # mailモード用の一時リスト
                        for url_res, pdf_res, content_res, mails_res in results_tuples:
                            url_list.append(url_res) # URLは常に記録
                            if attribute_name.lower() == 'pdf': pdf_list.append(pdf_res)
                            if attribute_name.lower() == 'content': content_list.append(content_res)
                            if attribute_name.lower() == 'mail' and mails_res: flat_mails.extend(mails_res) # 全メールを一時リストへ

                        # mail モードの場合、ドメインユニーク化
                        if attribute_name.lower() == 'mail':
                             unique_emails_this_step = []
                             for email in flat_mails:
                                 if isinstance(email, str) and '@' in email:
                                     try:
                                         domain = email.split('@',1)[1].lower()
                                         if domain and domain not in processed_domains:
                                             unique_emails_this_step.append(email)
                                             processed_domains.add(domain)
                                     except IndexError: continue # @マーク以降がない不正な形式
                             mail_list = unique_emails_this_step # ユニーク化後のリストを格納

                    # --- 通常の属性取得 ---
                    else:
                        async def get_single_attr(locator: Locator, attr_name: str, index: int, timeout_ms: int) -> Optional[str]:
                            """単一要素から属性値を取得"""
                            try:
                                return await locator.get_attribute(attr_name, timeout=timeout_ms)
                            except Exception as e:
                                logger.warning(f"Error getting attribute '{attr_name}' for element {index}: {e}")
                                return f"Error: {type(e).__name__}" # エラー時はエラーメッセージを返す

                        timeout_per_element = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                        tasks = [get_single_attr(loc, attribute_name, idx, timeout_per_element) for idx, (loc, _) in enumerate(found_elements_list)]
                        generic_list = await asyncio.gather(*tasks)

                    # --- 結果辞書への格納 ---
                    action_result_details["results_count"] = len(url_list) if attribute_name.lower() in ['href', 'pdf', 'content', 'mail'] else len(generic_list)
                    action_result_details["attribute"] = attribute_name # 取得した属性名を記録
                    if attribute_name.lower() in ['href', 'pdf', 'content', 'mail']: action_result_details["url_list"] = url_list # URLリストは常に追加
                    if attribute_name.lower() == 'pdf': action_result_details["pdf_texts"] = pdf_list
                    if attribute_name.lower() == 'content': action_result_details["scraped_texts"] = content_list
                    if attribute_name.lower() == 'mail': action_result_details["extracted_emails"] = mail_list
                    if generic_list: action_result_details["attribute_list"] = generic_list

                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_all_text_contents ---
            elif action == "get_all_text_contents":
                if not selector: raise ValueError("Action 'get_all_text_contents' requires 'selector'.")
                text_list: List[Optional[str]] = []
                if not found_elements_list:
                    logger.warning(f"No elements found for selector '{selector}'. Cannot get text contents.")
                    action_result_details["results_count"] = 0
                else:
                    num_found = len(found_elements_list)
                    logger.info(f"Getting textContent from {num_found} elements found by '{selector}'...")
                    async def get_single_text(locator: Locator, index: int, timeout_ms: int) -> Optional[str]:
                        """単一要素からtextContentを取得"""
                        try:
                            text = await locator.text_content(timeout=timeout_ms)
                            return text.strip() if text else "" # 前後空白除去
                        except Exception as e:
                            logger.warning(f"Error getting textContent for element {index}: {e}")
                            return f"Error: {type(e).__name__}"

                    timeout_per_element = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                    tasks = [get_single_text(loc, idx, timeout_per_element) for idx, (loc, _) in enumerate(found_elements_list)]
                    text_list = await asyncio.gather(*tasks)
                    action_result_details["results_count"] = len(text_list)

                action_result_details["text_list"] = text_list
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- wait_visible ---
            elif action == "wait_visible":
                if not target_element: raise PlaywrightError("Wait visible failed: Target element was not located.")
                # 要素特定時に wait_for('visible') しているので、ここでは確認ログのみ
                logger.info("Element visibility was confirmed during location.")
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- select_option (クリックしてから実行する改善版) ---
            elif action == "select_option":
                  if not target_element: raise PlaywrightError("Select option failed: Target element was not located.")
                  if option_type not in ['value', 'index', 'label'] or option_value is None:
                      raise ValueError("Invalid 'option_type' or 'option_value' for select_option action.")

                  logger.info(f"Selecting option (Type: {option_type}, Value: '{option_value}')..."); value_to_select: Union[str, Dict]
                  if option_type == 'value': value_to_select = {'value': str(option_value)}
                  elif option_type == 'index':
                      try: index_val = int(option_value)
                      except (ValueError, TypeError): raise ValueError("Option type 'index' requires an integer value.")
                      value_to_select = {'index': index_val}
                  else: value_to_select = {'label': str(option_value)}

                  try:
                      # 1. 要素が表示され有効になるまで待機
                      logger.info(f"Waiting for select element '{action_result_details.get('selector', 'N/A')}' to be visible and enabled...")
                      await target_element.wait_for(state='visible', timeout=max(1000, action_wait_time // 4))
                      await target_element.is_enabled(timeout=max(1000, action_wait_time // 4))
                      logger.info("Select element appears ready.")
                      # 2. 要素をクリック (フォーカスとドロップダウン表示を期待)
                      logger.info("Clicking the select element first...")
                      await target_element.click(timeout=max(1000, action_wait_time // 4))
                      await asyncio.sleep(0.3) # ドロップダウンが開くのを少し待つ
                      # 3. select_option を実行
                      logger.info("Attempting select_option after click...")
                      selected_values = await target_element.select_option(value_to_select, timeout=max(5000, int(action_wait_time * 0.7)))
                  except Exception as e:
                       logger.error(f"Select option failed: {e}", exc_info=True)
                       # エラー発生時、より詳細なエラーメッセージを試みる
                       try:
                           options_available = await target_element.evaluate("el => Array.from(el.options).map(opt => ({value: opt.value, text: opt.innerText, index: opt.index}))")
                           logger.info(f"Available options: {options_available}")
                       except Exception as eval_e:
                           logger.warning(f"Could not retrieve available options: {eval_e}")
                       raise # 元のエラーを再送出

                  logger.info(f"Select option success. Selected: {selected_values}")
                  action_result_details.update({"option_type": option_type, "option_value": option_value, "selected_actual_values": selected_values})
                  results.append({**step_result_base, "status": "success", **action_result_details})

            # --- scroll_to_element ---
            elif action == "scroll_to_element":
                   if not target_element: raise PlaywrightError("Scroll to element failed: Target element was not located.")
                   logger.info("Scrolling element into view if needed...")
                   await target_element.scroll_into_view_if_needed(timeout=action_wait_time)
                   await asyncio.sleep(0.3) # スクロール後の安定待ち
                   logger.info("Scroll successful.")
                   results.append({**step_result_base, "status": "success", **action_result_details})

            # --- press_key ---
            elif action == "press_key":
                if not key_to_press or not isinstance(key_to_press, str):
                    raise ValueError("Action 'press_key' requires a 'value' specifying the key name (e.g., 'Enter', 'ArrowDown', 'Tab').")

                # Playwrightがサポートするキー名のリスト (必要に応じて拡張)
                valid_keys = [
                    "Enter", "ArrowUp", "ArrowDown", "ArrowRight", "ArrowLeft", "Tab",
                    "Escape", "Backspace", "Delete", "Home", "End", "PageUp", "PageDown",
                    "Shift", "Control", "Alt", "Meta", "CapsLock" # 修飾キーも含む
                ]
                # 大文字小文字を区別せずにキー名を正規化
                normalized_key = next((k for k in valid_keys if k.lower() == key_to_press.lower()), None)

                if not normalized_key:
                     # サポートリストにない場合、単一の英数字か記号かチェック
                     # (Playwrightは "a", "B", "1", "$", "*" なども受け付ける)
                     if len(key_to_press) == 1:
                         normalized_key = key_to_press
                     else:
                         raise ValueError(f"Invalid or unsupported key specified for 'press_key': {key_to_press}. Supported examples: {valid_keys}, 'a', '1', '$'.")

                logger.info(f"Pressing key '{normalized_key}' {press_count} times...")

                target_for_press: Union[Page, Locator] = page # デフォルトはページ全体
                if target_element:
                    logger.info("Target element found, attempting to focus and press on the element.")
                    try:
                        # キー入力前に要素にフォーカスを試みる
                        await target_element.focus(timeout=max(1000, action_wait_time // 4))
                        target_for_press = target_element # フォーカス成功したら要素に対して実行
                    except Exception as focus_e:
                        logger.warning(f"Failed to focus on target element before pressing key: {focus_e}. Pressing key on page level.")
                else:
                     logger.info("No target element specified or found. Pressing key on page level (current focused element).")

                press_delay_ms = 50 # キー入力間のディレイ (ms)
                for k_idx in range(press_count):
                    # press メソッドを呼び出し
                    await target_for_press.press(normalized_key, timeout=max(1000, action_wait_time // press_count if press_count > 0 else action_wait_time))
                    logger.debug(f"Pressed '{normalized_key}' (Count: {k_idx + 1}/{press_count})")
                    if press_count > 1: # 複数回押す場合はディレイを入れる
                        await asyncio.sleep(press_delay_ms / 1000)

                logger.info(f"Key '{normalized_key}' pressed {press_count} times successfully.")
                action_result_details.update({"key_pressed": normalized_key, "count": press_count})
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- screenshot ---
            elif action == "screenshot":
                  # ファイル名決定 (valueがあればそれ、なければデフォルト)
                  filename_base = str(value).strip() if value else f"screenshot_step{step_num}"
                  # 拡張子 .png がなければ追加
                  filename = f"{filename_base}.png" if not filename_base.lower().endswith(('.png', '.jpg', '.jpeg')) else filename_base
                  # ファイル名に使えない文字を除去
                  safe_filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
                  screenshot_path = Path(config.DEFAULT_SCREENSHOT_DIR) / safe_filename # Pathオブジェクトで結合
                  # 出力ディレクトリ作成
                  screenshot_path.parent.mkdir(parents=True, exist_ok=True)

                  logger.info(f"Saving screenshot to '{screenshot_path}'...")
                  ss_timeout = max(10000, action_wait_time) # スクショはタイムアウト長め

                  if target_element: # 要素が指定されていれば要素のみ
                       await target_element.screenshot(path=screenshot_path, timeout=ss_timeout)
                       logger.info("Element screenshot saved.")
                  else: # 要素がなければページ全体
                       await root_page.screenshot(path=screenshot_path, full_page=True, timeout=ss_timeout*2) # fullpageは時間かかるかも
                       logger.info("Page screenshot saved.")

                  action_result_details["filename"] = str(screenshot_path.resolve()) # 結果には絶対パスを記録
                  results.append({**step_result_base, "status": "success", **action_result_details})


            # --- 未知のアクション ---
            else:
                 known_actions = single_element_actions + multiple_elements_actions + ["switch_to_iframe", "switch_to_parent_frame", "wait_page_load", "sleep", "scroll_page_to_bottom", "screenshot"]
                 if action not in known_actions:
                     logger.warning(f"Undefined or unknown action '{action}'. Skipping this step.")
                     results.append({**step_result_base, "status": "skipped", "message": f"Undefined action: {action}"})
                     continue # スキップして次のステップへ

            # --- ステップ正常終了時のURL更新 ---
            final_url = root_page.url # 各ステップ終了時にURLを再確認

        # --- ステップごとのエラーハンドリング ---
        except (PlaywrightTimeoutError, PlaywrightError, ValueError, IndexError, Exception) as e:
            error_message = f"ステップ {step_num} ({action}) でエラー発生: {type(e).__name__} - {e}"
            logger.error(error_message, exc_info=True) # スタックトレース付きでログ出力
            error_screenshot_path = None
            current_url_on_error = "Unknown (Page might be closed)"
            try: # エラー時でもURL取得を試みる
                if root_page and not root_page.is_closed(): current_url_on_error = root_page.url
                elif page and not page.is_closed(): current_url_on_error = page.url
            except Exception: pass

            # エラー時のスクリーンショット取得試行
            if root_page and not root_page.is_closed():
                 timestamp = time.strftime("%Y%m%d_%H%M%S")
                 error_ss_filename = f"error_step{step_num}_{timestamp}.png"
                 error_ss_path = Path(config.DEFAULT_SCREENSHOT_DIR) / error_ss_filename
                 try:
                     error_ss_path.parent.mkdir(parents=True, exist_ok=True)
                     await root_page.screenshot(path=error_ss_path, full_page=True, timeout=15000) # タイムアウト短縮
                     error_screenshot_path = str(error_ss_path.resolve())
                     logger.info(f"Error screenshot saved: {error_ss_path.name}")
                 except Exception as ss_e:
                     logger.error(f"Failed to save error screenshot for step {step_num}: {ss_e}")

            # エラー詳細を結果に追加
            error_details = {
                **step_result_base,
                "status": "error",
                "selector": selector, # エラー時のセレクター情報も記録
                "target_hints": target_hints, # 試行したヒント
                "locator_attempts": locator_attempt_logs, # ヒント試行ログ
                "message": str(e), # エラーメッセージ本体
                "full_error": error_message, # より詳細なエラー情報
                "traceback": traceback.format_exc(), # スタックトレース
                "url_on_error": current_url_on_error # エラー発生時のURL
            }
            if error_screenshot_path:
                error_details["error_screenshot"] = error_screenshot_path
            results.append(error_details)

            # エラー発生時は False を返して処理全体を中断
            return False, results, current_url_on_error

    # --- 全ステップ正常完了 ---
    logger.info("All steps processed successfully.")
    final_url = root_page.url if root_page and not root_page.is_closed() else final_url
    return True, results, final_url # 成功時は True を返す