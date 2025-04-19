# --- ファイル: playwright_actions.py (press_key, Tab, select_option改善, PDF判定修正 含む完全版) ---
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
    Response, # Response をインポート
    expect
)

# --- 必要なモジュール ---
try:
    import config
except ImportError:
    class ConfigMock:
        DEFAULT_ACTION_TIMEOUT = 10000
        IFRAME_LOCATOR_TIMEOUT = 5000
        PDF_DOWNLOAD_TIMEOUT = 60000
        NEW_PAGE_EVENT_TIMEOUT = 4000
        DYNAMIC_SEARCH_MAX_DEPTH = 2
        DEFAULT_SCREENSHOT_DIR = 'screenshots'
    config = ConfigMock()
    logging.warning("config.py が見つかりません。デフォルト値を使用します。")

try:
    import utils
except ImportError:
    utils = None
    logging.warning("utils.py が見つかりません。PDF抽出などの機能は利用できません。")

try:
    from playwright_finders import find_element_dynamically, find_all_elements_dynamically
    from playwright_helper_funcs import get_page_inner_text
except ImportError as e:
     logging.critical(f"playwright_actions の依存モジュールが見つかりません: {e.name}")
     raise

logger = logging.getLogger(__name__)

# --- メールアドレス抽出用ヘルパー関数 ---
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
async def _extract_emails_from_page_async(context: BrowserContext, url: str, timeout: int) -> List[str]:
    """指定されたURLにアクセスし、メールアドレスを抽出する"""
    page = None; emails_found: Set[str] = set(); start_time = time.monotonic()
    page_access_timeout = max(int(timeout * 0.8), 15000)
    logger.info(f"URLからメール抽出開始: {url} (タイムアウト: {page_access_timeout}ms)")
    try:
        page = await context.new_page()
        nav_timeout = max(int(page_access_timeout * 0.9), 10000)
        await page.goto(url, wait_until="load", timeout=nav_timeout)
        remaining_time_for_text = page_access_timeout - (time.monotonic() - start_time) * 1000
        if remaining_time_for_text <= 1000: raise PlaywrightTimeoutError("Not enough time for text extraction")
        text_timeout = int(remaining_time_for_text)
        try:
            page_text = await page.locator(':root').inner_text(timeout=text_timeout)
            if page_text:
                found_in_text = EMAIL_REGEX.findall(page_text)
                if found_in_text: emails_found.update(found_in_text)
        except Exception as text_err: logger.warning(f"ページテキスト取得中にエラー ({url}): {text_err}")
        try:
            mailto_links = await page.locator("a[href^='mailto:']").all()
            if mailto_links:
                for link in mailto_links:
                    try:
                        href = await link.get_attribute('href', timeout=500)
                        if href and href.startswith('mailto:'):
                            email_part = href[len('mailto:'):].split('?')[0]
                            if email_part and EMAIL_REGEX.match(email_part): emails_found.add(email_part)
                    except Exception: pass
        except Exception as mailto_err: logger.warning(f"mailtoリンク取得中にエラー ({url}): {mailto_err}")
        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(f"メール抽出完了 ({url}). ユニーク候補: {len(emails_found)} ({elapsed:.0f}ms)")
        if emails_found and utils and hasattr(utils, 'write_emails_to_file'):
            unique_emails_list = list(emails_found)
            try: await asyncio.to_thread(utils.write_emails_to_file, unique_emails_list, "extracted_mails.txt")
            except Exception as mfwe: logger.error(f"Mail file write error: {mfwe}")
        return list(emails_found)
    except Exception as e:
        logger.warning(f"メール抽出処理全体でエラー ({url}): {type(e).__name__} - {e}", exc_info=False)
        logger.debug(f"Traceback:", exc_info=True)
        return []
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
    if not hints or not isinstance(hints, list):
        logger.warning(f"[{action_name}] No valid hints provided.")
        return None, None, []

    hint_count = len(hints)
    base_validation_timeout = max(500, min(2000, overall_timeout // max(1, hint_count)))
    logger.info(f"[{action_name}] Trying {hint_count} locator hints (overall timeout: {overall_timeout}ms, validation_timeout per hint: {base_validation_timeout}ms)...")
    hints.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("confidence", "low"), 2))

    for i, hint in enumerate(hints):
        elapsed_ms = (time.monotonic() - start_time) * 1000
        if elapsed_ms >= overall_timeout:
            logger.warning(f"[{action_name}] Overall timeout reached during hint processing ({elapsed_ms:.0f}ms).")
            break
        remaining_ms = overall_timeout - elapsed_ms
        current_validation_timeout = min(base_validation_timeout, max(100, int(remaining_ms * 0.8)))

        hint_type = hint.get("type"); hint_value = hint.get("value"); hint_name = hint.get("name")
        hint_level = hint.get("level"); hint_common_selector = hint.get("common_selector")
        hint_index = hint.get("index"); hint_confidence = hint.get("confidence", "low")

        locator_description = f"Hint {i+1}/{hint_count}: type={hint_type}, confidence={hint_confidence}"
        value_str = str(hint_value); value_log = f"{value_str[:50]}{'...' if len(value_str) > 50 else ''}"
        logging.info(f"  Trying: {locator_description} (value: {value_log})")

        attempt_log = {"hint": hint, "status": "Skipped", "reason": "No locator generated"}
        locator_to_try: Optional[Locator] = None

        try:
            options = {}
            if hint_type == "role_and_text":
                 actual_role = hint_value if isinstance(hint_value, str) else hint.get("role")
                 if not isinstance(actual_role, str) or not actual_role: raise ValueError("Role missing/invalid in 'role_and_text' hint")
                 if isinstance(hint_name, str):
                     if hint_name.startswith('/') and hint_name.endswith('/') and len(hint_name) > 2:
                          try: pattern = re.compile(hint_name[1:-1]); options["name"] = pattern
                          except re.error as re_err: options["name"] = hint_name; logging.warning(f"Invalid regex in hint, using as exact text: {re_err}")
                     else: options["name"] = hint_name
                 if actual_role == "heading" and isinstance(hint_level, int): options["level"] = hint_level
                 locator_to_try = target_scope.get_by_role(actual_role, **options)
            elif hint_type == "test_id" and isinstance(hint_value, str): locator_to_try = target_scope.get_by_test_id(hint_value)
            elif hint_type == "text_exact" and isinstance(hint_value, str): locator_to_try = target_scope.get_by_text(hint_value, exact=True)
            elif hint_type == "placeholder" and isinstance(hint_value, str): locator_to_try = target_scope.get_by_placeholder(hint_value)
            elif hint_type == "aria_label" and isinstance(hint_value, str): locator_to_try = target_scope.get_by_label(hint_value, exact=True)
            elif hint_type == "css_selector_candidate" and isinstance(hint_value, str): locator_to_try = target_scope.locator(hint_value)
            elif hint_type == "nth_child" and isinstance(hint_common_selector, str) and isinstance(hint_index, int) and hint_index >= 0:
                locator_to_try = target_scope.locator(hint_common_selector).nth(hint_index)
            else: attempt_log["reason"] = f"Invalid/unknown hint type or value: {hint}"; logging.warning(f"    -> {attempt_log['reason']}")

            if locator_to_try:
                attempt_log["reason"] = "Checking validity...";
                try: attempt_log["final_selector_attempted"] = repr(locator_to_try)
                except Exception: attempt_log["final_selector_attempted"] = "[repr failed]"
                count = await locator_to_try.count(); attempt_log["count"] = count; logging.debug(f"    -> Count: {count}")
                if count == 1:
                    is_visible = False
                    try: await locator_to_try.wait_for(state='visible', timeout=current_validation_timeout); is_visible = True; logging.debug(f"    -> Visible.")
                    except PlaywrightTimeoutError: logging.debug(f"    -> Not visible within timeout.")
                    except Exception as vis_err: logging.warning(f"    -> Visibility check error: {vis_err}")
                    attempt_log["is_visible"] = is_visible
                    if is_visible:
                        successful_locator = locator_to_try; successful_hint = hint; attempt_log["status"] = "Success"; attempt_log["reason"] = "Found unique and visible element."; logging.info(f"    -> Success!"); attempt_logs.append(attempt_log); break
                    else: attempt_log["status"] = "Fail"; attempt_log["reason"] = "Element found but not visible."; logging.info(f"    -> Fail: Not visible.")
                elif count > 1: attempt_log["status"] = "Fail"; attempt_log["reason"] = f"Multiple elements found ({count}). Hint is not specific enough."; logging.info(f"    -> Fail: Multiple elements.")
                else: attempt_log["status"] = "Fail"; attempt_log["reason"] = "Element not found."; logging.info(f"    -> Fail: Not found.")
            else: attempt_log["status"] = "Fail"; attempt_log["reason"] = "Could not generate locator from hint."
        except ValueError as ve: attempt_log["status"] = "Error"; attempt_log["reason"] = f"ValueError during locator generation: {ve}"; logging.warning(f"    -> ValueError: {ve}")
        except PlaywrightError as pe: attempt_log["status"] = "Error"; attempt_log["reason"] = f"PlaywrightError during locator check: {pe}"; logging.warning(f"    -> PlaywrightError: {pe}")
        except Exception as e: attempt_log["status"] = "Error"; attempt_log["reason"] = f"Unexpected Error: {e}"; logging.error(f"    -> Unexpected error during hint processing: {e}", exc_info=True)
        attempt_logs.append(attempt_log)

    if successful_locator: logger.info(f"[{action_name}] Located element successfully using hint: {successful_hint}")
    else: logger.warning(f"[{action_name}] Failed to locate element using any of the provided hints.")
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
    (PDF判定修正・press_key追加・select_option改善 含む完全版)
    """
    results: List[Dict[str, Any]] = []
    current_target: Union[Page, FrameLocator] = page
    root_page: Page = page
    current_context: BrowserContext = root_page.context
    iframe_stack: List[Union[Page, FrameLocator]] = []
    final_url: str = page.url

    for i, step_data in enumerate(actions):
        step_num = i + 1
        action = step_data.get("action", "").lower()
        selector = step_data.get("selector"); target_hints = step_data.get("target_hints")
        iframe_selector_input = step_data.get("iframe_selector"); value = step_data.get("value")
        attribute_name = step_data.get("attribute_name"); option_type = step_data.get("option_type")
        option_value = step_data.get("option_value");
        key_to_press = step_data.get("value") if action == "press_key" else None
        press_count = int(step_data.get("count", 1)) if action == "press_key" else 1
        action_wait_time = step_data.get("wait_time_ms", default_timeout)

        logger.info(f"--- ステップ {step_num}/{len(actions)}: Action='{action}' ---")
        step_info = { "selector": selector, "target_hints_count": len(target_hints) if isinstance(target_hints, list) else None, "iframe(指定)": iframe_selector_input, "value": value, "attribute_name": attribute_name, "option_type": option_type, "option_value": option_value, }
        if action == "press_key":
             step_info["key_to_press"] = key_to_press
             step_info["press_count"] = press_count
        step_info_str = ", ".join([f"{k}='{str(v)[:50]}...'" if isinstance(v, str) and len(v) > 50 else f"{k}='{v}'" for k, v in step_info.items() if v is not None])
        logger.info(f"詳細: {step_info_str} (timeout: {action_wait_time}ms)")
        if isinstance(target_hints, list): logger.debug(f"Target Hints:\n{pprint.pformat(target_hints)}")

        step_result_base = {"step": step_num, "action": action}
        if step_data.get("memo"): step_result_base["memo"] = step_data["memo"]

        try:
            if root_page.is_closed(): raise PlaywrightError(f"Root page closed.")
            final_url = root_page.url
            current_base_url = final_url

            # --- Iframe/Parent Frame 切替 ---
            if action == "switch_to_iframe":
                if not iframe_selector_input: raise ValueError("Action 'switch_to_iframe' requires 'iframe_selector'.")
                logger.info(f"Switching to iframe using selector: '{iframe_selector_input}'")
                target_frame_locator = current_target.frame_locator(iframe_selector_input)
                try: await target_frame_locator.locator(':root').wait_for(state='attached', timeout=action_wait_time)
                except PlaywrightTimeoutError as e: raise PlaywrightTimeoutError(f"Iframe '{iframe_selector_input}' not found or timed out ({action_wait_time}ms).") from e
                if id(current_target) not in [id(s) for s in iframe_stack]: iframe_stack.append(current_target)
                current_target = target_frame_locator; logger.info(f"Successfully switched to FrameLocator: {iframe_selector_input}")
                results.append({**step_result_base, "status": "success", "selector": iframe_selector_input})
                final_url = root_page.url
                continue

            elif action == "switch_to_parent_frame":
                status="warning"; message=None
                if not iframe_stack: logger.warning("Already at top-level frame or iframe stack is empty."); message="Already at top-level or stack empty."
                else: parent_target = iframe_stack.pop(); current_target = parent_target; logger.info(f"Switched back to parent target: {type(current_target).__name__}"); status="success"
                if status == "warning" and isinstance(current_target, FrameLocator): current_target = root_page; logger.info("Current target was FrameLocator, reset to root Page.")
                results.append({**step_result_base, "status": status, "message": message})
                final_url = root_page.url
                continue

            # --- ページ全体操作 ---
            if action in ["wait_page_load", "sleep", "scroll_page_to_bottom"]:
                 if action == "wait_page_load": await root_page.wait_for_load_state("load", timeout=action_wait_time); logger.info("Page load complete.")
                 elif action == "sleep":
                     try: seconds = float(value) if value is not None else 1.0; assert seconds >= 0
                     except (TypeError, ValueError, AssertionError): raise ValueError("Invalid 'value' for sleep action. Must be a non-negative number (seconds).")
                     logger.info(f"Sleeping for {seconds:.2f} seconds..."); await asyncio.sleep(seconds)
                     results.append({**step_result_base, "status": "success", "duration_sec": seconds});
                     final_url = root_page.url; continue
                 elif action == "scroll_page_to_bottom": await root_page.evaluate('window.scrollTo(0, document.body.scrollHeight)'); await asyncio.sleep(0.5); logger.info("Scrolled page to bottom.")
                 results.append({**step_result_base, "status": "success"})
                 final_url = root_page.url; continue

            # --- 要素操作準備と要素特定ロジック ---
            target_element: Optional[Locator] = None
            found_elements_list: List[Tuple[Locator, Union[Page, FrameLocator]]] = []
            found_scope: Optional[Union[Page, FrameLocator]] = None
            successful_hint_info: Optional[Dict[str, Any]] = None
            locator_attempt_logs: List[Dict[str, Any]] = []
            single_element_actions = ["click", "input", "hover", "get_inner_text", "get_text_content", "get_inner_html", "get_attribute", "wait_visible", "select_option", "scroll_to_element", "press_key"]
            multiple_elements_actions = ["get_all_attributes", "get_all_text_contents"]
            is_screenshot_action = action == "screenshot"
            needs_single_element = action in single_element_actions or (is_screenshot_action and (target_hints or selector))
            needs_multiple_elements = action in multiple_elements_actions

            if needs_single_element:
                element_located_by = None
                allow_no_element = action == "press_key"

                if target_hints and isinstance(target_hints, list) and target_hints:
                    logger.info(f"Locating element based on {len(target_hints)} target_hints...")
                    target_element, successful_hint_info, locator_attempt_logs = await try_locators_sequentially(current_target, target_hints, action, action_wait_time)
                    step_result_base["locator_attempts"] = locator_attempt_logs
                    if target_element: logger.info(f"Element located using target_hints: {successful_hint_info}"); step_result_base["locator_hint_used"] = successful_hint_info; found_scope = current_target; element_located_by = 'hints'
                    else: logger.warning("Failed to locate element using target_hints.")

                if not target_element and selector:
                    fallback_reason = "target_hints not provided" if not target_hints else "target_hints failed"
                    logger.warning(f"{fallback_reason}, falling back to dynamic search using selector: '{selector}'")
                    required_state = 'visible' if action not in ['get_attribute', 'get_all_attributes', 'get_all_text_contents'] else 'attached'
                    target_element, found_scope = await find_element_dynamically(current_target, selector, max_depth=config.DYNAMIC_SEARCH_MAX_DEPTH, timeout=action_wait_time, target_state=required_state)
                    if target_element and found_scope: logger.info(f"Element located using fallback selector '{selector}'."); step_result_base["selector_used_as_fallback"] = selector; element_located_by = 'fallback_selector'
                    else: logger.warning(f"Failed to locate element using fallback selector '{selector}' as well.")

                if not target_element and not allow_no_element:
                    if not target_hints and not selector: raise ValueError(f"Action '{action}' requires 'target_hints' or 'selector'.")
                    error_msg = f"Failed to locate required single element for action '{action}'."
                    tried_hints = step_result_base.get("locator_attempts") is not None; tried_fallback = step_result_base.get("selector_used_as_fallback") is not None; original_selector = step_data.get("selector")
                    if tried_hints and tried_fallback: error_msg += f" Tried target_hints and fallback selector '{step_result_base['selector_used_as_fallback']}'."
                    elif tried_hints and not original_selector: error_msg += " Tried target_hints, but no fallback selector was provided."
                    elif tried_hints: error_msg += f" Tried target_hints and fallback selector '{original_selector}' (failed)."
                    elif original_selector: error_msg += f" Tried selector '{original_selector}'."
                    raise PlaywrightError(error_msg)
                elif not target_element and allow_no_element: logger.info(f"Action '{action}' allows execution without a specific target element.")

            elif needs_multiple_elements:
                if not selector: raise ValueError(f"Action '{action}' requires a 'selector'.")
                logger.info(f"Locating multiple elements using selector '{selector}' with dynamic search...")
                found_elements_list = await find_all_elements_dynamically(current_target, selector, max_depth=config.DYNAMIC_SEARCH_MAX_DEPTH, timeout=action_wait_time)
                if not found_elements_list: logger.warning(f"No elements found matching selector '{selector}' within dynamic search.")
                step_result_base["selector"] = selector

            # --- スコープ更新処理 ---
            if needs_single_element and found_scope and id(found_scope) != id(current_target):
                logger.info(f"Updating operating scope from {type(current_target).__name__} to {type(found_scope).__name__}")
                if id(current_target) not in [id(s) for s in iframe_stack]: iframe_stack.append(current_target)
                current_target = found_scope
            logger.info(f"Final operating scope for this step: {type(current_target).__name__}")

            # --- 各アクション実行 ---
            action_result_details = {}
            if step_result_base.get("locator_hint_used"): action_result_details["locator_hint_used"] = step_result_base["locator_hint_used"]; action_result_details["locator_method"] = "hints"
            elif step_result_base.get("selector_used_as_fallback"): action_result_details["selector"] = step_result_base["selector_used_as_fallback"]; action_result_details["locator_method"] = "fallback_selector"
            elif step_result_base.get("selector"): action_result_details["selector"] = step_result_base["selector"]; action_result_details["locator_method"] = "selector"

            # --- click ---
            if action == "click":
                if not target_element: raise PlaywrightError("Click failed: Target element was not located.")
                logger.info("Clicking element...")
                context_for_click = root_page.context; new_page: Optional[Page] = None
                try:
                    async with context_for_click.expect_page(timeout=config.NEW_PAGE_EVENT_TIMEOUT) as new_page_info:
                        await target_element.click(timeout=action_wait_time)
                    new_page = await new_page_info.value
                    new_page_url = new_page.url; logger.info(f"New page opened: {new_page_url}")
                    try: await new_page.wait_for_load_state("load", timeout=action_wait_time); logger.info("New page load complete.")
                    except PlaywrightTimeoutError: logger.warning(f"New page load timed out.")
                    root_page, current_target, current_context, iframe_stack = new_page, new_page, new_page.context, []
                    action_result_details.update({"new_page_opened": True, "new_page_url": new_page_url})
                    logger.info("Scope reset to the new page.")
                except PlaywrightTimeoutError:
                    logger.info(f"Click completed, but no new page opened within {config.NEW_PAGE_EVENT_TIMEOUT}ms.")
                    action_result_details["new_page_opened"] = False
                except Exception as click_err: logger.error(f"Click action failed: {click_err}", exc_info=True); raise click_err
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- input ---
            elif action == "input":
                 if not target_element: raise PlaywrightError("Input failed: Target element was not located.")
                 if value is None: raise ValueError("Action 'input' requires a 'value'.")
                 input_value_str = str(value)
                 logger.info(f"Inputting text into element: '{input_value_str[:50]}{'...' if len(input_value_str) > 50 else ''}'")
                 try:
                     await target_element.wait_for(state='visible', timeout=max(1000, action_wait_time // 3))
                     logger.debug("Clicking element to ensure focus before input.")
                     await target_element.click(timeout=max(1000, action_wait_time // 3))
                     await target_element.fill(input_value_str, timeout=action_wait_time)
                     await asyncio.sleep(0.1); logger.info("Input successful.")
                     action_result_details["value"] = value; results.append({**step_result_base, "status": "success", **action_result_details})
                 except Exception as e: logger.error(f"Input action failed: {e}", exc_info=True); raise PlaywrightError(f"Failed to input value: {e}") from e

            # --- hover ---
            elif action == "hover":
                 if not target_element: raise PlaywrightError("Hover failed: Target element was not located.")
                 logger.info("Hovering over element..."); await target_element.hover(timeout=action_wait_time); logger.info("Hover successful.")
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_inner_text ---
            elif action == "get_inner_text":
                 if not target_element: raise PlaywrightError("Get innerText failed: Target element was not located.")
                 logger.info("Getting innerText..."); text = await target_element.inner_text(timeout=action_wait_time); text = text.strip() if text else ""
                 logger.info(f"Got innerText: '{text[:100]}{'...' if len(text) > 100 else ''}'"); action_result_details["text"] = text
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_text_content ---
            elif action == "get_text_content":
                 if not target_element: raise PlaywrightError("Get textContent failed: Target element was not located.")
                 logger.info("Getting textContent..."); text = await target_element.text_content(timeout=action_wait_time); text = text.strip() if text else ""
                 logger.info(f"Got textContent: '{text[:100]}{'...' if len(text) > 100 else ''}'"); action_result_details["text"] = text
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_inner_html ---
            elif action == "get_inner_html":
                 if not target_element: raise PlaywrightError("Get innerHTML failed: Target element was not located.")
                 logger.info("Getting innerHTML..."); html_content = await target_element.inner_html(timeout=action_wait_time)
                 logger.info(f"Got innerHTML (first 500 chars): {html_content[:500]}..."); action_result_details["html"] = html_content
                 results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_attribute (PDF判定修正) ---
            elif action == "get_attribute":
                if not target_element: raise PlaywrightError("Get attribute failed: Element not located.")
                if not attribute_name: raise ValueError("Action 'get_attribute' requires 'attribute_name'.")
                logger.info(f"Getting attribute '{attribute_name}' from element...")
                attr_value = await target_element.get_attribute(attribute_name, timeout=action_wait_time)
                pdf_text_content = None; processed_value = attr_value; scraped_content = None

                if attribute_name.lower() == 'href' and attr_value is not None:
                     try: absolute_url = urljoin(current_base_url, attr_value); processed_value = absolute_url
                     except Exception as url_e: logger.error(f"Failed to resolve href URL '{attr_value}': {url_e}"); processed_value = f"Error resolving URL: {url_e}"

                elif attribute_name.lower() == 'pdf' and attr_value is not None:
                    pdf_text_content = f"Warning: Could not process PDF link."
                    try:
                        absolute_url = urljoin(current_base_url, attr_value)
                        processed_value = absolute_url
                        logger.info(f"Checking if link is PDF: {absolute_url}")
                        if not utils: raise RuntimeError("utils module (for PDF processing) is not available.")
                        response: Optional[Response] = await utils.download_pdf_async(api_request_context, absolute_url)
                        if response:
                            content_type = response.headers.get('content-type', '').lower()
                            logger.info(f"  Content-Type: '{content_type}'")
                            if 'application/pdf' in content_type:
                                logger.info(f"  Content-Type indicates PDF. Extracting text...")
                                pdf_bytes = await response.body()
                                if pdf_bytes:
                                    pdf_text_content = await asyncio.to_thread(utils.extract_text_from_pdf_sync, pdf_bytes)
                                    if pdf_text_content is None or pdf_text_content.startswith("Error:"): logger.error(f"  PDF text extraction failed: {pdf_text_content}")
                                    else: logger.info(f"  PDF text extracted successfully (Length: {len(pdf_text_content)}).")
                                else: pdf_text_content = "Error: PDF download succeeded but body was empty."
                            else: pdf_text_content = f"Warning: Content-Type is not 'application/pdf' ('{content_type}')."
                        else: pdf_text_content = "Error: PDF download/access failed."
                    except Exception as pdf_e: logger.error(f"PDF processing error for URL '{attr_value}': {pdf_e}", exc_info=True); pdf_text_content = f"Error processing PDF link: {pdf_e}"

                elif attribute_name.lower() == 'content' and attr_value is not None:
                     try:
                         absolute_url = urljoin(current_base_url, attr_value); processed_value = absolute_url
                         parsed_url = urlparse(absolute_url)
                         if parsed_url.scheme in ['http', 'https'] and not absolute_url.lower().endswith('.pdf'):
                              success_content, content_or_error = await get_page_inner_text(current_context, absolute_url, action_wait_time); scraped_content = content_or_error
                         elif absolute_url.lower().endswith('.pdf'): scraped_content = "Warning: Link points to a PDF. Use attribute_name='pdf'."
                         else: scraped_content = f"Warning: Cannot fetch content. Invalid URL scheme or not HTTP/S: {absolute_url}"
                     except Exception as content_e: logger.error(f"Content scraping error for URL '{attr_value}': {content_e}", exc_info=True); scraped_content = f"Error scraping content: {content_e}"

                logger.info(f"Got attribute '{attribute_name}': '{str(processed_value)[:100]}{'...' if isinstance(processed_value, str) and len(processed_value) > 100 else ''}'")
                action_result_details.update({"attribute": attribute_name, "value": processed_value})
                if pdf_text_content is not None: action_result_details["pdf_text"] = pdf_text_content
                if scraped_content is not None: action_result_details["scraped_text"] = scraped_content
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_all_attributes (PDF判定修正) ---
            elif action == "get_all_attributes":
                if not selector: raise ValueError("Action 'get_all_attributes' requires 'selector'.")
                if not attribute_name: raise ValueError("Action 'get_all_attributes' requires 'attribute_name'.")

                if not found_elements_list:
                    logger.warning(f"No elements found for '{selector}'. Skipping attribute/content extraction.")
                    action_result_details["results_count"] = 0
                    if attribute_name.lower() in ['href', 'pdf', 'content', 'mail']: action_result_details["url_list"] = []
                    if attribute_name.lower() == 'pdf': action_result_details["pdf_texts"] = []
                    if attribute_name.lower() == 'content': action_result_details["scraped_texts"] = []
                    if attribute_name.lower() == 'mail': action_result_details["extracted_emails"] = []
                    if attribute_name.lower() not in ['href', 'pdf', 'content', 'mail']: action_result_details["attribute_list"] = []
                else:
                    num_found = len(found_elements_list)
                    logger.info(f"Getting attribute/content '{attribute_name}' from {num_found} elements found by '{selector}'...")
                    url_list, pdf_list, content_list, mail_list, generic_list = [],[],[],[],[]
                    processed_domains = set()

                    if attribute_name.lower() in ['href', 'pdf', 'content', 'mail']:
                        CONCURRENT_LIMIT=5; semaphore=asyncio.Semaphore(CONCURRENT_LIMIT)
                        logger.info(f"Processing {num_found} links concurrently (limit: {CONCURRENT_LIMIT}) for mode '{attribute_name.lower()}'...")

                        async def process_single_element_for_href_related(
                            locator: Locator, index: int, base_url: str, attr_mode: str, sem: asyncio.Semaphore
                        ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
                            original_href: Optional[str] = None; absolute_url: Optional[str] = None
                            pdf_text: Optional[str] = None; scraped_text: Optional[str] = None
                            emails_from_page: Optional[List[str]] = None
                            async with sem:
                                try:
                                    logger.debug(f"  [{index+1}/{num_found}] Processing link...")
                                    href_timeout = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                                    original_href = await locator.get_attribute("href", timeout=href_timeout)
                                    if original_href is None: return None, None, None, None
                                    try: absolute_url = urljoin(base_url, original_href); parsed_url = urlparse(absolute_url)
                                    except Exception: return f"Error converting URL: {original_href}", None, None, None
                                    if parsed_url.scheme not in ['http', 'https']: return absolute_url, None, None, None

                                    # --- PDF モード (Content-Type チェック) ---
                                    if attr_mode == 'pdf':
                                        pdf_start = time.monotonic()
                                        response: Optional[Response] = await utils.download_pdf_async(api_request_context, absolute_url) if utils else None
                                        if response:
                                            content_type = response.headers.get('content-type', '').lower()
                                            logger.debug(f"  [{index+1}] PDF check: Content-Type='{content_type}' URL: {absolute_url}")
                                            if 'application/pdf' in content_type:
                                                pdf_bytes = await response.body()
                                                if pdf_bytes: pdf_text = await asyncio.to_thread(utils.extract_text_from_pdf_sync, pdf_bytes)
                                                else: pdf_text = "Error: PDF download succeeded but body was empty."
                                            else: pdf_text = f"Warning: Content-Type is not PDF ('{content_type}')."
                                        else: pdf_text = "Error: PDF download/access failed."
                                        pdf_elapsed = (time.monotonic() - pdf_start) * 1000
                                        logger.info(f"  [{index+1}/{num_found}] PDF processing finished ({pdf_elapsed:.0f}ms).")
                                    elif attr_mode == 'content' and not absolute_url.lower().endswith('.pdf'):
                                        content_start = time.monotonic()
                                        success_content, content_or_error = await get_page_inner_text(current_context, absolute_url, action_wait_time)
                                        scraped_text = content_or_error
                                        content_elapsed = (time.monotonic() - content_start) * 1000
                                        logger.info(f"  [{index+1}/{num_found}] Content scraping finished ({content_elapsed:.0f}ms). Success: {success_content}")
                                    elif attr_mode == 'mail':
                                        mail_start = time.monotonic()
                                        emails_from_page = await _extract_emails_from_page_async(current_context, absolute_url, action_wait_time)
                                        mail_elapsed = (time.monotonic() - mail_start) * 1000
                                        logger.info(f"  [{index+1}/{num_found}] Mail extraction finished ({mail_elapsed:.0f}ms). Found: {len(emails_from_page) if emails_from_page else 0}")
                                    return absolute_url, pdf_text, scraped_text, emails_from_page
                                except (PlaywrightError, RuntimeError, Exception) as e:
                                    error_msg = f"Error processing element {index} for {attr_mode}: {type(e).__name__}"; logger.warning(error_msg, exc_info=False)
                                    err_detail = f"Error: {type(e).__name__}"
                                    return absolute_url or f"Error getting href for element {index}", err_detail if attr_mode=='pdf' else None, err_detail if attr_mode=='content' else None, [err_detail] if attr_mode=='mail' else None

                        tasks = [process_single_element_for_href_related(loc, idx, current_base_url, attribute_name.lower(), semaphore) for idx, (loc, _) in enumerate(found_elements_list)]
                        results_tuples = await asyncio.gather(*tasks)
                        flat_mails=[]
                        for url_res, pdf_res, content_res, mails_res in results_tuples:
                            url_list.append(url_res)
                            if attribute_name.lower() == 'pdf': pdf_list.append(pdf_res)
                            if attribute_name.lower() == 'content': content_list.append(content_res)
                            if attribute_name.lower() == 'mail' and mails_res: flat_mails.extend(mails_res)
                        if attribute_name.lower() == 'mail':
                             unique_emails_this_step = [];
                             for email in flat_mails:
                                 if isinstance(email, str) and '@' in email:
                                     try: domain = email.split('@',1)[1].lower()
                                     except IndexError: continue
                                     if domain and domain not in processed_domains: unique_emails_this_step.append(email); processed_domains.add(domain); logger.debug(f"    Added unique domain email: {email}")
                             mail_list = unique_emails_this_step
                             logger.info(f"ドメイン重複排除完了。ユニークドメインメールアドレス数: {len(mail_list)}")
                    else: # 通常属性
                        async def get_single_attr(locator: Locator, attr_name: str, index: int, timeout_ms: int) -> Optional[str]:
                            try: return await locator.get_attribute(attr_name, timeout=timeout_ms)
                            except Exception as e: logger.warning(f"Error getting attr '{attr_name}' idx {index}: {e}"); return f"Error: {type(e).__name__}"
                        timeout = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                        tasks = [get_single_attr(loc, attribute_name, idx, timeout) for idx, (loc, _) in enumerate(found_elements_list)]
                        generic_list = await asyncio.gather(*tasks)

                    # 結果格納
                    action_result_details["results_count"] = len(url_list) if attribute_name.lower() in ['href', 'pdf', 'content', 'mail'] else len(generic_list)
                    action_result_details["attribute"] = attribute_name
                    if attribute_name.lower() in ['href', 'pdf', 'content', 'mail']: action_result_details["url_list"] = url_list
                    if attribute_name.lower() == 'pdf': action_result_details["pdf_texts"] = pdf_list
                    if attribute_name.lower() == 'content': action_result_details["scraped_texts"] = content_list
                    if attribute_name.lower() == 'mail': action_result_details["extracted_emails"] = mail_list
                    if generic_list: action_result_details["attribute_list"] = generic_list

                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- get_all_text_contents (変更なし) ---
            elif action == "get_all_text_contents":
                 # ... (実装は省略) ...
                if not selector: raise ValueError("get_all_text_contents requires 'selector'.")
                text_list: List[Optional[str]] = []
                if not found_elements_list: logger.warning(f"No elements for '{selector}'."); action_result_details["results_count"] = 0
                else:
                    num_found = len(found_elements_list); logger.info(f"Getting textContent from {num_found} elements...")
                    async def get_single_text(locator: Locator, index: int, timeout_ms: int) -> Optional[str]:
                        try: text = await locator.text_content(timeout=timeout_ms); return text.strip() if text else ""
                        except Exception as e: logger.warning(f"Error getting text idx {index}: {e}"); return f"Error: {type(e).__name__}"
                    timeout = max(500, action_wait_time // num_found if num_found > 5 else action_wait_time // 3)
                    tasks = [get_single_text(loc, idx, timeout) for idx, (loc, _) in enumerate(found_elements_list)]
                    text_list = await asyncio.gather(*tasks)
                    action_result_details["results_count"] = len(text_list)
                action_result_details["text_list"] = text_list
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- wait_visible (変更なし) ---
            elif action == "wait_visible":
                 # ... (実装は省略) ...
                 if not target_element: raise PlaywrightError("Wait visible failed: Target element was not located.")
                 logger.info("Element visibility confirmed."); results.append({**step_result_base, "status": "success", **action_result_details})

            # --- select_option (変更なし) ---
            elif action == "select_option":
                 # ... (実装は省略 - clickしてから実行する版) ...
                  if not target_element: raise PlaywrightError("Select option failed: Element not located.")
                  if option_type not in ['value', 'index', 'label'] or option_value is None: raise ValueError("Invalid 'option_type' or 'option_value'.")
                  logger.info(f"Selecting option (Type: {option_type}, Value: '{option_value}')..."); value_to_select: Union[str, Dict]
                  if option_type == 'value': value_to_select = {'value': str(option_value)}
                  elif option_type == 'index':
                      try: index_val = int(option_value)
                      except (ValueError, TypeError): raise ValueError("Option type 'index' requires an integer value.")
                      value_to_select = {'index': index_val}
                  else: value_to_select = {'label': str(option_value)}
                  try:
                      logger.info(f"Waiting for select element '{action_result_details.get('selector', 'N/A')}'...")
                      await target_element.wait_for(state='visible', timeout=max(1000, action_wait_time // 4))
                      await target_element.is_enabled(timeout=max(1000, action_wait_time // 4))
                      logger.info("Clicking the select element first...")
                      await target_element.click(timeout=max(1000, action_wait_time // 4))
                      await asyncio.sleep(0.3)
                      logger.info("Attempting select_option after click...")
                      selected_values = await target_element.select_option(value_to_select, timeout=max(5000, int(action_wait_time * 0.7)))
                  except Exception as e:
                       logger.error(f"Select option failed: {e}", exc_info=True)
                       try: options_available = await target_element.evaluate("el => Array.from(el.options).map(opt => ({value: opt.value, text: opt.innerText, index: opt.index}))"); logger.info(f"Available options: {options_available}")
                       except Exception as eval_e: logger.warning(f"Could not retrieve available options: {eval_e}")
                       raise
                  logger.info(f"Select option success. Selected: {selected_values}")
                  action_result_details.update({"option_type": option_type, "option_value": option_value, "selected_actual_values": selected_values})
                  results.append({**step_result_base, "status": "success", **action_result_details})

            # --- scroll_to_element (変更なし) ---
            elif action == "scroll_to_element":
                   # ... (実装は省略) ...
                   if not target_element: raise PlaywrightError("Scroll failed: Element not located.")
                   logger.info("Scrolling element into view..."); await target_element.scroll_into_view_if_needed(timeout=action_wait_time); await asyncio.sleep(0.3); logger.info("Scroll successful.")
                   results.append({**step_result_base, "status": "success", **action_result_details})

            # --- press_key (変更なし) ---
            elif action == "press_key":
                 # ... (実装は省略) ...
                if not key_to_press or not isinstance(key_to_press, str): raise ValueError("Action 'press_key' requires 'value' (key name).")
                valid_keys = ["Enter", "ArrowUp", "ArrowDown", "ArrowRight", "ArrowLeft", "Tab", "Escape", "Backspace", "Delete", "Home", "End", "PageUp", "PageDown", "Shift", "Control", "Alt", "Meta", "CapsLock"]
                normalized_key = next((k for k in valid_keys if k.lower() == key_to_press.lower()), None)
                if not normalized_key:
                     if len(key_to_press) == 1: normalized_key = key_to_press
                     else: raise ValueError(f"Invalid key for 'press_key': {key_to_press}. Supported: {valid_keys}, or single chars.")
                logger.info(f"Pressing key '{normalized_key}' {press_count} times...")
                target_for_press: Union[Page, Locator] = page
                if target_element:
                    logger.info("Target element found, focusing..."); 
                    try: 
                        await target_element.focus(timeout=max(1000, action_wait_time // 4)); 
                        target_for_press = target_element
                    except Exception as focus_e: 
                        logger.warning(f"Focus failed: {focus_e}. Pressing on page.")
                else: logger.info("No target element. Pressing on page.")
                press_delay_ms = 50
                for k_idx in range(press_count):
                    await target_for_press.press(normalized_key, timeout=max(1000, action_wait_time // press_count if press_count > 0 else action_wait_time))
                    logger.debug(f"Pressed '{normalized_key}' ({k_idx + 1}/{press_count})")
                    if press_count > 1: await asyncio.sleep(press_delay_ms / 1000)
                logger.info(f"Key '{normalized_key}' pressed {press_count} times successfully.")
                action_result_details.update({"key_pressed": normalized_key, "count": press_count})
                results.append({**step_result_base, "status": "success", **action_result_details})

            # --- screenshot (変更なし) ---
            elif action == "screenshot":
                  # ... (実装は省略) ...
                  filename_base = str(value).strip() if value else f"screenshot_step{step_num}"
                  filename = f"{filename_base}.png" if not filename_base.lower().endswith(('.png', '.jpg', '.jpeg')) else filename_base
                  safe_filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
                  screenshot_path = Path(config.DEFAULT_SCREENSHOT_DIR) / safe_filename
                  screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                  logger.info(f"Saving screenshot to '{screenshot_path}'...")
                  ss_timeout = max(10000, action_wait_time)
                  if target_element: await target_element.screenshot(path=screenshot_path, timeout=ss_timeout); logger.info("Element screenshot saved.")
                  else: await root_page.screenshot(path=screenshot_path, full_page=True, timeout=ss_timeout*2); logger.info("Page screenshot saved.")
                  action_result_details["filename"] = str(screenshot_path.resolve())
                  results.append({**step_result_base, "status": "success", **action_result_details})

            # --- 未知のアクション (変更なし) ---
            else:
                 # ... (実装は省略) ...
                 known_actions = single_element_actions + multiple_elements_actions + ["switch_to_iframe", "switch_to_parent_frame", "wait_page_load", "sleep", "scroll_page_to_bottom", "screenshot"]
                 if action not in known_actions: logger.warning(f"Undefined action '{action}'. Skipping."); results.append({**step_result_base, "status": "skipped", "message": f"Undefined action: {action}"}); continue

            final_url = root_page.url

        # --- ステップごとのエラーハンドリング ---
        except (PlaywrightTimeoutError, PlaywrightError, ValueError, IndexError, Exception) as e:
            error_message = f"ステップ {step_num} ({action}) でエラー発生: {type(e).__name__} - {e}"
            logger.error(error_message, exc_info=True)
            error_screenshot_path = None
            current_url_on_error = "Unknown (Page might be closed)"
            try:
                if root_page and not root_page.is_closed(): current_url_on_error = root_page.url
                elif page and not page.is_closed(): current_url_on_error = page.url
            except Exception: pass
            if root_page and not root_page.is_closed():
                 timestamp = time.strftime("%Y%m%d_%H%M%S")
                 error_ss_filename = f"error_step{step_num}_{timestamp}.png"
                 error_ss_path = Path(config.DEFAULT_SCREENSHOT_DIR) / error_ss_filename
                 try:
                     error_ss_path.parent.mkdir(parents=True, exist_ok=True)
                     await root_page.screenshot(path=error_ss_path, full_page=True, timeout=15000)
                     error_screenshot_path = str(error_ss_path.resolve()); logger.info(f"Error screenshot saved: {error_ss_path.name}")
                 except Exception as ss_e: logger.error(f"Failed to save error screenshot: {ss_e}")
            error_details = { **step_result_base, "status": "error", "selector": selector, "target_hints": target_hints, "locator_attempts": locator_attempt_logs, "message": str(e), "full_error": error_message, "traceback": traceback.format_exc(), "url_on_error": current_url_on_error }
            if error_screenshot_path: error_details["error_screenshot"] = error_screenshot_path
            results.append(error_details)
            return False, results, current_url_on_error

    logger.info("All steps processed.")
    final_url = root_page.url if root_page and not root_page.is_closed() else final_url
    return True, results, final_url