# -*- coding: utf-8 -*-
# --- ファイル: batch_test_runner.py (アプローチ2: 直接制御・完全コード) ---

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os
import re
import shutil # ファイル削除用

# --- 必要なモジュールをインポート ---
# LLM関連
try:
    # ファイル名を generate_action_json_from_llm に修正
    from generate_action_json_from_llm import get_element_info_list_with_fallback
except ImportError:
    logging.error("generate_action_json_from_llm.py またはLLM関数が見つかりません。")
    # ダミー関数で続行できるようにする（エラーは出るが起動はする）
    async def get_element_info_list_with_fallback(*args, **kwargs):
        logging.warning("ダミーのLLM関数を使用します。")
        return None, None, None # 戻り値のタプルの要素数を合わせる

# HTML処理関連
try:
    from html_processor import cleanup_html, DO_CLEANUP
except ImportError:
    logging.error("html_processor.py が見つかりません。HTMLクリーンアップはスキップされます。")
    def cleanup_html(html_content: str) -> str:
        logging.warning("ダミーのHTMLクリーンアップ関数を使用します。")
        return html_content
    DO_CLEANUP = False

# Playwright関連
try:
    import playwright_launcher
    import config
    import utils
    # ★★★ playwright_actions を直接インポート ★★★
    import playwright_actions
except ImportError as e:
    logging.critical(f"必須モジュール(playwright_launcher, config, utils, playwright_actions)が見つかりません: {e.name}")
    print(f"Error: Missing required module - {e.name}")
    exit()

# MCPクライアントコアは不要

# Playwright オブジェクトの型ヒント
from playwright.async_api import Playwright, Browser, BrowserContext, Page

# --- 設定 ---
SCREENSHOT_BASE_DIR = Path(config.DEFAULT_SCREENSHOT_DIR)
INPUT_JSON_FILE     = Path("./web_runner_llm_batch_test_runner.json") # ★入力ファイル名を修正★
OUTPUT_SUMMARY_FILE = Path("./web_runner_llm_batch_output.txt")
OUTPUT_JSON_DIR     = Path("./output_generated_json_for_batch")
OUTPUT_DETAILS_DIR  = Path("./output_result")

# --- ログ・ファイル削除 ---
def delete_directory_contents(directory_path):
  """指定されたディレクトリ内のファイルやサブディレクトリを削除する（ディレクトリ自体は残す）"""
  path = Path(directory_path)
  if not path.is_dir():
      print(f"ディレクトリ '{path}' が存在しないか、ディレクトリではありません。")
      return
  try:
    for item in path.iterdir():
      try:
        if item.is_file() or item.is_link(): item.unlink()
        elif item.is_dir(): shutil.rmtree(item)
      except Exception as e: print(f"  削除失敗: {item}. Error: {e}")
    print(f"ディレクトリ '{path}' 内クリア完了。")
  except Exception as e: print(f"ディレクトリクリアエラー ({path}): {e}")

print("--- 出力ディレクトリのクリア ---")
delete_directory_contents(OUTPUT_JSON_DIR)
delete_directory_contents(OUTPUT_DETAILS_DIR)
if OUTPUT_SUMMARY_FILE.exists():
    try: OUTPUT_SUMMARY_FILE.unlink(); print(f"既存サマリファイル削除: {OUTPUT_SUMMARY_FILE}")
    except Exception as e: print(f"サマリファイル削除失敗: {e}")
print("-----------------------------")


# --- ロギング設定 ---
log_file = "batch_runner.log"; log_dir = Path("./logs"); log_dir.mkdir(exist_ok=True); log_path = log_dir / log_file
log_format = '%(asctime)s - [%(levelname)s] [%(name)s:%(lineno)d] %(message)s'
file_handler = logging.FileHandler(log_path, encoding='utf-8'); file_handler.setFormatter(logging.Formatter(log_format))
stream_handler = logging.StreamHandler(); stream_handler.setFormatter(logging.Formatter(log_format))
root_logger = logging.getLogger();
# 既存のハンドラをクリア (二重ログ防止)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(file_handler); root_logger.addHandler(stream_handler); root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__) # このファイル用のロガー

# --- HTML取得関数 ---
async def get_actual_html_for_batch(url: str) -> Optional[str]:
    """PlaywrightでHTMLを取得し、クリーンアップする (バッチ用)"""
    logger.info(f"HTML取得開始: {url}")
    html_content = None # 初期化
    try:
        get_html_action = [{"action": "get_inner_html", "selector": "html", "memo": "Get full HTML for batch"}]
        # タイムアウト設定: config.pyの値があればそれを使い、なければデフォルトを設定
        default_timeout = getattr(config, 'DEFAULT_ACTION_TIMEOUT', 10000)
        fetch_timeout = max(60000, default_timeout * 3) # 最低60秒、またはデフォルトの3倍

        success, results = await playwright_launcher.run_playwright_automation_async(
            target_url=url,
            actions=get_html_action,
            headless_mode=False, # ★ヘッドレスOFFで実行★
            slow_motion=0,
            default_timeout=fetch_timeout
        )
        if success and results and isinstance(results[0].get("html"), str):
            html_content = results[0]["html"]
            logger.info(f"HTML取得成功 (URL: {url}, Length: {len(html_content)})")
            if DO_CLEANUP:
                # クリーンアップはCPU負荷がかかる可能性があるので別スレッドで実行
                cleaned_html = await asyncio.to_thread(cleanup_html, html_content)
                logger.info(f"クリーンアップ後のHTMLサイズ: {len(cleaned_html)}")
                return cleaned_html
            else:
                logger.info("HTMLクリーンアップはスキップされました。")
                return html_content
        else:
            logger.error(f"HTML取得失敗 (URL: {url}). Result: {results}")
            return None # 失敗時は None を返す
    except Exception as e:
        logger.error(f"HTML取得中に予期せぬエラー (URL: {url}): {e}", exc_info=True)
        return None # エラー時も None を返す


# --- 結果出力関数 (サマリファイル用) ---
def append_summary_result(case_name: str, result_line: str):
    """整形済みの結果行をサマリ結果ファイルに追記する"""
    try:
        with open(OUTPUT_SUMMARY_FILE, "a", encoding="utf-8") as f:
            f.write(result_line + "\n")
        logger.info(f"サマリ結果追記: {OUTPUT_SUMMARY_FILE.name} <- {case_name}")
    except Exception as e:
        logger.error(f"サマリ結果ファイルへの書き込みエラー ({case_name}): {e}")

# --- 詳細結果ファイル保存関数 ---
def save_detailed_result(filename_prefix: str, success: bool, data: Union[str, List[Dict], Dict], output_dir: Path): # dataの型ヒント修正
    """ケースごとの詳細結果やエラー情報をファイルに保存する"""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_details.json" if success else "_error.json"
    safe_prefix = re.sub(r'[\\/*?:"<>|]', "", filename_prefix) # ファイル名に使えない文字を除去
    filename = output_dir / f"{safe_prefix}{suffix}"
    try:
        content_to_write = ""
        # ★★★ Web-Runner の結果は通常 List[Dict] か、エラー時に Dict ★★★
        if isinstance(data, (list, dict)):
            try:
                content_to_write = json.dumps(data, indent=2, ensure_ascii=False)
                # 拡張子は .json のまま
            except Exception as e:
                logger.warning(f"結果/エラーのJSON変換失敗 ({filename_prefix}): {e}。文字列保存。")
                content_to_write = str(data)
                filename = filename.with_suffix(".txt")
        elif isinstance(data, str): # MCPを使っていないので文字列が返ることは基本ないはずだが念のため
             content_to_write = f"Unexpected String Data:\n\n{data}"
             filename = filename.with_suffix(".txt")
        else: # その他の予期しない形式
            content_to_write = f"Unexpected Data Format Received ({type(data)}):\n\n{str(data)}"
            filename = filename.with_suffix(".txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content_to_write)
        logger.info(f"詳細結果/エラー情報を保存: {filename.name}")
    except Exception as e:
        logger.error(f"詳細結果/エラー情報のファイル保存エラー ({filename}): {e}")


# --- メインのバッチ処理関数 (Playwright直接制御・完全版) ---
async def run_batch_tests():
    logger.info(f"--- バッチテスト開始 (入力: {INPUT_JSON_FILE}) ---")
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True); SCREENSHOT_BASE_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DETAILS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 入力JSON読み込み
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f: test_cases: List[Dict[str, Any]] = json.load(f)
        logger.info(f"{len(test_cases)} 件のテストケースを読み込みました。")
    except Exception as e: logger.critical(f"入力ファイルエラー: {e}", exc_info=True); return

    # --- Playwright 起動 ---
    playwright: Optional[Playwright] = None
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None
    try:
        logger.info("Playwrightを起動します...")
        playwright, browser = await playwright_launcher.launch_browser(headless_mode=False, slow_motion=50)
        context, page = await playwright_launcher.create_context_and_page(browser)
        logger.info("Playwrightの起動とページ作成が完了しました。")
    except Exception as e:
        logger.critical(f"Playwrightの起動に失敗しました: {e}", exc_info=True)
        await playwright_launcher.close_browser(playwright, browser, context)
        return

    # 2. 各テストケースを処理
    for i, case in enumerate(test_cases):
        case_name = case.get("case_name", f"UnnamedCase_{i+1}")
        start_url = case.get("start_url")
        steps_instructions = case.get("steps")
        logger.info("\n" + "=" * 40); logger.info(f"[{i+1}/{len(test_cases)}] 実行中: {case_name}")
        logger.info(f"  開始URL: {start_url}")

        if not start_url or not steps_instructions or not isinstance(steps_instructions, list):
            logger.error("テストケース形式不正。スキップ。")
            append_summary_result(case_name, f"テスト結果: {case_name},結果:ERROR - テストケース形式不正,N/A")
            continue

        # --- ケース開始時の状態 ---
        current_url: str = start_url # 必ず文字列
        case_success = True
        last_step_result_summary = "ケース未実行"
        last_screenshot_filename = None
        step_results_for_case: List[Dict] = [] # ケース内の全ステップ結果

        try:
            # --- 最初のページへ移動 ---
            logger.info(f"最初のページに移動: {start_url}")
            await page.goto(start_url, wait_until="load", timeout=60000)
            current_url = page.url # 移動後の実際のURLを取得
            logger.info(f"移動完了。現在のURL: {current_url}")

            # --- ケース内のステップを順番に処理 ---
            for j, step_info in enumerate(steps_instructions):
                step_num = j + 1
                instruction = step_info.get("instruction")
                logger.info(f"--- ステップ {step_num}/{len(steps_instructions)}: {case_name} ---")
                logger.info(f"  指示: {instruction}")

                if not instruction: logger.error("ステップ指示なし。スキップ。"); last_step_result_summary = f"ERROR - ステップ{step_num} 指示なし"; case_success = False; break
                if not current_url: logger.error("URL不明。続行不可。"); last_step_result_summary = f"ERROR - ステップ{step_num} URL不明"; case_success = False; break

                step_success = False # 各ステップの成功フラグ
                step_result_details: List[Dict] = [{"step": step_num, "status": "error", "message": "Step init failed"}] # デフォルトはエラー
                action_obj : Optional[Dict[str, Any]] = None # 生成されたアクションJSON

                try:
                    # 2a. 現在のページのHTML取得 & クリーンアップ
                    logger.info(f"ステップ{step_num}: 現在のページHTMLを取得中 (URL: {current_url})...")
                    current_html = await page.content() # ★ page オブジェクトから直接取得 ★
                    if not current_html: raise RuntimeError("現在のページのHTML取得失敗")
                    logger.info(f"HTML取得成功 (Length: {len(current_html)})")
                    if DO_CLEANUP:
                         current_html_for_llm = await asyncio.to_thread(cleanup_html, current_html)
                         logger.info(f"クリーンアップ後HTMLサイズ: {len(current_html_for_llm)}")
                    else:
                         current_html_for_llm = current_html
                         logger.info("HTMLクリーンアップ skip")

                    # 2b. LLMでJSON生成
                    logger.info(f"ステップ{step_num}: LLMでアクションJSON生成中...")
                    target_hints, fallback_selector, action_details = await asyncio.to_thread(
                        get_element_info_list_with_fallback, current_html_for_llm, instruction
                    )
                    if not action_details or action_details.get("action_type") == "unknown": raise ValueError(f"ステップ{step_num}: LLMアクションタイプ特定失敗")
                    action_obj = { "memo": instruction, "action": action_details.get("action_type"), "selector": fallback_selector, "target_hints": target_hints if target_hints is not None else [], }
                    for key in ["value", "attribute_name", "option_type", "option_value"]:
                        if action_details.get(key) is not None: action_obj[key] = action_details[key]

                    # 生成JSON保存
                    try:
                        json_filename = f"{case_name}_step{step_num}_generated.json"; output_path = OUTPUT_JSON_DIR / json_filename
                        with open(output_path, "w", encoding="utf-8") as f: json.dump(action_obj, f, indent=2, ensure_ascii=False)
                        logger.info(f"生成アクションJSON保存: {output_path.name}")
                    except Exception as e_save: logger.error(f"生成JSON保存エラー: {e_save}")

                    # 2c. アクションを直接実行
                    logger.info(f"ステップ{step_num}: Web-Runnerアクション '{action_obj['action']}' を直接実行...")
                    # ★★★ playwright_actions.execute_actions_async を呼び出す ★★★
                    step_success, step_result_details, final_url = await playwright_actions.execute_actions_async(
                        page=page, # 現在のページオブジェクト
                        actions=[action_obj], # このステップのアクションのみ
                        api_request_context=context.request, # コンテキストから取得
                        default_timeout=config.DEFAULT_ACTION_TIMEOUT # configから取得
                    )
                    # ★★★ URLを更新 ★★★
                    if final_url != current_url:
                         logger.info(f"URLが変更されました: {current_url} -> {final_url}")
                         current_url = final_url # 実行後のURLで更新
                    else:
                         logger.info(f"URLは変更されませんでした: {current_url}")


                    # 2d. ステップ結果の解析
                    if step_success and step_result_details:
                        logger.info(f"ステップ {step_num} 成功.")
                        last_step_result_summary = f"ステップ{step_num}({action_obj['action']}): 成功"
                        # 状態更新のための previous_action_type は不要になった
                    elif not step_success and step_result_details:
                        action_result = step_result_details[0]
                        logger.error(f"ステップ {step_num} アクション失敗: {action_result.get('message')}")
                        last_step_result_summary = f"ERROR - ステップ{step_num}({action_obj['action']}) 失敗: {str(action_result.get('message'))[:50]}"
                        case_success = False; break # ケース失敗としてループ中断
                    else: # 予期しない結果
                        raise RuntimeError("execute_actions_async から予期しない結果")

                # --- ステップ内のエラーハンドリング ---
                except (ValueError, RuntimeError, TypeError, Exception) as step_e:
                     logger.error(f"ステップ {step_num} 処理中にエラー: {step_e}", exc_info=True)
                     last_step_result_summary = f"ERROR - ステップ{step_num} 内部エラー: {str(step_e)[:50]}"
                     case_success = False
                     # エラー時の詳細情報も保存
                     save_detailed_result(f"{case_name}_step{step_num}", False, {"error": str(step_e), "traceback": traceback.format_exc()}, OUTPUT_DETAILS_DIR)
                     break # エラー発生時はケース中断

                # 各ステップの詳細結果保存
                save_detailed_result(f"{case_name}_step{step_num}", step_success, step_result_details, OUTPUT_DETAILS_DIR)
                step_results_for_case.extend(step_result_details) # ケースの結果リストに追加

                await asyncio.sleep(0.5) # ステップ間に待機

            # --- ケース内のループ終了後 ---

            # --- ケース全体の最後にスクリーンショットとSleep ---
            if case_success: # ケースがエラーなく完了した場合のみ
                logger.info("ケース最終状態のスクリーンショットと待機を実行します...")
                try:
                    screenshot_filename_base = f"{case_name}_final_screenshot"
                    safe_ss_base = re.sub(r'[\\/*?:"<>|]', "_", screenshot_filename_base)
                    ss_action = {"action": "screenshot", "value": safe_ss_base}
                    sl_action = {"action": "sleep", "value": 2}
                    # ★ スクショとSleepも同じページオブジェクトに対して実行 ★
                    ss_success, ss_results, _ = await playwright_actions.execute_actions_async(
                        page, [ss_action, sl_action], context.request, config.DEFAULT_ACTION_TIMEOUT * 2
                    )
                    step_results_for_case.extend(ss_results) # スクショとSleepの結果も全体結果に追加
                    if ss_success and len(ss_results) > 0 and ss_results[0].get("status") == "success":
                        ss_path = ss_results[0].get("filename")
                        if ss_path: last_screenshot_filename = Path(ss_path).name
                        logger.info(f"最終スクリーンショット保存完了: {last_screenshot_filename}")
                    else:
                        logger.warning("最終スクリーンショットの取得またはSleepに失敗しました。")
                except Exception as e_ss:
                    logger.error(f"最終スクリーンショット取得/Sleep失敗 ({case_name}): {e_ss}")

        # --- ケース全体のエラーハンドリング ---
        except (ValueError, RuntimeError, TypeError, Exception) as case_e:
             logger.error(f"ケース {case_name} 処理中に予期せぬエラー: {case_e}", exc_info=True)
             last_step_result_summary = f"ERROR - ケース処理エラー: {str(case_e)[:50]}"
             case_success = False
             save_detailed_result(f"{case_name}_case_error", False, {"error": str(case_e), "traceback": traceback.format_exc()}, OUTPUT_DETAILS_DIR)


        # --- ケース終了後のサマリ出力 ---
        final_summary_line = ""
        if case_success:
            final_summary_line = f"テスト結果: {case_name},結果:成功 ({last_step_result_summary}),{last_screenshot_filename if last_screenshot_filename else 'N/A'}"
        else:
            # 失敗した場合、最後のステップのサマリ結果を使う
            final_summary_line = f"テスト結果: {case_name},結果:{last_step_result_summary},{last_screenshot_filename if last_screenshot_filename else 'N/A'}"
        append_summary_result(case_name, final_summary_line)

        # ★ ケースごとの詳細結果全体もファイルに保存 ★
        save_detailed_result(f"{case_name}_all_steps", case_success, step_results_for_case, OUTPUT_DETAILS_DIR)

        await asyncio.sleep(1) # ケース間に待機

    # --- 全テストケース終了後 ---
    logger.info(f"--- バッチテスト完了 ---")

    # --- Playwright 終了処理 ---
    await playwright_launcher.close_browser(playwright, browser, context)


# --- 実行ブロック ---
if __name__ == "__main__":
    # ライブラリ存在チェック
    try:
        import anyio; import playwright; import google.generativeai; import bs4
    except ImportError as e:
        logger.critical(f"必須ライブラリ '{e.name}' が見つかりません。pip install で導入してください。")
        exit()
    # anyio.run() を使ってメインの非同期関数を実行
    try:
        anyio.run(run_batch_tests)
    except Exception as e:
        logger.critical(f"バッチ処理の開始に失敗しました: {e}", exc_info=True)