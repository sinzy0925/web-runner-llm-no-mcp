# -*- coding: utf-8 -*-
# --- ファイル: web_runner_llm_text_batch_runner.py (テキスト入力・連続バッチ処理版) ---

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
import argparse # コマンドライン引数用

# --- 必要なモジュールをインポート ---
# LLM関連
try:
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
    import playwright_actions
except ImportError as e:
    logging.critical(f"必須モジュール(playwright_launcher, config, utils, playwright_actions)が見つかりません: {e.name}")
    print(f"Error: Missing required module - {e.name}")
    exit()

# Playwright オブジェクトの型ヒント
from playwright.async_api import Playwright, Browser, BrowserContext, Page

# --- 設定 ---
SCREENSHOT_BASE_DIR = Path(config.DEFAULT_SCREENSHOT_DIR)
DEFAULT_INPUT_TEXT_DIR = Path("./text_inputs") # ★ デフォルトの入力テキストディレクトリ ★
OUTPUT_SUMMARY_FILE = Path("./web_runner_llm_a_text_batch_output.txt")
OUTPUT_JSON_DIR     = Path("./output_generated_json_for_batch")
OUTPUT_DETAILS_DIR  = Path("./output_result")
STEALTH_MODE        =True # False

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
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
root_logger.addHandler(file_handler); root_logger.addHandler(stream_handler); root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

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
def save_detailed_result(filename_prefix: str, success: bool, data: Union[str, List[Dict], Dict], output_dir: Path):
    """ケースごとの詳細結果やエラー情報をファイルに保存する"""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_details.json" if success else "_error.json"
    safe_prefix = re.sub(r'[\\/*?:"<>|]', "", filename_prefix)
    filename = output_dir / f"{safe_prefix}{suffix}"
    try:
        content_to_write = ""
        if isinstance(data, (list, dict)):
            try:
                content_to_write = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"結果/エラーのJSON変換失敗 ({filename_prefix}): {e}。文字列保存。")
                content_to_write = str(data)
                filename = filename.with_suffix(".txt")
        elif isinstance(data, str):
             content_to_write = f"Unexpected String Data:\n\n{data}"
             filename = filename.with_suffix(".txt")
        else:
            content_to_write = f"Unexpected Data Format Received ({type(data)}):\n\n{str(data)}"
            filename = filename.with_suffix(".txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content_to_write)
        logger.info(f"詳細結果/エラー情報を保存: {filename.name}")
    except Exception as e:
        logger.error(f"詳細結果/エラー情報のファイル保存エラー ({filename}): {e}")

# --- テキストファイルからテストケースを読み込む関数 ---
def load_test_case_from_text(file_path: Path) -> Optional[Dict[str, Any]]:
    """テキストファイルからURLと指示リストを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()] # 空行を除外

        if len(lines) < 2:
            logger.warning(f"ファイル形式不正 (URLと指示が必要): {file_path}")
            return None

        start_url = lines[0]
        instructions = lines[1:]

        # URLの簡易バリデーション (http/httpsで始まるか)
        if not start_url.startswith(("http://", "https://")):
            logger.warning(f"無効なURL形式です: {start_url} (ファイル: {file_path})")
            return None

        case_name = file_path.stem # ファイル名をケース名とする
        logger.info(f"テストケース読み込み: {case_name} (URL: {start_url}, 指示: {len(instructions)}件)")
        return {"case_name": case_name, "start_url": start_url, "steps": instructions}

    except FileNotFoundError:
        logger.error(f"テストケースファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        logger.error(f"テストケースファイル読み込みエラー ({file_path}): {e}", exc_info=True)
        return None


# --- メインのバッチ処理関数 (Playwright直接制御・テキスト入力対応版) ---
async def run_text_batch_tests(input_dir: Path):
    logger.info(f"--- テキストファイルバッチテスト開始 (Playwright直接制御, 入力ディレクトリ: {input_dir}) ---")
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DETAILS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 入力テキストファイルリスト取得
    if not input_dir.is_dir():
        logger.critical(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        return

    test_files = sorted([f for f in input_dir.glob("*.txt") if f.is_file()])
    if not test_files:
        logger.warning(f"入力ディレクトリ '{input_dir}' に処理対象の .txt ファイルが見つかりません。")
        return

    logger.info(f"{len(test_files)} 件のテストケースファイルが見つかりました。")

    # --- Playwright 起動 (全テストケースで共有) ---
    playwright: Optional[Playwright] = None
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None
    try:
        logger.info("Playwrightを起動します (全テストケースで共有)...")
        headless_mode = False # ブラウザ表示
        slow_motion_ms = 50   # 遅延
        playwright, browser = await playwright_launcher.launch_browser(
            headless_mode=headless_mode, slow_motion=slow_motion_ms
        )
        context, page = await playwright_launcher.create_context_and_page(
            browser,
            default_timeout=config.DEFAULT_ACTION_TIMEOUT,
            apply_stealth=STEALTH_MODE
        )
        logger.info("Playwrightの起動と共有ページ作成が完了しました。")
    except Exception as e:
        logger.critical(f"Playwrightの起動に失敗しました: {e}", exc_info=True)
        await playwright_launcher.close_browser(playwright, browser, context)
        return

    # --- 全体的なtry...finallyでPlaywrightの終了を保証 ---
    try:
        # 2. 各テストケースファイルを処理
        for i, txt_file in enumerate(test_files):
            logger.info("\n" + "=" * 40)
            logger.info(f"[{i+1}/{len(test_files)}] 処理開始: {txt_file.name}")

            # 2a. テキストファイルからテストケース情報を読み込む
            case_data = load_test_case_from_text(txt_file)
            if not case_data:
                logger.error(f"テストケース '{txt_file.name}' の読み込みに失敗。スキップします。")
                append_summary_result(txt_file.stem, f"テスト結果: {txt_file.stem},結果:ERROR - ファイル読み込み失敗,N/A")
                continue

            case_name = case_data["case_name"]
            start_url = case_data["start_url"]
            steps_instructions = case_data["steps"] # ここは文字列のリストのはず

            # --- ケース開始時の状態 ---
            current_url: str = start_url
            case_success = True
            last_step_result_summary = "ケース未実行"
            last_screenshot_filename = None
            step_results_for_case: List[Dict] = [] # ケース内の全ステップ結果

            try:
                # --- 最初のページへ移動 ---
                logger.info(f"テストケース '{case_name}': 最初のページに移動: {start_url}")
                await page.goto(start_url, wait_until="load", timeout=60000)
                current_url = page.url # 移動後の実際のURLを取得
                logger.info(f"移動完了。現在のURL: {current_url}")

                # --- ケース内のステップを順番に処理 ---
                for j, instruction in enumerate(steps_instructions): # 指示リストを直接ループ
                    step_num = j + 1

                    if not instruction: # 指示が空文字列の場合もスキップ
                        logger.warning(f"ステップ {step_num}: 指示が空です。スキップ。")
                        continue
                    if not current_url:
                        logger.error(f"ステップ {step_num}: URL不明。続行不可。")
                        last_step_result_summary = f"ERROR - ステップ{step_num} URL不明"
                        case_success = False
                        break # ケース失敗としてループ中断

                    # --- リトライロジック開始 ---
                    max_retries = 1 # 最大リトライ回数 (1 = 合計2回試行)
                    attempt = 0
                    step_success = False # このステップが最終的に成功したか
                    step_result_details: Union[List[Dict], Dict] = [] # このステップの最終的な結果

                    while attempt <= max_retries:
                        current_attempt_success = False # この試行が成功したか
                        current_attempt_result: Union[List[Dict], Dict] = [] # この試行の結果
                        action_obj : Optional[Dict[str, Any]] = None # 生成されたアクションJSON

                        try:
                            logger.info(f"--- ステップ {step_num} (試行 {attempt + 1}/{max_retries + 1}): {case_name} ---")
                            logger.info(f"  指示: {instruction}")

                            # 2a. HTML取得 & クリーンアップ
                            logger.info(f"ステップ{step_num} (試行 {attempt + 1}): 現在のページHTMLを取得中 (URL: {current_url})...")
                            current_html = await page.content()
                            if not current_html: raise RuntimeError("現在のページのHTML取得失敗")
                            logger.info(f"HTML取得成功 (Length: {len(current_html)})")
                            if DO_CLEANUP:
                                 current_html_for_llm = await asyncio.to_thread(cleanup_html, current_html)
                                 logger.info(f"クリーンアップ後HTMLサイズ: {len(current_html_for_llm)}")
                            else:
                                 current_html_for_llm = current_html
                                 logger.info("HTMLクリーンアップ skip")

                            # 2b. LLMでJSON生成
                            logger.info(f"ステップ{step_num} (試行 {attempt + 1}): LLMでアクションJSON生成中...")
                            target_hints, fallback_selector, action_details = await asyncio.to_thread(
                                get_element_info_list_with_fallback, current_html_for_llm, instruction
                            )
                            if not action_details or action_details.get("action_type") == "unknown":
                                raise ValueError(f"ステップ{step_num}: LLMアクションタイプ特定失敗")

                            action_obj = {
                                "memo": instruction, "action": action_details.get("action_type"),
                                "selector": fallback_selector, "target_hints": target_hints if target_hints is not None else [],
                            }
                            for key in ["value", "attribute_name", "option_type", "option_value"]:
                                if action_details.get(key) is not None: action_obj[key] = action_details[key]

                            # 生成JSON保存
                            try:
                                json_filename = f"{case_name}_step{step_num}_attempt{attempt+1}_generated.json"; output_path = OUTPUT_JSON_DIR / json_filename
                                with open(output_path, "w", encoding="utf-8") as f: json.dump(action_obj, f, indent=2, ensure_ascii=False)
                                logger.info(f"生成アクションJSON保存: {output_path.name}")
                            except Exception as e_save: logger.error(f"生成JSON保存エラー: {e_save}")

                            # 2c. アクションを直接実行
                            logger.info(f"ステップ{step_num} (試行 {attempt + 1}): Web-Runnerアクション '{action_obj['action']}' を直接実行...")
                            current_attempt_success, current_attempt_result, final_url = await playwright_actions.execute_actions_async(
                                page=page, actions=[action_obj], api_request_context=context.request, default_timeout=config.DEFAULT_ACTION_TIMEOUT
                            )

                            # URL更新
                            if final_url != current_url:
                                 logger.info(f"URLが変更されました: {current_url} -> {final_url}")
                                 current_url = final_url
                            else:
                                 logger.info(f"URLは変更されませんでした: {current_url}")

                            # 2d. ステップ結果の解析 (この試行の結果)
                            if current_attempt_success:
                                logger.info(f"ステップ {step_num} (試行 {attempt + 1}) 成功.")
                                step_success = True
                                step_result_details = current_attempt_result
                                break # ★★★ 成功したらリトライループを抜ける ★★★
                            else:
                                error_msg = "Playwrightアクション実行失敗"
                                if current_attempt_result and isinstance(current_attempt_result, list) and len(current_attempt_result) > 0:
                                    error_msg = current_attempt_result[0].get('message', error_msg)
                                elif isinstance(current_attempt_result, dict):
                                     error_msg = current_attempt_result.get('message', error_msg)
                                logger.warning(f"ステップ {step_num} (試行 {attempt + 1}) アクション失敗: {error_msg}。")
                                step_result_details = current_attempt_result
                                # ★ リトライ条件 (Playwright失敗) をチェック ★
                                if attempt < max_retries:
                                     logger.warning("リトライします...")
                                else:
                                     logger.error("最大リトライ回数に達しました。")
                                     # step_success は False のまま
                                     break # リトライ上限なのでループを抜ける

                        except (ValueError, RuntimeError, Exception) as step_e:
                            is_llm_error = isinstance(step_e, ValueError) and "LLMアクションタイプ特定失敗" in str(step_e)
                            if is_llm_error and attempt < max_retries:
                                logger.warning(f"ステップ {step_num} (試行 {attempt + 1}) でLLM生成エラー発生: {step_e}。リトライします...")
                                step_result_details = [{"step": step_num, "status": "error", "message": f"Attempt {attempt+1} failed: {step_e}", "traceback": traceback.format_exc()}]
                            else:
                                logger.error(f"ステップ {step_num} (試行 {attempt + 1}) で致命的エラー発生、またはリトライ上限到達: {step_e}", exc_info=True)
                                last_step_result_summary = f"ERROR - ステップ{step_num} 内部エラー: {str(step_e)[:50]}"
                                case_success = False
                                step_result_details = [{"step": step_num, "status": "error", "message": f"Fatal error or max retries reached on attempt {attempt+1}: {step_e}", "traceback": traceback.format_exc()}]
                                break # ★★★ リトライループを抜ける ★★★

                        # --- リトライ処理 ---
                        attempt += 1
                        if attempt <= max_retries:
                            logger.info(f"ステップ {step_num}: リトライ待機 (1秒)...")
                            await asyncio.sleep(1)

                    # --- while ループ終了後 (リトライループ後) ---
                    if not step_success:
                         logger.error(f"ステップ {step_num}: 最大リトライ回数 ({max_retries + 1}回) を試行しましたが成功しませんでした。")
                         action_name_log = action_obj['action'] if action_obj else "N/A"
                         error_msg_log = "Unknown error"
                         if step_result_details and isinstance(step_result_details, list) and len(step_result_details) > 0:
                              error_msg_log = step_result_details[0].get("message", error_msg_log)
                         elif isinstance(step_result_details, dict):
                              error_msg_log = step_result_details.get("message", error_msg_log)

                         last_step_result_summary = f"ERROR - ステップ{step_num}({action_name_log}) リトライ失敗: {str(error_msg_log)[:50]}"
                         case_success = False
                         save_detailed_result(f"{case_name}_step{step_num}", False, step_result_details, OUTPUT_DETAILS_DIR)
                         break # ★★★ ステップループを中断 ★★★
                    else:
                         # last_step_result_summary はループ内で更新済み
                         save_detailed_result(f"{case_name}_step{step_num}", True, step_result_details, OUTPUT_DETAILS_DIR)
                         if isinstance(step_result_details, list): step_results_for_case.extend(step_result_details)
                         elif isinstance(step_result_details, dict): step_results_for_case.append(step_result_details)

                         # スクショファイル名の更新
                         ss_result = next((res for res in step_result_details if res.get("action") == "screenshot"), None) if isinstance(step_result_details, list) else None
                         if ss_result and ss_result.get("status") == "success":
                             ss_path = ss_result.get("filename")
                             if ss_path: last_screenshot_filename = Path(ss_path).name

                    if not case_success: break # ケース失敗が確定したらステップループを抜ける
                    await asyncio.sleep(0.5)

                # --- ステップループ終了後 ---

                # --- ケース全体の最後にスクリーンショットとSleep ---
                if case_success:
                    logger.info("ケース最終状態のスクリーンショットと待機を実行します...")
                    try:
                        screenshot_filename_base = f"{case_name}_final_screenshot"
                        safe_ss_base = re.sub(r'[\\/*?:"<>|]', "_", screenshot_filename_base)
                        ss_action = {"action": "screenshot", "value": safe_ss_base}
                        sl_action = {"action": "sleep", "value": 2}
                        ss_success, ss_results, _ = await playwright_actions.execute_actions_async(
                            page, [ss_action, sl_action], context.request, config.DEFAULT_ACTION_TIMEOUT * 2
                        )
                        if isinstance(ss_results, list): step_results_for_case.extend(ss_results)
                        elif isinstance(ss_results, dict): step_results_for_case.append(ss_results)
                        if ss_success and isinstance(ss_results, list) and len(ss_results) > 0 and ss_results[0].get("status") == "success":
                             ss_path = ss_results[0].get("filename")
                             if ss_path: last_screenshot_filename = Path(ss_path).name
                        if last_screenshot_filename: logger.info(f"最終スクリーンショット保存完了: {last_screenshot_filename}")
                        else: logger.warning("最終スクリーンショットの取得またはSleepに失敗しました。")
                    except Exception as e_ss: logger.error(f"最終スクリーンショット取得/Sleep失敗 ({case_name}): {e_ss}")

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
                final_summary_line = f"テスト結果: {case_name},結果:{last_step_result_summary},{last_screenshot_filename if last_screenshot_filename else 'N/A'}"
            append_summary_result(case_name, final_summary_line)

            # ★ ケースごとの詳細結果全体もファイルに保存 ★
            save_detailed_result(f"{case_name}_all_steps", case_success, step_results_for_case, OUTPUT_DETAILS_DIR)

            await asyncio.sleep(1) # ケース間に待機

        # --- 全テストケースファイル処理終了後 ---
        logger.info(f"--- 全テストケース処理完了 ---")

    # --- Playwright 終了処理 ---
    finally:
        print('ow')
        await playwright_launcher.close_browser(playwright, browser, context)


# --- 実行ブロック ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Web Runner LLM batch tests from text files.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_TEXT_DIR,
        help=f"Directory containing the input text files (default: {DEFAULT_INPUT_TEXT_DIR})"
    )
    args = parser.parse_args()

    # ライブラリ存在チェック
    try:
        import anyio; import playwright; import google.generativeai; import bs4
    except ImportError as e:
        logger.critical(f"必須ライブラリ '{e.name}' が見つかりません。pip install で導入してください。")
        exit()

    try:
        anyio.run(run_text_batch_tests, args.input_dir) # 引数を渡す
    except Exception as e:
        logger.critical(f"バッチ処理の開始に失敗しました: {e}", exc_info=True)