# --- ファイル: utils.py (PDFエラー詳細化・結果追記機能追加版) ---
import json
import logging
import os
import sys
import asyncio
import time
import traceback
import fitz  # PyMuPDF
from playwright.async_api import APIRequestContext, TimeoutError as PlaywrightTimeoutError, Response # <<< Responseを追加
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urljoin

try:
    import config
except ImportError:
    class ConfigMock: # config.py がない場合のデフォルト
        LOG_FILE = 'output_web_runner.log'
        DEFAULT_ACTION_TIMEOUT = 10000
        PDF_DOWNLOAD_TIMEOUT = 60000
        MCP_SERVER_LOG_FILE = 'output/web_runner_mcp.log'
    config = ConfigMock()
    logging.warning("config.py が見つかりません。デフォルト値を使用します。")


logger = logging.getLogger(__name__)

# --- ロギング設定 ---
def setup_logging_for_standalone(log_file_path: str = config.LOG_FILE):
    """Web-Runner単体実行用のロギング設定を行います。"""
    log_level = logging.INFO
    root_logger = logging.getLogger()
    # 既存のハンドラをクリアして二重ログを防ぐ
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers = [console_handler]
    log_target = "Console"

    # ファイルハンドラ (エラー処理強化)
    file_handler = None
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True) # ディレクトリが存在しなければ作成
            print(f"DEBUG [utils]: Ensuring directory exists: '{log_dir}'") # デバッグ出力
        # ファイル書き込み権限チェック (より丁寧な方法)
        if os.path.exists(log_file_path):
            if not os.access(log_file_path, os.W_OK):
                 raise PermissionError(f"Write permission denied for log file: {log_file_path}")
        else:
             # ファイルが存在しない場合、ディレクトリに書き込み権限があるか
             if log_dir and not os.access(log_dir, os.W_OK):
                   raise PermissionError(f"Write permission denied for log directory: {log_dir}")

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        log_target += f" and File ('{log_file_path}')"
        print(f"DEBUG [utils]: FileHandler created for '{log_file_path}'")
    except PermissionError as pe:
         print(f"警告 [utils]: ログファイルへの書き込み権限がありません: {pe}", file=sys.stderr)
         # ファイルハンドラなしで続行
    except Exception as e:
        print(f"警告 [utils]: ログファイル '{log_file_path}' のハンドラ設定に失敗しました: {e}", file=sys.stderr)
        # ファイルハンドラなしで続行

    # ロギング基本設定
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True # 既存の設定を上書き
    )
    logging.getLogger('playwright').setLevel(logging.WARNING) # Playwrightのログレベル抑制
    current_logger = logging.getLogger(__name__)
    current_logger.info(f"Standalone logger setup complete. Level: {logging.getLevelName(log_level)}. Target: {log_target}")
    print(f"DEBUG [utils]: Logging setup finished. Root handlers: {logging.getLogger().handlers}")


# --- JSON入力読み込み ---
def load_input_from_json(filepath: str) -> Dict[str, Any]:
    """指定されたJSONファイルから入力データを読み込む。"""
    logger.info(f"入力ファイル '{filepath}' の読み込みを開始します...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # --- バリデーション強化 ---
        if not isinstance(data, dict):
            raise TypeError("JSONのルートがオブジェクトではありません。")
        if "target_url" not in data or not data["target_url"]:
            raise ValueError("JSONファイルに必須キー 'target_url' が存在しないか、値が空です。")
        if "actions" not in data or not isinstance(data["actions"], list):
            raise ValueError("JSONファイルに必須キー 'actions' が存在しないか、リスト形式ではありません。")
        if not data["actions"]:
             logger.warning(f"入力ファイル '{filepath}' の 'actions' リストが空です。実行するアクションがありません。")
        # --- ここまで ---
        logger.info(f"入力ファイル '{filepath}' を正常に読み込みました。")
        return data
    except FileNotFoundError:
        logger.error(f"入力ファイルが見つかりません: {filepath}")
        raise # エラーを再送出
    except json.JSONDecodeError as e:
        logger.error(f"JSON形式のエラーです ({filepath}): {e}")
        raise
    except (ValueError, TypeError) as ve: # TypeErrorも捕捉
        logger.error(f"入力データの形式が不正です ({filepath}): {ve}")
        raise
    except Exception as e:
        logger.error(f"入力ファイルの読み込み中に予期せぬエラーが発生しました ({filepath}): {e}", exc_info=True)
        raise


# --- PDFテキスト抽出 (エラー詳細化版) ---
def extract_text_from_pdf_sync(pdf_data: bytes) -> Optional[str]:
    """PDFのバイトデータからテキストを抽出する (同期的)。エラー時はエラーメッセージ文字列を返す。"""
    doc = None
    try:
        logger.info(f"PDFデータ (サイズ: {len(pdf_data)} bytes) からテキスト抽出を開始します...")
        if not pdf_data:
             logger.error("PDF処理エラー: 入力データが空です。")
             return "Error: Input PDF data is empty."

        # --- fitz.open を try-except で囲む ---
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            if not doc or len(doc) == 0: # ドキュメントが開けたか、ページがあるか確認
                 logger.error("PDF処理エラー: ドキュメントが開けません、またはページ数が0です。")
                 if doc: doc.close() # 開けていたら閉じる
                 return "Error: Cannot open PDF document or it has no pages."
            logger.info(f"PDFページ数: {len(doc)}")
        except Exception as open_err:
             # fitz が出す可能性のある様々なエラーを捕捉
             logger.error(f"fitz.open でエラーが発生しました: {type(open_err).__name__} - {open_err}", exc_info=True)
             return f"Error opening PDF: {type(open_err).__name__} - {open_err}"
        # --- ここまで ---

        text_parts = []
        for page_num in range(len(doc)):
            page_start_time = time.monotonic()
            try:
                page = doc.load_page(page_num)
                # --- page.get_text を try-except で囲む ---
                try:
                    page_text = page.get_text("text", sort=True) # テキストとして取得、ソート有効
                    if page_text:
                        text_parts.append(page_text.strip()) # 前後の空白を除去して追加
                except Exception as get_text_err:
                     logger.warning(f"ページ {page_num + 1} の get_text でエラー: {type(get_text_err).__name__} - {get_text_err}", exc_info=True)
                     text_parts.append(f"--- Error getting text from page {page_num + 1}: {type(get_text_err).__name__} ---")
                # --- ここまで ---
                page_elapsed = (time.monotonic() - page_start_time) * 1000
                logger.debug(f"ページ {page_num + 1} 処理完了 ({page_elapsed:.0f}ms)。")
            except Exception as page_e: # load_page 自体のエラー
                logger.warning(f"ページ {page_num + 1} のロード中にエラー: {page_e}", exc_info=True)
                text_parts.append(f"--- Error loading page {page_num + 1}: {page_e} ---")

        # 結合と整形
        full_text = "\n\n--- Page Separator ---\n\n".join(text_parts) # ページ区切りを分かりやすく
        # 空行を除去して再結合
        cleaned_text = '\n'.join([line.strip() for line in full_text.splitlines() if line.strip()])
        logger.info(f"PDFテキスト抽出完了。総文字数 (整形後): {len(cleaned_text)}")
        # テキストが全く抽出できなかった場合も考慮
        return cleaned_text if cleaned_text else "(No text extracted from PDF)"

    # --- fitz 固有のエラーも捕捉 ---
    except fitz.fitz.TryingToReadFromEmptyFileError: # fitz の specific error
         logger.error("PDF処理エラー: ファイルデータが空または破損しています。")
         return "Error: PDF data is empty or corrupted."
    except fitz.fitz.FileDataError as e: # fitz の specific error
         logger.error(f"PDF処理エラー (PyMuPDF FileDataError): {e}", exc_info=False)
         return f"Error: PDF file data error - {e}"
    except RuntimeError as e: # fitz が出す可能性のある他のランタイムエラー
        logger.error(f"PDF処理エラー (PyMuPDF RuntimeError): {e}", exc_info=True)
        return f"Error: PDF processing failed (PyMuPDF RuntimeError) - {e}"
    except Exception as e: # その他の予期せぬエラー
        logger.error(f"PDFテキスト抽出中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return f"Error: Unexpected error during PDF text extraction - {type(e).__name__}: {e}"
    finally:
        if doc:
            try: doc.close(); logger.debug("PDFドキュメントを閉じました。")
            except Exception as close_e: logger.warning(f"PDFドキュメントのクローズ中にエラーが発生しました (無視): {close_e}")


# --- PDFダウンロード (Responseオブジェクトを返す版) ---
async def download_pdf_async(api_request_context: APIRequestContext, url: str) -> Optional[Response]:
    """指定されたURLからファイルを非同期でダウンロードし、Responseオブジェクトを返す。失敗時はNoneを返す。"""
    logger.info(f"ファイルを非同期でダウンロード/アクセス中: {url} (Timeout: {config.PDF_DOWNLOAD_TIMEOUT}ms)")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        response = await api_request_context.get(url, headers=headers, timeout=config.PDF_DOWNLOAD_TIMEOUT, fail_on_status_code=False)

        if not response.ok:
            logger.error(f"ダウンロード失敗 ({url}) - Status: {response.status} {response.status_text}")
            try:
                error_body = await response.text(timeout=5000)
                logger.debug(f"エラーレスポンスボディ (一部): {error_body[:500]}")
            except Exception as body_err: logger.warning(f"エラーレスポンスボディの読み取り中にエラー: {body_err}")
            return None
        # ★★★ Content-Type のチェックは呼び出し元で行うため削除 ★★★
        # content_type = response.headers.get('content-type', '').lower()
        # if 'application/pdf' not in content_type:
        #     logger.warning(f"レスポンスのContent-TypeがPDFではありません ({url}): '{content_type}'。ダウンロードは続行しますが、後続処理で失敗する可能性があります。")
        # body = await response.body() # body の取得も呼び出し元で行う
        # if not body:
        #      logger.warning(f"PDFダウンロード成功 ({url}) Status: {response.status} ですが、レスポンスボディが空です。")
        #      return None

        logger.info(f"ダウンロード/アクセス成功 ({url}) - Status: {response.status}")
        return response # Response オブジェクトをそのまま返す

    except PlaywrightTimeoutError:
        logger.error(f"ダウンロード中にタイムアウトが発生しました ({url})。設定タイムアウト: {config.PDF_DOWNLOAD_TIMEOUT}ms")
        return None
    except Exception as e:
        logger.error(f"非同期ダウンロード中に予期せぬエラーが発生しました ({url}): {e}", exc_info=True)
        return None


# --- 結果ファイル書き込み ---
def write_results_to_file(
    results: List[Dict[str, Any]],
    filepath: str,
    final_summary_data: Optional[Dict[str, Any]] = None
):
    """実行結果を指定されたファイルに書き込む。"""
    logger.info(f"実行結果を '{filepath}' に書き込みます...")
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"出力ディレクトリ '{output_dir}' を作成しました。")

        with open(filepath, "w", encoding="utf-8") as file:
            file.write("--- Web Runner 実行結果 ---\n\n")
            if not results:
                 file.write("(実行ステップ結果がありません)\n")
            for i, res in enumerate(results):
                if not isinstance(res, dict):
                    file.write(f"--- Step {i+1}: Invalid Format ---\n")
                    file.write(f"Received non-dict data: {res}\n\n")
                    continue

                step_num = res.get('step', i + 1)
                action_type = res.get('action', 'Unknown')
                status = res.get('status', 'Unknown')
                selector = res.get('selector')

                file.write(f"--- Step {step_num}: {action_type} ({status}) ---\n")
                if res.get('memo'): file.write(f"Memo: {res.get('memo')}\n")
                if selector: file.write(f"Selector: {selector}\n")
                if res.get('iframe_selector'): file.write(f"IFrame Selector (for switch): {res.get('iframe_selector')}\n")
                if res.get('required_state'): file.write(f"Required Element State: {res.get('required_state')}\n")
                if res.get('locator_method'): file.write(f"Locator Method: {res.get('locator_method')}\n")
                if res.get('locator_hint_used'): file.write(f"Hint Used: {res.get('locator_hint_used')}\n")


                if status == "error":
                     file.write(f"Message: {res.get('message')}\n")
                     if res.get('full_error') and res.get('full_error') != res.get('message'): file.write(f"Details: {res.get('full_error')}\n")
                     if res.get('error_screenshot'): file.write(f"Screenshot: {res.get('error_screenshot')}\n")
                     if res.get('url_on_error'): file.write(f"URL on Error: {res.get('url_on_error')}\n")
                     if res.get('traceback'): file.write(f"Traceback:\n{res.get('traceback')}\n")
                elif status == "success":
                    # 共通キーを除いた詳細情報を取得
                    details_to_write = {k: v for k, v in res.items() if k not in [
                        'step', 'status', 'action', 'selector', 'iframe_selector',
                        'required_state', 'memo', 'locator_method', 'locator_hint_used'
                    ]}

                    # アクションタイプに応じた整形出力
                    if action_type == 'get_all_attributes':
                        attr_name = details_to_write.pop('attribute', 'N/A')
                        results_count = details_to_write.pop('results_count', 0)
                        url_list = details_to_write.pop('url_list', None)
                        pdf_texts = details_to_write.pop('pdf_texts', None)
                        scraped_texts = details_to_write.pop('scraped_texts', None)
                        email_list = details_to_write.pop('extracted_emails', None)
                        attr_list = details_to_write.pop('attribute_list', None)
                        file.write(f"Requested Attribute/Content: {attr_name}\n")
                        file.write(f"Results ({results_count} items processed):\n")
                        if results_count > 0:
                            # 各リストが存在する場合のみ出力
                            if email_list is not None:
                                file.write("  Extracted Emails (Unique Domains for this step):\n")
                                if email_list: file.write('\n'.join(f"  - {email}" for email in email_list) + "\n")
                                else: file.write("    (No unique domain emails found for this step)\n")
                            if url_list is not None and attr_name.lower() != 'mail':
                                file.write(f"  Processed URLs ({len(url_list)}):\n")
                                for idx, url in enumerate(url_list): file.write(f"    [{idx+1}] {url if url else '(URL not processed or invalid)'}\n")
                            if pdf_texts is not None:
                                file.write("  Extracted PDF Texts:\n")
                                for idx, pdf_content in enumerate(pdf_texts):
                                    file.write(f"    [{idx+1}]")
                                    if pdf_content is None: file.write(" (None)\n")
                                    elif isinstance(pdf_content, str) and pdf_content.startswith("Error:") or pdf_content.startswith("Warning:"): file.write(f" ({pdf_content})\n")
                                    elif pdf_content == "(No text extracted from PDF)": file.write(": (No text extracted)\n")
                                    else:
                                        file.write(f" (Length: {len(pdf_content or '')}):\n")
                                        # テキストをインデントして書き出す
                                        indented_content = "\n".join(["      " + line for line in str(pdf_content).splitlines()])
                                        file.write(indented_content + "\n")
                            if scraped_texts is not None:
                                file.write("  Scraped Page Texts:\n")
                                for idx, scraped_content in enumerate(scraped_texts):
                                    file.write(f"    [{idx+1}]")
                                    if scraped_content is None: file.write(" (None)\n")
                                    elif isinstance(scraped_content, str) and (scraped_content.startswith("Error") or scraped_content.startswith("Warning")): file.write(f" ({scraped_content})\n")
                                    else:
                                        file.write(f" (Length: {len(scraped_content or '')}):\n")
                                        indented_content = "\n".join(["      " + line for line in str(scraped_content).splitlines()])
                                        file.write(indented_content + "\n")
                            if attr_list is not None:
                                file.write(f"  Attribute '{attr_name}' Values:\n")
                                for idx, attr_content in enumerate(attr_list): file.write(f"    [{idx+1}] {attr_content}\n")
                        else: file.write("  (No items found matching the selector for this step)\n")

                    elif action_type == 'get_all_text_contents':
                        text_list_result = details_to_write.pop('text_list', [])
                        results_count = details_to_write.pop('results_count', len(text_list_result) if isinstance(text_list_result, list) else 0)
                        file.write(f"Result Text List ({results_count} items):\n")
                        if isinstance(text_list_result, list):
                            valid_texts = [str(text) for text in text_list_result if text is not None] # Noneを除外
                            if valid_texts: file.write('\n'.join(f"- {text}" for text in valid_texts) + "\n")
                            else: file.write("(No text content found)\n")
                        else: file.write("(Invalid format received for text_list)\n")

                    elif action_type in ['get_text_content', 'get_inner_text'] and 'text' in details_to_write:
                        file.write(f"Result Text:\n{details_to_write.pop('text', '')}\n")

                    elif action_type == 'get_inner_html' and 'html' in details_to_write:
                        file.write(f"Result HTML:\n{details_to_write.pop('html', '')}\n")

                    elif action_type == 'get_attribute':
                        attr_name = details_to_write.pop('attribute', ''); attr_value = details_to_write.pop('value', None)
                        file.write(f"Result Attribute ('{attr_name}'): {attr_value}\n")
                        if 'pdf_text' in details_to_write:
                            pdf_text = details_to_write.pop('pdf_text', '')
                            prefix = "Extracted PDF Text"
                            if isinstance(pdf_text, str) and (pdf_text.startswith("Error:") or pdf_text.startswith("Warning:")): file.write(f"{prefix}: {pdf_text}\n")
                            elif pdf_text == "(No text extracted from PDF)": file.write(f"{prefix}: (No text extracted)\n")
                            else: file.write(f"{prefix}:\n{pdf_text}\n")
                        if 'scraped_text' in details_to_write:
                            scraped_text = details_to_write.pop('scraped_text', '')
                            prefix = "Scraped Page Text"
                            if isinstance(scraped_text, str) and (scraped_text.startswith("Error:") or scraped_text.startswith("Warning:")): file.write(f"{prefix}: {scraped_text}\n")
                            else: file.write(f"{prefix}:\n{scraped_text}\n")

                    elif action_type == 'screenshot' and 'filename' in details_to_write:
                        file.write(f"Screenshot saved to: {details_to_write.pop('filename')}\n")

                    elif action_type == 'click' and 'new_page_opened' in details_to_write:
                        if details_to_write.get('new_page_opened'): file.write(f"New page opened: {details_to_write.get('new_page_url')}\n")
                        else: file.write("New page did not open within timeout.\n")
                        # 処理済みなので削除
                        details_to_write.pop('new_page_opened', None)
                        details_to_write.pop('new_page_url', None)

                    # その他の残った詳細を出力
                    if details_to_write:
                        file.write("Other Details:\n")
                        for key, val in details_to_write.items():
                            # 長すぎるリストや辞書は省略表示
                            if isinstance(val, (list, dict)) and len(str(val)) > 200:
                                file.write(f"  {key}: {type(val).__name__} with {len(val)} items (Content omitted)\n")
                            else:
                                file.write(f"  {key}: {val}\n")

                elif status == "skipped" or status == "warning":
                    file.write(f"Message: {res.get('message', 'No message provided.')}\n")
                else: # status が success/error/skipped/warning 以外の場合
                    file.write(f"Raw Data: {res}\n") # 不明な場合は生データを書き出す

                file.write("\n") # ステップ間の空行

            # --- 最終集約結果の追記処理 ---
            if final_summary_data:
                file.write("--- Final Aggregated Summary ---\n\n")
                for summary_key, summary_value in final_summary_data.items():
                    file.write(f"{summary_key}:\n")
                    if isinstance(summary_value, list):
                        try:
                             # JSON形式で見やすく出力
                             json_output = json.dumps(summary_value, indent=2, ensure_ascii=False)
                             file.write(json_output + "\n")
                        except TypeError:
                             # JSONシリアライズできない場合は repr で
                             file.write(repr(summary_value) + "\n")
                    else:
                        file.write(f"{summary_value}\n")
                file.write("\n")

        logger.info(f"結果の書き込みが完了しました: '{filepath}'")
    except IOError as e:
        logger.error(f"結果ファイル '{filepath}' の書き込み中にIOエラーが発生しました: {e}")
    except Exception as e:
        logger.error(f"結果の処理またはファイル書き込み中に予期せぬエラーが発生しました: {e}", exc_info=True)

# --- メールアドレスをファイルに追記する関数（オプション）---
def write_emails_to_file(emails: List[str], filepath: str):
    """メールアドレスのリストを指定されたファイルに追記する"""
    if not emails: return
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            for email in emails:
                f.write(email + '\n')
        logger.info(f"Appended {len(emails)} emails to {filepath}")
    except Exception as e:
        logger.error(f"Error writing emails to file {filepath}: {e}")


# --- utils.py のテスト用コード ---
if __name__ == "__main__":
    print("--- Testing logging setup from utils.py ---")
    # MCPサーバーログファイルパスをテストに使用
    TEST_LOG_FILE = config.MCP_SERVER_LOG_FILE
    print(f"Test log file path: {TEST_LOG_FILE}")
    # 既存ログファイル削除試行
    if os.path.exists(TEST_LOG_FILE):
        try:
            os.remove(TEST_LOG_FILE)
            print(f"Removed existing test log file: {TEST_LOG_FILE}")
        except Exception as e:
            print(f"Could not remove existing test log file: {e}")
    # ロギング設定実行とテストログ出力
    try:
        setup_logging_for_standalone(log_file_path=TEST_LOG_FILE)
        test_logger = logging.getLogger("utils_test")
        print("\nAttempting to log messages...")
        test_logger.info("INFO message from utils_test.")
        test_logger.warning("WARNING message from utils_test.")
        test_logger.error("ERROR message from utils_test.")
        print(f"\nLogging test complete.")
        print(f"Please check the console output above and the content of the file: {os.path.abspath(TEST_LOG_FILE)}")
        print(f"Root logger handlers: {logging.getLogger().handlers}")
    except Exception as e:
        print(f"\n--- Error during logging test ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # PDF抽出テスト (ダミーデータ)
    print("\n--- Testing PDF extraction ---")
    # ダミーのPDFバイトデータ (内容は不問、構造だけ模倣)
    dummy_pdf_bytes = b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n5 0 obj<</Length 44>>stream\nBT /F1 24 Tf 100 700 Td (Dummy PDF Text) Tj ET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000058 00000 n \n0000000113 00000 n \n0000000218 00000 n \n0000000271 00000 n \ntrailer <</Size 6/Root 1 0 R>>\nstartxref\n356\n%%EOF'
    extracted = extract_text_from_pdf_sync(dummy_pdf_bytes)
    print(f"Extracted text from dummy PDF: {extracted}")

    # エラーケースのテスト
    print("\n--- Testing PDF extraction with invalid data ---")
    extracted_err = extract_text_from_pdf_sync(b'invalid pdf data')
    print(f"Result with invalid data: {extracted_err}")

    # 書き込みテスト (ダミーデータ)
    print("\n--- Testing result file writing ---")
    dummy_results = [
        {"step": 1, "action": "click", "status": "success", "selector": "#button1"},
        {"step": 2, "action": "input", "status": "error", "message": "Element not found", "selector": "#text-input"}
    ]
    dummy_summary = {"Total Time": "10.5s"}
    test_output_file = "test_output_results.txt"
    write_results_to_file(dummy_results, test_output_file, dummy_summary)
    print(f"Test results written to {test_output_file}")
    # try: os.remove(test_output_file) # テスト後は削除
    # except OSError: pass