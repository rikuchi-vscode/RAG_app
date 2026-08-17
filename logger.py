"""
システムログ管理モジュール
- アプリケーションのライフサイクル（起動・停止・セッション開始）の記録
- 外部API通信エラー、ファイル読み込み・解析エラーの記録
- メモリ使用量監視および OOM / クラッシュ（未捕捉例外）の追跡
- ログローテーション（RotatingFileHandler）
- 機密情報（APIキー等）のサニタイズ（マスキング）
- 前回プロセスの異常終了（OOM / 強制終了）の事後検知
"""
import os
import sys
import json
import time
import atexit
import signal
import logging
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


LOGS_DIR = "logs"
SYSTEM_LOG_FILE = os.path.join(LOGS_DIR, "system.log")
STATUS_FILE = os.path.join(LOGS_DIR, ".app_status.json")

# ロガー名定数
LOGGER_NAME = "rag_system"

# 機密情報検出・マスキング用キーワード
SENSITIVE_ENV_KEYS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "ANTHROPIC_API_KEY",
    "API_KEY",
    "SECRET"
]


import re

class SensitiveDataFilter(logging.Filter):
    """
    ログメッセージに含まれる可能性のあるAPIキーなどの機密情報を自動マスキングするフィルター
    """
    # 一般的なAPIキーパターンの正規表現
    API_KEY_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),          # OpenAI / Claude
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),          # Google / Gemini
        re.compile(r"tvly-[a-zA-Z0-9_\-]{20,}"),       # Tavily
    ]

    def __init__(self):
        super().__init__()
        # 環境変数からマスク対象の秘密文字列を収集
        self.secret_values = set()
        for key in SENSITIVE_ENV_KEYS:
            val = os.getenv(key, "").strip()
            if val and len(val) >= 8 and not val.startswith("your_"):
                self.secret_values.add(val)

    def filter(self, record: logging.LogRecord) -> bool:
        if not isinstance(record.msg, str):
            return True

        msg = record.msg
        # 環境変数の秘密値をマスキング
        for secret in self.secret_values:
            if secret in msg:
                masked = secret[:4] + "*" * max(4, len(secret) - 8) + secret[-4:] if len(secret) > 8 else "********"
                msg = msg.replace(secret, masked)

        # パターンマッチによるマスキング
        for pat in self.API_KEY_PATTERNS:
            msg = pat.sub(lambda m: m.group(0)[:4] + "********" + m.group(0)[-4:], msg)

        record.msg = msg
        return True



def _record_process_status(status: str, extra: Optional[Dict[str, Any]] = None):
    """プロセスの状態（RUNNING / STOPPED / CRASHED）をステータスファイルに記録"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    data = {
        "pid": os.getpid(),
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
    }
    if extra:
        data.update(extra)
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _check_previous_crash(logger: logging.Logger):
    """前回のプロセスが正常終了したか検証し、異常終了（OOM/強制終了など）の場合はログに記録"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                prev_status = json.load(f)
            prev_pid = prev_status.get("pid")
            prev_state = prev_status.get("status")
            prev_time = prev_status.get("timestamp")

            if prev_state == "RUNNING" and prev_pid != os.getpid():
                logger.warning(
                    f"[Crash Recovery] 前回のプロセス (PID: {prev_pid}, 起動時刻: {prev_time}) "
                    f"は正常終了処理 (atexit) を通らずに終了しました。"
                    f"メモリ不足 (OOM Killer) または OS/タスクによる強制終了の可能性があります。"
                )
        except Exception as e:
            logger.debug(f"ステータスファイルの読み込みに失敗しました: {e}")


def get_memory_info() -> Dict[str, Any]:
    """現在のシステムのメモリ使用状況および自プロセスのメモリ使用量を取得"""
    if not PSUTIL_AVAILABLE:
        return {"available": False}

    try:
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        proc_mem = process.memory_info()
        return {
            "available": True,
            "total_mb": round(vm.total / (1024 * 1024), 1),
            "available_mb": round(vm.available / (1024 * 1024), 1),
            "used_mb": round(vm.used / (1024 * 1024), 1),
            "percent": vm.percent,
            "process_rss_mb": round(proc_mem.rss / (1024 * 1024), 1),
        }
    except Exception:
        return {"available": False}


def check_memory_usage(threshold_percent: float = 85.0, context: str = "") -> Dict[str, Any]:
    """
    メモリ使用率をチェックし、閾値を超えている場合は警告・クリティカルログを出力する
    
    Args:
        threshold_percent (float): 警告を出すRAM使用率の閾値（%）
        context (str): チェック時の処理コンテキスト（例: "PDF分割処理前", "FAISS作成中"）
    """
    mem_info = get_memory_info()
    if not mem_info.get("available"):
        return mem_info

    logger = get_logger()
    percent = mem_info["percent"]
    proc_mb = mem_info["process_rss_mb"]
    avail_mb = mem_info["available_mb"]
    ctx_str = f" [{context}]" if context else ""

    if percent >= 95.0:
        logger.critical(
            f"危険なメモリ逼迫を検出{ctx_str}: システムRAM使用率 {percent}% (空き: {avail_mb} MB, 本プロセス: {proc_mb} MB) - OOMクラッシュの危険があります"
        )
    elif percent >= threshold_percent:
        logger.warning(
            f"高メモリ使用率を検出{ctx_str}: システムRAM使用率 {percent}% (空き: {avail_mb} MB, 本プロセス: {proc_mb} MB)"
        )

    return mem_info


_logger_initialized = False


def setup_system_logger(
    log_file_path: str = SYSTEM_LOG_FILE,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 5,
    level: int = logging.INFO
) -> logging.Logger:
    """
    システムロガーの初期化・設定
    - ローテーションファイルハンドラの設定
    - コンソール出力ハンドラの設定
    - 未捕捉例外フック（クラッシュ捕捉）の設定
    - シャットダウンフック（atexit/signal）の設定
    - 前回クラッシュ検出
    """
    global _logger_initialized
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # 既にハンドラが設定されている場合は再初期化をスキップ
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [PID:%(process)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        sensitive_filter = SensitiveDataFilter()

        # 1. ローテーションファイルハンドラ
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        logger.addHandler(file_handler)

        # 2. 標準出力（コンソール）ハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        logger.addHandler(console_handler)

    if not _logger_initialized:
        _logger_initialized = True

        # 前回クラッシュ検知
        _check_previous_crash(logger)

        # アプリ起動状態をステータスファイルに記録
        mem = get_memory_info()
        _record_process_status("RUNNING", extra={"memory_at_start": mem})

        logger.info(
            f"=== システムロガーが初期化されました (PID: {os.getpid()}, Python: {sys.version.split()[0]}) ==="
        )
        if mem.get("available"):
            logger.info(
                f"システムメモリ情報: 全体 {mem['total_mb']} MB, 空き {mem['available_mb']} MB (使用率: {mem['percent']}%)"
            )

        # 未処理例外ハンドラ（グローバルクラッシュ捕捉）
        def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical(
                "【未捕捉例外によるクラッシュ】想定外のエラーが発生してプロセスが停止しました:",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            _record_process_status("CRASHED", extra={"error": str(exc_value)})

        sys.excepthook = handle_uncaught_exception

        # スレッド内の未処理例外ハンドラ（Python 3.8+）
        if hasattr(threading, "excepthook"):
            def handle_thread_exception(args):
                if issubclass(args.exc_type, KeyboardInterrupt):
                    return
                logger.critical(
                    f"【未捕捉スレッド例外】スレッド '{args.thread.name}' で例外が発生しました:",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
                )
            threading.excepthook = handle_thread_exception

        # 正常終了時（atexit）のハンドラ
        def on_exit():
            mem_end = get_memory_info()
            logger.info(
                f"=== アプリケーション終了処理を実行しました (Graceful Shutdown, PID: {os.getpid()}) ==="
            )
            _record_process_status("STOPPED", extra={"memory_at_stop": mem_end})

        atexit.register(on_exit)

        # シグナルハンドラ（SIGINT / SIGTERM）
        def handle_signal(sig, frame):
            sig_name = "SIGINT" if sig == signal.SIGINT else f"Signal {sig}"
            logger.info(f"シグナルを受信しました ({sig_name})。終了処理を開始します。")
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, handle_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, handle_signal)
        except (ValueError, AttributeError):
            # メインスレッド以外での実行時など
            pass

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    初期化済みロガーを取得する（未初期化の場合は自動初期化）
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_system_logger()
    return logger


if __name__ == "__main__":
    test_logger = setup_system_logger()
    test_logger.info("ロガーの動作確認テスト (INFO)")
    test_logger.warning("警告テスト (WARNING)")
    test_logger.error("エラーテスト (ERROR)")
    check_memory_usage(threshold_percent=10.0, context="テスト実行")
