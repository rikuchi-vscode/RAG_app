"""
環境設定およびAPIキーの管理モジュール
"""
import os
from dotenv import load_dotenv

# .env ファイルの読み込み
load_dotenv()

# APIキーの取得
def _get_api_key(key: str) -> str:
    """
    環境変数（.env含む）または Streamlit Secrets (st.secrets) からAPIキーを安全に取得する
    """
    # 1. 環境変数の確認
    val = os.getenv(key, "").strip()
    if val and not val.startswith("your_"):
        return val

    # 2. Streamlit secrets の確認
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = str(st.secrets[key]).strip()
            if secret_val and not secret_val.startswith("your_"):
                # ライブラリ内部のos.environ参照にも対応できるように同期
                os.environ[key] = secret_val
                return secret_val
    except Exception:
        pass

    return val

OPENAI_API_KEY = _get_api_key("OPENAI_API_KEY")
GOOGLE_API_KEY = _get_api_key("GOOGLE_API_KEY")

def get_openai_api_key() -> str:
    """最新のOpenAI APIキーを取得する"""
    global OPENAI_API_KEY
    val = _get_api_key("OPENAI_API_KEY")
    if val:
        OPENAI_API_KEY = val
    return OPENAI_API_KEY

def get_google_api_key() -> str:
    """最新のGoogle APIキーを取得する"""
    global GOOGLE_API_KEY
    val = _get_api_key("GOOGLE_API_KEY")
    if val:
        GOOGLE_API_KEY = val
    return GOOGLE_API_KEY

def set_custom_api_key(key_name: str, key_value: str):
    """ブラウザ上などで手動入力されたAPIキーを動的に設定・反映する"""
    global OPENAI_API_KEY, GOOGLE_API_KEY
    clean_val = key_value.strip()
    os.environ[key_name] = clean_val
    if key_name == "OPENAI_API_KEY":
        OPENAI_API_KEY = clean_val
    elif key_name == "GOOGLE_API_KEY":
        GOOGLE_API_KEY = clean_val

def get_available_llm_provider() -> str:
    """
    設定されているAPIキーに基づいて利用可能なLLMプロバイダを判定して返す
    戻り値: 'openai', 'gemini', または 'none'
    """
    o_key = get_openai_api_key()
    g_key = get_google_api_key()
    if o_key and not o_key.startswith("your_"):
        return "openai"
    elif g_key and not g_key.startswith("your_"):
        return "gemini"
    return "none"

def validate_environment():
    """
    環境設定が正しく行われているかを検証する
    """
    provider = get_available_llm_provider()
    if provider == "none":
        print("[WARNING] 有効なAPIキー (OPENAI_API_KEY または GOOGLE_API_KEY) が .env または st.secrets に設定されていません。")
    else:
        print(f"[OK] LLMプロバイダ '{provider}' のAPIキーが検出されました。")
    return provider

if __name__ == "__main__":
    validate_environment()
