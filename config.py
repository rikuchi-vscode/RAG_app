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

def get_available_llm_provider() -> str:
    """
    設定されているAPIキーに基づいて利用可能なLLMプロバイダを判定して返す
    戻り値: 'openai', 'gemini', または 'none'
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your_"):
        return "openai"
    elif GOOGLE_API_KEY and not GOOGLE_API_KEY.startswith("your_"):
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
