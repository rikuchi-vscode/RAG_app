"""
環境設定およびAPIキーの管理モジュール
"""
import os
from dotenv import load_dotenv

# .env ファイルの読み込み
load_dotenv()

# APIキーの取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

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
        print("[WARNING] 有効なAPIキー (OPENAI_API_KEY または GOOGLE_API_KEY) が .env に設定されていません。")
    else:
        print(f"[OK] LLMプロバイダ '{provider}' のAPIキーが検出されました。")
    return provider

if __name__ == "__main__":
    validate_environment()
