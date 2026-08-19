"""
RAG（検索拡張生成）ロジックおよびLangChainチェーン構築モジュール
LCEL (LangChain Expression Language) を用いて最新かつ安定したRAGチェーンを実装
"""
import os
import time
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    get_openai_api_key,
    get_google_api_key,
    get_available_llm_provider
)
from vector_store import build_or_load_vector_store, FAISS
from logger import get_logger

logger = get_logger()


GEMINI_CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
]


def get_llm(model_name: Optional[str] = None, temperature: float = 0.2) -> BaseChatModel:
    """
    設定されている環境変数に基づいてChatModel (LLM) を初期化して返す
    """
    provider = get_available_llm_provider()

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            selected_model = model_name or "gpt-4o-mini"
            logger.debug(f"OpenAI LLMを初期化: model={selected_model}, temp={temperature}")
            return ChatOpenAI(
                model=selected_model,
                temperature=temperature,
                openai_api_key=get_openai_api_key(),
                max_retries=3
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            selected_model = model_name or GEMINI_CANDIDATE_MODELS[0]
            logger.debug(f"Gemini LLMを初期化: model={selected_model}, temp={temperature}")
            return ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=temperature,
                google_api_key=get_google_api_key(),
                max_retries=2
            )
        else:
            err_msg = "有効なAPIキーが設定されていません。.env に OPENAI_API_KEY または GOOGLE_API_KEY を設定してください。"
            logger.error(f"[API設定エラー] {err_msg}")
            raise ValueError(err_msg)
    except Exception as e:
        logger.error(f"[LLM初期化エラー] LLMインスタンスの生成に失敗しました: {e}", exc_info=True)
        raise


def get_rag_prompt() -> ChatPromptTemplate:
    """
    専門領域特化型Q&Aボット用のシステムプロンプトテンプレートを作成する
    """
    system_prompt = (
        "あなたは半導体工学や電磁気学などの専門講義資料・論文を熟知した専門チューターAIです。\n"
        "以下の【提供された講義資料・コンテキスト】に記載されている内容のみに基づいて、質問に対して正確・論理的・分かりやすく回答してください。\n\n"
        "【回答時のガイドライン】\n"
        "1. 提供されたコンテキストに記載がない事項については、憶測で答えず「提供された講義資料には該当する記載がありません」と明記してください。\n"
        "2. 数式や物理法則が含まれる場合は、分かりやすくLaTeX表記（例: $E_g$, $\\nabla \\cdot \\mathbf{{E}}$）を交えて解説してください。\n"
        "3. 学習者・受講生の試験対策や理解促進につながるよう、重要ポイントを箇条書きなどで整理して説明してください。\n\n"
        "【提供された講義資料・コンテキスト】\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    検索されたDocumentリストをプロンプト埋め込み用の文字列にフォーマットする
    """
    formatted_chunks = []
    for doc in docs:
        source_name = doc.metadata.get("source_name", "不明")
        chunk_id = doc.metadata.get("chunk_id", "-")
        formatted_chunks.append(
            f"--- [資料: {source_name} (Chunk: {chunk_id})] ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def create_rag_chain(vectorstore: Optional[FAISS] = None, k: int = 3, llm: Optional[BaseChatModel] = None):
    """
    検索機 (Retriever) と LLM を統合した LCEL ベースの RAG チェーンを作成する

    Args:
        vectorstore (Optional[FAISS]): ベクトルストア（Noneの場合は自動ロード）
        k (int): 検索で取得する類似ドキュメント数
        llm (Optional[BaseChatModel]): 使用するLLMインスタンス

    Returns:
        tuple: (rag_chain, retriever)
    """
    if vectorstore is None:
        vectorstore = build_or_load_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    if llm is None:
        llm = get_llm()
    prompt = get_rag_prompt()

    # ドキュメントを渡して回答を生成するサブチェーン
    generation_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_for_prompt(x["context"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # 1. 質問に基づいてRetrieverから関連文書を抽出
    # 2. 抽出文書と質問を結合してLLMに渡し回答を生成
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "input": RunnablePassthrough()
        })
        .assign(answer=generation_chain)
    )

    return rag_chain, retriever


def query_rag(
    question: str,
    rag_chain=None,
    vectorstore: Optional[FAISS] = None
) -> Dict[str, Any]:
    """
    質問を受け取り、RAGチェーンを実行して回答と参照ソースを返す（モデルフォールバック機能付き）

    Args:
        question (str): ユーザーの質問
        rag_chain: 既存のRAGチェーン（Noneの場合は新規作成）
        vectorstore (Optional[FAISS]): ベクトルストア

    Returns:
        Dict[str, Any]: {
            "answer": str (回答テキスト),
            "sources": List[Dict] (参照ドキュメント情報)
        }
    """
    provider = get_available_llm_provider()
    logger.info(f"RAGクエリ実行開始: '{question[:60]}...' (Provider: {provider})")
    
    if rag_chain is not None:
        try:
            response = rag_chain.invoke(question)
            logger.info("既存のRAGチェーンで回答生成に成功しました。")
            return _format_response(response)
        except Exception as e:
            logger.warning(f"[外部APIエラー] 既存RAGチェーンの実行に失敗したため、モデルフォールバックを開始します: {e}")

    # ベクトルストアの準備
    if vectorstore is None:
        vectorstore = build_or_load_vector_store()

    models_to_try = GEMINI_CANDIDATE_MODELS if provider == "gemini" else ["gpt-4o-mini"]
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(f"モデル '{model_name}' を用いてRAG推論を実行中...")
            current_llm = get_llm(model_name=model_name)
            chain, _ = create_rag_chain(vectorstore=vectorstore, llm=current_llm)
            response = chain.invoke(question)
            logger.info(f"モデル '{model_name}' による回答生成が完了しました。")
            return _format_response(response)
        except Exception as e:
            last_error = e
            logger.warning(
                f"[外部API通信エラー] モデル '{model_name}' の呼び出しに失敗しました: {e}。次のモデル候補を試行します。"
            )
            time.sleep(1)
            continue

    logger.error(f"[RAG推論致命的エラー] すべてのモデル候補での推論が失敗しました: {last_error}", exc_info=True)
    raise last_error



def _format_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAG実行結果を統一フォーマットに整形する
    """
    answer_text = response.get("answer", "")
    context_docs: List[Document] = response.get("context", [])
    sources = []
    for doc in context_docs:
        sources.append({
            "source_name": doc.metadata.get("source_name", "不明"),
            "chunk_id": doc.metadata.get("chunk_id", None),
            "content": doc.page_content
        })

    return {
        "answer": answer_text,
        "sources": sources
    }


if __name__ == "__main__":
    print("=== Step 4: RAGロジックの実装と回答生成テスト ===")
    
    # テスト質問 1: 半導体工学
    q1 = "シリコン（Si）とガリウムヒ素（GaAs）のバンドギャップの違いと、それぞれの用途への適性を説明してください。"
    print(f"\n==========================================")
    print(f"[Q1] {q1}")
    print(f"==========================================")
    result1 = query_rag(q1)
    print(f"\n[AIの回答]\n{result1['answer']}\n")
    print(f"[参照ソース: {len(result1['sources'])} 件]")
    for s in result1['sources']:
        print(f"- ファイル: {s['source_name']} (Chunk {s['chunk_id']})")

    # テスト質問 2: 電磁気学
    q2 = "マクスウェル方程式の第3式（ファラデーの電磁誘導の法則）の物理的な意味を教えてください。"
    print(f"\n==========================================")
    print(f"[Q2] {q2}")
    print(f"==========================================")
    result2 = query_rag(q2)
    print(f"\n[AIの回答]\n{result2['answer']}\n")
    print(f"[参照ソース: {len(result2['sources'])} 件]")
    for s in result2['sources']:
        print(f"- ファイル: {s['source_name']} (Chunk {s['chunk_id']})")
