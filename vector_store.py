"""
テキストのベクトル化とベクトルデータベース (FAISS) への保存・読み込み・検索モジュール
"""
import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from config import OPENAI_API_KEY, GOOGLE_API_KEY, get_available_llm_provider
from document_loader import load_and_split_documents
from logger import get_logger, check_memory_usage

logger = get_logger()

DEFAULT_INDEX_DIR = "faiss_index"


def get_embedding_model() -> Embeddings:
    """
    設定されているAPIキーに応じて最適なEmbeddingモデルを返す
    """
    provider = get_available_llm_provider()

    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            logger.info("Embeddingモデルを初期化: OpenAI (text-embedding-3-small)")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY
            )
        elif provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info("Embeddingモデルを初期化: Google Gemini (models/gemini-embedding-001)")
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
        else:
            err_msg = "有効なAPIキーが設定されていません。.env に OPENAI_API_KEY または GOOGLE_API_KEY を設定してください。"
            logger.error(f"[API設定エラー] {err_msg}")
            raise ValueError(err_msg)
    except Exception as e:
        logger.error(f"[Embedding初期化エラー] Embeddingモデルの生成に失敗しました: {e}", exc_info=True)
        raise


def create_vector_store(
    documents: List[Document],
    save_path: str = DEFAULT_INDEX_DIR
) -> FAISS:
    """
    DocumentリストからFAISSベクトルデータベースを作成し、ローカルに保存する

    Args:
        documents (List[Document]): ベクトル化対象のチャンクDocumentリスト
        save_path (str): インデックス保存先ディレクトリ

    Returns:
        FAISS: 作成されたベクトルストアオブジェクト
    """
    if not documents:
        logger.error("[ベクトルストア作成失敗] ベクトル化するドキュメントが空です。")
        raise ValueError("ベクトル化するドキュメントが空です。")

    check_memory_usage(threshold_percent=80.0, context="ベクトルストア作成前")
    logger.info(f"FAISSベクトルストアの構築開始: {len(documents)} 件のチャンクをベクトル化中...")

    try:
        embeddings = get_embedding_model()
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        
        # ローカルディレクトリに保存
        vectorstore.save_local(save_path)
        logger.info(f"FAISSインデックスを正常に保存しました: path={save_path}")
        check_memory_usage(threshold_percent=85.0, context="ベクトルストア作成後")
        return vectorstore
    except MemoryError as e:
        logger.critical(f"[メモリ不足エラー] ベクトル化中にメモリが不足しました (MemoryError): {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[外部API/ベクトル化エラー] FAISSインデックスの作成・保存に失敗しました: {e}", exc_info=True)
        raise


def load_vector_store(
    save_path: str = DEFAULT_INDEX_DIR,
    allow_dangerous_deserialization: bool = True
) -> Optional[FAISS]:
    """
    保存されたFAISSインデックスをローカルから読み込む

    Args:
        save_path (str): インデックス保存先ディレクトリ
        allow_dangerous_deserialization (bool): Pickleのデシリアライズを許可するか

    Returns:
        Optional[FAISS]: 読み込まれたベクトルストア（存在しない場合はNone）
    """
    if not os.path.exists(save_path):
        logger.debug(f"インデックスディレクトリが存在しません: {save_path}")
        return None

    try:
        logger.info(f"FAISSインデックスの読み込みを開始: path={save_path}")
        embeddings = get_embedding_model()
        vectorstore = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        logger.info(f"FAISSインデックスの読み込みに成功しました: path={save_path}")
        return vectorstore
    except Exception as e:
        logger.error(f"[インデックス読込エラー] FAISSインデックスの読み込みに失敗しました ({save_path}): {e}", exc_info=True)
        return None


def build_or_load_vector_store(
    data_source_path: str = "data",
    save_path: str = DEFAULT_INDEX_DIR,
    force_rebuild: bool = False
) -> FAISS:
    """
    既存のインデックスがあれば読み込み、なければドキュメントから新規作成する

    Args:
        data_source_path (str): 資料ディレクトリまたはファイルパス
        save_path (str): インデックス保存先
        force_rebuild (bool): 強制的に再構築するか

    Returns:
        FAISS: ベクトルストアオブジェクト
    """
    if not force_rebuild:
        vectorstore = load_vector_store(save_path)
        if vectorstore is not None:
            logger.info(f"既存のインデックス '{save_path}' を再利用します。")
            return vectorstore

    logger.info(f"'{data_source_path}' から資料を読み込んでインデックスを新規構築します...")
    chunks = load_and_split_documents(data_source_path)
    return create_vector_store(chunks, save_path)



def search_similar_documents(
    query: str,
    vectorstore: FAISS,
    k: int = 3
) -> List[Tuple[Document, float]]:
    """
    クエリに意味的に類似するドキュメントを検索する

    Args:
        query (str): 検索クエリ（例: "フェルミ準位の定義とは？"）
        vectorstore (FAISS): FAISSベクトルストア
        k (int): 取得する件数

    Returns:
        List[Tuple[Document, float]]: (Document, 類似度スコア/距離) のリスト
    """
    return vectorstore.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    print("=== Step 3: テキストのベクトル化とデータベース保存 (FAISS) のテスト ===")
    
    # 1. ベクトルストアの作成・保存
    vs = build_or_load_vector_store(data_source_path="data", force_rebuild=True)

    # 2. セマンティック検索のテスト
    test_queries = [
        "シリコンとガリウムヒ素のバンドギャップの違いと特徴は？",
        "マクスウェル方程式の第3式は何を表している？"
    ]

    for query in test_queries:
        print(f"\n==========================================")
        print(f"[Query] {query}")
        print(f"==========================================")
        results = search_similar_documents(query, vs, k=2)
        for i, (doc, score) in enumerate(results):
            print(f"\n[Result #{i+1}] (Score/Distance: {score:.4f})")
            print(f"File: {doc.metadata.get('source_name')} (Chunk ID: {doc.metadata.get('chunk_id')})")
            print(f"Content excerpt:\n{doc.page_content[:180]}...\n")
