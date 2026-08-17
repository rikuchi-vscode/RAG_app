"""
資料の読み込みとテキスト分割 (Chunking) を行うモジュール
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import get_logger, check_memory_usage

logger = get_logger()


def load_pdf(file_path: str) -> List[Document]:
    """
    指定されたPDFファイルを読み込み、Documentオブジェクトのリストを返す

    Args:
        file_path (str): PDFファイルのパス

    Returns:
        List[Document]: ページごとのDocumentリスト
    """
    if not os.path.exists(file_path):
        logger.error(f"[ファイル読込失敗] 指定されたPDFファイルが存在しません: {file_path}")
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    logger.info(f"PDFファイルの読み込みを開始: {file_path} (サイズ: {os.path.getsize(file_path)} bytes)")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"PDFファイルの読み込み完了: {file_path} (ページ数: {len(docs)})")
        return docs
    except Exception as e:
        logger.error(f"[ファイル読込エラー] PDFファイルの読み込みに失敗しました: {file_path} (エラー: {e})", exc_info=True)
        raise


def load_document(file_path: str) -> List[Document]:
    """
    拡張子に応じて適切なローダーを選択し、ファイルを読み込む
    （PDF, TXT, MD に対応）

    Args:
        file_path (str): ドキュメントファイルのパス

    Returns:
        List[Document]: 読み込まれたDocumentリスト
    """
    if not os.path.exists(file_path):
        logger.error(f"[ファイル読込失敗] ファイルが存在しません: {file_path}")
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8での読み込みに失敗したため、shift_jis/cp932でリトライします: {file_path}")
                loader = TextLoader(file_path, encoding="cp932")
                docs = loader.load()
        else:
            err_msg = f"未対応のファイル形式です: {ext} (対応形式: .pdf, .txt, .md)"
            logger.warning(f"[非対応フォーマット] {file_path} - {err_msg}")
            raise ValueError(err_msg)

        if ext != ".txt" and ext != ".md":
            docs = loader.load()

        # ソースファイル名をメタデータに正規化して保持
        for doc in docs:
            doc.metadata["source_name"] = os.path.basename(file_path)
        logger.info(f"ドキュメント読み込み成功: {file_path} ({len(docs)} 件)")
        return docs
    except Exception as e:
        if not isinstance(e, ValueError):
            logger.error(f"[ファイル読込エラー] ドキュメントの読み込み・解析に失敗しました ({file_path}): {e}", exc_info=True)
        raise


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    指定ディレクトリ内の対応ドキュメント (.pdf, .txt, .md) をすべて読み込む

    Args:
        directory_path (str): ディレクトリパス

    Returns:
        List[Document]: 読み込まれた全Documentのリスト
    """
    if not os.path.exists(directory_path):
        logger.error(f"[ディレクトリ読込失敗] ディレクトリが見つかりません: {directory_path}")
        raise FileNotFoundError(f"ディレクトリが見つかりません: {directory_path}")

    logger.info(f"ディレクトリからのドキュメント一括読み込みを開始: {directory_path}")
    check_memory_usage(threshold_percent=80.0, context="ディレクトリ読込前")
    supported_extensions = {".pdf", ".txt", ".md"}
    all_documents = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                try:
                    docs = load_document(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"[ドキュメント読込警告] スキップされたファイル ({file_path}): {e}")

    logger.info(f"ディレクトリ読み込み完了: {directory_path} (総ドキュメント数: {len(all_documents)})")
    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    読み込んだDocumentリストを意味のあるまとまり（チャンク）に分割する

    Args:
        documents (List[Document]): 分割前のDocumentリスト
        chunk_size (int): 1チャンクあたりの最大文字数（デフォルト: 800）
        chunk_overlap (int): チャンク間の重複文字数（デフォルト: 150）

    Returns:
        List[Document]: 分割後のDocumentリスト
    """
    check_memory_usage(threshold_percent=80.0, context="チャンク分割前")
    separators = [
        "\n\n",   # 段落区切り
        "\n",     # 改行
        "。",     # 句点
        "！", "？",
        "、",     # 読点
        " ",      # 空白
        ""        # 任意の文字
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len
    )

    try:
        splits = text_splitter.split_documents(documents)
    except MemoryError as e:
        logger.critical(f"[メモリ不足エラー] チャンク分割中にMemoryErrorが発生しました: {e}", exc_info=True)
        raise

    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i

    logger.info(f"チャンク分割完了: {len(documents)} 件のドキュメント -> {len(splits)} 件のチャンクに分割")
    check_memory_usage(threshold_percent=85.0, context="チャンク分割後")
    return splits


def load_and_split_documents(
    target_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    単一ファイルまたはディレクトリから読み込み、チャンク分割までを一括実行する

    Args:
        target_path (str): ファイルパスまたはディレクトリパス
        chunk_size (int): チャンクサイズ
        chunk_overlap (int): 重複サイズ

    Returns:
        List[Document]: 分割されたDocumentリスト
    """
    logger.info(f"ドキュメント読み込み・分割パイプライン開始: target={target_path}, chunk_size={chunk_size}")
    if os.path.isdir(target_path):
        docs = load_documents_from_directory(target_path)
    else:
        docs = load_document(target_path)

    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)



if __name__ == "__main__":
    print("=== Step 2: 資料の読み込みと分割 (Chunking) のテスト ===")
    
    data_dir = "data"
    if os.path.exists(data_dir):
        chunks = load_and_split_documents(data_dir, chunk_size=500, chunk_overlap=100)
        print(f"ディレクトリ '{data_dir}' から合計 {len(chunks)} 個のチャンクを作成しました。\n")
        
        for i, chunk in enumerate(chunks[:3]):  # 最初の3件を表示
            print(f"--- [Chunk {i+1}/{len(chunks)}] ファイル: {chunk.metadata.get('source_name')} ---")
            print(f"文字数: {len(chunk.page_content)}")
            print(f"内容抜粋:\n{chunk.page_content[:200]}...\n")
