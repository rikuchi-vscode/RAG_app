"""
専門領域特化型Q&Aボット（RAG） & 自律型Webリサーチ & LLMOps評価 Streamlit Webアプリケーション
"""
import os
import shutil
import base64
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pypdf import PdfReader

from config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    get_openai_api_key,
    get_google_api_key,
    set_custom_api_key,
    get_available_llm_provider
)
from document_loader import load_and_split_documents
from vector_store import (
    build_or_load_vector_store,
    create_vector_store,
    load_vector_store,
    DEFAULT_INDEX_DIR
)
from rag_chain import query_rag, create_rag_chain
from web_research_agent import WebResearchAgent
from llmops import (
    LLMOpsTracker,
    calculate_kpis,
    get_timeseries_dataframe,
    run_benchmark,
    DEFAULT_BENCHMARK_DATASET
)
from logger import setup_system_logger, get_logger, check_memory_usage

# システムロガーの初期化
system_logger = setup_system_logger()


# ページ設定
st.set_page_config(
    page_title="専門領域特化型 Q&A & Webリサーチ & LLMOps",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #4a90e2;
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #e3f2fd;
        color: #1976d2;
        font-weight: bold;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: bold;
        background-color: #e8f5e9;
        color: #2e7d32;
        margin-left: 8px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """セッションステートの初期化"""
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = True
        system_logger.info("新規Streamlitユーザーセッションが開始されました。")
        check_memory_usage(threshold_percent=80.0, context="新規セッション初期化")

    if st.session_state.get("custom_api_key_set"):
        if st.session_state.get("custom_google_api_key"):
            set_custom_api_key("GOOGLE_API_KEY", st.session_state.custom_google_api_key)
        if st.session_state.get("custom_openai_api_key"):
            set_custom_api_key("OPENAI_API_KEY", st.session_state.custom_openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "こんにちは！私は専門資料に特化したQ&Aチューターです。\n\n"
                    "専門資料の内容について、専門用語の解説など何でも質問してください。"
                ),
                "sources": [],
                "log_id": None,
                "eval_scores": None
            }
        ]
    if "vectorstore" not in st.session_state:
        loaded_files = get_loaded_documents_info()
        if os.path.exists(DEFAULT_INDEX_DIR) and loaded_files:
            st.session_state.vectorstore = load_vector_store()
        elif loaded_files:
            st.session_state.vectorstore = build_or_load_vector_store("data")
        else:
            st.session_state.vectorstore = None

    if "latest_research_result" not in st.session_state:
        st.session_state.latest_research_result = None

    if "research_topic_field" not in st.session_state:
        st.session_state.research_topic_field = ""

    if "llmops_tracker" not in st.session_state:
        st.session_state.llmops_tracker = LLMOpsTracker()

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None



def get_loaded_documents_info():
    """現在読み込まれているドキュメントのファイル名一覧を取得"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith((".pdf", ".txt", ".md"))]
    return sorted(files)


def delete_document(filename: str):
    """指定された資料を削除し、ベクトルストアを再構築する"""
    file_path = os.path.join("data", filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # 残りのドキュメントを確認
    remaining = get_loaded_documents_info()
    if remaining:
        chunks = load_and_split_documents("data")
        st.session_state.vectorstore = create_vector_store(chunks)
    else:
        # 資料が0件になった場合はインデックスフォルダを削除
        if os.path.exists(DEFAULT_INDEX_DIR):
            shutil.rmtree(DEFAULT_INDEX_DIR)
        st.session_state.vectorstore = None


def clear_all_documents():
    """すべての資料とインデックスを削除する"""
    data_dir = "data"
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            if os.path.isfile(p):
                os.remove(p)

    if os.path.exists(DEFAULT_INDEX_DIR):
        shutil.rmtree(DEFAULT_INDEX_DIR)
    st.session_state.vectorstore = None


def save_research_to_knowledge_base(topic: str, report_content: str) -> str:
    """Webリサーチ結果のレポートをナレッジベース（data/）に保存してベクトルストアを更新する"""
    os.makedirs("data", exist_ok=True)
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_topic = safe_topic.replace(" ", "_")[:30]
    filename = f"web_report_{safe_topic}.md"
    file_path = os.path.join("data", filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    chunks = load_and_split_documents("data")
    st.session_state.vectorstore = create_vector_store(chunks)
    return filename


@st.dialog("📄 資料プレビュー", width="large")
def preview_document_dialog(filename: str):
    """講義資料の内容をモーダルダイアログで表示する"""
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        st.error(f"ファイルが見つかりません: {filename}")
        return

    ext = os.path.splitext(filename)[1].lower()
    file_size_kb = os.path.getsize(file_path) / 1024

    st.caption(f"📁 ファイル名: `{filename}` | サイズ: {file_size_kb:.1f} KB")

    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        tab1, tab2 = st.tabs(["👁️ フォーマット表示", "📝 生テキスト"])
        with tab1:
            st.markdown(content)
        with tab2:
            st.code(content, language="markdown" if ext == ".md" else "text")

    elif ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

            tab1, tab2, tab3 = st.tabs([
                "🖼️ PDFビューワー（図・レイアウト表示）",
                "📝 抽出テキスト",
                "📥 ダウンロード"
            ])

            with tab1:
                pdf_display = f"""
                <iframe
                    src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=1"
                    width="100%"
                    height="650"
                    type="application/pdf"
                    style="border: none; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);"
                >
                    <p>お使いのブラウザはPDFの直接埋め込み表示に対応していません。「📥 ダウンロード」タブからダウンロードしてご確認ください。</p>
                </iframe>
                """
                st.markdown(pdf_display, unsafe_allow_html=True)

            with tab2:
                reader = PdfReader(file_path)
                num_pages = len(reader.pages)
                st.caption(f"総ページ数: {num_pages} ページ")

                page_num = st.selectbox(
                    "表示するページを選択",
                    options=list(range(1, num_pages + 1)),
                    format_func=lambda x: f"ページ {x}"
                )

                if page_num:
                    page_text = reader.pages[page_num - 1].extract_text()
                    st.markdown(f"#### 📖 ページ {page_num} のテキスト内容")
                    if page_text.strip():
                        st.text_area("テキスト内容", page_text, height=350)
                    else:
                        st.info("このページから抽出されたテキストはありません（画像または空白ページの可能性があります）。")

            with tab3:
                st.download_button(
                    label="📥 このPDFファイルをダウンロード",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"PDFの読み込み中にエラーが発生しました: {e}")


def main():
    init_session()
    provider = get_available_llm_provider()
    tracker: LLMOpsTracker = st.session_state.llmops_tracker

    # サイドバー
    with st.sidebar:
        st.title("⚙️ ナレッジベース設定")
        
        # APIステータス表示
        if provider == "gemini":
            st.success("🟢 Google Gemini API 接続中")
            if st.session_state.get("custom_api_key_set"):
                if st.button("🔄 設定したキーをクリア", key="btn_clear_custom_key", use_container_width=True):
                    st.session_state.custom_google_api_key = ""
                    st.session_state.custom_openai_api_key = ""
                    st.session_state.custom_api_key_set = False
                    set_custom_api_key("GOOGLE_API_KEY", "")
                    set_custom_api_key("OPENAI_API_KEY", "")
                    st.toast("設定したAPIキーをクリアしました。")
                    st.rerun()
        elif provider == "openai":
            st.success("🟢 OpenAI API 接続中")
            if st.session_state.get("custom_api_key_set"):
                if st.button("🔄 設定したキーをクリア", key="btn_clear_custom_key", use_container_width=True):
                    st.session_state.custom_google_api_key = ""
                    st.session_state.custom_openai_api_key = ""
                    st.session_state.custom_api_key_set = False
                    set_custom_api_key("GOOGLE_API_KEY", "")
                    set_custom_api_key("OPENAI_API_KEY", "")
                    st.toast("設定したAPIキーをクリアしました。")
                    st.rerun()
        else:
            st.error("🔴 APIキーが未設定です。")
            with st.container(border=True):
                st.markdown("##### 🔑 APIキーの手動入力")
                st.caption(".env や Secrets が未設定の場合は以下に入力してください。")
                api_choice = st.radio("利用するプロバイダ", ["Google Gemini", "OpenAI"], horizontal=True, key="sidebar_api_choice")
                if api_choice == "Google Gemini":
                    input_gemini = st.text_input("Google API Key", type="password", placeholder="AIzaSy...", key="input_sidebar_gemini")
                    if st.button("設定して接続", key="btn_apply_sidebar_gemini", use_container_width=True):
                        if input_gemini.strip():
                            st.session_state.custom_google_api_key = input_gemini.strip()
                            st.session_state.custom_api_key_set = True
                            set_custom_api_key("GOOGLE_API_KEY", input_gemini.strip())
                            system_logger.info("ブラウザ上からGoogle Gemini APIキーが設定されました。")
                            st.toast("Google Gemini APIキーを設定しました！", icon="🟢")
                            st.rerun()
                        else:
                            st.warning("APIキーを入力してください。")
                else:
                    input_openai = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="input_sidebar_openai")
                    if st.button("設定して接続", key="btn_apply_sidebar_openai", use_container_width=True):
                        if input_openai.strip():
                            st.session_state.custom_openai_api_key = input_openai.strip()
                            st.session_state.custom_api_key_set = True
                            set_custom_api_key("OPENAI_API_KEY", input_openai.strip())
                            system_logger.info("ブラウザ上からOpenAI APIキーが設定されました。")
                            st.toast("OpenAI APIキーを設定しました！", icon="🟢")
                            st.rerun()
                        else:
                            st.warning("APIキーを入力してください。")

        st.markdown("---")
        st.subheader("📚 登録済みの講義・リサーチ資料")
        loaded_files = get_loaded_documents_info()
        if loaded_files:
            for f in loaded_files:
                col_name, col_view, col_del = st.columns([0.64, 0.18, 0.18])
                with col_name:
                    if f.startswith("web_report_"):
                        st.markdown(f"🌐 `{f}`")
                    else:
                        st.markdown(f"📄 `{f}`")
                with col_view:
                    if st.button("👁️", key=f"view_{f}", help=f"『{f}』の中身をプレビュー"):
                        preview_document_dialog(f)
                with col_del:
                    if st.button("🗑️", key=f"del_{f}", help=f"『{f}』を削除"):
                        with st.spinner(f"『{f}』を削除中..."):
                            try:
                                system_logger.info(f"ドキュメント削除要求: {f}")
                                delete_document(f)
                                st.toast(f"『{f}』を削除しました", icon="🗑️")
                                st.rerun()
                            except Exception as e:
                                system_logger.error(f"ドキュメント削除エラー ({f}): {e}", exc_info=True)
                                st.error(f"削除中にエラーが発生しました: {e}")

            if len(loaded_files) > 1:
                if st.button("⚠️ 全ての資料を削除", use_container_width=True):
                    with st.spinner("すべての資料を削除中..."):
                        try:
                            system_logger.info("全ドキュメント一括削除要求")
                            clear_all_documents()
                            st.toast("すべての資料を削除しました", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            system_logger.error(f"全ドキュメント削除エラー: {e}", exc_info=True)
                            st.error(f"削除中にエラーが発生しました: {e}")
        else:
            st.info("資料が登録されていません。下のフォームまたはWebリサーチから追加してください。")

        st.markdown("---")
        st.subheader("📤 新しい資料の追加")
        uploaded_file = st.file_uploader(
            "PDFまたはテキストファイルをアップロード",
            type=["pdf", "txt", "md"],
            help="講義スライド、レジュメ、論文ファイルを追加できます。"
        )

        if uploaded_file is not None:
            if st.button("ナレッジベースに取り込む", use_container_width=True):
                with st.spinner("ファイルを解析してベクトル化中..."):
                    try:
                        system_logger.info(f"ファイルアップロード受付: {uploaded_file.name} (サイズ: {uploaded_file.size} bytes)")
                        os.makedirs("data", exist_ok=True)
                        save_path = os.path.join("data", uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        chunks = load_and_split_documents("data")
                        st.session_state.vectorstore = create_vector_store(chunks)
                        system_logger.info(f"ファイル取り込み・ベクトル化成功: {uploaded_file.name}")
                        st.success(f"『{uploaded_file.name}』を取り込みました！")
                        st.rerun()
                    except Exception as e:
                        system_logger.error(f"ファイル取り込み失敗 ({uploaded_file.name}): {e}", exc_info=True)
                        st.error(f"ファイルの取り込み中にエラーが発生しました: {e}")


        st.markdown("---")
        if st.button("🧹 会話履歴をクリア", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # メインコンテンツ タブ構成 (3タブ構成)
    tab_rag, tab_research, tab_llmops = st.tabs([
        "🎓 専門資料 Q&A (RAG)",
        "🌐 自律型 Webリサーチ＆レポート生成",
        "📈 LLMOps & 評価分析"
    ])

    # ==========================================================
    # 【タブ1】🎓 専門資料 Q&A (RAG) - 既存機能を100%完全維持
    # ==========================================================
    with tab_rag:
        st.markdown('<div class="main-header">🎓 専門領域特化型 Q&A ボット</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">読み込んだ講義資料・論文に基づいた高精度なRAG対話システム</div>', unsafe_allow_html=True)

        # 過去の会話履歴を描画
        for idx_msg, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # 参照ソースがある場合はアコーディオン表示
                if msg.get("sources"):
                    with st.expander("📚 参照した講義資料の箇所"):
                        for s_idx, src in enumerate(msg["sources"]):
                            st.markdown(
                                f'<div class="source-box">'
                                f'<span class="badge">参照 {s_idx+1}</span> <b>{src.get("source_name", "不明")}</b> '
                                f'(Chunk {src.get("chunk_id", "-")})<br>'
                                f'<div style="margin-top: 4px; color: #444;">{src.get("content", "")}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                # アシスタントの回答に対するLLMOpsフィードバックUI
                if msg["role"] == "assistant" and msg.get("log_id"):
                    log_id = msg["log_id"]
                    ev = msg.get("eval_scores") or {}
                    col_fb1, col_fb2, col_score = st.columns([0.06, 0.06, 0.88])
                    with col_fb1:
                        if st.button("👍", key=f"up_{log_id}_{idx_msg}", help="良い回答として評価"):
                            tracker.update_feedback(log_id, "up")
                            st.toast("高評価（👍）を記録しました！LLMOpsダッシュボードに反映されます。", icon="✨")
                    with col_fb2:
                        if st.button("👎", key=f"down_{log_id}_{idx_msg}", help="改善が必要な回答として評価"):
                            tracker.update_feedback(log_id, "down")
                            st.toast("低評価（👎）を記録しました。改善データとして蓄積されます。", icon="📝")
                    with col_score:
                        if ev.get("overall_score"):
                            st.caption(f"🎯 AI品質スコア: **{ev.get('overall_score')}/5.0** (忠実性: {ev.get('faithfulness')}/5 | 適合性: {ev.get('answer_relevance')}/5)")

        # ユーザー入力
        prompt = st.chat_input("専門用語の意味など質問を入力...")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

            with st.chat_message("assistant"):
                if provider == "none":
                    error_msg = "APIキーが設定されていないため回答を生成できません。.env を確認してください。"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
                elif not get_loaded_documents_info() or st.session_state.vectorstore is None:
                    warn_msg = "現在ナレッジベースに登録されている講義資料がありません。サイドバーの「新しい資料の追加」からPDFやテキストをアップロードしてください。"
                    st.warning(warn_msg)
                    st.session_state.messages.append({"role": "assistant", "content": warn_msg, "sources": []})
                else:
                    with st.spinner("講義資料を検索して回答を生成中..."):
                        start_time = time.time()
                        try:
                            result = query_rag(
                                question=prompt,
                                vectorstore=st.session_state.vectorstore
                            )
                            latency = time.time() - start_time
                            answer = result["answer"]
                            sources = result["sources"]

                            # LLMOps ロギング (自動品質評価付き)
                            model_used = "gemini-3.5-flash" if provider == "gemini" else "gpt-4o-mini"
                            log_record = tracker.log_event(
                                feature_type="rag_qa",
                                question=prompt,
                                answer=answer,
                                model_name=model_used,
                                provider=provider,
                                latency_sec=latency,
                                context_docs=sources,
                                auto_eval=True
                            )

                            st.markdown(answer)

                            if sources:
                                with st.expander("📚 参照した講義資料の箇所"):
                                    for s_idx, src in enumerate(sources):
                                        st.markdown(
                                            f'<div class="source-box">'
                                            f'<span class="badge">参照 {s_idx+1}</span> <b>{src.get("source_name", "不明")}</b> '
                                            f'(Chunk {src.get("chunk_id", "-")})<br>'
                                            f'<div style="margin-top: 4px; color: #444;">{src.get("content", "")}</div>'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "log_id": log_record["log_id"],
                                "eval_scores": log_record.get("eval_scores")
                            })
                        except Exception as e:
                            system_logger.error(f"[Q&Aエラー] 回答生成処理中に例外が発生しました: {e}", exc_info=True)
                            err = f"回答生成中にエラーが発生しました: {e}"
                            st.error(err)
                            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})

    # ==========================================================
    # 【タブ2】🌐 自律型 Webリサーチ＆レポート生成 - 既存機能を完全維持
    # ==========================================================
    with tab_research:
        st.markdown('<div class="main-header">🌐 自律型 Webリサーチ＆レポート生成</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">テーマを入力すると、AIが自律的にWeb検索を行い（思考 ⇄ 検索 ⇄ 観察 ⇄ 再思考）、体系的なマークダウンレポートを自動生成します。</div>',
            unsafe_allow_html=True
        )

        def set_sample_topic(topic_text: str):
            st.session_state.research_topic_field = topic_text

        col_input, col_settings = st.columns([0.7, 0.3])
        with col_input:
            topic_input = st.text_input(
                "調査したいテーマ・キーワードを入力してください",
                placeholder="例: 次世代半導体2nmプロセスの実用化動向とRapidusの戦略",
                key="research_topic_field"
            )

            st.caption("💡 サンプル候補をクリックして入力:")
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            with sample_col1:
                st.button(
                    "🔬 2nm半導体とRapidus動向",
                    on_click=set_sample_topic,
                    args=("次世代半導体2nmプロセスの実用化動向とRapidusの戦略",),
                    use_container_width=True
                )
            with sample_col2:
                st.button(
                    "📡 6Gとテラヘルツ波通信",
                    on_click=set_sample_topic,
                    args=("6G通信に向けたテラヘルツ波デバイスの最新研究動向",),
                    use_container_width=True
                )
            with sample_col3:
                st.button(
                    "📦 先端パッケージングCoWoS",
                    on_click=set_sample_topic,
                    args=("AI半導体を支える先端パッケージング技術（CoWoS / チップレット）の市場と技術動向",),
                    use_container_width=True
                )

        with col_settings:
            with st.expander("⚙️ リサーチ詳細設定"):
                max_iter = st.slider("最大自律ループ回数", min_value=1, max_value=4, value=2, help="情報不足時に再検索を行う最大回数")
                st.caption(f"現在のLLMプロバイダ: **{provider.upper()}**")

        start_research = st.button("🚀 自律リサーチ＆レポート生成を開始", type="primary", use_container_width=True)

        if start_research:
            if not topic_input.strip():
                st.warning("調査したいテーマを入力してください。")
            elif provider == "none":
                st.error("APIキーが設定されていないためリサーチを実行できません。.env を確認してください。")
            else:
                progress_container = st.container()
                with progress_container:
                    with st.status("🔍 自律リサーチエージェントが調査を開始しました...", expanded=True) as status:
                        agent = WebResearchAgent(
                            provider=provider,
                            max_iterations=max_iter
                        )

                        step_logs = []
                        def on_progress(step_type: str, message: str, details: dict):
                            step_logs.append((step_type, message, details))
                            if step_type == "thought":
                                st.markdown(f"💭 **[思考 (Thought)]** {message}")
                            elif step_type == "action":
                                st.markdown(f"🔍 **[行動 (Action)]** {message}")
                            elif step_type == "observation":
                                st.markdown(f"👀 **[観察 (Observation)]** {message}")
                            elif step_type == "re_thought":
                                st.markdown(f"🤔 **[再思考 (Re-Thought)]** {message}")
                            elif step_type == "report_gen":
                                st.markdown(f"📝 **[レポート作成]** {message}")

                        start_t = time.time()
                        try:
                            system_logger.info(f"Webリサーチ開始: '{topic_input.strip()}' (max_iter={max_iter})")
                            result = agent.run_research(
                                topic=topic_input.strip(),
                                progress_callback=on_progress
                            )
                            latency_r = time.time() - start_t
                            status.update(label="✅ リサーチ＆レポート作成が完了しました！", state="complete", expanded=False)
                            st.session_state.latest_research_result = result
                            system_logger.info(f"Webリサーチ完了: '{topic_input.strip()}' ({latency_r:.2f}s)")

                            # LLMOps ロギング
                            tracker.log_event(
                                feature_type="web_research",
                                question=topic_input.strip(),
                                answer=result["report_markdown"][:1000] + "...",
                                model_name="gemini-3.5-flash" if provider == "gemini" else "gpt-4o-mini",
                                provider=provider,
                                latency_sec=latency_r,
                                context_docs=result.get("sources", []),
                                auto_eval=False
                            )

                            st.toast("レポートの生成が完了しました！", icon="🎉")
                        except Exception as e:
                            system_logger.error(f"Webリサーチ実行エラー ('{topic_input.strip()}'): {e}", exc_info=True)
                            status.update(label="❌ リサーチ中にエラーが発生しました", state="error")
                            st.error(f"エラー詳細: {e}")


        # リサーチ結果の表示
        if st.session_state.latest_research_result:
            res = st.session_state.latest_research_result
            st.markdown("---")
            st.subheader(f"📊 調査レポート: {res['topic']}")

            btn_col1, btn_col2 = st.columns([0.5, 0.5])
            with btn_col1:
                st.download_button(
                    label="📥 マークダウンレポート (.md) をダウンロード",
                    data=res["report_markdown"],
                    file_name=f"report_{res['topic'][:20]}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with btn_col2:
                if st.button("📚 このレポートをナレッジベースに追加 (Q&A対象化)", use_container_width=True):
                    with st.spinner("レポートをナレッジベースに取り込んでベクトル化中..."):
                        saved_name = save_research_to_knowledge_base(res["topic"], res["report_markdown"])
                        st.success(f"ナレッジベースに『{saved_name}』として追加しました！")
                        st.toast("『🎓 専門資料 Q&A』タブですぐに質問できます！", icon="🚀")
                        st.rerun()

            st.markdown("### 📄 レポート本文")
            st.markdown(res["report_markdown"])

            if res.get("steps"):
                with st.expander("🧠 エージェントの自律思考・調査プロセス（ReActログ）"):
                    for s in res["steps"]:
                        st.markdown(f"**Iteration {s.get('iteration', '-')} - {s.get('phase', '')}**")
                        if "content" in s:
                            st.markdown(f"> {s['content']}")
                        if "queries" in s and s["queries"]:
                            st.caption(f"検索クエリ: {', '.join(s['queries'])}")
                        st.markdown("---")

            if res.get("sources"):
                with st.expander("🌐 参照したWebリソース一覧"):
                    for s_idx, src in enumerate(res["sources"]):
                        st.markdown(f"{s_idx+1}. [{src.get('title', 'リンク')}]({src.get('url', '#')})")
                        if src.get("snippet"):
                            st.caption(f"抜粋: {src['snippet'][:150]}...")

    # ==========================================================
    # 【タブ3】📈 LLMOps & 評価分析 - 新規追加機能
    # ==========================================================
    with tab_llmops:
        st.markdown('<div class="main-header">📈 LLMOps & 評価ダッシュボード</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">LLM実行のトレーサビリティ（レイテンシ・トークン・コスト）とRAG品質（接地性・適合性・ハルシネーション検出）の総合分析</div>',
            unsafe_allow_html=True
        )

        all_logs = tracker.load_all_logs()
        kpis = calculate_kpis(all_logs)

        # 1. サマリーKPIカード
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{kpis["total_calls"]} <span style="font-size: 1rem; color:#888;">回</span></div>'
                f'<div class="kpi-label">🚀 総リクエスト数</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with kpi_col2:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{kpis["avg_latency"]:.2f} <span style="font-size: 1rem; color:#888;">秒</span></div>'
                f'<div class="kpi-label">⏱️ 平均応答レイテンシ</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with kpi_col3:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{kpis["avg_overall_score"]:.1f} <span style="font-size: 1rem; color:#888;">/ 5.0</span></div>'
                f'<div class="kpi-label">🎯 RAG総合品質スコア</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with kpi_col4:
            pos_rate_str = f"{kpis['positive_feedback_rate']:.0f}%" if kpis['feedback_count'] > 0 else "N/A"
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value">{pos_rate_str}</div>'
                f'<div class="kpi-label">👍 ユーザー高評価率 ({kpis["feedback_count"]}件)</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # LLMOps サブタブ
        ops_tab1, ops_tab2, ops_tab3, ops_tab4 = st.tabs([
            "📊 パフォーマンス・トレンド",
            "🎯 RAG品質分析 (LLM-as-a-Judge)",
            "📋 実行トレース・ログ一覧",
            "🧪 ベンチマーク評価 (テスト実行)"
        ])

        # ----------------------------------------------------
        # サブタブ1: パフォーマンス・トレンド
        # ----------------------------------------------------
        with ops_tab1:
            st.subheader("📈 応答時間とトークン消費の推移")
            df_logs = get_timeseries_dataframe(all_logs)

            if not df_logs.empty and len(df_logs) >= 1:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    fig_lat = px.line(
                        df_logs,
                        x="日時",
                        y="応答時間 (秒)",
                        color="機能",
                        markers=True,
                        title="⏱️ 応答レイテンシの推移 (秒)",
                        template="plotly_white"
                    )
                    fig_lat.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_lat, use_container_width=True)

                with chart_col2:
                    fig_tok = px.bar(
                        df_logs,
                        x="日時",
                        y="トークン数",
                        color="モデル",
                        title="🔢 トークン消費量の推移",
                        template="plotly_white"
                    )
                    fig_tok.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_tok, use_container_width=True)

                st.caption(f"💰 累計概算コスト: **${kpis['total_cost_usd']:.4f} USD** (総トークン数: {kpis['total_tokens']:,})")
            else:
                st.info("まだ十分な実行ログがありません。Q&AやWebリサーチを実行すると推移グラフが表示されます。")

        # ----------------------------------------------------
        # サブタブ2: RAG品質分析 (LLM-as-a-Judge)
        # ----------------------------------------------------
        with ops_tab2:
            st.subheader("🎯 RAG Triad 品質評価スコアボード")
            st.markdown(
                "LLM-as-a-Judge が各対話を分析し、**【接地性・忠実性】**（ハルシネーション抑制）、**【回答適合性】**（意図への的確性）、**【コンテキスト適合性】**（検索精度）を5段階で評価した結果です。"
            )

            if kpis["total_calls"] > 0:
                triad_col1, triad_col2 = st.columns([0.45, 0.55])
                with triad_col1:
                    # レーダーチャート
                    categories = ["接地性・忠実性\n(ハルシネーション抑制)", "回答適合性\n(的確さ)", "検索適合性\n(資料抽出精度)"]
                    values = [kpis["avg_faithfulness"], kpis["avg_answer_relevance"], kpis["avg_context_relevance"]]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(26, 115, 232, 0.2)',
                        line=dict(color='#1a73e8', width=2),
                        name="平均スコア"
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 5])
                        ),
                        showlegend=False,
                        height=350,
                        margin=dict(l=40, r=40, t=30, b=30)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with triad_col2:
                    st.markdown("#### 📋 指標別平均スコア")
                    st.progress(kpis["avg_faithfulness"] / 5.0, text=f"🛡️ 接地性・忠実性 (Faithfulness): {kpis['avg_faithfulness']:.1f} / 5.0")
                    st.progress(kpis["avg_answer_relevance"] / 5.0, text=f"🎯 回答適合性 (Answer Relevance): {kpis['avg_answer_relevance']:.1f} / 5.0")
                    st.progress(kpis["avg_context_relevance"] / 5.0, text=f"🔍 コンテキスト適合性 (Context Relevance): {kpis['avg_context_relevance']:.1f} / 5.0")

                    if kpis["hallucination_rate"] > 0:
                        st.warning(f"⚠️ ハルシネーションの疑いがある回答率: **{kpis['hallucination_rate']:.1f}%**")
                    else:
                        st.success("✅ ハルシネーション検出率: **0.0%**（極めて高い忠実性を維持しています）")
            else:
                st.info("対話ログがありません。「🎓 専門資料 Q&A」で質問を行うと自動採点されます。")

        # ----------------------------------------------------
        # サブタブ3: 実行トレース・ログ一覧
        # ----------------------------------------------------
        with ops_tab3:
            st.subheader("📋 実行トレース & 詳細ログ一覧")
            
            top_col1, top_col2 = st.columns([0.8, 0.2])
            with top_col2:
                if st.button("🗑️ ログを全削除", help="蓄積されたログファイルを初期化"):
                    tracker.clear_all_logs()
                    st.toast("ログを削除しました", icon="🗑️")
                    st.rerun()

            if all_logs:
                for idx, log_item in enumerate(all_logs[:30]):
                    ts = log_item.get("timestamp", "")
                    ft = "🎓 専門Q&A" if log_item.get("feature_type") == "rag_qa" else "🌐 Webリサーチ"
                    q_text = log_item.get("question", "")[:60]
                    ev = log_item.get("eval_scores") or {}
                    score_str = f"⭐ {ev.get('overall_score', '-')}/5.0" if ev else ""
                    fb_str = "👍" if log_item.get("user_feedback", {}).get("rating") == "up" else ("👎" if log_item.get("user_feedback", {}).get("rating") == "down" else "")

                    exp_label = f"[{ts}] {ft} | {q_text}... {score_str} {fb_str}"
                    with st.expander(exp_label):
                        col_l1, col_l2, col_l3 = st.columns(3)
                        with col_l1:
                            st.caption(f"**モデル**: `{log_item.get('model_name')}` ({log_item.get('provider')})")
                        with col_l2:
                            st.caption(f"**レイテンシ**: `{log_item.get('latency_sec')} 秒`")
                        with col_l3:
                            st.caption(f"**トークン数**: `{log_item.get('total_tokens')} tokens` (${log_item.get('cost_usd'):.5f})")

                        st.markdown(f"**【ユーザーの入力/質問】**\n{log_item.get('question')}")
                        st.markdown(f"**【AIの回答】**\n{log_item.get('answer')}")

                        if ev:
                            st.markdown(f"**【AIジャッジによる採点理由】**")
                            st.info(f"忠実性: {ev.get('faithfulness')}/5 | 適合性: {ev.get('answer_relevance')}/5 | 検索精度: {ev.get('context_relevance')}/5\n\n💬 {ev.get('reasoning')}")

                        if log_item.get("context_sources"):
                            st.caption(f"参照された資料: {', '.join(log_item.get('context_sources'))}")
            else:
                st.info("現在記録されているログはありません。")

        # ----------------------------------------------------
        # サブタブ4: オフライン・ベンチマーク評価
        # ----------------------------------------------------
        with ops_tab4:
            st.subheader("🧪 オフライン・ベンチマーク評価（自動リグレッションテスト）")
            st.markdown(
                "ハルシネーション抑制確認など、あらかじめ用意された標準テストセットに対して一括でRAG質問を実行し、回答品質を自動検証します。"
            )

            if st.button("🚀 ベンチマークテストを実行 (4問一括評価)", type="primary"):
                if provider == "none":
                    st.error("APIキーが設定されていないためベンチマークを実行できません。")
                elif st.session_state.vectorstore is None:
                    st.warning("ナレッジベースが空です。資料をアップロードしてから実行してください。")
                else:
                    prog_bar = st.progress(0, text="ベンチマーク実行中...")
                    status_text = st.empty()

                    def benchmark_progress(current, total, question):
                        prog_bar.progress(current / total, text=f"テスト実行中 ({current}/{total}): {question[:30]}...")

                    try:
                        system_logger.info("LLMOpsベンチマーク自動評価開始")
                        bm_result = run_benchmark(
                            vectorstore=st.session_state.vectorstore,
                            progress_callback=benchmark_progress
                        )
                        st.session_state.benchmark_results = bm_result
                        prog_bar.empty()
                        system_logger.info("LLMOpsベンチマーク自動評価完了")
                        st.success("🎉 ベンチマーク評価が完了しました！")
                    except Exception as e:
                        prog_bar.empty()
                        system_logger.error(f"ベンチマーク評価実行エラー: {e}", exc_info=True)
                        st.error(f"ベンチマーク評価中にエラーが発生しました: {e}")


            if st.session_state.benchmark_results:
                bm = st.session_state.benchmark_results
                st.markdown("---")
                st.markdown(f"### 📊 ベンチマーク総合スコア: **{bm['avg_overall_score']} / 5.0**")
                
                bm_col1, bm_col2, bm_col3 = st.columns(3)
                with bm_col1:
                    st.metric("テスト総数", f"{bm['total_tests']} 問")
                with bm_col2:
                    st.metric("平均接地性・忠実性", f"{bm['avg_faithfulness']} / 5.0")
                with bm_col3:
                    st.metric("平均回答適合性", f"{bm['avg_answer_relevance']} / 5.0")

                st.markdown("#### 📝 各テスト項目の詳細結果")
                for item in bm["items"]:
                    ev = item["eval_scores"]
                    with st.expander(f"[{item['id']}] {item['category']} : {item['question'][:40]}... (スコア: {ev.get('overall_score')}/5.0)"):
                        st.markdown(f"**質問**: {item['question']}")
                        st.markdown(f"**期待される正解の要点**: {item['expected_key_points']}")
                        st.markdown(f"**AIの回答**: {item['answer']}")
                        st.info(f"**AIジャッジの評価**: 忠実性={ev.get('faithfulness')}/5, 適合性={ev.get('answer_relevance')}/5\n\n💬 {ev.get('reasoning')}")


if __name__ == "__main__":
    main()
