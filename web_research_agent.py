"""
Webリサーチ＆レポート自動生成エージェント モジュール
web-research.md の仕様に基づく ReAct 自律サイクル（思考 → 行動 → 観察 → 再思考）を実装
"""
import os
import time
import json
import re
import warnings
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from config import OPENAI_API_KEY, GOOGLE_API_KEY, get_available_llm_provider
from logger import get_logger

logger = get_logger()

# .env の再読み込み
load_dotenv()

# 利用可能なGeminiモデル候補（フォールバック順）
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]


class WebResearchAgent:
    """
    自律型Webリサーチ＆レポート生成エージェント
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 3
    ):
        self.provider = provider or get_available_llm_provider()
        self.model_name = model_name
        self.max_iterations = max_iterations

        if self.provider == "none":
            logger.error("[APIキー未設定] WebResearchAgent初期化失敗: 有効なAPIキーがありません。")
            raise ValueError(
                "有効なAPIキーが設定されていません。.env に GOOGLE_API_KEY または OPENAI_API_KEY を設定してください。"
            )
        logger.info(f"WebResearchAgentを初期化: provider={self.provider}, max_iterations={self.max_iterations}")

    def _call_llm(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        設定されたプロバイダ（Gemini / OpenAI）に応じてLLMを呼び出す（フォールバック機能付き）
        """
        if self.provider == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GOOGLE_API_KEY)
            models_to_try = [self.model_name] if self.model_name else GEMINI_MODELS
            last_error = None

            for m in models_to_try:
                try:
                    config = types.GenerateContentConfig(
                        temperature=0.3,
                    )
                    if system_instruction:
                        config.system_instruction = system_instruction

                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=config
                    )
                    return response.text.strip()
                except Exception as e:
                    last_error = e
                    logger.warning(f"[WebResearch API警告] Geminiモデル '{m}' 呼び出し失敗: {e}。フォールバックします。")
                    time.sleep(1)
                    continue

            logger.error(f"[WebResearch APIエラー] Gemini API全モデル呼び出し失敗: {last_error}", exc_info=True)
            raise RuntimeError(f"Gemini API呼び出しに失敗しました: {last_error}")

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            selected_model = self.model_name or "gpt-4o-mini"
            try:
                chat = ChatOpenAI(
                    model=selected_model,
                    temperature=0.3,
                    openai_api_key=OPENAI_API_KEY
                )
                messages = []
                if system_instruction:
                    messages.append(SystemMessage(content=system_instruction))
                messages.append(HumanMessage(content=prompt))

                response = chat.invoke(messages)
                return response.content.strip()
            except Exception as e:
                logger.error(f"[WebResearch APIエラー] OpenAI '{selected_model}' 呼び出し失敗: {e}", exc_info=True)
                raise

        else:
            logger.error("利用可能なLLMプロバイダが設定されていません。")
            raise ValueError("利用可能なLLMプロバイダが設定されていません。")

    def search_web(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """
        Web検索ツール（DDGS / DuckDuckGo）を実行し、タイトル・スニペット・URLを取得する
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            logger.info(f"Web検索クエリ実行: '{query}' (上限 {max_results} 件)")
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                formatted = []
                for r in raw_results:
                    title = r.get("title", "").strip()
                    url = (r.get("href") or r.get("url") or "").strip()
                    snippet = (r.get("body") or r.get("snippet") or "").strip()
                    if title and url:
                        formatted.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        })
                logger.info(f"Web検索完了: '{query}' -> {len(formatted)} 件取得")
                return formatted
        except Exception as e:
            logger.warning(f"[Web検索ツールエラー] クエリ '{query}' の実行に失敗しました: {e}", exc_info=True)
            return []


    def run_research(
        self,
        topic: str,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        指定されたテーマに対して自律型ReActリサーチを実行し、レポートを生成する

        Args:
            topic (str): リサーチしたいテーマや質問
            progress_callback (Optional[Callable]): 進捗更新コールバック関数
                (step_type, message, details)

        Returns:
            Dict[str, Any]: {
                "topic": str,
                "report_markdown": str,
                "steps": List[Dict],
                "sources": List[Dict],
                "provider": str
            }
        """
        def notify(step_type: str, message: str, details: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(step_type, message, details or {})

        logs: List[Dict[str, Any]] = []
        collected_sources: List[Dict[str, str]] = []
        seen_urls = set()
        knowledge_base: List[Dict[str, Any]] = []

        # ==========================================
        # Step 1. 思考 (Initial Thought): 検索方針と初期キーワードの策定
        # ==========================================
        notify("thought", f"リサーチテーマ『{topic}』の分析と検索計画を立案中...")
        plan_prompt = f"""
あなたは高度なリサーチ能力を持つWeb調査アナリストAIです。
ユーザーから以下のリサーチテーマが与えられました。

【テーマ】: {topic}

このテーマについて包括的で専門性の高いレポートを作成するため、まず調査すべき検索キーワード（日本語）を2〜3個策定してください。
回答は必ず以下のJSON形式のみで出力してください（マークダウンコードブロックも不要です）。
{{
    "thought": "調査方針の簡単な説明",
    "search_queries": ["検索キーワード1", "検索キーワード2"]
}}
"""
        initial_plan_raw = self._call_llm(plan_prompt)
        try:
            # JSON抽出
            cleaned_json = re.sub(r"^```json\s*|\s*```$", "", initial_plan_raw.strip(), flags=re.MULTILINE)
            plan_data = json.loads(cleaned_json)
            thought_text = plan_data.get("thought", "初期検索方針を策定しました。")
            search_queries = plan_data.get("search_queries", [topic])
        except Exception:
            thought_text = f"テーマ『{topic}』に関する基礎情報と最新動向を調査します。"
            search_queries = [topic, f"{topic} 最新動向"]

        logs.append({
            "iteration": 1,
            "phase": "Thought (思考)",
            "content": thought_text,
            "queries": search_queries
        })
        notify("thought", f"【思考】{thought_text}", {"queries": search_queries})

        # ==========================================
        # ReActループ（行動 → 観察 → 再思考）
        # ==========================================
        for iteration in range(1, self.max_iterations + 1):
            if not search_queries:
                break

            # 2. 行動 (Action): Web検索の実行
            iteration_results = []
            for query in search_queries:
                notify("action", f"Web検索を実行中: 『{query}』", {"query": query})
                results = self.search_web(query, max_results=3)
                for r in results:
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        collected_sources.append(r)
                        iteration_results.append(r)
                        knowledge_base.append(r)

            # 3. 観察 (Observation): 検索結果の把握
            obs_text = f"{len(iteration_results)} 件の新規Web情報を取得・分析しました。"
            logs.append({
                "iteration": iteration,
                "phase": "Action & Observation (行動と観察)",
                "queries": search_queries,
                "found_count": len(iteration_results),
                "items": iteration_results
            })
            notify("observation", f"【観察】{obs_text}", {"found_items": iteration_results})

            # 最大ループに達した場合は再思考をスキップしてレポート生成へ
            if iteration >= self.max_iterations:
                break

            # 4. 再思考 (Re-Thought / Evaluation): 情報の十分性を評価
            notify("re_thought", "収集した情報の十分性を評価し、追加調査の要否を検討中...")
            snippets_summary = "\n\n".join([
                f"- タイトル: {k['title']}\n  内容: {k['snippet']}"
                for k in knowledge_base[-8:]
            ])

            eval_prompt = f"""
あなたはWebリサーチアナリストです。
現在、テーマ『{topic}』に関するレポートを作成するため情報を収集中です。

【これまでに収集した主な情報】
{snippets_summary}

【評価タスク】
1. 現在収集した情報で、テーマの全体像・背景・最新動向・技術/市場課題を網羅した詳細レポートが作成できるか判断してください。
2. もしまだ不足している情報（具体的な数値、最新の事例、競合・業界動向、課題など）があれば、追加で検索すべきキーワードを1〜2個指定してください。
3. すでに十分であれば is_sufficient を true にしてください。

必ず以下のJSON形式のみで回答してください:
{{
    "evaluation": "情報の十分性に関する考察",
    "is_sufficient": true または false,
    "next_queries": ["追加キーワード1"] (不足時のみ)
}}
"""
            eval_raw = self._call_llm(eval_prompt)
            try:
                cleaned_eval_json = re.sub(r"^```json\s*|\s*```$", "", eval_raw.strip(), flags=re.MULTILINE)
                eval_data = json.loads(cleaned_eval_json)
                is_sufficient = eval_data.get("is_sufficient", False)
                eval_thought = eval_data.get("evaluation", "情報の精査を完了しました。")
                next_queries = eval_data.get("next_queries", [])
            except Exception:
                is_sufficient = (iteration >= 2)
                eval_thought = "情報収集を継続します。"
                next_queries = [f"{topic} 課題 展望"] if not is_sufficient else []

            logs.append({
                "iteration": iteration,
                "phase": "Re-Thought (再思考)",
                "content": eval_thought,
                "is_sufficient": is_sufficient,
                "next_queries": next_queries
            })
            notify("re_thought", f"【再思考】{eval_thought}", {"is_sufficient": is_sufficient, "next_queries": next_queries})

            if is_sufficient or not next_queries:
                break
            else:
                search_queries = next_queries

        # ==========================================
        # Step 5. マークダウンレポートの生成
        # ==========================================
        notify("report_gen", "収集した知見を統合し、構造化されたマークダウンレポートを作成中...")

        all_context_text = "\n\n".join([
            f"【情報源 {idx+1}】\nタイトル: {item['title']}\nURL: {item['url']}\n内容抜粋: {item['snippet']}"
            for idx, item in enumerate(knowledge_base)
        ])

        report_prompt = f"""
あなたは専門調査機関のシニアテクニカルリサーチャーです。
以下の【Webリサーチ結果】に基づいて、テーマ『{topic}』に関する包括的で信頼性の高いマークダウンレポートを作成してください。

【Webリサーチ結果】
{all_context_text}

【レポート作成の要件】
1. **構成**:
   - **# {topic} に関する調査レポート**（魅力的なメインタイトル）
   - **## 1. エグゼクティブサマリー**（要点・キーファインディングを3〜4行で簡潔に要約）
   - **## 2. 背景と基礎知識**（なぜ今このテーマが注目されているか、基本概念）
   - **## 3. 最新動向と技術・市場の現状**（具体的な企業名、製品、数値データ、最新トレンドなど詳細に解説）
   - **## 4. 主な課題と今後の展望**（ボトルネック、技術的・経済的障壁、今後のロードマップ）
   - **## 5. まとめ**（総括）
   - **## 6. 参照Webリソース一覧**（収集した情報源の [タイトル](URL) リンクを箇条書きで記載）
2. **トーン & スタイル**:
   - 客観的かつ論理的、読みやすい日本語で記述してください。
   - 箇条書きや強調（太字）、表などを効果的に活用してください。
   - コンテキストにない不確かな推測は避け、収集されたファクトに基づいて記述してください。
"""

        final_report = self._call_llm(
            report_prompt,
            system_instruction="あなたは正確・客観的かつ体系的な技術調査レポートを作成する専門リサーチャーです。"
        )

        notify("done", "マークダウンレポートの生成が完了しました！")

        return {
            "topic": topic,
            "report_markdown": final_report,
            "steps": logs,
            "sources": collected_sources,
            "provider": self.provider
        }


def run_quick_research(topic: str) -> Dict[str, Any]:
    """クイック実行用ヘルパー関数"""
    agent = WebResearchAgent()
    return agent.run_research(topic)


if __name__ == "__main__":
    print("=== WebResearchAgent 単体テスト ===")
    test_topic = "次世代半導体2nmプロセスの実用化動向とRapidusの戦略"
    print(f"調査テーマ: {test_topic}\n")

    def progress_print(step_type, message, details):
        print(f"[{step_type.upper()}] {message}")

    agent = WebResearchAgent(max_iterations=2)
    result = agent.run_research(test_topic, progress_callback=progress_print)

    print("\n" + "=" * 50)
    print("【生成されたレポート】")
    print("=" * 50)
    print(result["report_markdown"][:1000] + "\n...")
    print(f"\n参照ソース数: {len(result['sources'])} 件")
