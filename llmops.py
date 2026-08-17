"""
LLMOps (LLMの可視化・トレーサビリティ・品質評価) モジュール
- 実行ログ・レイテンシ・トークン消費・コストの自動記録
- RAG Triad (忠実性 / 回答適合性 / コンテキスト適合性) に基づく LLM-as-a-Judge 自動評価
- ユーザーフィードバック (👍 / 👎) の収集とメトリクス集計
- オフラインベンチマークテストの自動実行
"""
import os
import time
import json
import uuid
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from config import OPENAI_API_KEY, GOOGLE_API_KEY, get_available_llm_provider
from logger import get_logger

logger = get_logger()

warnings.filterwarnings("ignore")
load_dotenv()

LOGS_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOGS_DIR, "llmops_history.json")

# 推定コスト定数 (1,000トークンあたりのUSD換算目安)
# Gemini 2.5/3.5 Flash: 入力 ~$0.000075 / 出力 ~$0.00030
# GPT-4o-mini: 入力 ~$0.00015 / 出力 ~$0.00060
COST_PER_1K_TOKENS = {
    "gemini": {"input": 0.000075, "output": 0.00030},
    "openai": {"input": 0.000150, "output": 0.00060},
}


def _estimate_tokens(text: str) -> int:
    """日本語・英語混在テキストの概算トークン数を算出（日本語約1.2文字/トークン）"""
    if not text:
        return 0
    return max(1, int(len(text) / 1.5))


def _calculate_cost(input_tokens: int, output_tokens: int, provider: str) -> float:
    """トークン数から概算コスト(USD)を計算"""
    rates = COST_PER_1K_TOKENS.get(provider, COST_PER_1K_TOKENS["gemini"])
    cost = (input_tokens / 1000.0) * rates["input"] + (output_tokens / 1000.0) * rates["output"]
    return round(cost, 6)


class LLMOpsTracker:
    """
    LLM実行ログの記録・永続化・管理を行うクラス
    """

    def __init__(self, log_path: str = LOG_FILE_PATH):
        self.log_path = log_path
        self._ensure_log_file()

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def load_all_logs(self) -> List[Dict[str, Any]]:
        """保存されている全ログを読み込む"""
        self._ensure_log_file()
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
                return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
        except Exception as e:
            logger.error(f"[LLMOpsログ読込エラー] ログファイルの読み込みに失敗しました ({self.log_path}): {e}", exc_info=True)
            return []

    def save_logs(self, logs: List[Dict[str, Any]]):
        """全ログをファイルに保存する"""
        self._ensure_log_file()
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[LLMOpsログ保存エラー] ログファイルの保存に失敗しました ({self.log_path}): {e}", exc_info=True)


    def log_event(
        self,
        feature_type: str,
        question: str,
        answer: str,
        model_name: str,
        provider: str,
        latency_sec: float,
        context_docs: Optional[List[Dict[str, Any]]] = None,
        auto_eval: bool = True
    ) -> Dict[str, Any]:
        """
        LLM呼び出しイベントをロギングする

        Returns:
            Dict[str, Any]: 作成されたログレコード
        """
        context_docs = context_docs or []
        context_text = "\n\n".join([c.get("content", "") for c in context_docs])

        input_text = f"{context_text}\n\n{question}" if context_text else question
        input_tokens = _estimate_tokens(input_text)
        output_tokens = _estimate_tokens(answer)
        cost = _calculate_cost(input_tokens, output_tokens, provider)

        log_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat(timespec="seconds")

        eval_scores = None
        if auto_eval:
            judge = RAGQualityJudge(provider=provider)
            eval_scores = judge.evaluate(question=question, context_text=context_text, answer=answer)

        record: Dict[str, Any] = {
            "log_id": log_id,
            "timestamp": timestamp,
            "feature_type": feature_type,  # "rag_qa" or "web_research"
            "model_name": model_name,
            "provider": provider,
            "latency_sec": round(latency_sec, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost,
            "question": question,
            "answer": answer,
            "context_count": len(context_docs),
            "context_sources": [c.get("source_name", "不明") for c in context_docs],
            "eval_scores": eval_scores,
            "user_feedback": {
                "rating": None,  # "up", "down", or None
                "comment": ""
            }
        }

        logs = self.load_all_logs()
        logs.append(record)
        self.save_logs(logs)
        return record

    def update_feedback(self, log_id: str, rating: str, comment: str = "") -> bool:
        """
        ユーザーフィードバック (👍/👎) を特定のログレコードに更新・付与する
        """
        logs = self.load_all_logs()
        updated = False
        for entry in logs:
            if entry.get("log_id") == log_id:
                entry["user_feedback"] = {
                    "rating": rating,
                    "comment": comment,
                    "updated_at": datetime.now().isoformat(timespec="seconds")
                }
                updated = True
                break

        if updated:
            self.save_logs(logs)
        return updated

    def clear_all_logs(self):
        """ログ履歴をクリアする"""
        self.save_logs([])


class RAGQualityJudge:
    """
    RAG Triad (接地性・忠実性 / 回答適合性 / 検索適合性) に基づく LLM-as-a-Judge 評価器
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or get_available_llm_provider()

    def evaluate(
        self,
        question: str,
        context_text: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        質問・参照コンテキスト・AIの回答を評価し、各指標のスコア (1〜5点) と理由を返す
        """
        if self.provider == "none":
            return {
                "faithfulness": 4,
                "answer_relevance": 4,
                "context_relevance": 4,
                "overall_score": 4.0,
                "has_hallucination": False,
                "reasoning": "APIキー未設定のためデフォルト値を記録しました。"
            }

        eval_prompt = f"""
あなたはRAG (検索拡張生成) システムの回答品質を厳格に評価するAIジャッジです。
以下の【ユーザーの質問】、【提供された講義資料コンテキスト】、【AIの回答】を精査し、3つの指標で1〜5点（整数）で採点してください。

【ユーザーの質問】
{question}

【提供された講義資料コンテキスト】
{context_text if context_text.strip() else "(コンテキストなし / 該当資料なし)"}

【AIの回答】
{answer}

【評価指標の定義】
1. **接地性・忠実性 (faithfulness)**: 回答が提供されたコンテキストのみに基づいており、ハルシネーション（資料にない勝手な創作や虚偽）が含まれていないか。
   - 5点: 完全に資料の記述のみに基づいており、虚偽や過度な推測が一切ない。また「資料に記載がありません」と正確に答えている場合も5点。
   - 3点: 大筋は合っているが、資料にない一般知識が混ざっている。
   - 1点: 資料の記述と矛盾している、または重大なハルシネーションがある。

2. **回答適合性 (answer_relevance)**: ユーザーの質問の意図に対して、直接的・論理的・過不足なく回答しているか。
   - 5点: 質問に対して完璧に回答しており、不要な脱線がない。
   - 3点: 質問に関連しているが、核心を突いていない、または冗長。
   - 1点: 質問の意図と全く異なる回答をしている。

3. **コンテキスト適合性 (context_relevance)**: 検索されたコンテキストが、質問に回答するために必要十分な情報を含んでいたか。
   - 5点: 質問の回答に不可欠な核心部分が的確に含まれている。
   - 3点: 一部関連する情報はあるが、不足している。
   - 1点: 質問と無関係な資料が抽出されている（またはコンテキストが空）。

必ず以下のJSON形式のみで回答してください（コードブロック不要）:
{{
    "faithfulness": 5,
    "answer_relevance": 5,
    "context_relevance": 5,
    "has_hallucination": false,
    "reasoning": "採点理由の簡潔な解説（日本語で1〜2文）"
}}
"""
        try:
            raw_text = self._call_judge_llm(eval_prompt)
            import re
            cleaned_json = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
            data = json.loads(cleaned_json)

            f_score = max(1, min(5, int(data.get("faithfulness", 4))))
            a_score = max(1, min(5, int(data.get("answer_relevance", 4))))
            c_score = max(1, min(5, int(data.get("context_relevance", 4))))
            overall = round((f_score + a_score + c_score) / 3.0, 1)

            return {
                "faithfulness": f_score,
                "answer_relevance": a_score,
                "context_relevance": c_score,
                "overall_score": overall,
                "has_hallucination": bool(data.get("has_hallucination", f_score <= 2)),
                "reasoning": data.get("reasoning", "評価が完了しました。")
            }
        except Exception as e:
            return {
                "faithfulness": 4,
                "answer_relevance": 4,
                "context_relevance": 4,
                "overall_score": 4.0,
                "has_hallucination": False,
                "reasoning": f"自動評価中にエラーが発生しました ({e})。"
            }

    def _call_judge_llm(self, prompt: str) -> str:
        """評価用LLM呼び出し（複数モデル自動フォールバック付き）"""
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GOOGLE_API_KEY)
            candidate_models = [
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.5-flash",
                "gemini-flash-latest"
            ]
            last_err = None
            for m in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    return response.text.strip()
                except Exception as e:
                    last_err = e
                    continue
            raise last_err or RuntimeError("Gemini Judge API call failed.")
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY)
            res = chat.invoke([HumanMessage(content=prompt)])
            return res.content.strip()
        return "{}"


# ==========================================================
# メトリクス集計 & 分析ヘルパー関数
# ==========================================================

def calculate_kpis(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """サマリーKPIを算出する"""
    total_calls = len(logs)
    if total_calls == 0:
        return {
            "total_calls": 0,
            "avg_latency": 0.0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevance": 0.0,
            "avg_context_relevance": 0.0,
            "avg_overall_score": 0.0,
            "hallucination_rate": 0.0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "positive_feedback_rate": 0.0,
            "feedback_count": 0
        }

    total_latency = sum(l.get("latency_sec", 0.0) for l in logs)
    total_tokens = sum(l.get("total_tokens", 0) for l in logs)
    total_cost = sum(l.get("cost_usd", 0.0) for l in logs)

    # 評価スコア集計
    f_scores, a_scores, c_scores, o_scores = [], [], [], []
    hallucination_count = 0

    for l in logs:
        ev = l.get("eval_scores")
        if ev and isinstance(ev, dict):
            f_scores.append(ev.get("faithfulness", 4))
            a_scores.append(ev.get("answer_relevance", 4))
            c_scores.append(ev.get("context_relevance", 4))
            o_scores.append(ev.get("overall_score", 4.0))
            if ev.get("has_hallucination", False):
                hallucination_count += 1

    eval_count = len(o_scores)
    avg_f = sum(f_scores) / eval_count if eval_count else 0.0
    avg_a = sum(a_scores) / eval_count if eval_count else 0.0
    avg_c = sum(c_scores) / eval_count if eval_count else 0.0
    avg_o = sum(o_scores) / eval_count if eval_count else 0.0
    hallucination_rate = (hallucination_count / eval_count * 100) if eval_count else 0.0

    # フィードバック集計
    thumbs_up = 0
    feedback_total = 0
    for l in logs:
        fb = l.get("user_feedback", {})
        rating = fb.get("rating") if isinstance(fb, dict) else None
        if rating in ("up", "down"):
            feedback_total += 1
            if rating == "up":
                thumbs_up += 1

    pos_rate = (thumbs_up / feedback_total * 100) if feedback_total else 0.0

    return {
        "total_calls": total_calls,
        "avg_latency": round(total_latency / total_calls, 2),
        "avg_faithfulness": round(avg_f, 1),
        "avg_answer_relevance": round(avg_a, 1),
        "avg_context_relevance": round(avg_c, 1),
        "avg_overall_score": round(avg_o, 1),
        "hallucination_rate": round(hallucination_rate, 1),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "positive_feedback_rate": round(pos_rate, 1),
        "feedback_count": feedback_total
    }


def get_timeseries_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """時系列グラフ用のDataFrameを作成"""
    if not logs:
        return pd.DataFrame(columns=["timestamp", "latency_sec", "total_tokens", "overall_score"])

    rows = []
    for l in logs:
        ev = l.get("eval_scores") or {}
        rows.append({
            "日時": l.get("timestamp", ""),
            "応答時間 (秒)": l.get("latency_sec", 0.0),
            "トークン数": l.get("total_tokens", 0),
            "品質スコア (1-5)": ev.get("overall_score", 4.0),
            "機能": "専門Q&A" if l.get("feature_type") == "rag_qa" else "Webリサーチ",
            "モデル": l.get("model_name", "不明")
        })

    df = pd.DataFrame(rows)
    df["日時"] = pd.to_datetime(df["日時"], errors="coerce")
    return df.sort_values("日時")


# ==========================================================
# ベンチマーク評価 (オフラインテスト)
# ==========================================================

DEFAULT_BENCHMARK_DATASET = [
    {
        "id": "BM-01",
        "category": "半導体工学",
        "question": "シリコン（Si）とガリウムヒ素（GaAs）のバンドギャップの違いと、それぞれの用途への適性を説明してください。",
        "key_points": "Siは約1.12eVで間接遷移型（集積回路向き）、GaAsは約1.42eVで直接遷移型（発光素子・レーザー向き）"
    },
    {
        "id": "BM-02",
        "category": "半導体工学",
        "question": "n型半導体とp型半導体における多数キャリアとドーピング不純物の違いは何ですか？",
        "key_points": "n型は5価（P, As等）をドープし電子が多数キャリア。p型は3価（B, Ga等）をドープし正孔が多数キャリア。"
    },
    {
        "id": "BM-03",
        "category": "電磁気学 / 未記載テスト",
        "question": "マクスウェル方程式の第3式（ファラデーの電磁誘導の法則）の意味を説明してください。",
        "key_points": "講義資料に電磁気学の記述がない場合は「該当する記載がありません」と答えることが正解（ハルシネーション抑制確認）"
    },
    {
        "id": "BM-04",
        "category": "半導体デバイス",
        "question": "MOSFETのしきい値電圧（Vth）の物理的な意味を教えてください。",
        "key_points": "反転層（チャネル）が形成され、ドレイン電流が流れ始めるゲート電圧。"
    }
]


def run_benchmark(
    vectorstore,
    dataset: Optional[List[Dict[str, Any]]] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    登録されたテストデータセットに対して一括でRAG回答生成とLLM-as-a-Judge評価を実行する
    """
    from rag_chain import query_rag

    dataset = dataset or DEFAULT_BENCHMARK_DATASET
    results = []
    judge = RAGQualityJudge()

    for idx, item in enumerate(dataset):
        q = item["question"]
        if progress_callback:
            progress_callback(idx + 1, len(dataset), q)

        start_t = time.time()
        try:
            rag_res = query_rag(question=q, vectorstore=vectorstore)
            latency = time.time() - start_t
            ans = rag_res.get("answer", "")
            sources = rag_res.get("sources", [])
            context_str = "\n\n".join([s.get("content", "") for s in sources])

            eval_res = judge.evaluate(question=q, context_text=context_str, answer=ans)
        except Exception as e:
            latency = time.time() - start_t
            ans = f"エラー: {e}"
            sources = []
            eval_res = {
                "faithfulness": 1,
                "answer_relevance": 1,
                "context_relevance": 1,
                "overall_score": 1.0,
                "has_hallucination": False,
                "reasoning": f"実行エラー: {e}"
            }

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "expected_key_points": item["key_points"],
            "answer": ans,
            "latency_sec": round(latency, 2),
            "sources_count": len(sources),
            "eval_scores": eval_res
        })

    # 総合スコア集計
    avg_overall = round(sum(r["eval_scores"]["overall_score"] for r in results) / len(results), 2)
    avg_faith = round(sum(r["eval_scores"]["faithfulness"] for r in results) / len(results), 2)
    avg_rel = round(sum(r["eval_scores"]["answer_relevance"] for r in results) / len(results), 2)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_tests": len(results),
        "avg_overall_score": avg_overall,
        "avg_faithfulness": avg_faith,
        "avg_answer_relevance": avg_rel,
        "items": results
    }


if __name__ == "__main__":
    print("=== LLMOpsTracker & RAGQualityJudge 単体テスト ===")
    tracker = LLMOpsTracker()
    test_record = tracker.log_event(
        feature_type="rag_qa",
        question="テスト質問: バンドギャップとは？",
        answer="価電子帯と伝導帯の間の禁制帯エネルギー差です。",
        model_name="gemini-3.5-flash",
        provider="gemini",
        latency_sec=1.2,
        context_docs=[{"source_name": "test.txt", "content": "バンドギャップの説明..."}],
        auto_eval=True
    )
    print(f"Log ID: {test_record['log_id']}")
    print(f"Eval Scores: {test_record['eval_scores']}")
    print(f"Total Logs count: {len(tracker.load_all_logs())}")
