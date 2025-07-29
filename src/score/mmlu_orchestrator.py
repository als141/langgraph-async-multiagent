"""
MMLU専用オーケストレーター - 既存システムを活用
"""

import time
import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

# 既存システムのインポート
from .config import MMLU_AGENTS_CONFIG
from .data_loader import MMLUProblem
from .answer_extractor import AnswerExtractor

# MMLU用の構造化結論生成関数
def _clean_transcript_entry(entry: str) -> str:
    """議論記録のエントリーからJSONのresponseフィールドのみを抽出"""
    import json
    import re
    
    # エージェント名と内容を分離
    if ': ' not in entry:
        return entry
    
    agent_name, content = entry.split(': ', 1)
    
    # 複数行JSONや不完全JSONの処理
    content = content.strip()
    
    # JSONの開始を検出
    if content.startswith('{') or 'json' in content.lower():
        # JSONブロックを抽出する複数のパターンを試行
        json_patterns = [
            r'\{.*?\}',  # 単一行JSON
            r'\{[^}]*"response"\s*:\s*"([^"]*)"[^}]*\}',  # responseフィールドを直接抽出
            r'"response"\s*:\s*"([^"]*)"',  # responseフィールドのみ
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                if pattern == r'"response"\s*:\s*"([^"]*)"':
                    # responseフィールドの値を直接取得
                    return f"{agent_name}: {matches[0]}"
                else:
                    # JSON全体から解析
                    for match in matches:
                        try:
                            if isinstance(match, str) and match.startswith('{'):
                                parsed = json.loads(match)
                                if 'response' in parsed:
                                    return f"{agent_name}: {parsed['response']}"
                        except (json.JSONDecodeError, KeyError, TypeError):
                            continue
        
        # JSONマーカーを削除して通常テキストとして処理
        content = re.sub(r'^.*?json\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'\{.*?\}', '', content, flags=re.DOTALL)
        content = content.strip()
        
        if content:
            return f"{agent_name}: {content}"
    
    return entry

async def generate_mmlu_structured_conclusion(
    full_transcript: List[str], 
    topic: str, 
    available_choices: List[str],
    final_comments: List[str] = None,
    generated_conclusion: str = None
) -> str:
    """MMLU問題用の構造化された結論を生成"""
    from .structured_output import MMLUStructuredExtractor
    
    # 構造化抽出器を使用
    extractor = MMLUStructuredExtractor()
    
    try:
        # 議論記録をクリーンアップ（JSONのresponseフィールドのみ抽出）
        cleaned_transcript = [_clean_transcript_entry(entry) for entry in full_transcript]
        
        # 最終コメントがある場合は追加
        if final_comments:
            # 最終コメントセクションを追加
            cleaned_transcript.append("=== エージェント最終意見 ===")
            for comment in final_comments:
                cleaned_transcript.append(_clean_transcript_entry(comment))
        
        # 生成された最終結論を直接使用して構造化された回答を抽出
        structured_result = extractor.extract_final_answer(
            cleaned_transcript, 
            topic, 
            available_choices, 
            final_conclusion=generated_conclusion
        )
        
        # 結論文を構築（構造化された情報を使用）
        conclusion = f"""
## 議論分析と最終判定

**参加者の議論要約:**
{chr(10).join(cleaned_transcript[-5:])}  # クリーンアップされた最後の5発言（最終意見を含む）

**分析結果:**
{structured_result.reasoning_summary}

**判定結果:**
- 選択された回答: {structured_result.final_answer}
- 信頼度: {structured_result.confidence:.2f}

**最終回答: {structured_result.final_answer}**
"""
        
        print(f"構造化抽出成功: {structured_result.final_answer} (信頼度: {structured_result.confidence:.3f})")
        return conclusion
        
    except Exception as e:
        print(f"構造化抽出でエラー: {e}")
        
        # フォールバック: 従来の方法
        from .graph import llm
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import SystemMessage, HumanMessage
        
        available_letters = [chr(ord('A') + i) for i in range(len(available_choices))]
        
        # フォールバック用の議論記録を作成（最終コメントを含む）
        fallback_transcript = full_transcript.copy()
        if final_comments:
            fallback_transcript.append("=== エージェント最終意見 ===")
            fallback_transcript.extend(final_comments)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""議論を分析し、最適な選択肢を決定してください。
利用可能な選択肢: {', '.join(available_letters)}
必ず「最終回答: [文字]」の形式で回答してください。"""),
            HumanMessage(content=f"""
議論記録:
{chr(10).join(fallback_transcript)}

問題:
{topic}
""")
        ])
        
        chain = prompt | llm
        full_conclusion = ""
        
        async for chunk in chain.astream({}):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if isinstance(content, list):
                    chunk_text = ""
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            chunk_text += item['text']
                        elif isinstance(item, dict) and len(item) == 1 and 'index' in item:
                            continue
                    full_conclusion += chunk_text
                else:
                    content_str = str(content)
                    if content_str.strip() not in ["{'index': 0}", "{'index':0}", "{\"index\":0}", "{\"index\": 0}"]:
                        full_conclusion += content_str
        
        return full_conclusion

# 修正されたrun_graph関数を使用
async def run_mmlu_graph(topic: str, max_turns: int = 15) -> Dict[str, Any]:
    """MMLU用の議論グラフを実行"""
    from .orchestrator import run_graph
    from .config import MMLU_AGENTS_CONFIG
    
    # 既存のrun_graph関数を使用するが、MMLU専用エージェントを設定
    original_config = None
    try:
        # 一時的にMMLU_AGENTS_CONFIGを使用
        import src.score.config as config_module
        original_config = config_module.AGENTS_CONFIG
        config_module.AGENTS_CONFIG = MMLU_AGENTS_CONFIG
        
        # 議論を実行（結論生成前まで）
        results = []
        full_transcript = []
        
        # エージェント別のメッセージバッファ
        agent_messages = {}
        final_comments = []
        generated_conclusion = None  # conclusion_nodeで生成された結論を格納
        
        async for event in run_graph(topic, max_turns):
            results.append(event)
            
            # 最終コメント収集
            if event.get("type") == "final_comments_complete":
                final_comments = event.get("content", [])
                print(f"最終コメント収集: {len(final_comments)}件")
            
            if event.get("type") == "agent_message_chunk":
                # ストリーミングメッセージのチャンクを結合
                agent_name = event.get('agent_name', 'Unknown')
                chunk = event.get('chunk', '')
                
                if agent_name not in agent_messages:
                    agent_messages[agent_name] = ""
                agent_messages[agent_name] += chunk
                
            elif event.get("type") == "agent_message_complete":
                # エージェントメッセージ完了時に記録
                agent_name = event.get('agent_name', 'Unknown')
                if agent_name in agent_messages:
                    complete_message = agent_messages[agent_name].strip()
                    
                    # JSONの場合はresponseフィールドを抽出
                    extracted_message = complete_message
                    try:
                        import json
                        if complete_message.strip().startswith('{') and complete_message.strip().endswith('}'):
                            parsed = json.loads(complete_message.strip())
                            if 'response' in parsed:
                                extracted_message = parsed['response']
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # JSONでない場合や解析失敗時はそのまま使用
                        pass
                    
                    full_transcript.append(f"{agent_name}: {extracted_message}")
                    agent_messages[agent_name] = ""  # リセット
                    
            elif event.get("type") == "agent_message":
                # 非ストリーミングメッセージ（フォールバック）
                agent_name = event.get('agent_name', 'Unknown')
                message = event.get('message', '')
                full_transcript.append(f"{agent_name}: {message}")
                print(f"[Turn {len(full_transcript)}] {agent_name}: {message}")
                
            elif event.get("type") == "conclusion_complete":
                # conclusion_nodeで生成された最終結論を取得
                generated_conclusion = event.get("conclusion", "")
                print(f"Generated conclusion captured: {len(generated_conclusion)} characters")
                break
        
        # MMLU専用の構造化結論生成
        # 選択肢を抽出
        options_match = re.search(r'\*\*選択肢:\*\*\n(.*?)(?=\n\*\*|$)', topic, re.DOTALL)
        available_choices = []
        if options_match:
            options_text = options_match.group(1)
            # A) B) C) D) 形式の選択肢を抽出
            option_pattern = r'([A-Z])\)\s*([^\n]+)'
            found_options = re.findall(option_pattern, options_text)
            available_choices = [opt[1].strip() for opt in found_options]
        
        if not available_choices:
            available_choices = ["選択肢A", "選択肢B", "選択肢C", "選択肢D"]  # フォールバック
        
        mmlu_conclusion = await generate_mmlu_structured_conclusion(
            full_transcript, 
            topic, 
            available_choices, 
            final_comments, 
            generated_conclusion  # conclusion_nodeで生成された結論を渡す
        )
        
        return {
            "conclusion": mmlu_conclusion,
            "full_transcript": full_transcript,
            "turn_count": len([e for e in results if e.get("type") == "agent_message"]),
            "events": results
        }
        
    finally:
        # 元の設定を復元
        if original_config is not None:
            config_module.AGENTS_CONFIG = original_config


@dataclass
class DebateResult:
    """議論結果のデータクラス"""
    question_id: str
    final_conclusion: str
    full_transcript: List[str]
    turn_count: int
    debate_duration: float
    facilitator_interventions: int
    consensus_score: float


class MMLUOrchestrator:
    """MMLU専用の議論オーケストレーター"""
    
    def __init__(self, max_turns: int = 15, log_dir: str = "logs"):
        self.max_turns = max_turns
        self.log_dir = log_dir
        self.answer_extractor = AnswerExtractor()
    
    def format_mmlu_topic(self, problem: MMLUProblem) -> str:
        """MMLU問題を議論トピック形式にフォーマット"""
        formatted_options = []
        for i, option in enumerate(problem.options):
            option_letter = chr(ord('A') + i)
            formatted_options.append(f"{option_letter}) {option}")
        
        options_text = "\n".join(formatted_options)
        
        topic = f"""**問題解決議論**

**問題:**
{problem.question_ja}

**選択肢:**
{options_text}

**議論のガイドライン:**
1. 問題を正確に理解し、重要なポイントを特定してください
2. 各選択肢について具体的な根拠とともに評価してください
3. あなたの専門知識を活用して判断してください
4. 他の参加者の意見を聞き、建設的に議論を深めてください
5. 最終的に「答えは○○です」の形で明確に選択肢を示してください

**重要な制約:**
- 必ず選択肢A、B、C、D、E、F、G、H、I、Jの中から一つを選んでください
- 「分からない」や「判断できない」は避けてください
- 根拠を示して論理的に説明してください"""
        
        return topic
    
    async def run_single_problem_debate(self, problem: MMLUProblem) -> DebateResult:
        """単一問題の議論を実行"""
        start_time = time.time()
        
        print(f"🔄 問題 {problem.question_id} の議論開始")
        
        # 議論トピックの作成
        topic = self.format_mmlu_topic(problem)
        
        try:
            # 既存のマルチエージェントシステムで議論を実行
            result = await run_mmlu_graph(topic, self.max_turns)
            
            debate_duration = time.time() - start_time
            
            return DebateResult(
                question_id=problem.question_id,
                final_conclusion=result.get("conclusion", ""),
                full_transcript=result.get("full_transcript", []),
                turn_count=result.get("turn_count", 0),
                debate_duration=debate_duration,
                facilitator_interventions=0,  # 今後実装
                consensus_score=0.0  # 今後実装
            )
            
        except Exception as e:
            print(f"問題 {problem.question_id} で議論エラー: {e}")
            debate_duration = time.time() - start_time
            
            # エラー時のダミー結果
            return DebateResult(
                question_id=problem.question_id,
                final_conclusion=f"議論中にエラーが発生しました: {e}",
                full_transcript=[],
                turn_count=0,
                debate_duration=debate_duration,
                facilitator_interventions=0,
                consensus_score=0.0
            )
    
    def extract_answer_from_debate(self, debate_result: DebateResult, problem: MMLUProblem) -> str:
        """議論結果から回答を抽出（MMLU用の直接形式）"""
        conclusion = debate_result.final_conclusion
        
        # まず「最終回答: X」形式を探す
        final_answer_pattern = r'最終回答:\s*([A-Z])'
        match = re.search(final_answer_pattern, conclusion)
        
        if match:
            answer = match.group(1)
            # 有効な選択肢の範囲内かチェック
            if answer in [chr(ord('A') + i) for i in range(len(problem.options))]:
                print(f"問題 {problem.question_id}: 直接抽出された回答={answer} (最終回答形式)")
                return answer
        
        # フォールバック: 従来の抽出方法
        extraction = self.answer_extractor.extract_with_confidence(
            conclusion,
            problem.options
        )
        
        print(f"問題 {problem.question_id}: 抽出された回答={extraction.extracted_answer}, 信頼度={extraction.confidence_score:.3f} (フォールバック)")
        
        return extraction.extracted_answer or "A"  # 最終フォールバック
    
    async def run_batch_problems(
        self, 
        problems: List[MMLUProblem],
        progress_callback: Optional[callable] = None
    ) -> List[DebateResult]:
        """複数問題のバッチ実行"""
        results = []
        
        for i, problem in enumerate(problems):
            print(f"\n=== 問題 {i+1}/{len(problems)} ===")
            
            try:
                result = await self.run_single_problem_debate(problem)
                results.append(result)
                
                # 結果サマリー表示
                extracted_answer = self.extract_answer_from_debate(result, problem)
                is_correct = extracted_answer == problem.correct_answer
                status = "✓ 正解" if is_correct else f"✗ 不正解 (正解: {problem.correct_answer})"
                print(f"結果: {status}, ターン数: {result.turn_count}, 時間: {result.debate_duration:.1f}秒")
                
                if progress_callback:
                    progress_callback(i + 1, len(problems), result)
                    
            except Exception as e:
                print(f"問題 {problem.question_id} でエラー: {e}")
                # エラー時のダミー結果
                error_result = DebateResult(
                    question_id=problem.question_id,
                    final_conclusion=f"エラー: {e}",
                    full_transcript=[],
                    turn_count=0,
                    debate_duration=0.0,
                    facilitator_interventions=0,
                    consensus_score=0.0
                )
                results.append(error_result)
        
        return results


# 単体テスト用の関数
async def test_single_mmlu_problem():
    """単一MMLU問題のテスト"""
    from .data_loader import MMLUDataLoader
    
    # データ読み込み
    loader = MMLUDataLoader("data/mmlu_pro_100.csv")
    problems = loader.load_and_validate()
    
    if not problems:
        print("テスト用の問題が見つかりません")
        return
    
    # 最初の問題でテスト
    test_problem = problems[0]
    print(f"テスト問題: {test_problem.question_ja}")
    print(f"選択肢: {test_problem.options}")
    print(f"正解: {test_problem.correct_answer}")
    
    # オーケストレーター実行
    orchestrator = MMLUOrchestrator(max_turns=10)
    result = await orchestrator.run_single_problem_debate(test_problem)
    
    print(f"\n=== 議論結果 ===")
    print(f"ターン数: {result.turn_count}")
    print(f"所要時間: {result.debate_duration:.2f}秒")
    print(f"最終結論: {result.final_conclusion}")
    
    # 回答抽出
    extracted_answer = orchestrator.extract_answer_from_debate(result, test_problem)
    is_correct = extracted_answer == test_problem.correct_answer
    
    print(f"\n=== 評価結果 ===")
    print(f"抽出された回答: {extracted_answer}")
    print(f"正解: {test_problem.correct_answer}")
    print(f"正答: {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    import asyncio
    print("=== MMLU Orchestrator 単体テスト ===")
    asyncio.run(test_single_mmlu_problem())