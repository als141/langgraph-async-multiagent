"""
MMLU専用構造化出力モジュール
最終回答部分のみを厳格に型定義
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List
from enum import Enum
import re
import logging
from langchain_openai import ChatOpenAI


class ChoiceOption(str, Enum):
    """有効な選択肢の厳格な定義"""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"


class MMLUStructuredAnswer(BaseModel):
    """MMLU問題の構造化された最終回答"""
    
    final_answer: ChoiceOption = Field(
        description="最終的な選択肢。必ずA-Jのいずれかを選択してください。"
    )
    
    confidence: float = Field(
        description="回答の信頼度（0.0-1.0）",
        ge=0.0,
        le=1.0
    )
    
    reasoning_summary: str = Field(
        description="選択した理由の簡潔な要約",
        min_length=10,
        max_length=100
    )
    
    @field_validator('reasoning_summary')
    @classmethod
    def validate_reasoning(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("推論要約は10文字以上である必要があります")
        if len(v.strip()) > 100:
            # 100文字を超える場合は切り詰める
            return v.strip()[:100]
        return v.strip()
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class MMLUStructuredExtractor:
    """MMLU用の構造化回答抽出器"""
    
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.logger = logging.getLogger(__name__)
        
        # API キーが設定されているかチェック
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.5,
            streaming=False,
            api_key=api_key,
            use_responses_api=True,
            use_previous_response_id=True,
        )
        self.structured_model = self._setup_structured_model()
        self.metrics = {"success": 0, "fallback": 0, "error": 0}
    
    def _setup_structured_model(self):
        """構造化モデルのセットアップ（Responses API対応）"""
        try:
            # Method 1: Responses APIでbind()を使用
            return self.model.bind(
                response_format=MMLUStructuredAnswer,
                strict=True
            )
        except Exception as e:
            self.logger.warning(f"Method 1 (bind) failed: {e}")
            try:
                # Method 2: bind_toolsで空のツールと一緒に使用
                return self.model.bind_tools(
                    [],  # 空のツールリスト
                    response_format=MMLUStructuredAnswer,
                    strict=True
                )
            except Exception as e2:
                self.logger.warning(f"Method 2 (bind_tools) failed: {e2}")
                try:
                    # Method 3: 従来のwith_structured_output
                    return self.model.with_structured_output(
                        MMLUStructuredAnswer,
                        method="function_calling"
                    )
                except Exception as e3:
                    self.logger.warning(f"Method 3 (function_calling) failed: {e3}")
                    # Method 4: JSON mode フォールバック
                    return self.model.with_structured_output(
                        MMLUStructuredAnswer,
                        method="json_mode"
                    )
    
    def extract_final_answer(
        self, 
        full_transcript: List[str], 
        topic: str, 
        available_choices: List[str]
    ) -> MMLUStructuredAnswer:
        """
        議論の記録から構造化された最終回答を抽出
        
        Args:
            full_transcript: 議論の完全な記録
            topic: 問題のトピック（選択肢情報を含む）
            available_choices: 利用可能な選択肢のリスト
            
        Returns:
            MMLUStructuredAnswer: 構造化された最終回答
        """
        
        # 利用可能な選択肢を特定
        valid_choices = self._determine_valid_choices(available_choices)
        
        # プロンプト作成（システムとユーザーを分離）
        prompt_data = self._create_extraction_prompt(full_transcript, topic, valid_choices)
        
        # 構造化出力で回答を抽出（リトライ機能付き）
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # システムプロンプトとユーザープロンプトを分離して送信
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=prompt_data["system"]),
                    HumanMessage(content=prompt_data["user"])
                ]
                response = self.structured_model.invoke(messages)
                
                # Responses APIの場合、additional_kwargsからparsedを取得
                if hasattr(response, 'additional_kwargs') and 'parsed' in response.additional_kwargs:
                    result = response.additional_kwargs['parsed']
                elif hasattr(response, 'final_answer'):
                    result = response
                else:
                    # レスポンスがMMLUStructuredAnswerオブジェクトでない場合
                    raise ValueError(f"予期しないレスポンス形式: {type(response)}")
                
                # 選択肢の有効性を追加チェック
                if result.final_answer not in valid_choices:
                    raise ValueError(f"無効な選択肢: {result.final_answer}")
                
                self.metrics["success"] += 1
                self.logger.info(f"構造化抽出成功: {result.final_answer} (信頼度: {result.confidence:.3f})")
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"構造化抽出失敗 (試行 {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # リトライ時はより明示的なプロンプトを使用
                    prompt_data = self._create_explicit_prompt(full_transcript, topic, valid_choices)
                    continue
        
        # 全ての試行が失敗した場合のフォールバック
        return self._fallback_extraction(full_transcript, topic, valid_choices, last_error)
    
    def _determine_valid_choices(self, available_choices: List[str]) -> List[str]:
        """利用可能な選択肢から有効な文字を決定"""
        choice_count = len(available_choices)
        return [chr(ord('A') + i) for i in range(min(choice_count, 10))]  # A-J
    
    def _extract_question_and_options(self, topic: str) -> tuple[str, str]:
        """topicから問題文と選択肢のみを抽出"""
        import re
        
        # 問題文を抽出
        question_match = re.search(r'\*\*問題:\*\*\s*\n(.*?)(?=\n\*\*|$)', topic, re.DOTALL)
        question = question_match.group(1).strip() if question_match else "問題文が見つかりません"
        
        # 選択肢を抽出
        options_match = re.search(r'\*\*選択肢:\*\*\s*\n(.*?)(?=\n\*\*|$)', topic, re.DOTALL)
        options = options_match.group(1).strip() if options_match else "選択肢が見つかりません"
        
        return question, options
    
    def _create_extraction_prompt(
        self, 
        full_transcript: List[str], 
        topic: str, 
        valid_choices: List[str]
    ) -> dict:
        """構造化抽出用のプロンプト作成（システムとユーザーを分離）"""
        
        # topicから問題文と選択肢のみを抽出
        question, options = self._extract_question_and_options(topic)
        
        system_prompt = f"""あなたは多肢選択問題の議論分析専門家です。

以下の指示に従って分析してください:
- 議論の内容を客観的に分析してください
- 参加者の合意や論理的な根拠を重視してください
- 必ず {', '.join(valid_choices)} のいずれかを選択してください
- 信頼度は議論の一致度と論理的妥当性に基づいて設定してください
- 推論要約は100文字以内で簡潔に説明してください

回答は以下のJSON形式で行ってください:
{{
  "final_answer": "選択した選択肢",
  "confidence": 信頼度(0.0-1.0),
  "reasoning_summary": "選択理由の要約"
}}"""

        user_prompt = f"""以下の議論記録を分析し、参加者の意見を総合して最適な選択肢を決定してください。

**問題:**
{question}

**選択肢:**
{options}

**議論記録:**
{chr(10).join(full_transcript)}"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _create_explicit_prompt(
        self, 
        full_transcript: List[str], 
        topic: str, 
        valid_choices: List[str]
    ) -> dict:
        """より明示的なリトライ用プロンプト（システムとユーザーを分離）"""
        
        # topicから問題文と選択肢のみを抽出
        question, options = self._extract_question_and_options(topic)
        
        system_prompt = f"""あなたは多肢選択問題の議論分析専門家です。

【重要】これはリトライです。より注意深く分析してください。

必須要件:
- final_answer: {valid_choices} のいずれか一つを厳密に選択
- confidence: 0.0から1.0の数値
- reasoning_summary: 10文字以上100文字以下の説明

回答は必ずJSON形式で行ってください:
{{
  "final_answer": "選択した選択肢",
  "confidence": 信頼度(0.0-1.0),
  "reasoning_summary": "選択理由の要約"
}}"""

        user_prompt = f"""以下の議論記録を分析し、参加者の意見を総合して最適な選択肢を決定してください。

**問題:**
{question}

**選択肢:**
{options}

**議論内容（最新5発言）:**
{chr(10).join(full_transcript[-5:]) if full_transcript else "議論記録なし"}"""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _fallback_extraction(
        self, 
        full_transcript: List[str], 
        topic: str, 
        valid_choices: List[str], 
        error: Exception
    ) -> MMLUStructuredAnswer:
        """フォールバック: パターンマッチングによる抽出"""
        
        self.metrics["fallback"] += 1
        self.logger.error(f"構造化抽出が完全に失敗: {error}")
        
        # 議論記録から選択肢を探す
        full_text = " ".join(full_transcript)
        
        # パターンマッチングで選択肢を検索
        choice_counts = {}
        for choice in valid_choices:
            # 様々なパターンで選択肢を検索
            patterns = [
                f"答えは{choice}",
                f"選択肢{choice}",
                f"{choice}が正しい",
                f"{choice}だと思う",
                f"{choice}です",
                f"{choice})",
                f"「{choice})",
            ]
            
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, full_text, re.IGNORECASE))
            choice_counts[choice] = count
        
        # 最も多く言及された選択肢を選択
        if choice_counts and max(choice_counts.values()) > 0:
            best_choice = max(choice_counts, key=choice_counts.get)
            confidence = min(choice_counts[best_choice] / 5.0, 0.8)  # 最大0.8
        else:
            # 完全にフォールバック
            best_choice = valid_choices[0]  # 最初の選択肢
            confidence = 0.1
        
        return MMLUStructuredAnswer(
            final_answer=ChoiceOption(best_choice),
            confidence=confidence,
            reasoning_summary=f"パターン抽出で{best_choice}を選択"
        )
    
    def get_metrics(self) -> dict:
        """処理メトリクスを取得"""
        total = sum(self.metrics.values())
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "success_rate": self.metrics["success"] / total,
            "fallback_rate": self.metrics["fallback"] / total,
            "error_rate": self.metrics["error"] / total,
            "total_processed": total
        }


# テスト関数
def test_structured_extraction():
    """構造化抽出のテスト"""
    
    # API キーがない場合はフォールバックテストを実行
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY環境変数が設定されていません。フォールバックテストを実行します。")
        
        print("\n=== 基本機能テスト（構造化なし） ===")
        try:
            from .answer_extractor import AnswerExtractor
            basic_extractor = AnswerExtractor()
            
            test_conclusion = "この問題を検討した結果、答えはCのLIFO memoryです。"
            test_options = ["FIFO memory", "Flash memory", "LIFO memory", "LILO memory"]
            
            basic_result = basic_extractor.extract_with_confidence(test_conclusion, test_options)
            print(f"基本抽出結果: {basic_result.extracted_answer}")
            print(f"信頼度: {basic_result.confidence_score:.3f}")
            print(f"手法: {basic_result.extraction_method}")
            
            return basic_result
            
        except Exception as basic_error:
            print(f"基本テストも失敗: {basic_error}")
            return None
    
    try:
        extractor = MMLUStructuredExtractor()
        
        # テストデータ
        test_transcript = [
            "論理分析者: この問題はスタックの特徴について聞いています。",
            "知識統合者: スタックはLIFO（後入れ先出し）の特徴を持ちます。",
            "批判的検証者: 選択肢を検討すると、Cが最も適切です。",
            "論理分析者: 同意します。答えはCのLIFO memoryです。"
        ]
        
        test_topic = """**問題:**
スタックは別名何と呼ばれますか

**選択肢:**
A) FIFO memory
B) Flash memory  
C) LIFO memory
D) LILO memory"""
        
        test_choices = ["FIFO memory", "Flash memory", "LIFO memory", "LILO memory"]
        
        result = extractor.extract_final_answer(test_transcript, test_topic, test_choices)
        
        print("=== 構造化抽出テスト結果 ===")
        print(f"最終回答: {result.final_answer}")
        print(f"信頼度: {result.confidence:.3f}")
        print(f"推論要約: {result.reasoning_summary}")
        print(f"メトリクス: {extractor.get_metrics()}")
        
        return result
        
    except Exception as e:
        print(f"テスト失敗: {e}")
        
        # フォールバックテスト: 構造化なしでの基本テスト
        print("\n=== 基本機能テスト（構造化なし） ===")
        try:
            from .answer_extractor import AnswerExtractor
            basic_extractor = AnswerExtractor()
            
            test_conclusion = "この問題を検討した結果、答えはCのLIFO memoryです。"
            test_options = ["FIFO memory", "Flash memory", "LIFO memory", "LILO memory"]
            
            basic_result = basic_extractor.extract_with_confidence(test_conclusion, test_options)
            print(f"基本抽出結果: {basic_result.extracted_answer}")
            print(f"信頼度: {basic_result.confidence_score:.3f}")
            
            return basic_result
            
        except Exception as basic_error:
            print(f"基本テストも失敗: {basic_error}")
            return None


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    print("=== MMLU構造化出力モジュール テスト ===")
    
    # テスト実行
    test_result = test_structured_extraction()
    
    if test_result:
        if hasattr(test_result, 'final_answer'):
            print(f"\n✅ 構造化抽出成功: {test_result.final_answer}")
        else:
            print(f"\n✅ 基本抽出成功: {test_result.extracted_answer}")
    else:
        print("\n❌ 全てのテストが失敗")