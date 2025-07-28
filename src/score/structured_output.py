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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


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
        min_length=20,
        max_length=200
    )
    
    @field_validator('reasoning_summary')
    @classmethod
    def validate_reasoning(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("推論要約は20文字以上である必要があります")
        return v.strip()
    
    class Config:
        use_enum_values = True
        validate_assignment = True


class MMLUStructuredExtractor:
    """MMLU用の構造化回答抽出器"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.logger = logging.getLogger(__name__)
        
        # API キーが設定されているかチェック
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0.0,
            streaming=False,
            api_key=api_key
        )
        self.structured_model = self._setup_structured_model()
        self.metrics = {"success": 0, "fallback": 0, "error": 0}
    
    def _setup_structured_model(self):
        """構造化モデルのセットアップ"""
        try:
            return self.model.with_structured_output(
                MMLUStructuredAnswer,
                method="function_calling"  # より確実な方法
            )
        except Exception as e:
            self.logger.warning(f"Function calling setup failed: {e}, using json_mode")
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
        
        # プロンプト作成
        prompt = self._create_extraction_prompt(full_transcript, topic, valid_choices)
        
        # 構造化出力で回答を抽出（リトライ機能付き）
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = self.structured_model.invoke(prompt)
                
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
                    prompt = self._create_explicit_prompt(full_transcript, topic, valid_choices)
                    continue
        
        # 全ての試行が失敗した場合のフォールバック
        return self._fallback_extraction(full_transcript, topic, valid_choices, last_error)
    
    def _determine_valid_choices(self, available_choices: List[str]) -> List[str]:
        """利用可能な選択肢から有効な文字を決定"""
        choice_count = len(available_choices)
        return [chr(ord('A') + i) for i in range(min(choice_count, 10))]  # A-J
    
    def _create_extraction_prompt(
        self, 
        full_transcript: List[str], 
        topic: str, 
        valid_choices: List[str]
    ) -> str:
        """構造化抽出用のプロンプト作成"""
        
        return f"""あなたは多肢選択問題の議論分析専門家です。

以下の議論記録を分析し、参加者の意見を総合して最適な選択肢を決定してください。

**問題:**
{topic}

**議論記録:**
{chr(10).join(full_transcript)}

**利用可能な選択肢:** {', '.join(valid_choices)}

**重要な指示:**
1. 議論の内容を客観的に分析してください
2. 参加者の合意や論理的な根拠を重視してください
3. 必ず {', '.join(valid_choices)} のいずれかを選択してください
4. 信頼度は議論の一致度と論理的妥当性に基づいて設定してください
5. 推論要約では、なぜその選択肢が最適かを簡潔に説明してください

構造化された形式で回答してください。"""
    
    def _create_explicit_prompt(
        self, 
        full_transcript: List[str], 
        topic: str, 
        valid_choices: List[str]
    ) -> str:
        """より明示的なリトライ用プロンプト"""
        
        return f"""【重要】これはリトライです。より注意深く分析してください。

問題分析タスク:
{topic}

議論内容:
{chr(10).join(full_transcript[-5:])}  # 最後の5発言に絞る

選択可能な選択肢: {valid_choices}

必須要件:
- final_answer: {valid_choices} のいずれか一つを厳密に選択
- confidence: 0.0から1.0の数値
- reasoning_summary: 20文字以上200文字以下の説明

指定された構造で必ず回答してください。"""
    
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
            reasoning_summary=f"構造化抽出に失敗したため、パターンマッチングで選択肢{best_choice}を抽出。エラー: {str(error)[:50]}"
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