"""
議論結論からの回答抽出モジュール
"""

import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AnswerExtraction:
    """回答抽出結果"""
    extracted_answer: Optional[str]
    confidence_score: float
    extraction_method: str
    alternative_candidates: List[Tuple[str, float]]
    reasoning: str


class ExtractionStrategy(ABC):
    """回答抽出戦略の基底クラス"""
    
    @abstractmethod
    def extract(self, conclusion: str, options: List[str]) -> AnswerExtraction:
        pass


class PatternMatchingStrategy(ExtractionStrategy):
    """パターンマッチングによる回答抽出"""
    
    def __init__(self):
        # 回答を示すパターン
        self.answer_patterns = [
            r'答えは([A-Z])',
            r'正解は([A-Z])',
            r'選択肢([A-Z])',
            r'([A-Z])が正しい',
            r'([A-Z])だと思う',
            r'([A-Z])です',
            r'([A-Z])\s*:\s*',
            r'([A-Z])\s*番',
            r'([A-Z])\s*を選',
            r'結論.*?([A-Z])',
            r'選択肢は「([A-Z])\)',
            r'適切な選択肢は「([A-Z])\)',
            r'「([A-Z])\)\s*[^」]*」です',
            r'「([A-Z])\)\s*[^」]*」が.*適切',
            r'「([A-Z])\)\s*[^」]*」を.*選',
            r'最も適切な選択肢は.*「([A-Z])\)',
            r'最終的.*選択肢.*「([A-Z])\)',
            r'は「([A-Z])\).*」です',
        ]
    
    def extract(self, conclusion: str, options: List[str]) -> AnswerExtraction:
        candidates = {}
        
        # 各パターンでマッチング
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, conclusion, re.IGNORECASE)
            for match in matches:
                letter = match.upper()
                # 有効な選択肢の範囲内かチェック
                if letter in [chr(ord('A') + i) for i in range(len(options))]:
                    candidates[letter] = candidates.get(letter, 0) + 1
        
        if not candidates:
            return AnswerExtraction(
                extracted_answer=None,
                confidence_score=0.0,
                extraction_method="pattern_matching",
                alternative_candidates=[],
                reasoning="パターンマッチングで回答を抽出できませんでした"
            )
        
        # 最も言及回数の多い選択肢を選択
        best_answer = max(candidates, key=candidates.get)
        confidence = min(candidates[best_answer] / 3.0, 1.0)  # 3回以上で最大信頼度
        
        # 代替候補をスコア順でソート
        alternatives = [(letter, count/max(candidates.values())) 
                       for letter, count in candidates.items() 
                       if letter != best_answer]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return AnswerExtraction(
            extracted_answer=best_answer,
            confidence_score=confidence,
            extraction_method="pattern_matching",
            alternative_candidates=alternatives,
            reasoning=f"パターンマッチングで'{best_answer}'を{candidates[best_answer]}回検出"
        )


class SemanticSimilarityStrategy(ExtractionStrategy):
    """選択肢テキストとの意味的類似度による抽出"""
    
    def extract(self, conclusion: str, options: List[str]) -> AnswerExtraction:
        # 簡単な単語ベースの類似度計算
        similarities = []
        
        for i, option in enumerate(options):
            # 選択肢の重要な単語を抽出
            option_words = set(option.lower().split())
            conclusion_words = set(conclusion.lower().split())
            
            # Jaccard類似度を計算
            intersection = len(option_words & conclusion_words)
            union = len(option_words | conclusion_words)
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            similarities.append((chr(ord('A') + i), similarity))
        
        # 最も類似度の高い選択肢を選択
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if similarities[0][1] > 0.1:  # 最低閾値
            best_answer = similarities[0][0]
            confidence = similarities[0][1]
            alternatives = similarities[1:3]  # 上位2つの代替候補
            
            return AnswerExtraction(
                extracted_answer=best_answer,
                confidence_score=confidence,
                extraction_method="semantic_similarity",
                alternative_candidates=alternatives,
                reasoning=f"意味的類似度{confidence:.3f}で'{best_answer}'を選択"
            )
        else:
            return AnswerExtraction(
                extracted_answer=None,
                confidence_score=0.0,
                extraction_method="semantic_similarity",
                alternative_candidates=[],
                reasoning="十分な意味的類似度が見つかりませんでした"
            )


class LastResortStrategy(ExtractionStrategy):
    """最後の手段：結論の最後の文字などから推測"""
    
    def extract(self, conclusion: str, options: List[str]) -> AnswerExtraction:
        # 結論の最後の部分から選択肢を探す
        last_chars = conclusion[-50:] if len(conclusion) > 50 else conclusion
        
        # 有効な選択肢文字を検索
        valid_letters = [chr(ord('A') + i) for i in range(len(options))]
        found_letters = []
        
        for char in reversed(last_chars):
            if char.upper() in valid_letters:
                found_letters.append(char.upper())
        
        if found_letters:
            # 最後に見つかった文字を採用
            answer = found_letters[0]
            confidence = 0.3  # 低い信頼度
            
            return AnswerExtraction(
                extracted_answer=answer,
                confidence_score=confidence,
                extraction_method="last_resort",
                alternative_candidates=[],
                reasoning=f"結論の末尾から'{answer}'を推測（低信頼度）"
            )
        else:
            # 完全に失敗した場合はランダムにAを選択
            return AnswerExtraction(
                extracted_answer='A',
                confidence_score=0.1,
                extraction_method="random_fallback",
                alternative_candidates=[],
                reasoning="回答を抽出できなかったため、デフォルトでAを選択"
            )


class AnswerExtractor:
    """回答抽出器のメインクラス"""
    
    def __init__(self):
        self.extraction_strategies = [
            PatternMatchingStrategy(),
            SemanticSimilarityStrategy(),
            LastResortStrategy()
        ]
    
    def extract_with_confidence(
        self, 
        conclusion: str, 
        options: List[str]
    ) -> AnswerExtraction:
        """複数手法で回答を抽出し、信頼度を計算"""
        
        if not conclusion or not options:
            return AnswerExtraction(
                extracted_answer=None,
                confidence_score=0.0,
                extraction_method="input_validation_failed",
                alternative_candidates=[],
                reasoning="入力データが不正です"
            )
        
        best_extraction = None
        all_extractions = []
        
        # 各戦略を順番に試行
        for strategy in self.extraction_strategies:
            try:
                extraction = strategy.extract(conclusion, options)
                all_extractions.append(extraction)
                
                # 有効な回答が得られた場合
                if extraction.extracted_answer and extraction.confidence_score > 0:
                    if (best_extraction is None or 
                        extraction.confidence_score > best_extraction.confidence_score):
                        best_extraction = extraction
                        
                    # 高い信頼度が得られた場合は早期終了
                    if extraction.confidence_score > 0.8:
                        break
                        
            except Exception as e:
                print(f"抽出戦略でエラー: {type(strategy).__name__}: {e}")
                continue
        
        # 最良の抽出結果を返す
        if best_extraction:
            return best_extraction
        else:
            # 全ての戦略が失敗した場合のフォールバック
            return AnswerExtraction(
                extracted_answer='A',
                confidence_score=0.0,
                extraction_method="complete_failure",
                alternative_candidates=[],
                reasoning="全ての抽出戦略が失敗したため、デフォルトでAを選択"
            )
    
    def batch_extract(
        self, 
        conclusions: List[str], 
        options_list: List[List[str]]
    ) -> List[AnswerExtraction]:
        """複数の結論をバッチ処理で抽出"""
        results = []
        
        for conclusion, options in zip(conclusions, options_list):
            extraction = self.extract_with_confidence(conclusion, options)
            results.append(extraction)
        
        return results


if __name__ == "__main__":
    # テスト用のコード
    extractor = AnswerExtractor()
    
    test_cases = [
        {
            "conclusion": "この問題を検討した結果、答えはBです。",
            "options": ["選択肢A", "選択肢B", "選択肢C", "選択肢D"]
        },
        {
            "conclusion": "複数の観点から分析した結果、選択肢Cが最も適切だと思います。",
            "options": ["オプション1", "オプション2", "オプション3", "オプション4"]
        },
        {
            "conclusion": "様々な議論を経て、最終的にDが正しいという結論に達しました。",
            "options": ["A案", "B案", "C案", "D案"]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== テストケース {i+1} ===")
        print(f"結論: {test_case['conclusion']}")
        print(f"選択肢: {test_case['options']}")
        
        result = extractor.extract_with_confidence(
            test_case['conclusion'], 
            test_case['options']
        )
        
        print(f"抽出結果: {result.extracted_answer}")
        print(f"信頼度: {result.confidence_score:.3f}")
        print(f"手法: {result.extraction_method}")
        print(f"理由: {result.reasoning}")