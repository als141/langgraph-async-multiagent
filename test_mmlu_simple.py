#!/usr/bin/env python3
"""
MMLU システムの簡単なテスト
API キーなしでも動作する基本機能テスト
"""

import sys
import os
sys.path.append('.')

from src.score.data_loader import MMLUDataLoader
from src.score.answer_extractor import AnswerExtractor
from src.score.structured_output import test_structured_extraction

def test_data_loading():
    """データ読み込みテスト"""
    print("=== データ読み込みテスト ===")
    
    loader = MMLUDataLoader("data/mmlu_pro_100.csv")
    
    # 統計取得
    stats = loader.get_category_stats()
    print(f"有効問題数: {sum(stats.values())}")
    print(f"カテゴリ別: {stats}")
    
    # サンプル問題取得
    problems = loader.stratified_sample(2)  # 各カテゴリ2問ずつ
    print(f"サンプル問題数: {len(problems)}")
    
    if problems:
        first_problem = problems[0]
        print(f"\n問題例:")
        print(f"  ID: {first_problem.question_id}")
        print(f"  問題: {first_problem.question_ja}")
        print(f"  選択肢数: {len(first_problem.options)}")
        print(f"  正解: {first_problem.correct_answer}")
        
        return problems
    
    return []

def test_answer_extraction():
    """回答抽出テスト"""
    print("\n=== 回答抽出テスト ===")
    
    extractor = AnswerExtractor()
    
    test_cases = [
        {
            "conclusion": "議論の結果、答えはCのLIFO memoryです。",
            "options": ["FIFO memory", "Flash memory", "LIFO memory", "LILO memory"],
            "expected": "C"
        },
        {
            "conclusion": "最終回答: B",
            "options": ["選択肢A", "選択肢B", "選択肢C"],
            "expected": "B"
        },
        {
            "conclusion": "選択肢Dが最も適切だと判断します。",
            "options": ["オプション1", "オプション2", "オプション3", "オプション4"],
            "expected": "D"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        result = extractor.extract_with_confidence(
            test_case["conclusion"],
            test_case["options"]
        )
        
        is_correct = result.extracted_answer == test_case["expected"]
        status = "✓" if is_correct else "✗"
        
        print(f"テスト {i+1}: {status} 期待={test_case['expected']}, 結果={result.extracted_answer}, 信頼度={result.confidence_score:.3f}")
        
        if is_correct:
            success_count += 1
    
    print(f"成功率: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    
    return success_count == len(test_cases)

def test_structured_output():
    """構造化出力テスト"""
    print("\n=== 構造化出力テスト ===")
    
    # API キーの有無をチェック
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY未設定のため、基本テストのみ実行")
        result = test_structured_extraction()
        return result is not None
    else:
        print("OPENAI_API_KEY設定済み - 完全なテストを実行")
        result = test_structured_extraction()
        return result is not None and hasattr(result, 'final_answer')

def test_mmlu_topic_formatting():
    """MMLU トピックフォーマット テスト"""
    print("\n=== MMLU トピックフォーマット テスト ===")
    
    # MMLUOrchestratorをインポートせずに、フォーマット機能のみテスト
    from src.score.data_loader import MMLUProblem
    
    # テスト用の問題データ
    test_problem = MMLUProblem(
        question_id="test_001",
        question="What is a stack also known as?",
        question_ja="スタックは別名何と呼ばれますか",
        options=["FIFO memory", "Flash memory", "LIFO memory", "LILO memory"],
        correct_answer="C",
        correct_index=2,
        category="engineering",
        source="test",
        cot_content=""
    )
    
    # フォーマット処理を手動で実行
    formatted_options = []
    for i, option in enumerate(test_problem.options):
        option_letter = chr(ord('A') + i)
        formatted_options.append(f"{option_letter}) {option}")
    
    options_text = "\n".join(formatted_options)
    
    topic = f"""**問題解決議論**

**問題:**
{test_problem.question_ja}

**選択肢:**
{options_text}

**議論のガイドライン:**
1. 問題を正確に理解し、重要なポイントを特定してください
2. 各選択肢について具体的な根拠とともに評価してください
3. あなたの専門知識を活用して判断してください
4. 他の参加者の意見を聞き、建設的に議論を深めてください
5. 最終的に「答えは○○です」の形で明確に選択肢を示してください

**重要な制約:**
- 必ず選択肢A、B、C、D（またはそれ以上）の中から一つを選んてください
- 「分からない」や「判断できない」は避けてください
- 根拠を示して論理的に説明してください"""
    
    print("フォーマットされたトピック:")
    print(topic[:200] + "...")
    
    # 選択肢の抽出テスト
    import re
    options_match = re.search(r'\*\*選択肢:\*\*\n(.*?)(?=\n\*\*|$)', topic, re.DOTALL)
    if options_match:
        options_text = options_match.group(1)
        option_pattern = r'([A-Z])\)\s*([^\n]+)'
        found_options = re.findall(option_pattern, options_text)
        
        print(f"抽出された選択肢: {len(found_options)}個")
        for letter, text in found_options:
            print(f"  {letter}: {text}")
        
        return len(found_options) == len(test_problem.options)
    
    return False

def main():
    """メインテスト実行"""
    print("🎯 MMLU システム 基本機能テスト")
    print("="*50)
    
    tests = [
        ("データ読み込み", test_data_loading),
        ("回答抽出", test_answer_extraction),
        ("構造化出力", test_structured_output),
        ("MMLU フォーマット", test_mmlu_topic_formatting),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            # データ読み込みテストの場合は結果が問题リスト
            if isinstance(result, list):
                status = "✅ 成功" if len(result) > 0 else "❌ 失敗"
            else:
                status = "✅ 成功" if result else "❌ 失敗"
            
            print(f"\n{test_name}: {status}")
            
        except Exception as e:
            print(f"\n{test_name}: ❌ エラー - {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n" + "="*50)
    print("📊 テスト結果サマリー")
    print("="*50)
    
    success_count = 0
    for test_name, result in results:
        if isinstance(result, list):
            success = len(result) > 0
        else:
            success = result
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        
        if success:
            success_count += 1
    
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("\n🎉 全てのテストが成功しました！")
        print("\nMMLU システムの基本機能は正常に動作しています。")
        print("実際のベンチマーク実行にはOPENAI_API_KEYの設定が必要です。")
    else:
        print(f"\n⚠️  {len(results) - success_count}個のテストが失敗しました。")

if __name__ == "__main__":
    main()