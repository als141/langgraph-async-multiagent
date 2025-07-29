#!/usr/bin/env python3
"""
シンプルスコアリングシステムのテスト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from simple_mmlu_scorer import SimpleMMLUScorer

def test_single_question():
    """単一問題のテスト"""
    print("🧪 単一問題テスト")
    
    # テスト用問題
    question = "スタックは別名何と呼ばれますか？"
    options = [
        "FIFO memory",
        "Flash memory", 
        "LIFO memory",
        "LILO memory"
    ]
    correct_answer = "C"
    
    try:
        scorer = SimpleMMLUScorer()
        
        print(f"問題: {question}")
        print("選択肢:")
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            print(f"  {letter}) {option}")
        print(f"正解: {correct_answer}")
        print()
        
        # 回答取得
        result = scorer.get_answer(question, options)
        
        print("結果:")
        print(f"  予測回答: {result['predicted_answer']}")
        print(f"  正解: {correct_answer}")
        print(f"  正答: {'✅' if result['predicted_answer'] == correct_answer else '❌'}")
        print(f"  応答時間: {result['response_time']:.2f}秒")
        print(f"  成功: {'✅' if result['success'] else '❌'}")
        
        if not result['success']:
            print(f"  エラー: {result.get('error', 'Unknown')}")
        
        if 'raw_response' in result:
            print(f"  生レスポンス: {result['raw_response']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False

def test_csv_loading():
    """CSV読み込みテスト"""
    print("\n🧪 CSV読み込みテスト")
    
    csv_path = project_root / "data" / "mmlu_pro_100.csv"
    
    if not csv_path.exists():
        print(f"❌ CSVファイルが見つかりません: {csv_path}")
        return False
    
    try:
        scorer = SimpleMMLUScorer()
        problems = scorer.load_csv_data(str(csv_path))
        
        print(f"✅ {len(problems)}問を読み込み成功")
        
        # 最初の問題を表示
        if problems:
            first_problem = problems[0]
            print(f"\n最初の問題例:")
            print(f"  ID: {first_problem['question_id']}")
            print(f"  問題: {first_problem['question'][:100]}...")
            print(f"  選択肢数: {len(first_problem['options'])}")
            print(f"  正解: {first_problem['correct_answer']}")
        
        return True
    
    except Exception as e:
        print(f"❌ CSV読み込みエラー: {e}")
        return False

def test_few_questions():
    """少数問題のテスト"""
    print("\n🧪 少数問題テスト（3問）")
    
    csv_path = project_root / "data" / "mmlu_pro_100.csv"
    
    if not csv_path.exists():
        print(f"❌ CSVファイルが見つかりません: {csv_path}")
        return False
    
    try:
        scorer = SimpleMMLUScorer()
        problems = scorer.load_csv_data(str(csv_path))
        
        # 最初の3問をテスト
        test_problems = problems[:3]
        results = []
        correct_count = 0
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n[{i}/3] 問題 {problem['question_id']}")
            print(f"問題: {problem['question'][:80]}...")
            
            answer_result = scorer.get_answer(problem['question'], problem['options'])
            is_correct = answer_result['predicted_answer'] == problem['correct_answer']
            
            if is_correct:
                correct_count += 1
            
            status = "✅" if is_correct else "❌"
            print(f"結果: {status} 予測={answer_result['predicted_answer']}, 正解={problem['correct_answer']}")
            
            results.append({
                'question_id': problem['question_id'],
                'predicted': answer_result['predicted_answer'],
                'correct': problem['correct_answer'],
                'is_correct': is_correct,
                'time': answer_result['response_time']
            })
        
        accuracy = correct_count / len(test_problems)
        print(f"\n🎯 テスト結果: {correct_count}/{len(test_problems)} ({accuracy:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ 少数問題テストエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("="*60)
    print("🧪 シンプルMMLUスコアラー テストスイート")
    print("="*60)
    
    # APIキー確認
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY環境変数が設定されていません")
        return 1
    else:
        print("✅ OPENAI_API_KEY が設定されています")
    
    # テスト実行
    tests = [
        ("単一問題テスト", test_single_question),
        ("CSV読み込みテスト", test_csv_loading),
        ("少数問題テスト", test_few_questions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"実行中: {test_name}")
        print(f"{'='*40}")
        
        try:
            if test_func():
                print(f"✅ {test_name} 成功")
                passed += 1
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 例外: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎯 テスト結果: {passed}/{total} 成功")
    print(f"{'='*60}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())