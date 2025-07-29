#!/usr/bin/env python3
"""
シンプルスコアリングシステム実行スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from simple_mmlu_scorer import SimpleMMLUScorer

def main():
    """メイン実行"""
    print("="*60)
    print("🤖 シンプルGPT MMLUスコアリングシステム")
    print("="*60)
    
    # CSVファイルのパス
    csv_path = project_root / "data" / "mmlu_pro_100.csv"
    output_path = project_root / "results" / "simple_gpt41_results.json"
    
    if not csv_path.exists():
        print(f"❌ CSVファイルが見つかりません: {csv_path}")
        return 1
    
    try:
        # スコアラー初期化
        scorer = SimpleMMLUScorer()
        
        # 少数問題でテスト実行
        print("🧪 テスト実行（最初の5問）")
        results = scorer.score_all_problems(str(csv_path), str(output_path))
        
        print("\n" + "="*60)
        print("🎯 テスト結果サマリー")
        print("="*60)
        print(f"総問題数: {results['total_questions']}")
        print(f"正答数: {results['correct_answers']}")
        print(f"正答率: {results['overall_accuracy']:.1%}")
        
        # カテゴリ別統計（もしあれば）
        category_stats = {}
        for result in results['results']:
            # 問題IDからカテゴリを推測（簡易版）
            category = "unknown"  # 実際のCSVにcategoryがある場合は使用
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if result.is_correct:
                category_stats[category]["correct"] += 1
        
        print(f"\n📁 詳細結果: {output_path}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())