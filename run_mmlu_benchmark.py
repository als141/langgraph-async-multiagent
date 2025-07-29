#!/usr/bin/env python3
"""
MMLU PROベンチマーク実行スクリプト
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.score.mmlu_benchmark_runner import MMLUBenchmarkRunner, BenchmarkConfig


def create_output_directory(base_dir: str = "results") -> str:
    """出力ディレクトリを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"mmlu_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def print_banner():
    """バナー表示"""
    print("="*80)
    print("🎯 MMLU PRO マルチエージェント議論ベンチマーク")
    print("="*80)
    print()


def print_config_summary(config: BenchmarkConfig):
    """設定サマリーの表示"""
    print("📋 実行設定:")
    print(f"  データセット: {config.dataset_path}")
    print(f"  出力ディレクトリ: {config.output_dir}")
    print(f"  カテゴリあたり問題数: {config.questions_per_category}")
    print(f"  最大ターン数: {config.max_turns_per_question}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  同時実行数: {config.max_concurrent}")
    print(f"  タイムアウト: {config.timeout_per_question}秒")
    print()


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='MMLU PROベンチマークを実行します',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全100問実行（デフォルト）
  python run_mmlu_benchmark.py
  
  # 各カテゴリ10問ずつ実行
  python run_mmlu_benchmark.py --questions-per-category 10
  
  # 議論ターン数を制限
  python run_mmlu_benchmark.py --max-turns 10
  
  # カスタムデータセットを使用
  python run_mmlu_benchmark.py --dataset custom_data.csv
  
  # チェックポイントから再開
  python run_mmlu_benchmark.py --resume --output-dir results/mmlu_benchmark_20240101_120000
        """
    )
    
    # 引数定義
    parser.add_argument(
        '--dataset',
        default='data/mmlu_pro_100.csv',
        help='データセットのパス (デフォルト: data/mmlu_pro_100.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='出力ディレクトリ (デフォルト: 自動生成)'
    )
    
    parser.add_argument(
        '--questions-per-category',
        type=int,
        default=25,
        help='カテゴリあたり問題数 (デフォルト: 25)'
    )
    
    parser.add_argument(
        '--max-turns',
        type=int,
        default=15,
        help='問題あたり最大ターン数 (デフォルト: 15)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='バッチサイズ (デフォルト: 10)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=1,
        help='同時実行数 (デフォルト: 1、API制限に注意)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='問題あたりタイムアウト秒数 (デフォルト: 300)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='チェックポイントから再開'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='テストモード（各カテゴリ2問ずつ）'
    )
    
    parser.add_argument(
        '--experiment-mode',
        type=int,
        metavar='N',
        help='実験モード（各カテゴリN問ずつ、例: --experiment-mode 3）'
    )
    
    parser.add_argument(
        '--total-questions',
        type=int,
        metavar='N',
        help='実行する総問題数（指定した数だけ順番に実行）'
    )
    
    parser.add_argument(
        '--log-dir',
        default='logs',
        help='ログ出力ディレクトリ (デフォルト: logs)'
    )
    
    args = parser.parse_args()
    
    # バナー表示
    print_banner()
    
    # 実行モードの処理
    if args.test_mode:
        print("🧪 テストモードで実行します")
        args.questions_per_category = 2
        args.max_turns = 8
        args.batch_size = 2
    elif args.experiment_mode:
        print(f"🔬 実験モードで実行します（各カテゴリ{args.experiment_mode}問ずつ）")
        args.questions_per_category = args.experiment_mode
        args.max_turns = min(args.max_turns, 10)  # 実験モードでは短めに
        args.batch_size = min(args.batch_size, args.experiment_mode)
    elif args.total_questions:
        print(f"📊 指定問題数モードで実行します（総{args.total_questions}問）")
        # questions_per_categoryは後で調整
    
    # 出力ディレクトリの設定
    if not args.output_dir:
        args.output_dir = create_output_directory()
    
    # データセットファイルの存在確認
    if not os.path.exists(args.dataset):
        print(f"❌ エラー: データセットファイルが見つかりません: {args.dataset}")
        sys.exit(1)
    
    # max_concurrentがbatch_sizeより小さい場合は自動調整
    effective_max_concurrent = max(args.max_concurrent, args.batch_size)
    if effective_max_concurrent != args.max_concurrent:
        print(f"⚠️  max_concurrent を {args.max_concurrent} から {effective_max_concurrent} に自動調整しました（バッチサイズに合わせて並行実行）")
    
    # 設定作成
    config = BenchmarkConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_turns_per_question=args.max_turns,
        questions_per_category=args.questions_per_category,
        batch_size=args.batch_size,
        max_concurrent=effective_max_concurrent,
        timeout_per_question=args.timeout,
        save_intermediate_results=True,
        log_dir=args.log_dir
    )
    
    # 設定表示
    print_config_summary(config)
    
    # 確認プロンプト（テストモード・実験モード以外）
    if not args.test_mode and not args.experiment_mode:
        if args.total_questions:
            total_questions = args.total_questions
        else:
            total_questions = args.questions_per_category * 4  # 4カテゴリ想定
            
        estimated_time = total_questions * args.max_turns * 2 / 60  # 概算時間（分）
        
        print(f"📊 予想実行内容:")
        print(f"  総問題数: {total_questions}問")
        print(f"  推定実行時間: {estimated_time:.0f}分")
        print()
        
        response = input("実行を開始しますか？ (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("実行をキャンセルしました。")
            sys.exit(0)
    
    # ベンチマーク実行
    print("🚀 ベンチマークを開始します...")
    print()
    
    runner = MMLUBenchmarkRunner(config)
    
    try:
        if args.resume:
            print("📂 チェックポイントから再開します...")
            report = await runner.resume_benchmark()
        else:
            # 総問題数が指定されている場合はそれを渡す
            total_questions = args.total_questions if args.total_questions else None
            report = await runner.run_full_benchmark(total_questions=total_questions)
        
        # 結果サマリー表示
        print()
        print("="*80)
        print("🎉 ベンチマーク完了!")
        print("="*80)
        print(f"📈 結果サマリー:")
        print(f"  総問題数: {report.total_questions}")
        print(f"  正答数: {report.correct_answers}")
        print(f"  全体正答率: {report.overall_accuracy:.2%}")
        print(f"  信頼度重み付き正答率: {report.confidence_weighted_accuracy:.2%}")
        print(f"  平均議論ターン数: {report.average_turns:.1f}")
        print(f"  平均処理時間: {report.average_processing_time:.1f}秒")
        print(f"  総実行時間: {report.execution_time:.1f}秒")
        print()
        
        # カテゴリ別結果
        print("📊 カテゴリ別正答率:")
        for category, accuracy in report.category_accuracies.items():
            count = report.category_counts[category]
            correct = int(accuracy * count)
            print(f"  {category}: {correct}/{count} ({accuracy:.2%})")
        print()
        
        print(f"📁 詳細結果は以下に保存されました:")
        print(f"  {config.output_dir}")
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print()
        print("⏹️  ユーザーによって中断されました")
        print("💾 チェックポイントを保存中...")
        
        try:
            await runner._save_checkpoint()
            print("✅ チェックポイントを保存しました")
            print("🔄 --resumeオプションで再開できます")
        except Exception as save_error:
            print(f"❌ チェックポイント保存エラー: {save_error}")
        
        return 130  # SIGINT
        
    except Exception as e:
        print()
        print(f"❌ エラーが発生しました: {e}")
        print("💾 チェックポイントを保存中...")
        
        try:
            await runner._save_checkpoint()
            print("✅ チェックポイントを保存しました")
        except Exception as save_error:
            print(f"❌ チェックポイント保存エラー: {save_error}")
        
        # デバッグ情報表示
        import traceback
        print()
        print("🐛 デバッグ情報:")
        traceback.print_exc()
        
        return 1


def run_progress_monitor():
    """進捗監視の簡易版（将来の拡張用）"""
    # 実装予定：リアルタイム進捗表示
    pass


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  実行が中断されました")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)