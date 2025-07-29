"""
MMLUベンチマーク実行メインモジュール
"""

import os
import time
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .data_loader import MMLUDataLoader, MMLUProblem
from .mmlu_orchestrator import MMLUOrchestrator, DebateResult
from .mmlu_evaluator import MMLUEvaluator, BenchmarkReport


@dataclass
class BenchmarkConfig:
    """ベンチマーク設定"""
    dataset_path: str
    output_dir: str
    max_turns_per_question: int = 15
    questions_per_category: int = 25
    batch_size: int = 10
    max_concurrent: int = 1  # 同時実行数（API制限考慮）
    timeout_per_question: int = 300  # 秒
    save_intermediate_results: bool = True
    log_dir: str = "logs"


@dataclass
class BenchmarkProgress:
    """ベンチマーク進捗管理"""
    completed_questions: List[str]
    failed_questions: List[str]
    current_batch: int
    start_time: Optional[float]
    
    def save_checkpoint(self, filepath: str):
        """進捗をファイルに保存"""
        checkpoint_data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def load_checkpoint(self, filepath: str):
        """保存された進捗を読み込み"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.completed_questions = data.get('completed_questions', [])
                self.failed_questions = data.get('failed_questions', [])
                self.current_batch = data.get('current_batch', 0)
                self.start_time = data.get('start_time')
            return True
        return False


class MMLUBenchmarkRunner:
    """MMLUベンチマーク実行器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_loader = MMLUDataLoader(config.dataset_path)
        self.orchestrator = MMLUOrchestrator(
            max_turns=config.max_turns_per_question,
            log_dir=config.log_dir
        )
        self.evaluator = MMLUEvaluator()
        self.progress = BenchmarkProgress(
            completed_questions=[],
            failed_questions=[],
            current_batch=0,
            start_time=None
        )
        
        # 出力ディレクトリの作成
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # チェックポイントファイルのパス
        self.checkpoint_path = self.output_path / "checkpoint.json"
    
    async def run_full_benchmark(self, total_questions: Optional[int] = None) -> BenchmarkReport:
        """ベンチマークの完全実行"""
        
        print("MMLU PROベンチマークを開始します...")
        
        # データ読み込み
        if total_questions:
            # 指定された総問題数で実行
            problems = self._load_problems_by_total(total_questions)
            print(f"総問題数指定モード: {total_questions}問")
        else:
            # カテゴリ別サンプリング
            problems = self.data_loader.stratified_sample(self.config.questions_per_category)
            print(f"カテゴリ別サンプリング: 各カテゴリ{self.config.questions_per_category}問")
            
        if not problems:
            raise ValueError("問題データの読み込みに失敗しました")
        
        print(f"対象問題数: {len(problems)}")
        
        # 実行開始
        self.progress.start_time = time.time()
        
        try:
            # バッチ処理で実行
            debate_results = await self._run_batch_processing(problems)
            
            # 評価実行
            execution_time = time.time() - self.progress.start_time
            report = self.evaluator.evaluate_batch_results(
                debate_results, 
                problems, 
                execution_time
            )
            
            # 結果保存
            await self._save_results(report, debate_results, problems)
            
            print(f"ベンチマーク完了! 全体正答率: {report.overall_accuracy:.2%}")
            return report
            
        except Exception as e:
            print(f"ベンチマーク実行中にエラー: {e}")
            await self._save_checkpoint()
            raise
    
    def _load_problems_by_total(self, total_questions: int) -> List[MMLUProblem]:
        """指定された総問題数だけ順番に読み込み"""
        all_problems = self.data_loader.load_and_validate()
        
        if total_questions > len(all_problems):
            print(f"警告: 指定された問題数{total_questions}は利用可能な問題数{len(all_problems)}を超えています")
            return all_problems
        
        # 最初のN問を返す
        selected_problems = all_problems[:total_questions]
        
        # カテゴリ別統計を表示
        category_counts = {}
        for problem in selected_problems:
            category_counts[problem.category] = category_counts.get(problem.category, 0) + 1
        
        print("選択された問題のカテゴリ別内訳:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}問")
        
        return selected_problems
    
    async def resume_benchmark(self) -> BenchmarkReport:
        """チェックポイントから再開"""
        
        if not self.progress.load_checkpoint(self.checkpoint_path):
            print("チェックポイントファイルが見つかりません。新規実行を開始します。")
            return await self.run_full_benchmark()
        
        print(f"チェックポイントから再開します...")
        print(f"完了済み: {len(self.progress.completed_questions)}問")
        print(f"失敗: {len(self.progress.failed_questions)}問")
        
        # 残りの問題を取得
        problems = self.data_loader.stratified_sample(self.config.questions_per_category)
        remaining_problems = [
            p for p in problems 
            if p.question_id not in self.progress.completed_questions
        ]
        
        if not remaining_problems:
            print("すべての問題が完了済みです。")
            # 既存の結果を読み込んで返す
            return await self._load_existing_results()
        
        print(f"残り問題数: {len(remaining_problems)}")
        
        # 残りの問題を実行
        remaining_results = await self._run_batch_processing(remaining_problems)
        
        # 既存結果と結合
        all_results = await self._combine_results(remaining_results, problems)
        
        # 最終レポート生成
        execution_time = time.time() - self.progress.start_time
        report = self.evaluator.evaluate_batch_results(all_results, problems, execution_time)
        
        await self._save_results(report, all_results, problems)
        
        print(f"ベンチマーク完了! 全体正答率: {report.overall_accuracy:.2%}")
        return report
    
    async def _run_batch_processing(self, problems: List[MMLUProblem]) -> List[DebateResult]:
        """バッチ処理で問題を実行"""
        
        all_results = []
        
        # バッチに分割
        for i in range(0, len(problems), self.config.batch_size):
            batch = problems[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            print(f"\nバッチ {batch_num}/{(len(problems) - 1) // self.config.batch_size + 1} を実行中...")
            
            # バッチ内の問題を並列実行（制限付き）
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            batch_tasks = []
            active_problems = []
            
            for problem in batch:
                if problem.question_id in self.progress.completed_questions:
                    continue  # スキップ済み問題
                    
                task = self._run_single_problem_with_semaphore(semaphore, problem)
                batch_tasks.append(task)
                active_problems.append(problem)
            
            # バッチ内の問題を一覧表示（並行実行される問題）
            if active_problems:
                print(f"🔄 {len(active_problems)}問を並行実行開始:")
                for problem in active_problems:
                    short_question = problem.question_ja[:60] + "..." if len(problem.question_ja) > 60 else problem.question_ja
                    print(f"  - 問題 {problem.question_id}: {short_question}")
            
            # バッチ実行
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 結果処理
                for j, result in enumerate(batch_results):
                    problem = batch[j] if j < len(batch) else None
                    
                    if isinstance(result, Exception):
                        print(f"問題 {problem.question_id if problem else 'unknown'} でエラー: {result}")
                        if problem:
                            self.progress.failed_questions.append(problem.question_id)
                        continue
                    
                    if result and problem:
                        all_results.append(result)
                        self.progress.completed_questions.append(problem.question_id)
                        
                        # 進捗表示
                        progress_pct = len(self.progress.completed_questions) / len(problems) * 100
                        print(f"  問題 {problem.question_id} 完了 ({progress_pct:.1f}%)")
            
            # 中間保存
            if self.config.save_intermediate_results:
                await self._save_checkpoint()
                await self._save_intermediate_results(all_results, batch_num)
        
        return all_results
    
    async def _run_single_problem_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        problem: MMLUProblem
    ) -> DebateResult:
        """セマフォ付きで単一問題を実行"""
        
        async with semaphore:
            try:
                # タイムアウト付きで実行
                result = await asyncio.wait_for(
                    self.orchestrator.run_single_problem_debate(problem),
                    timeout=self.config.timeout_per_question
                )
                return result
                
            except asyncio.TimeoutError:
                print(f"問題 {problem.question_id} がタイムアウトしました")
                # ダミー結果を返す
                return DebateResult(
                    question_id=problem.question_id,
                    final_conclusion="タイムアウトにより議論が中断されました",
                    full_transcript=[],
                    turn_count=0,
                    debate_duration=self.config.timeout_per_question,
                    facilitator_interventions=0,
                    consensus_score=0.0
                )
            
            except Exception as e:
                print(f"問題 {problem.question_id} で予期しないエラー: {e}")
                raise
    
    async def _save_results(
        self, 
        report: BenchmarkReport, 
        debate_results: List[DebateResult], 
        problems: List[MMLUProblem]
    ):
        """結果の保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細レポート保存
        report_text = self.evaluator.generate_detailed_report(report)
        report_file = self.output_path / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # JSON形式レポート保存
        json_file = self.output_path / f"benchmark_report_{timestamp}.json"
        self.evaluator.save_report_json(report, json_file)
        
        # 詳細結果CSV保存
        csv_file = self.output_path / f"detailed_results_{timestamp}.csv"
        self.evaluator.save_detailed_results_csv(report.detailed_results, csv_file)
        
        # 議論結果の生データ保存
        raw_results = []
        for debate_result, problem in zip(debate_results, problems):
            raw_results.append({
                "problem": asdict(problem),
                "debate_result": asdict(debate_result)
            })
        
        raw_file = self.output_path / f"raw_results_{timestamp}.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)
        
        print(f"結果を保存しました: {self.output_path}")
        print(f"- レポート: {report_file.name}")
        print(f"- JSONレポート: {json_file.name}")
        print(f"- 詳細結果: {csv_file.name}")
        print(f"- 生データ: {raw_file.name}")
    
    async def _save_checkpoint(self):
        """チェックポイント保存"""
        self.progress.save_checkpoint(self.checkpoint_path)
    
    async def _save_intermediate_results(self, results: List[DebateResult], batch_num: int):
        """中間結果の保存"""
        intermediate_file = self.output_path / f"intermediate_batch_{batch_num}.json"
        
        results_data = [asdict(result) for result in results]
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    async def _load_existing_results(self) -> BenchmarkReport:
        """既存の結果を読み込み"""
        # 最新のレポートファイルを探す
        json_files = list(self.output_path.glob("benchmark_report_*.json"))
        if not json_files:
            raise FileNotFoundError("既存の結果ファイルが見つかりません")
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # BenchmarkReportオブジェクトに復元
        # 簡略化のため、主要な統計のみ表示
        print(f"既存結果を読み込み: {latest_file.name}")
        print(f"全体正答率: {report_data['overall_accuracy']:.2%}")
        
        return report_data  # 実際の実装では適切にオブジェクト復元
    
    async def _combine_results(
        self, 
        new_results: List[DebateResult], 
        all_problems: List[MMLUProblem]
    ) -> List[DebateResult]:
        """新しい結果と既存結果を結合"""
        # 簡略化実装：実際はより複雑な結合処理が必要
        return new_results
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """進捗サマリーの取得"""
        total_problems = self.config.questions_per_category * 4  # 4カテゴリ想定
        completed = len(self.progress.completed_questions)
        failed = len(self.progress.failed_questions)
        
        return {
            "total_problems": total_problems,
            "completed": completed,
            "failed": failed,
            "remaining": total_problems - completed - failed,
            "completion_rate": completed / total_problems if total_problems > 0 else 0.0,
            "elapsed_time": time.time() - self.progress.start_time if self.progress.start_time else 0.0
        }


if __name__ == "__main__":
    # テスト実行
    import asyncio
    
    async def test_runner():
        config = BenchmarkConfig(
            dataset_path="../../data/mmlu_pro_100.csv",
            output_dir="./test_results",
            max_turns_per_question=10,
            questions_per_category=2,  # テスト用に少なく設定
            batch_size=2,
            max_concurrent=1
        )
        
        runner = MMLUBenchmarkRunner(config)
        
        try:
            report = await runner.run_full_benchmark()
            print(f"テスト完了: 正答率 {report.overall_accuracy:.2%}")
        except Exception as e:
            print(f"テスト失敗: {e}")
    
    # asyncio.run(test_runner())