"""
MMLU評価エンジン
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from .data_loader import MMLUProblem
from .mmlu_orchestrator import DebateResult
from .answer_extractor import AnswerExtractor


@dataclass
class EvaluationResult:
    """単一問題の評価結果"""
    question_id: str
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    confidence_score: float
    debate_turns: int
    processing_time: float
    category: str
    extraction_method: str
    reasoning: str


@dataclass
class BenchmarkReport:
    """ベンチマーク全体のレポート"""
    total_questions: int
    correct_answers: int
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    category_counts: Dict[str, int]
    average_turns: float
    average_processing_time: float
    confidence_weighted_accuracy: float
    detailed_results: List[EvaluationResult]
    execution_time: float
    timestamp: str


class MMLUEvaluator:
    """MMLU専用評価器"""
    
    def __init__(self):
        self.answer_extractor = AnswerExtractor()
    
    def evaluate_single_result(
        self,
        debate_result: DebateResult,
        problem: MMLUProblem
    ) -> EvaluationResult:
        """単一問題の評価"""
        
        # 回答抽出
        extraction = self.answer_extractor.extract_with_confidence(
            debate_result.final_conclusion,
            problem.options
        )
        
        predicted_answer = extraction.extracted_answer
        is_correct = predicted_answer == problem.correct_answer if predicted_answer else False
        
        return EvaluationResult(
            question_id=problem.question_id,
            predicted_answer=predicted_answer,
            correct_answer=problem.correct_answer,
            is_correct=is_correct,
            confidence_score=extraction.confidence_score,
            debate_turns=debate_result.turn_count,
            processing_time=debate_result.debate_duration,
            category=problem.category,
            extraction_method=extraction.extraction_method,
            reasoning=extraction.reasoning
        )
        
    def evaluate_batch_results(
        self,
        debate_results: List[DebateResult],
        problems: List[MMLUProblem],
        execution_time: float = 0.0
    ) -> BenchmarkReport:
        """バッチ結果の評価"""
        
        if len(debate_results) != len(problems):
            raise ValueError("debate_resultsとproblemsの長さが一致しません")
        
        detailed_results = []
        
        # 各問題を評価
        for debate_result, problem in zip(debate_results, problems):
            eval_result = self.evaluate_single_result(debate_result, problem)
            detailed_results.append(eval_result)
        
        # 統計計算
        total_questions = len(detailed_results)
        correct_answers = sum(1 for r in detailed_results if r.is_correct)
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # カテゴリ別統計
        category_stats = self._calculate_category_stats(detailed_results)
        
        # その他の統計
        average_turns = sum(r.debate_turns for r in detailed_results) / total_questions if total_questions > 0 else 0.0
        average_processing_time = sum(r.processing_time for r in detailed_results) / total_questions if total_questions > 0 else 0.0
        
        # 信頼度重み付き正答率
        confidence_weighted_accuracy = self._calculate_confidence_weighted_accuracy(detailed_results)
        
        return BenchmarkReport(
            total_questions=total_questions,
            correct_answers=correct_answers,
            overall_accuracy=overall_accuracy,
            category_accuracies=category_stats["accuracies"],
            category_counts=category_stats["counts"],
            average_turns=average_turns,
            average_processing_time=average_processing_time,
            confidence_weighted_accuracy=confidence_weighted_accuracy,
            detailed_results=detailed_results,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _calculate_category_stats(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """カテゴリ別統計の計算"""
        category_correct = defaultdict(int)
        category_total = defaultdict(int)
        
        for result in results:
            category_total[result.category] += 1
            if result.is_correct:
                category_correct[result.category] += 1
        
        category_accuracies = {}
        for category in category_total:
            accuracy = category_correct[category] / category_total[category]
            category_accuracies[category] = accuracy
        
        return {
            "accuracies": category_accuracies,
            "counts": dict(category_total)
        }
    
    def _calculate_confidence_weighted_accuracy(self, results: List[EvaluationResult]) -> float:
        """信頼度重み付き正答率の計算"""
        if not results:
            return 0.0
        
        weighted_correct = 0.0
        total_confidence = 0.0
        
        for result in results:
            confidence = max(result.confidence_score, 0.1)  # 最低信頼度を設定
            total_confidence += confidence
            if result.is_correct:
                weighted_correct += confidence
        
        return weighted_correct / total_confidence if total_confidence > 0 else 0.0
    
    def generate_detailed_report(self, report: BenchmarkReport) -> str:
        """詳細レポートの生成"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MMLU PRO ベンチマーク評価レポート")
        report_lines.append("="*80)
        report_lines.append(f"実行日時: {report.timestamp}")
        report_lines.append(f"総実行時間: {report.execution_time:.2f}秒")
        report_lines.append("")
        
        # 全体統計
        report_lines.append("【全体統計】")
        report_lines.append(f"総問題数: {report.total_questions}")
        report_lines.append(f"正答数: {report.correct_answers}")
        report_lines.append(f"全体正答率: {report.overall_accuracy:.2%}")
        report_lines.append(f"信頼度重み付き正答率: {report.confidence_weighted_accuracy:.2%}")
        report_lines.append(f"平均議論ターン数: {report.average_turns:.1f}")
        report_lines.append(f"平均処理時間: {report.average_processing_time:.2f}秒")
        report_lines.append("")
        
        # カテゴリ別統計
        report_lines.append("【カテゴリ別統計】")
        for category, accuracy in report.category_accuracies.items():
            count = report.category_counts[category]
            correct = int(accuracy * count)
            report_lines.append(f"{category}: {correct}/{count} ({accuracy:.2%})")
        report_lines.append("")
        
        # 抽出方法別統計
        extraction_stats = self._calculate_extraction_method_stats(report.detailed_results)
        report_lines.append("【回答抽出方法別統計】")
        for method, stats in extraction_stats.items():
            report_lines.append(f"{method}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        report_lines.append("")
        
        # エラーケース分析
        error_analysis = self._analyze_error_cases(report.detailed_results)
        report_lines.append("【エラーケース分析】")
        for category, errors in error_analysis.items():
            if errors:
                report_lines.append(f"{category}:")
                for error in errors[:3]:  # 上位3件のみ表示
                    report_lines.append(f"  - 問題ID {error['question_id']}: {error['reasoning'][:100]}...")
        report_lines.append("")
        
        # 推奨改善点
        recommendations = self._generate_recommendations(report)
        if recommendations:
            report_lines.append("【推奨改善点】")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)
    
    def _calculate_extraction_method_stats(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """抽出方法別統計の計算"""
        method_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in results:
            method = result.extraction_method
            method_stats[method]["total"] += 1
            if result.is_correct:
                method_stats[method]["correct"] += 1
        
        # 正答率を計算
        for method, stats in method_stats.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return dict(method_stats)
    
    def _analyze_error_cases(self, results: List[EvaluationResult]) -> Dict[str, List[Dict[str, Any]]]:
        """エラーケースの分析"""
        error_cases = defaultdict(list)
        
        for result in results:
            if not result.is_correct:
                error_cases[result.category].append({
                    "question_id": result.question_id,
                    "predicted": result.predicted_answer,
                    "correct": result.correct_answer,
                    "confidence": result.confidence_score,
                    "reasoning": result.reasoning
                })
        
        # 信頼度の低い順でソート
        for category in error_cases:
            error_cases[category].sort(key=lambda x: x["confidence"])
        
        return dict(error_cases)
    
    def _generate_recommendations(self, report: BenchmarkReport) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []
        
        # 全体正答率が低い場合
        if report.overall_accuracy < 0.5:
            recommendations.append("全体正答率が50%を下回っています。エージェントのプロンプトや議論戦略の見直しを検討してください。")
        
        # カテゴリ間の性能差が大きい場合
        if report.category_accuracies:
            min_acc = min(report.category_accuracies.values())
            max_acc = max(report.category_accuracies.values())
            if max_acc - min_acc > 0.3:
                worst_category = min(report.category_accuracies, key=report.category_accuracies.get)
                recommendations.append(f"カテゴリ間の性能差が大きいです。特に{worst_category}分野の強化が必要です。")
        
        # 議論ターン数が多すぎる場合
        if report.average_turns > 12:
            recommendations.append("平均議論ターン数が多すぎます。議論の効率化や早期終了条件の見直しを検討してください。")
        
        # 信頼度重み付き正答率が低い場合
        if report.confidence_weighted_accuracy < report.overall_accuracy - 0.1:
            recommendations.append("回答抽出の信頼度が低いです。抽出パターンやアルゴリズムの改善を検討してください。")
        
        return recommendations
    
    def save_report_json(self, report: BenchmarkReport, filepath: str):
        """レポートをJSONファイルに保存"""
        report_dict = asdict(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
    
    def save_detailed_results_csv(self, results: List[EvaluationResult], filepath: str):
        """詳細結果をCSVファイルに保存"""
        import csv
        
        fieldnames = [
            'question_id', 'predicted_answer', 'correct_answer', 'is_correct',
            'confidence_score', 'debate_turns', 'processing_time', 'category',
            'extraction_method', 'reasoning'
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(asdict(result))


if __name__ == "__main__":
    # テスト用のコード
    from .data_loader import MMLUDataLoader
    
    # サンプルデータでテスト
    evaluator = MMLUEvaluator()
    
    # ダミーデータの作成
    sample_results = [
        EvaluationResult(
            question_id="test_1",
            predicted_answer="A",
            correct_answer="A",
            is_correct=True,
            confidence_score=0.8,
            debate_turns=5,
            processing_time=30.0,
            category="engineering",
            extraction_method="pattern_matching",
            reasoning="パターンマッチングで正答を抽出"
        ),
        EvaluationResult(
            question_id="test_2",
            predicted_answer="B",
            correct_answer="C",
            is_correct=False,
            confidence_score=0.6,
            debate_turns=8,
            processing_time=45.0,
            category="history",
            extraction_method="semantic_similarity",
            reasoning="意味的類似度で誤答を抽出"
        )
    ]
    
    # ダミーレポートの生成
    report = BenchmarkReport(
        total_questions=2,
        correct_answers=1,
        overall_accuracy=0.5,
        category_accuracies={"engineering": 1.0, "history": 0.0},
        category_counts={"engineering": 1, "history": 1},
        average_turns=6.5,
        average_processing_time=37.5,
        confidence_weighted_accuracy=0.57,
        detailed_results=sample_results,
        execution_time=120.0,
        timestamp="2024-01-01 12:00:00"
    )
    
    # レポート生成テスト
    detailed_report = evaluator.generate_detailed_report(report)
    print(detailed_report)