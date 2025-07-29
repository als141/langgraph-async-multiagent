"""
MMLU PROデータ読み込みモジュール
"""

import pandas as pd
import ast
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MMLUProblem:
    """MMLU問題のデータクラス"""
    question_id: str
    question: str
    question_ja: str  # 日本語版
    options: List[str]
    correct_answer: str  # "A", "B", "C", etc.
    correct_index: int
    category: str
    source: str
    cot_content: str


class MMLUDataLoader:
    """MMLU PROデータローダー"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def load_and_validate(self) -> List[MMLUProblem]:
        """CSVを読み込み、データを検証して返す"""
        df = pd.read_csv(self.csv_path)
        problems = []
        
        for _, row in df.iterrows():
            try:
                problem = self._row_to_problem(row)
                if self.validate_problem(problem):
                    problems.append(problem)
                else:
                    print(f"問題データが無効: {problem.question_id}")
            except Exception as e:
                print(f"問題データの読み込みエラー: {e}")
                continue
                
        print(f"有効な問題を{len(problems)}件読み込みました")
        return problems
    
    def _row_to_problem(self, row) -> MMLUProblem:
        """DataFrameの行をMMLUProblemに変換"""
        # 選択肢の解析
        options = self.preprocess_options(row['options'])
        
        return MMLUProblem(
            question_id=str(row['question_id']),
            question=str(row['question']),
            question_ja=str(row['question_ja']),
            options=options,
            correct_answer=str(row['answer']),
            correct_index=int(row['answer_index']),
            category=str(row['category']),
            source=str(row['src']),
            cot_content=str(row['cot_content']) if pd.notna(row['cot_content']) else ""
        )
    
    def preprocess_options(self, options_str: str) -> List[str]:
        """選択肢文字列を配列に変換"""
        try:
            # 標準的なリスト形式に変換を試行
            if options_str.startswith('[') and options_str.endswith(']'):
                # まずJSON形式での解析を試行（最も確実）
                import json
                try:
                    options = json.loads(options_str)
                    clean_options = []
                    for option in options:
                        clean_option = re.sub(r'\s+', ' ', str(option).strip())
                        if clean_option:
                            clean_options.append(clean_option)
                    return clean_options
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # JSON解析に失敗した場合、ダブルクォート用正規表現を試行
                pattern = r'\"([^\"]*?)\"'
                matches = re.findall(pattern, options_str, re.DOTALL)
                
                if matches:
                    # 改行文字や余分な空白を処理
                    clean_options = []
                    for match in matches:
                        # 改行文字を空白に置換し、連続する空白を単一に
                        clean_option = re.sub(r'\s+', ' ', match.strip())
                        if clean_option:  # 空文字列でない場合のみ追加
                            clean_options.append(clean_option)
                    return clean_options
                
                # シングルクォート用正規表現を試行
                pattern = r"'([^']*?)'"
                matches = re.findall(pattern, options_str, re.DOTALL)
                
                if matches:
                    clean_options = []
                    for match in matches:
                        clean_option = re.sub(r'\s+', ' ', match.strip())
                        if clean_option:
                            clean_options.append(clean_option)
                    return clean_options
                
                # 正規表現で失敗した場合、従来のast.literal_evalを試行
                try:
                    options = ast.literal_eval(options_str)
                    clean_options = []
                    for option in options:
                        clean_option = re.sub(r'\s+', ' ', str(option).strip())
                        if clean_option:
                            clean_options.append(clean_option)
                    return clean_options
                except (ValueError, SyntaxError):
                    pass
            
            # その他の形式への対応
            return []
            
        except Exception as e:
            print(f"選択肢のパースで予期しないエラー: {options_str[:50]}..., エラー: {e}")
            return []
    
    def validate_problem(self, problem: MMLUProblem) -> bool:
        """問題データの整合性チェック"""
        # 必須フィールドのチェック
        if not problem.question_id or not problem.question or not problem.question_ja:
            return False
        
        # 選択肢の数をチェック（最低2つ）
        if not problem.options or len(problem.options) < 2:
            return False
        
        # 正解インデックスの範囲チェック
        if problem.correct_index < 0 or problem.correct_index >= len(problem.options):
            return False
        
        # 正解記号と正解インデックスの整合性チェック
        expected_answer = chr(ord('A') + problem.correct_index)
        if problem.correct_answer != expected_answer:
            return False
        
        return True
    
    def stratified_sample(self, n_per_category: int = 25) -> List[MMLUProblem]:
        """カテゴリ別に均等サンプリング"""
        all_problems = self.load_and_validate()
        
        # カテゴリ別にグループ化
        category_problems: Dict[str, List[MMLUProblem]] = {}
        for problem in all_problems:
            if problem.category not in category_problems:
                category_problems[problem.category] = []
            category_problems[problem.category].append(problem)
        
        # 各カテゴリから指定数をサンプリング
        sampled_problems = []
        for category, problems in category_problems.items():
            sample_size = min(n_per_category, len(problems))
            sampled = problems[:sample_size]  # 先頭からn問を取得
            sampled_problems.extend(sampled)
            print(f"{category}: {sample_size}問をサンプリング")
        
        return sampled_problems
    
    def get_category_stats(self) -> Dict[str, int]:
        """カテゴリ別の問題数統計を取得"""
        problems = self.load_and_validate()
        stats = {}
        for problem in problems:
            stats[problem.category] = stats.get(problem.category, 0) + 1
        return stats


if __name__ == "__main__":
    # テスト実行
    loader = MMLUDataLoader("data/mmlu_pro_100.csv")
    stats = loader.get_category_stats()
    print("カテゴリ別統計:", stats)
    
    problems = loader.stratified_sample(25)
    print(f"サンプリング結果: {len(problems)}問")
    
    # 最初の問題を表示
    if problems:
        first_problem = problems[0]
        print(f"\n問題例:")
        print(f"ID: {first_problem.question_id}")
        print(f"問題: {first_problem.question_ja}")
        print(f"選択肢: {first_problem.options}")
        print(f"正解: {first_problem.correct_answer} ({first_problem.correct_index})")
        print(f"カテゴリ: {first_problem.category}")