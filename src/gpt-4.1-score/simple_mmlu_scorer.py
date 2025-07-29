#!/usr/bin/env python3
"""
シンプルなOpenAI GPT-4.1を使用したMMLU問題スコアリングシステム
システムプロンプトなし、ユーザープロンプトのみで回答を取得
"""

import os
import json
import time
import asyncio
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

from openai import OpenAI

@dataclass
class MMLUResult:
    """MMLU問題の結果"""
    question_id: str
    question: str
    options: List[str]
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    response_time: float

class SimpleMMLUScorer:
    """シンプルなMMLUスコアリングシステム"""
    
    def __init__(self):
        # OpenAI APIキーの確認
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        self.client = OpenAI(api_key=api_key)
        
        # JSONスキーマ定義
        self.json_schema = {
            "type": "object",
            "properties": {
                "final_answer": {
                    "type": "string",
                    "description": "選択した選択肢アルファベットのみ（A, B, C, D, E, F, G, H, I, Jのいずれか）",
                    "enum": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                }
            },
            "required": ["final_answer"],
            "additionalProperties": False
        }
    
    def load_csv_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """CSVファイルからデータを読み込み"""
        import pandas as pd
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
        df = pd.read_csv(csv_path)
        problems = []
        
        for _, row in df.iterrows():
            try:
                # 選択肢の解析
                options_str = row['options']
                options = self._parse_options(options_str)
                
                problem = {
                    'question_id': str(row['question_id']),
                    'question': str(row['question_ja']),  # 日本語版を使用
                    'options': options,
                    'correct_answer': str(row['answer']),
                    'category': str(row['category'])
                }
                problems.append(problem)
                
            except Exception as e:
                print(f"問題 {row.get('question_id', 'unknown')} の読み込みエラー: {e}")
                continue
        
        print(f"✅ {len(problems)}問を読み込みました")
        return problems
    
    def _parse_options(self, options_str: str) -> List[str]:
        """選択肢文字列を解析"""
        import json
        import re
        
        try:
            # JSON形式での解析を試行
            if options_str.startswith('[') and options_str.endswith(']'):
                options = json.loads(options_str)
                return [str(option).strip() for option in options]
        except:
            pass
        
        # フォールバック: 空のリストを返す
        return []
    
    def create_user_prompt(self, question: str, options: List[str]) -> str:
        """ユーザープロンプトを作成"""
        # 選択肢を整形
        formatted_options = []
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            formatted_options.append(f"{letter}) {option}")
        
        prompt = f"""必ず A, B, C, D, E, F, G, H, I, J のいずれかを選択してください
最適な選択肢を決定してください

問題:
{question}

選択肢:
{chr(10).join(formatted_options)}

回答は以下のJSON形式で行ってください:
{{
"final_answer": "選択した選択肢アルファベットのみ"
}}"""
        
        return prompt
    
    def get_answer(self, question: str, options: List[str]) -> Dict[str, Any]:
        """単一問題の回答を取得"""
        user_prompt = self.create_user_prompt(question, options)
        
        start_time = time.time()
        
        try:
            response = self.client.responses.create(
                model="o3-mini",
                input=[{"role": "user", "content": user_prompt}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "mmlu_answer",
                        "schema": self.json_schema,
                        "strict": True
                    }
                }
            )
            
            response_time = time.time() - start_time
            
            # レスポンスから回答を抽出
            if response.status == "completed":
                output_text = response.output_text
                try:
                    parsed_response = json.loads(output_text)
                    predicted_answer = parsed_response.get("final_answer", "")
                    
                    return {
                        "predicted_answer": predicted_answer,
                        "response_time": response_time,
                        "success": True,
                        "raw_response": output_text
                    }
                except json.JSONDecodeError:
                    return {
                        "predicted_answer": "",
                        "response_time": response_time,
                        "success": False,
                        "error": "JSON解析エラー",
                        "raw_response": output_text
                    }
            else:
                return {
                    "predicted_answer": "",
                    "response_time": response_time,
                    "success": False,
                    "error": f"API呼び出し失敗: {response.status}",
                    "raw_response": str(response)
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "predicted_answer": "",
                "response_time": response_time,
                "success": False,
                "error": str(e)
            }
    
    async def get_answer_async(self, question: str, options: List[str]) -> Dict[str, Any]:
        """非同期で単一問題の回答を取得"""
        user_prompt = self.create_user_prompt(question, options)
        
        start_time = time.time()
        
        try:
            # 非同期でAPI呼び出し（実際にはOpenAI Python SDKは同期のみなので、run_in_executorを使用）
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.responses.create(
                    model="o3-mini",
                    input=[{"role": "user", "content": user_prompt}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mmlu_answer",
                            "schema": self.json_schema,
                            "strict": True
                        }
                    }
                )
            )
            
            response_time = time.time() - start_time
            
            # レスポンスから回答を抽出
            if response.status == "completed":
                output_text = response.output_text
                try:
                    parsed_response = json.loads(output_text)
                    predicted_answer = parsed_response.get("final_answer", "")
                    
                    return {
                        "predicted_answer": predicted_answer,
                        "response_time": response_time,
                        "success": True,
                        "raw_response": output_text
                    }
                except json.JSONDecodeError:
                    return {
                        "predicted_answer": "",
                        "response_time": response_time,
                        "success": False,
                        "error": "JSON解析エラー",
                        "raw_response": output_text
                    }
            else:
                return {
                    "predicted_answer": "",
                    "response_time": response_time,
                    "success": False,
                    "error": f"API呼び出し失敗: {response.status}",
                    "raw_response": str(response)
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "predicted_answer": "",
                "response_time": response_time,
                "success": False,
                "error": str(e)
            }
    
    async def process_batch(self, problems: List[Dict[str, Any]], batch_size: int = 5) -> List[MMLUResult]:
        """バッチ処理で問題を並行処理"""
        results = []
        
        # バッチに分割
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(problems) - 1) // batch_size + 1
            
            print(f"\n🔄 バッチ {batch_num}/{total_batches} ({len(batch)}問) を並行処理中...")
            
            # バッチ内の問題を並行実行
            batch_tasks = []
            for problem in batch:
                task = self.get_answer_async(problem['question'], problem['options'])
                batch_tasks.append((problem, task))
            
            # 並行実行
            batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
            
            # 結果処理
            for j, ((problem, _), answer_result) in enumerate(zip(batch_tasks, batch_results)):
                if isinstance(answer_result, Exception):
                    print(f"  ❌ 問題 {problem['question_id']}: エラー - {answer_result}")
                    answer_result = {
                        "predicted_answer": "",
                        "response_time": 0.0,
                        "success": False,
                        "error": str(answer_result)
                    }
                
                # 正答判定
                is_correct = (answer_result['predicted_answer'] == problem['correct_answer'])
                
                # 結果記録
                result = MMLUResult(
                    question_id=problem['question_id'],
                    question=problem['question'],
                    options=problem['options'],
                    correct_answer=problem['correct_answer'],
                    predicted_answer=answer_result['predicted_answer'],
                    is_correct=is_correct,
                    response_time=answer_result['response_time']
                )
                results.append(result)
                
                # 進捗表示
                status = "✅" if is_correct else "❌"
                print(f"  {status} 問題 {problem['question_id']}: "
                      f"予測={answer_result['predicted_answer']}, 正解={problem['correct_answer']}")
                
                # エラー表示
                if not answer_result['success']:
                    print(f"    ⚠️ エラー: {answer_result.get('error', 'Unknown error')}")
            
            # バッチ間の待機（API制限対策）
            if i + batch_size < len(problems):
                print(f"  ⏳ 次のバッチまで1秒待機...")
                await asyncio.sleep(1)
        
        return results
    
    def score_all_problems(self, csv_path: str, output_path: str = None, batch_size: int = 1) -> Dict[str, Any]:
        """全問題をスコアリング（同期版・互換性のため）"""
        return asyncio.run(self.score_all_problems_async(csv_path, output_path, batch_size))
    
    async def score_all_problems_async(self, csv_path: str, output_path: str = None, batch_size: int = 5) -> Dict[str, Any]:
        """全問題をスコアリング（非同期版）"""
        problems = self.load_csv_data(csv_path)
        
        print(f"🚀 {len(problems)}問のスコアリングを開始... (バッチサイズ: {batch_size})")
        
        # バッチ処理で実行
        if batch_size > 1:
            results = await self.process_batch(problems, batch_size)
        else:
            # バッチサイズ1の場合は順次処理
            results = []
            for i, problem in enumerate(problems, 1):
                print(f"[{i}/{len(problems)}] 問題 {problem['question_id']} を処理中...")
                
                # 回答取得
                answer_result = self.get_answer(problem['question'], problem['options'])
                
                # 正答判定
                is_correct = (answer_result['predicted_answer'] == problem['correct_answer'])
                
                # 結果記録
                result = MMLUResult(
                    question_id=problem['question_id'],
                    question=problem['question'],
                    options=problem['options'],
                    correct_answer=problem['correct_answer'],
                    predicted_answer=answer_result['predicted_answer'],
                    is_correct=is_correct,
                    response_time=answer_result['response_time']
                )
                results.append(result)
                
                # 進捗表示
                accuracy = sum(1 for r in results if r.is_correct) / len(results)
                status = "✅" if is_correct else "❌"
                print(f"  {status} 予測: {answer_result['predicted_answer']}, 正解: {problem['correct_answer']} "
                      f"(現在の正答率: {accuracy:.1%})")
                
                # エラー表示
                if not answer_result['success']:
                    print(f"  ⚠️ エラー: {answer_result.get('error', 'Unknown error')}")
        
        # 最終結果集計
        correct_count = sum(1 for result in results if result.is_correct)
        overall_accuracy = correct_count / len(results) if results else 0
        
        summary = {
            "total_questions": len(problems),
            "correct_answers": correct_count,
            "overall_accuracy": overall_accuracy,
            "results": results,
            "batch_size": batch_size
        }
        
        # 結果保存
        if output_path:
            self._save_results(summary, output_path)
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any], output_path: str):
        """結果をファイルに保存"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSON形式で保存
        save_data = {
            "total_questions": summary["total_questions"],
            "correct_answers": summary["correct_answers"],
            "overall_accuracy": summary["overall_accuracy"],
            "results": []
        }
        
        for result in summary["results"]:
            save_data["results"].append({
                "question_id": result.question_id,
                "question": result.question,
                "options": result.options,
                "correct_answer": result.correct_answer,
                "predicted_answer": result.predicted_answer,
                "is_correct": result.is_correct,
                "response_time": result.response_time
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 結果を保存しました: {output_path}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='シンプルなMMLUスコアリングシステム')
    parser.add_argument('--csv', default='data/mmlu_pro_100.csv', help='CSVファイルのパス')
    parser.add_argument('--output', default='results/simple_scoring_results.json', help='出力ファイルのパス')
    parser.add_argument('--limit', type=int, help='処理する問題数の上限')
    parser.add_argument('--batch-size', type=int, default=1, help='バッチサイズ（並行処理数）')
    
    args = parser.parse_args()
    
    try:
        scorer = SimpleMMLUScorer()
        
        # CSV読み込み
        problems = scorer.load_csv_data(args.csv)
        
        # 問題数制限
        if args.limit and args.limit < len(problems):
            problems = problems[:args.limit]
            print(f"⚠️ 問題数を{args.limit}問に制限しました")
        
        # スコアリング実行
        results = scorer.score_all_problems(args.csv, args.output, args.batch_size)
        
        # 最終結果表示
        print("\n" + "="*60)
        print("🎉 スコアリング完了!")
        print("="*60)
        print(f"📊 総問題数: {results['total_questions']}")
        print(f"✅ 正答数: {results['correct_answers']}")
        print(f"📈 全体正答率: {results['overall_accuracy']:.1%}")
        print(f"⚡ バッチサイズ: {results['batch_size']}")
        print(f"📁 結果保存先: {args.output}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())