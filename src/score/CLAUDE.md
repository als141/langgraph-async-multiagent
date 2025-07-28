# MMLU PRO ベンチマーク評価システム設計書

## 1. 目標

### 1.1 主要目標
- **学術的評価の実現**: 既存のマルチエージェント議論システムをMMLU PROベンチマークで客観的に評価
- **知識推論能力の測定**: 複数エージェントの協調による問題解決能力の定量化
- **議論品質の評価**: 単純な正答率だけでなく、議論プロセスの質も測定

### 1.2 評価対象
- **データセット**: MMLU PRO（工学・その他・ビジネス・歴史の4分野から25問ずつ、計100問）
- **評価指標**: 
  - 正答率（Overall Accuracy）
  - 分野別正答率（Domain-specific Accuracy）
  - 議論効率性（Turn-to-Answer Ratio）
  - 合意形成度（Consensus Score）

## 2. 要件定義

### 2.1 機能要件

#### 2.1.1 データ処理要件
- CSVファイルからのMMLU PRO問題の自動読み込み
- 日本語問題文の適切な処理（question_jaフィールド使用）
- 選択肢の動的パース（配列形式への変換）
- 問題IDによる進捗管理

#### 2.1.2 議論実行要件
- 各問題を議論トピックとして設定
- エージェントが選択肢を認識して議論
- 最大ターン数の動的調整（問題の複雑さに応じて）
- タイムアウト機能（無限ループ防止）

#### 2.1.3 回答抽出要件
- 最終結論から選択肢（A, B, C, D等）の自動抽出
- 曖昧な回答の処理（複数候補の場合）
- 回答未決定の場合の処理

#### 2.1.4 評価・スコアリング要件
- 正答との自動比較
- リアルタイムスコア更新
- 分野別集計
- 詳細な評価レポート生成

### 2.2 非機能要件

#### 2.2.1 性能要件
- 100問の実行時間: 最大2時間以内
- メモリ使用量: 8GB以下
- API呼び出し制限の考慮（OpenAI rate limits）

#### 2.2.2 可用性要件
- エラー時の自動復旧機能
- 中断からの再開機能
- プログレスの永続化

#### 2.2.3 保守性要件
- ログ出力の充実
- 設定パラメータの外部化
- モジュラー設計による拡張性

## 3. システム設計

### 3.1 アーキテクチャ概要

```
┌─────────────────────┐
│   Benchmark Runner  │
│   (ベンチマーク実行)  │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   Problem Loader    │
│  (問題データ読み込み)  │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Debate Orchestrator │
│   (議論オーケストレーター) │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Answer Extractor   │
│   (回答抽出エンジン)   │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Evaluation Engine  │
│    (評価エンジン)     │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   Report Generator  │
│   (レポート生成)     │
└─────────────────────┘
```

### 3.2 コンポーネント設計

#### 3.2.1 Benchmark Runner
**責務**: ベンチマーク実行の全体制御
**主要機能**:
- 実行パラメータの管理
- プログレス追跡
- エラーハンドリング
- 結果集約

```python
class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.progress = BenchmarkProgress()
        
    async def run_benchmark(self) -> BenchmarkResult:
        # 実装詳細は後述
        pass
```

#### 3.2.2 Problem Loader
**責務**: MMLU PROデータの読み込みと前処理
**主な機能**:
- CSV解析
- データ検証
- 問題のフィルタリング・サンプリング

```python
class ProblemLoader:
    def load_problems(self, csv_path: str) -> List[MMLUProblem]:
        # CSVから問題を読み込み、MMLUProblemオブジェクトに変換
        pass
    
    def validate_problem(self, problem: MMLUProblem) -> bool:
        # 問題データの整合性チェック
        pass
```

#### 3.2.3 Answer Extractor
**責務**: 議論結論からの選択肢抽出
**抽出ロジック**:
1. **パターンマッチング**: 「答えはA」「選択肢Bが正しい」等の表現を検出
2. **選択肢テキストマッチング**: 結論文中の選択肢本文との類似度計算
3. **LLM支援抽出**: 専用プロンプトによる構造化抽出
4. **信頼度スコア**: 抽出結果の確信度を0-1で数値化

```python
class AnswerExtractor:
    def extract_answer(self, conclusion: str, options: List[str]) -> AnswerExtraction:
        # 複数手法を組み合わせた回答抽出
        pass
    
    def calculate_confidence(self, extraction_results: List[ExtractionResult]) -> float:
        # 抽出結果の信頼度計算
        pass
```

#### 3.2.4 Evaluation Engine
**責務**: 評価指標の計算と分析
**評価指標**:

1. **基本指標**:
   - Overall Accuracy: 全体正答率
   - Domain Accuracy: 分野別正答率
   - Confidence-weighted Accuracy: 信頼度重み付き正答率

2. **議論品質指標**:
   - Average Turns per Question: 問題あたり平均ターン数
   - Consensus Formation Time: 合意形成までの時間
   - Question Utilization Rate: 質問機能の活用率

3. **効率性指標**:
   - Time per Question: 問題あたり処理時間
   - Token Usage: トークン使用量
   - API Call Efficiency: API呼び出し効率

### 3.3 データモデル

#### 3.3.1 MMLUProblem
```python
@dataclass
class MMLUProblem:
    question_id: str
    question: str
    question_ja: str  # 日本語版
    options: List[str]
    correct_answer: str  # "A", "B", "C", etc.
    correct_index: int
    category: str
    source: str
    cot_content: str
```

#### 3.3.2 DebateResult
```python
@dataclass
class DebateResult:
    question_id: str
    final_conclusion: str
    full_transcript: List[str]
    turn_count: int
    debate_duration: float
    facilitator_interventions: int
    consensus_score: float
```

#### 3.3.3 AnswerExtraction
```python
@dataclass
class AnswerExtraction:
    extracted_answer: Optional[str]
    confidence_score: float
    extraction_method: str
    alternative_candidates: List[Tuple[str, float]]
    reasoning: str
```

#### 3.3.4 EvaluationResult
```python
@dataclass
class EvaluationResult:
    question_id: str
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    confidence_score: float
    debate_turns: int
    processing_time: float
    category: str
```

## 4. 実装指針

### 4.1 議論プロンプトの改良

#### 4.1.1 問題認識プロンプト
現在の汎用議論プロンプトを、選択肢問題に特化して改良：

```python
MMLU_DEBATE_PROMPT = """
**問題解決議論のルール:**

**現在の問題:**
{question_ja}

**選択肢:**
{formatted_options}

**議論の進め方:**
1. まず問題を理解し、重要なポイントを特定してください
2. 各選択肢について検討し、根拠を示して議論してください
3. 他の参加者の意見を聞き、建設的に議論を深めてください
4. 最終的に、最も適切だと思う選択肢を明確に述べてください

**重要**: 最終結論では「答えは○○です」の形で明確に選択肢を示してください。
"""
```

#### 4.1.2 結論抽出プロンプト
```python
CONCLUSION_EXTRACTION_PROMPT = """
以下の議論の結論を分析し、選択された答えを抽出してください。

議論の結論:
{conclusion}

選択肢:
{options}

以下のJSON形式で答えてください:
{
    "selected_answer": "A/B/C/D/等",
    "confidence": 0.0-1.0,
    "reasoning": "抽出根拠"
}
"""
```

### 4.2 エラーハンドリング戦略

#### 4.2.1 議論エラー処理
- **無限ループ防止**: 最大ターン数の強制適用
- **エージェント応答失敗**: 代替エージェントへの自動切り替え
- **API制限対応**: 指数バックオフによるリトライ

#### 4.2.2 回答抽出エラー処理
- **抽出失敗時**: 「不明」として記録し、統計に含める
- **複数候補時**: 最高信頼度の選択肢を採用
- **フォールバック**: 人間による手動確認のためのログ出力

### 4.3 実行制御

#### 4.3.1 バッチ処理設計
```python
async def run_batch_evaluation(
    problems: List[MMLUProblem],
    batch_size: int = 10,
    max_concurrent: int = 3
) -> List[EvaluationResult]:
    """
    問題をバッチ処理で実行
    - API制限を考慮した同時実行数制御
    - 中間結果の定期保存
    - エラー時の自動リトライ
    """
```

#### 4.3.2 プログレス管理
```python
class BenchmarkProgress:
    def __init__(self):
        self.completed_questions = set()
        self.failed_questions = set()
        self.current_batch = 0
        self.start_time = None
        
    def save_checkpoint(self, filepath: str):
        """進捗をファイルに保存"""
        
    def load_checkpoint(self, filepath: str):
        """保存された進捗を読み込み"""
```

## 5. 評価フレームワーク

### 5.1 統計分析

#### 5.1.1 基本統計
- 正答率の95%信頼区間
- 分野別性能比較（カイ二乗検定）
- 議論ターン数の分布分析

#### 5.1.2 相関分析
- 議論ターン数と正答率の相関
- 問題難易度と合意形成時間の関係
- エージェント発言量と最終答えの関係

### 5.2 可視化

#### 5.2.1 性能ダッシュボード
- リアルタイム正答率グラフ
- 分野別ヒートマップ
- 議論品質スコア推移

#### 5.2.2 詳細分析チャート
- 問題別議論深度vs正答性
- エージェント発言パターン分析
- エラーケース分類

## 6. 実装段階

### Phase 1: 基盤実装（1-2週間）
1. データローダーの実装
2. 基本的な議論システムの改良
3. 簡単な回答抽出機能

### Phase 2: 評価機能（1週間）
1. 評価エンジンの実装
2. 統計計算機能
3. 基本レポート生成

### Phase 3: 最適化・拡張（1週間）
1. エラーハンドリングの強化
2. 並列処理の実装
3. 詳細分析機能

### Phase 4: 検証・調整（1週間）
1. 小規模テストの実行
2. パフォーマンス調整
3. 最終検証

## 7. 設定例

### 7.1 benchmark_config.yaml
```yaml
benchmark:
  dataset_path: "data/mmlu_pro_100.csv"
  output_dir: "results/benchmark_run_{timestamp}"
  
  sampling:
    total_questions: 100
    questions_per_category: 25
    random_seed: 42
    
  debate_settings:
    max_turns_per_question: 15
    timeout_per_question: 300  # seconds
    facilitator_check_interval: 5
    
  evaluation:
    confidence_threshold: 0.7
    retry_failed_extractions: true
    save_intermediate_results: true
    
  performance:
    max_concurrent_debates: 3
    batch_size: 10
    api_delay: 1.0  # seconds between API calls
```

## 8. 実装の具体的指示

### 8.1 既存システムの改良ポイント

#### 8.1.1 config.pyの拡張
```python
# MMLU専用のエージェント設定を追加
MMLU_AGENTS_CONFIG = [
    {
        "name": "論理分析者",
        "persona": "論理的思考を重視し、選択肢を体系的に分析する専門家。根拠を明確に示して判断する。",
        "avatar": "🔍",
        "evaluation_focus": "logical_reasoning"
    },
    {
        "name": "知識統合者", 
        "persona": "幅広い知識を統合して総合的に判断する専門家。異なる視点を組み合わせて結論を導く。",
        "avatar": "📚",
        "evaluation_focus": "knowledge_integration"
    },
    {
        "name": "批判的検証者",
        "persona": "他の意見を批判的に検証し、誤りや見落としを指摘する専門家。反対意見も積極的に提示する。",
        "avatar": "⚖️", 
        "evaluation_focus": "critical_analysis"
    }
]
```

#### 8.1.2 state.pyの拡張
```python
class MMLUConversationState(ConversationState):
    """MMLU用の拡張状態"""
    # 既存フィールドに加えて以下を追加
    current_problem: Optional[MMLUProblem]
    extracted_answers: List[str]  # 各ターンで抽出された回答候補
    answer_confidences: List[float]  # 回答の信頼度履歴
    option_mentions: Dict[str, int]  # 各選択肢の言及回数
    reasoning_quality: float  # 推論の質的評価
```

### 8.2 新規ファイル作成指示

#### 8.2.1 src/mmlu_benchmark/data_loader.py
```python
"""
MMLU PROデータ読み込みモジュール
- CSV解析とバリデーション
- 問題の前処理
- カテゴリ別サンプリング
"""

class MMLUDataLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        
    def load_and_validate(self) -> List[MMLUProblem]:
        """CSVを読み込み、データを検証して返す"""
        
    def stratified_sample(self, n_per_category: int = 25) -> List[MMLUProblem]:
        """カテゴリ別に均等サンプリング"""
        
    def preprocess_options(self, options_str: str) -> List[str]:
        """選択肢文字列を配列に変換"""
```

#### 8.2.2 src/mmlu_benchmark/answer_extractor.py
```python
"""
議論結論からの回答抽出モジュール
- パターンマッチング
- 意味的類似度計算
- LLM支援抽出
"""

class AnswerExtractor:
    def __init__(self):
        self.extraction_strategies = [
            PatternMatchingStrategy(),
            SemanticSimilarityStrategy(), 
            LLMAssistedStrategy()
        ]
        
    def extract_with_confidence(
        self, 
        conclusion: str, 
        options: List[str]
    ) -> AnswerExtraction:
        """複数手法で回答を抽出し、信頼度を計算"""
```

#### 8.2.3 src/mmlu_benchmark/benchmark_runner.py
```python
"""
ベンチマーク実行のメインモジュール
- 問題の順次実行
- プログレス管理
- 結果集計
"""

class MMLUBenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_loader = MMLUDataLoader(config.dataset_path)
        self.evaluator = MMLUEvaluator()
        
    async def run_full_benchmark(self) -> BenchmarkReport:
        """100問の完全実行"""
        
    async def run_single_problem(self, problem: MMLUProblem) -> EvaluationResult:
        """単一問題の実行と評価"""
```

### 8.3 議論プロンプトの具体的改良

#### 8.3.1 agents.pyのプロンプト修正
現在のPROMPT_TEMPLATE_STRを以下のように変更：

```python
MMLU_PROMPT_TEMPLATE_STR = """
**あなたの情報:**
{persona}

**現在解決すべき問題:**
{question_ja}

**選択肢:**
{formatted_options}

**議論のガイドライン:**
1. **問題理解**: まず問題の核心を正確に把握してください
2. **選択肢分析**: 各選択肢について具体的な根拠とともに評価してください
3. **知識活用**: あなたの専門知識を活用して判断してください
4. **建設的議論**: 他の参加者の意見を聞き、建設的に議論を深めてください
5. **明確な結論**: 最終的に「答えは[選択肢]です」の形で明確に述べてください

**重要な制約:**
- 必ず選択肢A、B、C、D、E、F、G、H、I、Jの中から一つを選んでください
- 「分からない」や「判断できない」は避けてください
- 根拠を示して論理的に説明してください

現在の議論テーマ: {question_ja}
"""
```

### 8.4 評価システムの実装指示

#### 8.4.1 evaluation/evaluator.py
```python
class MMLUEvaluator:
    """MMLU専用評価器"""
    
    def __init__(self):
        self.answer_extractor = AnswerExtractor()
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate_single_result(
        self, 
        debate_result: DebateResult,
        problem: MMLUProblem
    ) -> EvaluationResult:
        """単一問題の評価"""
        
        # 1. 回答抽出
        extraction = self.answer_extractor.extract_with_confidence(
            debate_result.final_conclusion,
            problem.options
        )
        
        # 2. 正答判定
        is_correct = (extraction.extracted_answer == problem.correct_answer)
        
        # 3. 議論品質評価
        quality_metrics = self._evaluate_debate_quality(debate_result)
        
        return EvaluationResult(
            question_id=problem.question_id,
            predicted_answer=extraction.extracted_answer,
            correct_answer=problem.correct_answer,
            is_correct=is_correct,
            confidence_score=extraction.confidence_score,
            debate_turns=debate_result.turn_count,
            processing_time=debate_result.debate_duration,
            category=problem.category,
            quality_metrics=quality_metrics
        )
```

### 8.5 実行スクリプトの作成指示

#### 8.5.1 run_mmlu_benchmark.py
```python
#!/usr/bin/env python3
"""
MMLU PROベンチマーク実行スクリプト
"""

import asyncio
import argparse
from pathlib import Path
from src.mmlu_benchmark.benchmark_runner import MMLUBenchmarkRunner
from src.mmlu_benchmark.config import BenchmarkConfig

async def main():
    parser = argparse.ArgumentParser(description='Run MMLU PRO Benchmark')
    parser.add_argument('--config', default='config/benchmark_config.yaml')
    parser.add_argument('--dataset', default='data/mmlu_pro_100.csv')
    parser.add_argument('--output-dir', default='results/')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = BenchmarkConfig.from_yaml(args.config)
    config.dataset_path = args.dataset
    config.output_dir = args.output_dir
    
    # ベンチマーク実行
    runner = MMLUBenchmarkRunner(config)
    
    try:
        if args.resume:
            report = await runner.resume_benchmark()
        else:
            report = await runner.run_full_benchmark()
            
        print(f"ベンチマーク完了!")
        print(f"全体正答率: {report.overall_accuracy:.2%}")
        print(f"結果保存先: {report.output_path}")
        
    except KeyboardInterrupt:
        print("ベンチマーク中断 - チェックポイントを保存中...")
        await runner.save_checkpoint()
        
if __name__ == "__main__":
    asyncio.run(main())
```

### 8.6 結果分析ツールの指示

#### 8.6.1 analysis/result_analyzer.py
```python
class BenchmarkAnalyzer:
    """ベンチマーク結果の詳細分析"""
    
    def generate_comprehensive_report(self, results: List[EvaluationResult]) -> str:
        """包括的なレポート生成"""
        
        report_sections = [
            self._generate_summary_stats(results),
            self._generate_category_analysis(results), 
            self._generate_error_analysis(results),
            self._generate_debate_quality_analysis(results),
            self._generate_recommendations(results)
        ]
        
        return "\n\n".join(report_sections)
    
    def create_visualizations(self, results: List[EvaluationResult], output_dir: Path):
        """可視化グラフの生成"""
        # matplotlib/seabornを使用した可視化
        pass
```

### 8.7 テスト戦略

#### 8.7.1 単体テスト
```python
# tests/test_answer_extractor.py
def test_answer_extraction_patterns():
    """回答抽出パターンのテスト"""
    
def test_confidence_calculation():
    """信頼度計算のテスト"""

# tests/test_mmlu_loader.py  
def test_csv_loading():
    """CSV読み込みのテスト"""
    
def test_stratified_sampling():
    """層化サンプリングのテスト"""
```

#### 8.7.2 統合テスト
```python
# tests/integration/test_benchmark_flow.py
async def test_end_to_end_benchmark():
    """エンドツーエンドテスト（5問程度で実行）"""
```

### 8.8 実装の優先順位

#### 高優先度（必須機能）
1. `MMLUDataLoader`の実装
2. 既存の`orchestrator.py`のMMLU対応改良
3. `AnswerExtractor`の基本実装
4. 評価ロジックの実装

#### 中優先度（品質向上）
1. エラーハンドリングの強化
2. 並列処理の実装
3. 詳細分析機能
4. 可視化機能

#### 低優先度（拡張機能）
1. Web UIでのリアルタイム監視
2. 他のベンチマークデータセットへの拡張
3. 高度な統計分析機能

この設計に基づいて実装することで、学術的に有意義なマルチエージェント議論システムの評価が可能になります。実装時は高優先度の機能から順番に進め、各段階でテストを実行して品質を確保してください。