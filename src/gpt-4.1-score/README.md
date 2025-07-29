# シンプルGPT-4.1 MMLUスコアリングシステム

OpenAI GPT-4.1を使用してMMLU問題を直接解答するシンプルなスコアリングシステム。

## 特徴

- **システムプロンプトなし**: ユーザープロンプトのみで問題を提供
- **Structured Outputs**: JSON Schemaを使用して確実に選択肢を抽出
- **シンプルな構成**: 複雑な議論システムを使わず、直接AIに回答を求める
- **100問対応**: CSVから全100問を読み込み、順次処理
- **バッチ処理対応**: 複数問題を並行実行してスループットを向上

## 使用方法

### 1. 環境設定

```bash
# OpenAI APIキーを設定
export OPENAI_API_KEY="your-api-key-here"

# または .envファイルに記載
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. テスト実行

```bash
cd src/gpt-4.1-score
python test_simple_scorer.py
```

### 3. 全問題実行

```bash
cd src/gpt-4.1-score
python simple_mmlu_scorer.py --csv ../../data/mmlu_pro_100.csv --output ../../results/simple_results.json
```

### 4. 制限実行（例：10問のみ）

```bash
python simple_mmlu_scorer.py --csv ../../data/mmlu_pro_100.csv --limit 10
```

### 5. バッチ処理実行（例：バッチサイズ5で並行処理）

```bash
python simple_mmlu_scorer.py --csv ../../data/mmlu_pro_100.csv --batch-size 5
```

## ファイル構成

```
src/gpt-4.1-score/
├── simple_mmlu_scorer.py    # メインスコアリングシステム
├── test_simple_scorer.py    # テストスイート
├── run_simple_scoring.py    # 実行スクリプト
└── README.md               # このファイル
```

## システムの仕組み

### 1. ユーザープロンプト

```
必ず A, B, C, D, E, F, G, H, I, J のいずれかを選択してください
最適な選択肢を決定してください

問題:
[問題文]

選択肢:
A) [選択肢1]
B) [選択肢2]
...

回答は以下のJSON形式で行ってください:
{
"final_answer": "選択した選択肢アルファベットのみ"
}
```

### 2. JSON Schema

```json
{
    "type": "object",
    "properties": {
        "final_answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        }
    },
    "required": ["final_answer"],
    "additionalProperties": false
}
```

### 3. API呼び出し

```python
response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": user_prompt}],
    text={
        "format": {
            "type": "json_schema", 
            "schema": json_schema,
            "strict": True
        }
    }
)
```

## 出力形式

### コンソール出力例

```
✅ 100問を読み込みました
🚀 100問のスコアリングを開始...
[1/100] 問題 95 を処理中...
  ✅ 予測: J, 正解: J (現在の正答率: 100.0%)
[2/100] 問題 7520 を処理中...
  ✅ 予測: E, 正解: E (現在の正答率: 100.0%)
...
🎉 スコアリング完了!
📊 総問題数: 100
✅ 正答数: 85
📈 全体正答率: 85.0%
```

### JSON出力例

```json
{
  "total_questions": 100,
  "correct_answers": 85,
  "overall_accuracy": 0.85,
  "results": [
    {
      "question_id": "95",
      "question": "Where in the balance sheet...",
      "options": ["...", "..."],
      "correct_answer": "J",
      "predicted_answer": "J", 
      "is_correct": true,
      "response_time": 2.34
    }
  ]
}
```

## エラーハンドリング

- **API呼び出し失敗**: リトライなしで次の問題に進む
- **JSON解析エラー**: エラーを記録して次の問題に進む
- **無効な選択肢**: JSON Schemaで事前に制限
- **タイムアウト**: OpenAI APIのデフォルトタイムアウトを使用

## パフォーマンス

- **処理速度**: 1問あたり約2-3秒（APIレスポンス時間に依存）
- **コスト**: GPT-4.1の料金に準拠
- **制限**: OpenAI APIのレート制限に依存

## 比較対象

このシンプルシステムは、複雑なマルチエージェント議論システムと比較するためのベースラインとして機能します：

- **複雑度**: 最小限
- **推論プロセス**: なし（直接回答）
- **実行時間**: 高速
- **正答率**: ベースライン性能

## バッチ処理機能

### コマンドラインオプション

```bash
python simple_mmlu_scorer.py --help

options:
  --csv CSV             CSVファイルのパス (default: data/mmlu_pro_100.csv)
  --output OUTPUT       出力ファイルのパス (default: results/simple_scoring_results.json)
  --limit LIMIT         処理する問題数の上限
  --batch-size BATCH_SIZE
                        バッチサイズ（並行処理数） (default: 1)
```

### 使用例

```bash
# 順次処理（デフォルト）
python simple_mmlu_scorer.py --batch-size 1

# 3問ずつ並行処理
python simple_mmlu_scorer.py --batch-size 3

# 10問限定でバッチサイズ5
python simple_mmlu_scorer.py --limit 10 --batch-size 5
```

### バッチ処理の仕組み

1. **問題分割**: 全問題をバッチサイズごとに分割
2. **並行実行**: 各バッチ内の問題を`asyncio.gather()`で並行処理
3. **API制限対策**: バッチ間に1秒の待機時間を挿入
4. **プログレス表示**: バッチごとの進捗と個別問題の結果を表示

### 出力例（バッチ処理）

```
🚀 100問のスコアリングを開始... (バッチサイズ: 3)

🔄 バッチ 1/34 (3問) を並行処理中...
  ✅ 問題 95: 予測=J, 正解=J
  ❌ 問題 7520: 予測=A, 正解=E
  ✅ 問題 3367: 予測=C, 正解=C
  ⏳ 次のバッチまで1秒待機...

🔄 バッチ 2/34 (3問) を並行処理中...
  ...
```

## 拡張性

### 現在実装済み
- ✅ カテゴリ別統計
- ✅ 信頼度スコア  
- ✅ 並行処理（バッチ処理）
- ✅ 詳細なエラーログ

### 今後追加可能
- 結果の可視化
- より高度な信頼度計算
- API使用量の監視
- 結果のリアルタイム表示