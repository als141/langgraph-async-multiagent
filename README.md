# LangGraph Async Multi-Agent Debate System

## 概要

このプロジェクトは、**LangGraph**を使用した非同期マルチエージェント議論システムです。異なるペルソナを持つ複数のAIエージェントが、人間らしい議論を行い、自律的に結論に到達するシステムを実装しています。

## 主な特徴

- **非同期ストリーミング**: リアルタイムでエージェントの発言を表示
- **主観的視点**: 各エージェントが他のエージェントに対する独自の見方を持つ
- **自律的議論管理**: エージェントが自分で次の話者を指名
- **多層的終了判定**: 複数の条件を組み合わせた議論終了判定
- **ファシリテーター機能**: AIファシリテーターによる議論の進行管理

## システムアーキテクチャ

### 1. LangGraphによる状態管理

#### 状態定義 (src/multiagent_debate/state.py)

```python
class ConversationState(TypedDict):
    topic: str                          # 議論のトピック
    agent_states: Dict[str, AgentState] # エージェント個別の状態
    next_speaker: str                   # 次の発言者
    current_turn: int                   # 現在のターン数
    max_turns: int                      # 最大ターン数
    
    # 監視・評価フィールド
    convergence_score: float            # 発言の収束度（0-1）
    ready_flags: List[bool]             # 各エージェントの結論準備状況
    statement_embeddings: List[List[float]]  # 発言の埋め込みベクトル
    
    # ファシリテーター機能
    facilitator_action: Optional[str]   # ファシリテーターのアクション
    facilitator_message: Optional[str]  # ファシリテーターからのメッセージ
    
    # 議論の質評価
    discussion_depth: float             # 議論の深さ（0-1）
    pending_questions: List[str]        # 未回答の質問リスト
```

#### グラフ構造 (src/multiagent_debate/graph.py)

```
[agent_node] → [update_metrics_node] → [facilitator_node] → [conclusion_node]
     ↑              ↓                         ↓
     └─────────────────────────────────────────┘
```

**6つのノード**:
1. **agent_node**: エージェントの発言処理
2. **update_metrics_node**: 議論メトリクスの更新
3. **facilitator_node**: ファシリテーターによる進行判断
4. **pre_conclusion_node**: 暫定結論の作成
5. **final_comment_node**: 最終コメントの収集
6. **conclusion_node**: 最終結論の生成

### 2. エージェント設計理論

#### 主観的視点システム (src/multiagent_debate/config.py)

各エージェントは他のエージェントに対する**主観的な見方**を持ちます：

```python
AGENTS_CONFIG = [
    {
        "name": "佐藤",
        "persona": "議論の司会進行役。常に冷静沈着で、議論が円滑に進むように配慮する。",
        "subjective_views": {
            "鈴木": "少し懐疑的で、物事の裏側を考えがちだが、重要な指摘も多い。",
            "田中": "純粋で素直な視点を持っているが、議論が複雑になるとついていけない。"
        }
    },
    {
        "name": "鈴木", 
        "persona": "意見に対して懐疑的になり、あまり人の言う事を聞かない。",
        "subjective_views": {
            "佐藤": "議論をまとめようとするが、少し優等生すぎる。",
            "田中": "子供っぽい理想論を言うことが多い。"
        }
    }
]
```

#### 動的出力制御 (src/multiagent_debate/agents.py)

```python
# エージェント名をLiteral型で制限
ValidNextAgents = Literal[tuple(all_agent_names + ["Conclusion"])]

class AgentDecision(BaseModel):
    thoughts: str = Field(description="内部思考プロセス")
    response: str = Field(description="発言内容")
    next_agent: ValidNextAgents = Field(description="次の発言者")
    ready_to_conclude: bool = Field(description="結論準備完了フラグ")
```

**特徴**:
- **Literal型**による厳密な型チェック
- **JSON Mode**による構造化出力の強制
- **非同期ストリーミング**によるリアルタイム表示
- **フォールバック機能**（JSON解析失敗時の対応）

### 3. 議論終了の多層判定システム

#### 条件分岐ロジック (src/multiagent_debate/graph.py:434-449)

```python
def route_after_metrics(state: ConversationState) -> str:
    # 1. 強制終了条件
    if state["next_speaker"] == "Conclusion": 
        return "conclusion_node"
    if state["current_turn"] >= state["max_turns"]: 
        return "conclusion_node"
    
    # 2. 質問ベース継続判定（最優先）
    if state["pending_questions"]:
        print(f"継続: {len(state['pending_questions'])} 個の未回答質問")
        return "agent_node"
    
    # 3. ファシリテーター判定
    if state["current_turn"] % state["facilitator_check_interval"] == 0:
        return "facilitator_node"
    
    # 4. 高収束度での自動終了
    ready_ratio = sum(state["ready_flags"]) / len(state["ready_flags"])
    if (state["convergence_score"] > 0.98 and 
        ready_ratio > 0.8 and 
        state["discussion_depth"] > 0.7):
        return "conclusion_node"
    
    return "agent_node"
```

#### ファシリテーター判断基準 (src/multiagent_debate/graph.py:252-257)

```python
# 人間らしい会話を最優先とする判断基準
rules = [
    "NEVER interrupt if there are pending questions",
    "Continue if discussion_depth < 0.7", 
    "Continue if current_turn < max_turns * 0.6",
    "Only propose_conclusion if truly stagnant (3+ repetitions)",
    "Only propose_conclusion if ready_ratio > 80% AND remaining_turns < 3"
]
```

### 4. 議論品質の定量評価

#### 収束度計算 (src/multiagent_debate/graph.py:202-211)

```python
# OpenAI Embeddingsを使用したコサイン類似度計算
latest_embedding = await embeddings.aembed_query(spoken_content)
state["statement_embeddings"].append(latest_embedding)

# 前回発言との類似度計算
dot_product = np.dot(last_embedding, prev_embedding)
magnitude_1 = np.linalg.norm(last_embedding)
magnitude_2 = np.linalg.norm(prev_embedding)
convergence_score = dot_product / (magnitude_1 * magnitude_2)
```

#### 議論深度計算 (src/multiagent_debate/graph.py:228-229)

```python
# 質問応答比率による議論の深さ測定
total_questions_asked = state["current_turn"] - len(state["pending_questions"])
discussion_depth = total_questions_asked / max(1, state["current_turn"])
```

### 5. 段階的結論生成プロセス

#### 3段階の結論生成

1. **暫定結論生成** (pre_conclusion_node)
   - 議論全体を客観的に要約
   - 主要論点と各立場を明確化
   - 参加者に最終確認を求める

2. **最終コメント収集** (final_comment_node)
   - 各エージェントが暫定結論に対してコメント
   - 見落としや修正点の指摘
   - 並行処理による効率化

3. **最終結論統合** (conclusion_node)
   - 暫定結論と最終コメントを統合
   - 包括的で完全な結論を生成

## 実行方法

### 環境設定

```bash
# 依存関係のインストール
uv sync

# 環境変数の設定
cp .env.example .env
# .envファイルにOPENAI_API_KEYを設定
```

### Streamlit UI での実行

```bash
streamlit run app.py
```

### コマンドライン実行

```bash
python -m src.multiagent_debate.orchestrator
```

## プロジェクト構造

```
src/multiagent_debate/
├── __init__.py
├── agents.py          # エージェント実装
├── config.py          # エージェント設定
├── graph.py           # LangGraphワークフロー
├── orchestrator.py    # 実行制御
└── state.py           # 状態定義

app.py                 # Streamlit UI
nicegui_app.py        # NiceGUI UI
experiment_lab.py     # 実験用スクリプト
```

## 技術的特徴

### 非同期ストリーミング

- **リアルタイム表示**: エージェントの発言を文字単位でストリーミング
- **並行処理**: 複数エージェントのコメント生成を並列実行
- **エラーハンドリング**: ストリーミング失敗時の自動フォールバック

### 堅牢性

- **型安全性**: Pydantic + Literal型による厳密な型チェック
- **フォールバック機能**: JSON解析失敗時の緊急対応
- **自己修復**: 無効な次話者指名の自動修正

### 拡張性

- **設定駆動**: config.pyによる簡単なエージェント追加
- **モジュラー設計**: 各コンポーネントの独立性
- **プラグイン式**: 新しい判定ロジックの追加が容易

## 参考文献・理論背景

- **LangGraph**: 状態管理とワークフロー制御
- **Multi-Agent Systems**: 分散AIによる協調的問題解決
- **Conversational AI**: 自然な対話の生成と管理
- **Embedding-based Similarity**: 意味的類似度による収束判定

---

このシステムは、人間の議論に近い自然な流れを維持しながら、AIの能力を活用して効率的で質の高い議論を実現することを目指しています。