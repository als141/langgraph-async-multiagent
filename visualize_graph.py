#!/usr/bin/env python3
"""
LangGraph Visualization Script
グラフ構造とエージェント関係の可視化
"""
import os
import sys
from pathlib import Path
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import json

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.multiagent_debate.graph import create_debate_graph
from src.multiagent_debate.config import AGENTS_CONFIG

def visualize_langgraph_structure():
    """LangGraphの構造を複数の方法で可視化"""
    
    print("=== LangGraph 構造可視化 ===\n")
    
    # グラフを作成
    graph = create_debate_graph()
    
    print("1. ASCII アート表示:")
    print("-" * 50)
    try:
        # ASCII表示を試行
        ascii_graph = graph.get_graph().print_ascii()
        if ascii_graph:
            print(ascii_graph)
        else:
            print("ASCII表示は利用できません")
    except Exception as e:
        print(f"ASCII表示エラー: {e}")
    
    print("\n2. グラフの詳細情報:")
    print("-" * 50)
    try:
        graph_data = graph.get_graph()
        print(f"ノード数: {len(graph_data.nodes)}")
        print(f"エッジ数: {len(graph_data.edges)}")
        
        print("\nノード一覧:")
        for node in graph_data.nodes:
            print(f"  - {node}")
        
        print("\nエッジ一覧:")
        for edge in graph_data.edges:
            print(f"  - {edge}")
            
    except Exception as e:
        print(f"グラフ情報取得エラー: {e}")
    
    print("\n3. Mermaid PNG出力を試行:")
    print("-" * 50)
    try:
        png_data = graph.get_graph().draw_mermaid_png(
            output_file_path="graph_structure.png"
        )
        print("✓ graph_structure.png に保存しました")
        return png_data
    except Exception as e:
        print(f"Mermaid PNG出力エラー: {e}")
        return None

def visualize_agent_relationships():
    """エージェント間の関係を可視化"""
    
    print("\n=== エージェント関係図 ===\n")
    
    # NetworkXグラフを作成
    G = nx.DiGraph()
    
    # エージェントをノードとして追加
    for agent in AGENTS_CONFIG:
        G.add_node(agent["name"], 
                  persona=agent["persona"], 
                  avatar=agent["avatar"])
    
    # 主観的視点を基にエッジを追加
    for agent in AGENTS_CONFIG:
        if "subjective_views" in agent:
            for target, view in agent["subjective_views"].items():
                G.add_edge(agent["name"], target, view=view)
    
    # 可視化
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # ノードの描画
    node_colors = ['lightblue', 'lightcoral', 'lightgreen']
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors[:len(G.nodes)],
                          node_size=3000,
                          alpha=0.8)
    
    # エッジの描画
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          alpha=0.6,
                          connectionstyle="arc3,rad=0.1")
    
    # ラベルの描画
    labels = {}
    for agent in AGENTS_CONFIG:
        labels[agent["name"]] = f'{agent["avatar"]}\n{agent["name"]}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title("エージェント間の主観的関係図", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("agent_relationships.png", dpi=300, bbox_inches='tight')
    print("✓ agent_relationships.png に保存しました")
    plt.show()
    
    # 関係の詳細を表示
    print("\n主観的視点の詳細:")
    print("-" * 50)
    for agent in AGENTS_CONFIG:
        print(f"\n{agent['avatar']} {agent['name']}の視点:")
        if "subjective_views" in agent:
            for target, view in agent["subjective_views"].items():
                print(f"  → {target}: {view}")
        else:
            print("  （他のエージェントへの特別な視点なし）")

def create_system_architecture_diagram():
    """システムアーキテクチャ図を作成"""
    
    print("\n=== システムアーキテクチャ図 ===\n")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # ノードの位置定義
    nodes = {
        "agent_node": (2, 8),
        "update_metrics_node": (6, 8),
        "facilitator_node": (10, 8),
        "pre_conclusion_node": (6, 5),
        "final_comment_node": (6, 3),
        "conclusion_node": (6, 1),
        "END": (10, 1)
    }
    
    # ノードの描画
    for node_name, (x, y) in nodes.items():
        if node_name == "END":
            color = 'red'
            alpha = 0.7
        elif node_name == "agent_node":
            color = 'lightblue'
            alpha = 0.8
        elif "conclusion" in node_name:
            color = 'lightgreen'
            alpha = 0.8
        else:
            color = 'lightyellow'
            alpha = 0.8
            
        box = FancyBboxPatch((x-1, y-0.4), 2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=color,
                           edgecolor='black',
                           alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, node_name.replace('_', '\n'), 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # エッジの描画
    edges = [
        ("agent_node", "update_metrics_node"),
        ("update_metrics_node", "facilitator_node"),
        ("update_metrics_node", "agent_node"),  # ループバック
        ("facilitator_node", "agent_node"),  # ファシリテーター判断後の継続
        ("facilitator_node", "pre_conclusion_node"),
        ("pre_conclusion_node", "final_comment_node"),
        ("final_comment_node", "conclusion_node"),
        ("conclusion_node", "END"),
        ("update_metrics_node", "conclusion_node"),  # 直接終了
    ]
    
    for start, end in edges:
        start_pos = nodes[start]
        end_pos = nodes[end]
        
        # 矢印の色を決定
        if start == "update_metrics_node" and end == "agent_node":
            color = 'blue'  # ループバック
        elif start == "facilitator_node" and end == "agent_node":
            color = 'orange'  # ファシリテーター継続
        elif end == "conclusion_node":
            color = 'green'  # 終了への流れ
        else:
            color = 'black'  # 通常の流れ
        
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # 条件分岐の説明を追加
    ax.text(2, 6, "条件分岐:\n• 未回答質問あり → 継続\n• ファシリテーター判定\n• 高収束度 → 終了", 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.8),
            fontsize=8)
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('LangGraph ワークフロー構造', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("system_architecture.png", dpi=300, bbox_inches='tight')
    print("✓ system_architecture.png に保存しました")
    plt.show()

def export_graph_data():
    """グラフデータをJSONで出力"""
    
    print("\n=== グラフデータ出力 ===\n")
    
    try:
        graph = create_debate_graph()
        graph_data = graph.get_graph()
        
        # グラフデータを辞書形式で整理
        export_data = {
            "nodes": list(graph_data.nodes),
            "edges": [{"from": edge[0], "to": edge[1]} for edge in graph_data.edges],
            "agent_config": AGENTS_CONFIG,
            "graph_type": "StateGraph",
            "description": "LangGraph Async Multi-Agent Debate System"
        }
        
        # JSONファイルに出力
        with open("graph_data.json", "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print("✓ graph_data.json に保存しました")
        print(f"ノード数: {len(export_data['nodes'])}")
        print(f"エッジ数: {len(export_data['edges'])}")
        
    except Exception as e:
        print(f"グラフデータ出力エラー: {e}")

def main():
    """メイン実行関数"""
    print("🎨 LangGraph Multi-Agent Debate System Visualization")
    print("=" * 60)
    
    # 1. LangGraphの構造可視化
    png_data = visualize_langgraph_structure()
    
    # 2. エージェント関係図
    visualize_agent_relationships()
    
    # 3. システムアーキテクチャ図
    create_system_architecture_diagram()
    
    # 4. グラフデータ出力
    export_graph_data()
    
    print("\n🎉 可視化完了!")
    print("生成されたファイル:")
    print("  - graph_structure.png (LangGraph構造)")
    print("  - agent_relationships.png (エージェント関係)")
    print("  - system_architecture.png (システム構造)")
    print("  - graph_data.json (グラフデータ)")

if __name__ == "__main__":
    main()