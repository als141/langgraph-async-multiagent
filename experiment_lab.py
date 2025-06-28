"""
🧪 New Multi-Agent Debate System - Demonstration
"""
import asyncio
import json
from src.multiagent_debate.orchestrator import run_debate

def main():
    """Main function to run a sample debate."""
    
    # --- Debate Configuration ---
    topic = "リモートワークを導入すべきか？"
    agent_names = ["佐藤", "鈴木", "田中"]
    initial_speaker = "佐藤"
    max_turns = 7

    # --- Run the Debate ---
    result = run_debate(
        topic=topic,
        agent_names=agent_names,
        initial_speaker=initial_speaker,
        max_turns=max_turns
    )

    # --- Print the Results ---
    print("\n--- Debate Finished ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("-----------------------")

if __name__ == "__main__":
    main()
