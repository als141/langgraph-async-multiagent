import asyncio
import pprint
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from multiagent_debate.orchestrator import run_graph
from multiagent_debate.config import AGENTS_CONFIG

async def main():
    """Main function to run the debate from the command line."""
    topic = "消費税減税は日本経済にプラスか？"
    max_turns = 10

    print(f"--- Starting Debate ---")
    print(f"Topic: {topic}")
    agent_names = [agent["name"] for agent in AGENTS_CONFIG]
    print(f"Participants: {', '.join(agent_names)}")
    print(f"Max Turns: {max_turns}")
    print("-------------------------")

    final_conclusion = None

    async for event in run_graph(topic, max_turns=max_turns):
        if event["type"] == "agent_message":
            # The message is already printed by the graph node, so we just observe
            pass
        elif event["type"] == "status_update":
            print(f"\n--- {event['message']} ---\n")
        elif event["type"] == "facilitator_message":
            print(f"\n--- Facilitator: {event['message']} ---\n")
        elif event["type"] == "pre_conclusion":
            print("\n--- Preliminary Conclusion ---")
            pprint.pprint(event["content"])
        elif event["type"] == "final_comments":
            print("\n--- Final Comments ---")
            for comment in event["content"]:
                print(comment)
        elif event["type"] == "conclusion":
            final_conclusion = event["conclusion"]
            print("\n--- Final Conclusion ---")
            pprint.pprint(final_conclusion)
        elif event["type"] == "end_of_debate":
            print("\n--- End of Debate ---")

    if not final_conclusion:
        print("\nDebate ended without a formal conclusion.")

if __name__ == "__main__":
    asyncio.run(main())