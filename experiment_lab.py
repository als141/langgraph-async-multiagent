import asyncio
import pprint
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from multiagent_debate.orchestrator import run_graph
from multiagent_debate.config import AGENTS_CONFIG

def extract_response_from_chunk(text: str) -> str:
    """Extract response field content from JSON stream."""
    try:
        
        # Look for the response field
        if '"response"' in text and ':' in text:
            response_pos = text.find('"response"')
            after_response = text[response_pos:]
            colon_pos = after_response.find(':')
            if colon_pos != -1:
                after_colon = after_response[colon_pos + 1:].strip()
                if after_colon.startswith('"'):
                    # Find the content between quotes
                    quote_start = after_colon.find('"') + 1
                    remaining = after_colon[quote_start:]
                    
                    # Extract until closing quote (handle escapes)
                    content = ""
                    i = 0
                    while i < len(remaining):
                        char = remaining[i]
                        if char == '"' and (i == 0 or remaining[i-1] != '\\'):
                            break
                        content += char
                        i += 1
                    
                    return content
        
        return ""
    except:
        return ""

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

    current_agent_message = {}
    current_pre_conclusion = ""
    current_conclusion = ""
    final_conclusion = ""

    async for event in run_graph(topic, max_turns=max_turns):
        if event["type"] == "agent_message_chunk":
            agent_name = event["agent_name"]
            chunk = event["chunk"]
            if agent_name not in current_agent_message:
                current_agent_message[agent_name] = ""
                print(f"\n[{agent_name}]: ", end="", flush=True)
            
            # Extract response content from JSON chunk
            display_chunk = extract_response_from_chunk(current_agent_message[agent_name] + chunk)
            if display_chunk != current_agent_message[agent_name]:
                new_content = display_chunk[len(current_agent_message[agent_name]):]
                current_agent_message[agent_name] = display_chunk
                print(new_content, end="", flush=True)
        elif event["type"] == "agent_message_complete":
            agent_name = event["agent_name"]
            print() # New line after streaming completes
            current_agent_message[agent_name] = "" # Clear partial message
        elif event["type"] == "status_update":
            print(f"\n{event['message']}")
        elif event["type"] == "facilitator_message":
            print(f"\n--- Facilitator: {event['message']} ---")
        elif event["type"] == "pre_conclusion_chunk":
            if not current_pre_conclusion:
                print(f"\n--- Preliminary Conclusion ---\n", end="", flush=True)
            current_pre_conclusion += event["chunk"]
            print(event["chunk"], end="", flush=True)
        elif event["type"] == "pre_conclusion_complete":
            print() # New line after streaming completes
            current_pre_conclusion = ""
        elif event["type"] == "final_comments_complete":
            print("\n--- Final Comments ---")
            for comment in event["content"]:
                print(comment)
        elif event["type"] == "conclusion_chunk":
            if not current_conclusion:
                print(f"\n--- Final Conclusion ---\n", end="", flush=True)
            current_conclusion += event["chunk"]
            print(event["chunk"], end="", flush=True)
        elif event["type"] == "conclusion_complete":
            final_conclusion = event["conclusion"]
            print() # New line after streaming completes
            current_conclusion = ""
        elif event["type"] == "end_of_debate":
            print("\n--- End of Debate ---")

    if not final_conclusion:
        print("\nDebate ended without a formal conclusion.")

if __name__ == "__main__":
    asyncio.run(main())