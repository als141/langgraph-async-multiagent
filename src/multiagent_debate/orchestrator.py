

"""
Orchestrator with Subjective Views and Logging
"""
import time
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage

from .state import ConversationState, AgentState
from .graph import create_debate_graph
from .config import AGENTS_CONFIG # Import the single source of truth

def setup_debate_logger(log_dir="logs"):
    """Sets up a logger for the debate transcript."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"debate_{timestamp}.log")
    
    handler = logging.FileHandler(log_filename, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(f"debate_logger_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

def get_subjective_perspective_from_config(my_name: str, all_agents_config: List[Dict[str, Any]]) -> str:
    """Generates a subjective description of other agents from the config."""
    my_config = next((agent for agent in all_agents_config if agent["name"] == my_name), None)
    if not my_config or "subjective_views" not in my_config:
        return ""
    
    descriptions = []
    for target_name, view in my_config["subjective_views"].items():
        descriptions.append(f"- {target_name}: {view}")
    return "\n".join(descriptions)

async def run_graph(topic: str, max_turns: int = 10):
    """Asynchronous wrapper for running the debate graph and yielding events."""
    
    logger = setup_debate_logger()
    
    agent_names = [agent["name"] for agent in AGENTS_CONFIG]
    initial_speaker = agent_names[0] if agent_names else ""

    agent_states = {}
    for agent_config in AGENTS_CONFIG:
        agent_name = agent_config["name"]
        subjective_view = get_subjective_perspective_from_config(agent_name, AGENTS_CONFIG)
        agent_states[agent_name] = AgentState(
            name=agent_name,
            persona=agent_config["persona"],
            chat_history=[],
            subjective_view=subjective_view
        )

    initial_state = ConversationState(
        topic=topic,
        agent_states=agent_states,
        next_speaker=initial_speaker,
        current_turn=0,
        max_turns=max_turns,
        conclusion=None,
        full_transcript=[],
        logger=logger,
        convergence_score=0.0,
        ready_flags=[],
        statement_embeddings=[],
        facilitator_check_interval=8,
        facilitator_action=None,
        facilitator_message=None,
        preliminary_conclusion=None,
        final_comments=[],
        topic_diversity=0.0,
        discussion_depth=0.0,
        pending_questions=[]
    )

    app = create_debate_graph()

    async for event in app.astream_events(initial_state, version="v1", config={"recursion_limit": 50}):
        kind = event["event"]
        node_name = event["name"]

        if kind == "on_chain_start":
            if node_name == "pre_conclusion_node":
                yield {"type": "status_update", "message": "--- Pre-Conclusion: Preparing Draft Summary ---"}
            elif node_name == "final_comment_node":
                yield {"type": "status_update", "message": "--- Final Comments: Last Chance for Input ---"}
            elif node_name == "conclusion_node":
                yield {"type": "status_update", "message": "--- Generating Final Conclusion ---"}

        elif kind == "on_chain_end":
            output = event["data"].get("output")

            if node_name == "agent_node":
                if not output or not output.get("full_transcript"):
                    continue
                latest_transcript_entry = output.get("full_transcript", [])[-1]
                
                if ": " in latest_transcript_entry:
                    parts = latest_transcript_entry.split(": ", 1)
                    speaker_part = parts[0]
                    message = parts[1]
                    speaker_name = speaker_part.split("] ")[-1]
                else:
                    speaker_name = "Unknown"
                    message = latest_transcript_entry

                yield {"type": "agent_message", "agent_name": speaker_name, "message": message}

            elif node_name == "facilitator_node":
                if output and output.get("facilitator_message"):
                    yield {"type": "facilitator_message", "message": output.get("facilitator_message")}

            elif node_name == "pre_conclusion_node":
                if output and output.get("preliminary_conclusion"):
                    yield {"type": "pre_conclusion", "content": output.get("preliminary_conclusion")}

            elif node_name == "final_comment_node":
                if output and output.get("final_comments"):
                    yield {"type": "final_comments", "content": output.get("final_comments")}
            
            elif node_name == "conclusion_node":
                if output and output.get("conclusion"):
                    yield {"type": "conclusion", "conclusion": output.get("conclusion")}

    yield {"type": "end_of_debate"}
