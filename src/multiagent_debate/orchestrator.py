
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
from .agents import AGENT_PERSONAS

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

def get_subjective_perspective(my_name: str, all_agents: Dict[str, str]) -> str:
    """Generates a subjective description of other agents."""
    descriptions = []
    for name, persona in all_agents.items():
        if name == my_name:
            continue
        # A simple summary of the persona string
        role = persona.split('、')[1].replace('です。', '')
        descriptions.append(f"- {name}: {role}")
    return "\n".join(descriptions)

def run_debate(
    topic: str,
    initial_speaker: str,
    agent_names: List[str],
    max_turns: int = 10
) -> Dict[str, Any]:
    """Runs a debate with the new conversational agent system."""
    
    logger = setup_debate_logger()
    
    # 1. Initialize the state with subjective perspectives
    agent_states = {}
    for name in agent_names:
        subjective_view = get_subjective_perspective(name, AGENT_PERSONAS)
        agent_states[name] = AgentState(
            name=name,
            persona=AGENT_PERSONAS[name],
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
        # Initialize monitoring fields
        convergence_score=0.0,
        ready_flags=[],
        statement_embeddings=[]
    )

    # 2. Create the graph
    app = create_debate_graph()

    # 3. Run the debate
    print(f"--- Starting Debate ---")
    print(f"Topic: {topic}")
    print(f"Participants: {', '.join(agent_names)}")
    print(f"Initial Speaker: {initial_speaker}")
    print(f"Log file: {logger.handlers[0].baseFilename}")
    print("-------------------------")

    start_time = time.time()
    final_state = app.invoke(initial_state)
    end_time = time.time()

    # 4. Format and return the results
    result = {
        "topic": final_state["topic"],
        "conclusion": final_state["conclusion"],
        "full_transcript": final_state["full_transcript"],
        "execution_time": f"{end_time - start_time:.2f} seconds",
        "total_turns": final_state["current_turn"],
        "log_file": logger.handlers[0].baseFilename
    }

    return result

