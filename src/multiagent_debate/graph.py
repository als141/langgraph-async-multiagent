"""
New LangGraph Workflow for Conversational Agents
"""
import os
import numpy as np
import pathlib
from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from .state import ConversationState
from .agents import ConversationalAgent, llm

# Load .env from project root
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize embeddings for convergence score calculation
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# --- Graph Nodes ---

def agent_node(state: ConversationState) -> ConversationState:
    """Executes the current speaker's turn."""
    speaker_name = state["next_speaker"]
    agent_names = list(state["agent_states"].keys())

    # --- Fallback Logic for Robustness ---
    if speaker_name not in agent_names and speaker_name != "Conclusion":
        print(f"[Warning] Invalid next_speaker: '{speaker_name}'. Using round-robin.")
        last_speaker_name = state["full_transcript"][-1].split(":")[0].strip("[]").split(" ")[-1]
        last_speaker_index = agent_names.index(last_speaker_name)
        next_speaker_index = (last_speaker_index + 1) % len(agent_names)
        speaker_name = agent_names[next_speaker_index]
        state["next_speaker"] = speaker_name

    current_agent_state = state["agent_states"][speaker_name]

    # Initialize the agent for the current speaker
    agent = ConversationalAgent(current_agent_state, state["topic"], agent_names)

    # Get the agent's decision
    decision = agent.invoke()

    # The speaker's response, as an AIMessage for their own history
    ai_message = AIMessage(content=decision.response, name=speaker_name)
    
    # The same response, but as a HumanMessage for other agents' histories
    human_message = HumanMessage(content=f"（{speaker_name}の発言）: {decision.response}", name="human")

    # Update the state
    for name, agent_state in state["agent_states"].items():
        if name == speaker_name:
            agent_state["chat_history"].append(ai_message)
        else:
            agent_state["chat_history"].append(human_message)

    state["next_speaker"] = decision.next_agent
    state["current_turn"] += 1
    
    # Store the ready_to_conclude flag
    state["ready_flags"].append(decision.ready_to_conclude)
    
    # Log the turn
    turn_log = f"[Turn {state['current_turn']}] {speaker_name}: {decision.response}"
    state["full_transcript"].append(turn_log)
    state["logger"].info(turn_log)
    print(turn_log)
    print(f" -> Next Speaker: {decision.next_agent}")

    return state

def update_metrics_node(state: ConversationState) -> ConversationState:
    """Updates monitoring metrics after each turn."""
    current_turn = state["current_turn"]
    
    if current_turn > 0:
        # Get the latest statement for embedding
        latest_statement = state["full_transcript"][-1]
        # Extract just the spoken content (remove "[Turn X] Name: " prefix)
        spoken_content = latest_statement.split(": ", 1)[1] if ": " in latest_statement else latest_statement
        
        # Generate embedding for the latest statement
        try:
            latest_embedding = embeddings.embed_query(spoken_content)
            state["statement_embeddings"].append(latest_embedding)
            
            # Calculate convergence score (similarity between recent statements)
            if len(state["statement_embeddings"]) >= 2:
                # Compare last 2 statements for convergence
                last_embedding = state["statement_embeddings"][-1]
                prev_embedding = state["statement_embeddings"][-2]
                
                # Calculate cosine similarity
                dot_product = np.dot(last_embedding, prev_embedding)
                magnitude_1 = np.linalg.norm(last_embedding)
                magnitude_2 = np.linalg.norm(prev_embedding)
                
                if magnitude_1 > 0 and magnitude_2 > 0:
                    cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
                    state["convergence_score"] = max(0.0, cosine_similarity)  # Ensure non-negative
                else:
                    state["convergence_score"] = 0.0
            else:
                state["convergence_score"] = 0.0
                
        except Exception as e:
            print(f"[Warning] Failed to calculate embedding: {e}")
            state["convergence_score"] = 0.0
    
    # Calculate readiness ratio
    ready_count = sum(state["ready_flags"])
    total_flags = len(state["ready_flags"])
    readiness_ratio = ready_count / total_flags if total_flags > 0 else 0.0
    
    # Log metrics for debugging
    print(f" -> Metrics: Convergence Score: {state['convergence_score']:.3f}, Readiness: {ready_count}/{total_flags} ({readiness_ratio:.3f})")
    
    return state

def conclusion_node(state: ConversationState) -> ConversationState:
    """Generates the final conclusion for the debate."""
    print("\n--- Generating Conclusion ---")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="あなたは議論の結論をまとめる専門家です。以下の議論の完全な記録を読み、最終的な結論を客観的に要約してください。意見が分かれた場合は、それも明確に記述してください。議論のテーマ：" + state["topic"]),
        HumanMessage(content="\n".join(state["full_transcript"])) 
    ])
    
    chain = prompt | llm
    conclusion = chain.invoke({}).content
    state["conclusion"] = conclusion
    print(f"Conclusion: {conclusion}")
    return state

# --- Conditional Routing ---

def route_to_next_speaker(state: ConversationState) -> str:
    """Routes to the conclusion node or back to the agent node with enhanced termination conditions."""
    # Check if explicit conclusion was requested
    if state["next_speaker"] == "Conclusion":
        return "conclusion_node"
    
    # Check hard stop condition
    if state["current_turn"] >= state["max_turns"]:
        return "conclusion_node"
    
    # Enhanced termination conditions based on monitoring metrics
    ready_count = sum(state["ready_flags"])
    total_flags = len(state["ready_flags"])
    readiness_ratio = ready_count / total_flags if total_flags > 0 else 0.0
    
    # Terminate if convergence score is high AND readiness ratio exceeds threshold
    if state["convergence_score"] > 0.95 and readiness_ratio > 0.66:
        print(f" -> Auto-termination triggered: High convergence ({state['convergence_score']:.3f}) + High readiness ({readiness_ratio:.3f})")
        return "conclusion_node"
    
    return "agent_node"

# --- Graph Definition ---

def create_debate_graph() -> StateGraph:
    """Creates the LangGraph workflow for the debate."""
    workflow = StateGraph(ConversationState)

    workflow.add_node("agent_node", agent_node)
    workflow.add_node("update_metrics_node", update_metrics_node)
    workflow.add_node("conclusion_node", conclusion_node)

    workflow.set_entry_point("agent_node")

    # Agent node always goes to update_metrics_node
    workflow.add_edge("agent_node", "update_metrics_node")
    
    # Update metrics node uses conditional routing
    workflow.add_conditional_edges(
        "update_metrics_node",
        route_to_next_speaker,
        {
            "agent_node": "agent_node",
            "conclusion_node": "conclusion_node",
        }
    )
    workflow.add_edge("conclusion_node", END)

    return workflow.compile()