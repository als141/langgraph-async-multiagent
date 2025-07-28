"""
New State Management for Conversational Agents
"""
import logging
from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State for a single agent."""
    name: str
    persona: str
    subjective_view: str # New field for subjective perspective
    chat_history: List[BaseMessage]

class ConversationState(TypedDict):
    """Global state for the conversation."""
    topic: str
    agent_states: Dict[str, AgentState]
    next_speaker: str
    current_turn: int
    max_turns: int
    conclusion: Optional[str]
    full_transcript: List[str]
    logger: logging.Logger
    # --- New monitoring fields ---
    convergence_score: float  # Similarity score between recent statements
    ready_flags: List[bool]   # List of ready_to_conclude flags from agents
    statement_embeddings: List[List[float]]  # Store embeddings for convergence calculation
    # --- Facilitator fields ---
    facilitator_check_interval: int  # Check every N turns
    facilitator_action: Optional[str]  # "continue", "propose_conclusion", "call_vote"
    facilitator_message: Optional[str]  # Message from facilitator when taking action
    # --- Phased termination fields ---
    preliminary_conclusion: Optional[str]  # Draft conclusion from pre_conclusion_node
    final_comments: List[str]  # Final comments from agents before conclusion
    # --- Discussion quality metrics ---
    topic_diversity: float  # Diversity of topics discussed
    discussion_depth: float  # How deeply issues have been explored
    pending_questions: List[str]  # Questions that need responses
