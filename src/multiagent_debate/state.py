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
