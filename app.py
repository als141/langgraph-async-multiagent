
import streamlit as st
import asyncio
import sys
import os

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from multiagent_debate.orchestrator import run_graph
from multiagent_debate.config import AGENTS_CONFIG # Import the single source of truth

# --- UI Configuration ---
st.set_page_config(page_title="Multi-Agent Debate", layout="wide")
st.title("🧠 Multi-Agent Debate")

# Generate AGENT_AVATARS dynamically from the config
AGENT_AVATARS = {agent["name"]: agent["avatar"] for agent in AGENTS_CONFIG}
AGENT_AVATARS["user"] = "👤"
AGENT_AVATARS["status"] = "⚙️"

# --- Session State Initialization ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "_partial_agent_messages" not in st.session_state:
    st.session_state._partial_agent_messages = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conclusion_data" not in st.session_state:
    st.session_state.conclusion_data = {}
if "status_message" not in st.session_state:
    st.session_state.status_message = ""
if "debate_generator" not in st.session_state:
    st.session_state.debate_generator = None
if "event_loop" not in st.session_state:
    try:
        st.session_state.event_loop = asyncio.get_running_loop()
    except RuntimeError:
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)

# --- UI Rendering ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = AGENT_AVATARS.get(msg["role"], "👤")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

# --- Conclusion Rendering Section ---
conclusion_data = st.session_state.conclusion_data
if conclusion_data or st.session_state.status_message:
    st.subheader("🏆 Debate Conclusion Process")
    if st.session_state.status_message:
        st.info(st.session_state.status_message)
    if "pre_conclusion" in conclusion_data:
        with st.expander("**1. Preliminary Conclusion**", expanded=True):
            st.markdown(conclusion_data["pre_conclusion"])
    if "final_comments" in conclusion_data:
        with st.expander("**2. Final Comments**", expanded=True):
            for comment in conclusion_data["final_comments"]:
                st.markdown(comment)
    if "conclusion" in conclusion_data:
        with st.expander("**3. Final Conclusion**", expanded=True):
            st.success(conclusion_data["conclusion"])

st.divider()

# --- Input & Control Section ---
col1, col2 = st.columns([3, 1])
with col1:
    topic_input = st.text_input(
        "Enter the topic for debate:",
        placeholder="例：消費税減税は日本経済にプラスか？",
        disabled=st.session_state.is_running,
    )
with col2:
    max_turns_input = st.number_input(
        "Max Turns",
        min_value=2,
        max_value=50,
        value=30,
        step=1,
        disabled=st.session_state.is_running,
        help="Set the maximum number of turns for the debate."
    )

if st.button("Start Debate", disabled=st.session_state.is_running or not topic_input):
    st.session_state.messages = [{"role": "user", "content": f"**Topic:** {topic_input}"}]
    st.session_state.conclusion_data = {}
    st.session_state.status_message = ""
    st.session_state.is_running = True
    st.session_state.debate_generator = run_graph(
        topic_input, 
        max_turns=max_turns_input
    )
    st.rerun()

# --- Processing Loop ---
if st.session_state.is_running:
    try:
        loop = st.session_state.event_loop
        generator = st.session_state.debate_generator
        event = loop.run_until_complete(generator.__anext__())

        if event["type"] == "agent_message_chunk":
            agent_name = event["agent_name"]
            chunk = event["chunk"]
            if agent_name not in st.session_state._partial_agent_messages:
                st.session_state._partial_agent_messages[agent_name] = ""
            st.session_state._partial_agent_messages[agent_name] += chunk
            
            # Update or add message with streaming content
            found = False
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == agent_name and msg.get("is_streaming", False):
                    st.session_state.messages[i]["content"] = f"**{agent_name}:** {st.session_state._partial_agent_messages[agent_name]}"
                    found = True
                    break
            if not found:
                st.session_state.messages.append({
                    "role": agent_name, 
                    "content": f"**{agent_name}:** {st.session_state._partial_agent_messages[agent_name]}",
                    "is_streaming": True
                })

        elif event["type"] == "agent_message_complete":
            agent_name = event["agent_name"]
            message_content = event["message"]
            # Update the streaming message to complete
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == agent_name and msg.get("is_streaming", False):
                    st.session_state.messages[i]["content"] = f"**{agent_name}:** {message_content}"
                    st.session_state.messages[i]["is_streaming"] = False
                    break
            if agent_name in st.session_state._partial_agent_messages:
                del st.session_state._partial_agent_messages[agent_name]
        
        elif event["type"] == "facilitator_message":
            st.session_state.messages.append({"role": "Facilitator", "content": f'*_{event["message"]}_*'})

        elif event["type"] == "status_update":
            st.session_state.status_message = event["message"]

        elif event["type"] == "pre_conclusion_chunk":
            if "pre_conclusion" not in st.session_state.conclusion_data:
                st.session_state.conclusion_data["pre_conclusion"] = ""
            st.session_state.conclusion_data["pre_conclusion"] += event["chunk"]
            st.session_state.status_message = ""

        elif event["type"] == "pre_conclusion_complete":
            st.session_state.conclusion_data["pre_conclusion"] = event["content"]
            st.session_state.status_message = ""

        elif event["type"] == "final_comments_complete":
            st.session_state.conclusion_data["final_comments"] = event["content"]
            st.session_state.status_message = ""

        elif event["type"] == "conclusion_chunk":
            if "conclusion" not in st.session_state.conclusion_data:
                st.session_state.conclusion_data["conclusion"] = ""
            st.session_state.conclusion_data["conclusion"] += event["chunk"]
            st.session_state.status_message = ""

        elif event["type"] == "conclusion_complete":
            st.session_state.conclusion_data["conclusion"] = event["conclusion"]
            st.session_state.status_message = ""
        
        elif event["type"] == "end_of_debate":
            st.session_state.is_running = False
            st.balloons()

        # Force immediate rerun for smoother streaming
        st.rerun()

    except StopAsyncIteration:
        st.session_state.is_running = False
        if not st.session_state.conclusion_data.get("conclusion"):
             st.warning("The debate ended without a formal conclusion.")
        st.balloons()
        st.rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.is_running = False
        st.rerun()
