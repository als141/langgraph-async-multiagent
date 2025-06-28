import streamlit as st
import asyncio
import sys
import os

# --- Path Setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from multiagent_debate.orchestrator import run_graph
from multiagent_debate.agents import AGENT_PERSONAS

# --- UI Configuration ---
st.set_page_config(page_title="Multi-Agent Debate", layout="wide")
st.title("🧠 Multi-Agent Debate")
AGENT_AVATARS = {"佐藤": "🧑‍🏫", "鈴木": "😒", "田中": "👦", "Facilitator": "🤖", "user": "👤"}

# --- Session State Initialization ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
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
        value=10,
        step=1,
        disabled=st.session_state.is_running,
        help="Set the maximum number of turns for the debate."
    )

if st.button("Start Debate", disabled=st.session_state.is_running or not topic_input):
    st.session_state.messages = [{"role": "user", "content": f"**Topic:** {topic_input}"}]
    st.session_state.conclusion_data = {}
    st.session_state.status_message = ""
    st.session_state.is_running = True
    agent_names = list(AGENT_PERSONAS.keys())
    initial_speaker = "佐藤"
    st.session_state.debate_generator = run_graph(
        topic_input, 
        initial_speaker, 
        agent_names, 
        max_turns=max_turns_input
    )
    st.rerun()

# --- Processing Loop ---
if st.session_state.is_running:
    try:
        loop = st.session_state.event_loop
        generator = st.session_state.debate_generator
        event = loop.run_until_complete(generator.__anext__())

        if event["type"] == "agent_message":
            agent_name = event["agent_name"]
            message_content = f"**{agent_name}:** {event['message']}"
            st.session_state.messages.append({"role": agent_name, "content": message_content})
        
        elif event["type"] == "facilitator_message":
            st.session_state.messages.append({"role": "Facilitator", "content": f'*_{event["message"]}_*'})

        elif event["type"] == "status_update":
            st.session_state.status_message = event["message"]

        elif event["type"] == "pre_conclusion":
            st.session_state.conclusion_data["pre_conclusion"] = event["content"]
            st.session_state.status_message = ""

        elif event["type"] == "final_comments":
            st.session_state.conclusion_data["final_comments"] = event["content"]
            st.session_state.status_message = ""

        elif event["type"] == "conclusion":
            st.session_state.conclusion_data["conclusion"] = event["conclusion"]
            st.session_state.status_message = ""
        
        elif event["type"] == "end_of_debate":
            st.session_state.is_running = False
            st.balloons()

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