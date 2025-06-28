
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
AGENT_AVATARS = {"佐藤": "🧑‍🏫", "鈴木": "😒", "田中": "👦", "Facilitator": "🤖", "user": "👤", "status": "⚙️"}

# --- Session State Initialization ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conclusion" not in st.session_state:
    st.session_state.conclusion = None
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

if st.session_state.conclusion:
    st.subheader("🏆 Final Conclusion")
    st.success(st.session_state.conclusion)

st.divider()

# --- Input & Control Section ---
topic_input = st.text_input(
    "Enter the topic for debate:",
    placeholder="例：消費税減税は日本経済にプラスか？",
    disabled=st.session_state.is_running,
)

if st.button("Start Debate", disabled=st.session_state.is_running or not topic_input):
    st.session_state.messages = [{
        "role": "user", 
        "content": f"**Topic:** {topic_input}"
    }]
    st.session_state.conclusion = None
    st.session_state.is_running = True
    agent_names = list(AGENT_PERSONAS.keys())
    initial_speaker = "佐藤"
    st.session_state.debate_generator = run_graph(topic_input, initial_speaker, agent_names)
    st.rerun()

# --- Processing Loop ---
if st.session_state.is_running:
    with st.spinner("Debate in progress..."):
        try:
            loop = st.session_state.event_loop
            generator = st.session_state.debate_generator
            event = loop.run_until_complete(generator.__anext__())

            if event["type"] == "agent_message":
                agent_name = event["agent_name"]
                # Prepend the agent's name to the message for clear identification
                message_content = f"**{agent_name}:** {event['message']}"
                st.session_state.messages.append({"role": agent_name, "content": message_content})
            
            elif event["type"] == "status_update":
                # Display status updates with a distinct style
                st.session_state.messages.append({"role": "status", "content": f'*_{event["message"]}_*'})

            elif event["type"] == "conclusion":
                st.session_state.conclusion = event["conclusion"]
            
            elif event["type"] == "end_of_debate":
                st.session_state.is_running = False
                st.balloons()

            st.rerun()

        except StopAsyncIteration:
            st.session_state.is_running = False
            if not st.session_state.conclusion:
                 st.warning("The debate ended without a formal conclusion.")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.is_running = False
            st.rerun()
