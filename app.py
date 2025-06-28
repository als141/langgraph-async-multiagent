import streamlit as st
import asyncio
import sys
import os

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from multiagent_debate.orchestrator import run_graph
from multiagent_debate.agents import AGENT_PERSONAS
from src.multiagent_debate.agents import AGENT_PERSONAS

# --- UI Configuration ---
st.set_page_config(page_title="Multi-Agent Debate", layout="wide")
st.title("🧠 Multi-Agent Debate")

# Define avatars for agents
AGENT_AVATARS = {
    "佐藤": "🧑‍🏫",
    "鈴木": "😒",
    "田中": "👦",
    "Facilitator": "🤖"
}

# --- Main Application Logic ---

# Initialize session state for messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    avatar = AGENT_AVATARS.get(message["role"], "👤")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Input controls
topic = st.text_input("Enter the topic for debate:", placeholder="例：日本の教育制度の未来について")

if st.button("Start Debate"):
    if topic:
        # --- Start of a new debate ---
        st.session_state.messages = []
        
        # Display user's topic
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**Topic:** {topic}")
        st.session_state.messages.append({"role": "user", "content": f"**Topic:** {topic}"})

        # Define debate participants
        agent_names = list(AGENT_PERSONAS.keys())
        initial_speaker = "佐藤" # Facilitator starts

        # Placeholder for the final conclusion
        conclusion_placeholder = st.empty()

        # Run the debate asynchronously and stream results
        try:
            async def stream_debate():
                # Use a spinner to indicate progress
                with st.spinner("Debate in progress..."):
                    async for event in run_graph(topic, initial_speaker, agent_names):
                        if event["type"] == "agent_message":
                            agent_name = event["agent_name"]
                            message_content = event["message"]
                            avatar = AGENT_AVATARS.get(agent_name, "👤")
                            
                            with st.chat_message(agent_name, avatar=avatar):
                                st.markdown(message_content)
                            
                            st.session_state.messages.append({"role": agent_name, "content": message_content})

                        elif event["type"] == "facilitator_message":
                            message_content = event["message"]
                            avatar = AGENT_AVATARS.get("Facilitator", "🤖")

                            with st.chat_message("Facilitator", avatar=avatar):
                                st.info(message_content)
                            
                            st.session_state.messages.append({"role": "Facilitator", "content": message_content})

                        elif event["type"] == "conclusion":
                            conclusion_text = event["conclusion"]
                            with conclusion_placeholder.container():
                                st.success(f"**Conclusion:**\n\n{conclusion_text}")
                            
                            st.session_state.messages.append({"role": "Conclusion", "content": conclusion_text})

                        elif event["type"] == "end_of_debate":
                            st.info("The debate has concluded.")
                            break
                st.balloons()

            # Run the async function
            asyncio.run(stream_debate())

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter a topic to start the debate.")