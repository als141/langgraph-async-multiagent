#!/usr/bin/env python3
"""
NiceGUI implementation for Multi-Agent Debate System
with real-time streaming support
"""

import asyncio
import sys
import os
from nicegui import ui, app, run
from typing import Dict, List

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from multiagent_debate.orchestrator import run_graph
from multiagent_debate.config import AGENTS_CONFIG

# Global state
debate_state = {
    'is_running': False,
    'messages': [],
    'conclusion_data': {},
    'current_streaming_message': {},
    'status_message': '',
    'debate_task': None
}

# Agent avatars from config
AGENT_AVATARS = {agent["name"]: agent["avatar"] for agent in AGENTS_CONFIG}
AGENT_AVATARS["Facilitator"] = "🤖"
AGENT_AVATARS["Status"] = "⚙️"

class DebateUI:
    def __init__(self):
        self.chat_container = None
        self.conclusion_container = None
        self.topic_input = None
        self.max_turns_input = None
        self.start_button = None
        self.status_label = None
        
    def create_ui(self):
        """Create the main UI layout"""
        ui.page_title('🧠 Multi-Agent Debate System')
        
        with ui.header():
            ui.label('🧠 Multi-Agent Debate System').classes('text-h4')
        
        # Main container
        with ui.splitter(value=70).classes('w-full h-screen') as splitter:
            with splitter.before:
                # Chat area
                with ui.scroll_area().classes('w-full h-full p-4') as scroll:
                    self.chat_container = ui.column().classes('w-full')
                    
            with splitter.after:
                # Controls and conclusion
                with ui.column().classes('w-full p-4 gap-4'):
                    # Status
                    self.status_label = ui.label('Ready to start debate').classes('text-subtitle2 text-blue')
                    
                    ui.separator()
                    
                    # Topic input
                    ui.label('Debate Topic:').classes('text-subtitle1 font-bold')
                    self.topic_input = ui.input(
                        placeholder='例：少子高齢化の解決策は何か？',
                        value='少子高齢化の解決策は何か？'
                    ).classes('w-full')
                    
                    # Max turns input
                    ui.label('Max Turns:').classes('text-subtitle1 font-bold')
                    self.max_turns_input = ui.number(
                        value=10, 
                        min=2, 
                        max=50,
                        step=1
                    ).classes('w-full')
                    
                    # Start button
                    self.start_button = ui.button(
                        'Start Debate', 
                        on_click=self.start_debate
                    ).classes('w-full').props('color=primary')
                    
                    ui.separator()
                    
                    # Conclusion area
                    ui.label('🏆 Debate Conclusion').classes('text-h6 font-bold')
                    with ui.scroll_area().classes('w-full h-64 border rounded p-2'):
                        self.conclusion_container = ui.column().classes('w-full')
        
        # Update UI with existing messages
        self.update_chat_display()
        self.update_conclusion_display()
        
    def add_message(self, role: str, content: str, is_streaming: bool = False):
        """Add a message to the chat"""
        avatar = AGENT_AVATARS.get(role, "👤")
        
        # Find existing streaming message or create new one
        existing_msg = None
        for msg in debate_state['messages']:
            if msg.get('role') == role and msg.get('is_streaming', False):
                existing_msg = msg
                break
        
        if existing_msg and is_streaming:
            # Update existing streaming message
            existing_msg['content'] = content
        else:
            # Create new message
            msg = {
                'role': role,
                'content': content,
                'avatar': avatar,
                'is_streaming': is_streaming
            }
            if existing_msg:
                # Replace streaming message with final
                idx = debate_state['messages'].index(existing_msg)
                debate_state['messages'][idx] = msg
            else:
                debate_state['messages'].append(msg)
        
        self.update_chat_display()
    
    def update_chat_display(self):
        """Update the chat display"""
        if not self.chat_container:
            return
            
        self.chat_container.clear()
        
        for msg in debate_state['messages']:
            avatar = msg.get('avatar', '👤')
            role = msg.get('role', 'Unknown')
            content = msg.get('content', '')
            is_streaming = msg.get('is_streaming', False)
            
            with self.chat_container:
                with ui.row().classes('w-full gap-2 items-start'):
                    ui.label(avatar).classes('text-2xl')
                    with ui.column().classes('flex-1'):
                        ui.label(f'**{role}**').classes('font-bold')
                        if is_streaming:
                            ui.label(content + ' ▊').classes('whitespace-pre-wrap')  # Cursor indicator
                        else:
                            ui.label(content).classes('whitespace-pre-wrap')
                ui.separator()
    
    def update_conclusion_display(self):
        """Update the conclusion display"""
        if not self.conclusion_container:
            return
            
        self.conclusion_container.clear()
        
        conclusion_data = debate_state['conclusion_data']
        
        with self.conclusion_container:
            if debate_state['status_message']:
                ui.label(debate_state['status_message']).classes('text-blue font-bold')
                ui.separator()
            
            if 'pre_conclusion' in conclusion_data:
                ui.label('**1. Preliminary Conclusion**').classes('font-bold text-green')
                ui.label(conclusion_data['pre_conclusion']).classes('whitespace-pre-wrap text-sm')
                ui.separator()
            
            if 'final_comments' in conclusion_data:
                ui.label('**2. Final Comments**').classes('font-bold text-orange')
                for comment in conclusion_data['final_comments']:
                    ui.label(f'• {comment}').classes('whitespace-pre-wrap text-sm')
                ui.separator()
            
            if 'conclusion' in conclusion_data:
                ui.label('**3. Final Conclusion**').classes('font-bold text-purple')
                ui.label(conclusion_data['conclusion']).classes('whitespace-pre-wrap text-sm bg-purple-50 p-2 rounded')
    
    def update_status(self, message: str):
        """Update status message"""
        debate_state['status_message'] = message
        if self.status_label:
            self.status_label.text = message
        self.update_conclusion_display()
    
    async def start_debate(self):
        """Start the debate"""
        if debate_state['is_running']:
            return
        
        topic = self.topic_input.value
        max_turns = int(self.max_turns_input.value)
        
        if not topic.strip():
            ui.notify('Please enter a topic', type='negative')
            return
        
        # Reset state
        debate_state['is_running'] = True
        debate_state['messages'] = []
        debate_state['conclusion_data'] = {}
        debate_state['current_streaming_message'] = {}
        
        # Update UI
        self.start_button.text = 'Debate Running...'
        self.start_button.props('disabled')
        self.topic_input.props('disabled')
        self.max_turns_input.props('disabled')
        
        # Add initial message
        agent_names = [agent["name"] for agent in AGENTS_CONFIG]
        self.add_message(
            'Status', 
            f'**Starting Debate**\n\n**Topic:** {topic}\n**Participants:** {", ".join(agent_names)}\n**Max Turns:** {max_turns}'
        )
        
        # Start debate in background
        debate_state['debate_task'] = asyncio.create_task(self.run_debate(topic, max_turns))
    
    async def run_debate(self, topic: str, max_turns: int):
        """Run the debate and handle streaming events"""
        try:
            async for event in run_graph(topic, max_turns=max_turns):
                await self.handle_debate_event(event)
        except Exception as e:
            ui.notify(f'Debate error: {e}', type='negative')
            print(f"Debate error: {e}")
        finally:
            await self.end_debate()
    
    async def handle_debate_event(self, event: dict):
        """Handle individual debate events"""
        event_type = event.get('type')
        
        if event_type == 'agent_message_chunk':
            agent_name = event['agent_name']
            chunk = event['chunk']
            
            # Update or create streaming message
            if agent_name not in debate_state['current_streaming_message']:
                debate_state['current_streaming_message'][agent_name] = ''
            
            debate_state['current_streaming_message'][agent_name] += chunk
            self.add_message(
                agent_name, 
                debate_state['current_streaming_message'][agent_name], 
                is_streaming=True
            )
            
        elif event_type == 'agent_message_complete':
            agent_name = event['agent_name']
            message = event['message']
            
            # Finalize the message
            self.add_message(agent_name, message, is_streaming=False)
            if agent_name in debate_state['current_streaming_message']:
                del debate_state['current_streaming_message'][agent_name]
        
        elif event_type == 'facilitator_message':
            self.add_message('Facilitator', event['message'])
        
        elif event_type == 'status_update':
            self.update_status(event['message'])
        
        elif event_type == 'pre_conclusion_chunk':
            if 'pre_conclusion' not in debate_state['conclusion_data']:
                debate_state['conclusion_data']['pre_conclusion'] = ''
            debate_state['conclusion_data']['pre_conclusion'] += event['chunk']
            self.update_conclusion_display()
        
        elif event_type == 'pre_conclusion_complete':
            debate_state['conclusion_data']['pre_conclusion'] = event['content']
            self.update_conclusion_display()
        
        elif event_type == 'final_comments_complete':
            debate_state['conclusion_data']['final_comments'] = event['content']
            self.update_conclusion_display()
        
        elif event_type == 'conclusion_chunk':
            if 'conclusion' not in debate_state['conclusion_data']:
                debate_state['conclusion_data']['conclusion'] = ''
            debate_state['conclusion_data']['conclusion'] += event['chunk']
            self.update_conclusion_display()
        
        elif event_type == 'conclusion_complete':
            debate_state['conclusion_data']['conclusion'] = event['conclusion']
            self.update_conclusion_display()
        
        elif event_type == 'end_of_debate':
            self.add_message('Status', '🎉 **Debate Completed!** 🎉')
    
    async def end_debate(self):
        """Clean up after debate ends"""
        debate_state['is_running'] = False
        
        # Re-enable controls
        self.start_button.text = 'Start Debate'
        self.start_button.props(remove='disabled')
        self.topic_input.props(remove='disabled')
        self.max_turns_input.props(remove='disabled')
        
        self.update_status('Debate completed. Ready for next debate.')

# Create the UI instance
debate_ui = DebateUI()

@ui.page('/')
async def main_page():
    """Main page route"""
    debate_ui.create_ui()

if __name__ in {"__main__", "__mp_main__"}:
    # Configure NiceGUI
    ui.run(
        title='Multi-Agent Debate System',
        host='localhost',
        port=8080,
        reload=False,
        show=True,
        dark=False
    )