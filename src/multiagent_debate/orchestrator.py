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
    """Direct streaming wrapper for running the debate."""
    
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

    state = ConversationState(
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

    # Import here to avoid circular imports
    from .graph import agent_node_streaming, update_metrics_node, facilitator_node, pre_conclusion_node, final_comment_node, conclusion_node
    from .graph import route_after_metrics, route_after_facilitator

    try:
        while state["current_turn"] < state["max_turns"] and state["next_speaker"] != "Conclusion":
            # Agent speaking turn with streaming
            async for event in agent_node_streaming(state):
                yield event
                
            # Update metrics
            state = await update_metrics_node(state)
            
            # Route decision
            next_step = route_after_metrics(state)
            
            if next_step == "facilitator_node":
                state = await facilitator_node(state)
                yield {"type": "facilitator_message", "message": state.get("facilitator_message", "")}
                
                facilitator_route = route_after_facilitator(state)
                if facilitator_route == "pre_conclusion_node":
                    break
                    
            elif next_step == "conclusion_node":
                break
        
        # Final conclusion sequence
        if state["current_turn"] >= state["max_turns"] or state["next_speaker"] == "Conclusion":
            yield {"type": "status_update", "message": "--- Pre-Conclusion: Preparing Draft Summary ---"}
            async for event in pre_conclusion_node_streaming(state):
                yield event
            
            yield {"type": "status_update", "message": "--- Final Comments: Last Chance for Input ---"}
            state = await final_comment_node(state)
            if state.get("final_comments"):
                yield {"type": "final_comments_complete", "content": state["final_comments"]}
            
            yield {"type": "status_update", "message": "--- Generating Final Conclusion ---"}
            async for event in conclusion_node_streaming(state):
                yield event
    
    except Exception as e:
        print(f"Error in run_graph: {e}")
        yield {"type": "error", "message": str(e)}
    
    yield {"type": "end_of_debate"}

# Helper streaming functions
async def pre_conclusion_node_streaming(state):
    """Streaming version of pre_conclusion_node with improved OpenAI Responses API handling."""
    from .graph import llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""あなたは議論のファシリテータです。これまでの議論をまとめて、暫定的な結論案を作成してください。

**重要な指示:**
1. 議論全体を客観的に要約してください
2. 主要な論点とそれぞれの立場を明確に記述してください
3. 合意に至った点と意見が分かれた点を区別してください
4. 「これはまだ暫定的なまとめです」ということを明記してください
5. 参加者に補足や修正意見を求めてください

議論のテーマ: {topic}"""),
        HumanMessage(content=f"""
以下の議論の記録を基に、暫定的な結論案を作成してください：

{chr(10).join(state['full_transcript'])}

上記の議論を踏まえ、暫定的なまとめを作成し、参加者に最終確認を求めてください。
""")
    ])
    
    chain = prompt | llm
    full_response = ""
    
    try:
        async for chunk in chain.astream({"topic": state["topic"]}):
            if hasattr(chunk, 'content'):
                content_text = _extract_clean_content_from_chunk(chunk.content)
                if content_text:
                    full_response += content_text
                    yield {"type": "pre_conclusion_chunk", "chunk": content_text}
        
        state["preliminary_conclusion"] = full_response
        yield {"type": "pre_conclusion_complete", "content": full_response}
        
    except Exception as e:
        print(f"Pre-conclusion streaming failed: {e}")
        # Fallback to non-streaming
        try:
            result = await chain.ainvoke({"topic": state["topic"]})
            content = _extract_text_from_response(result)
            state["preliminary_conclusion"] = content
            yield {"type": "pre_conclusion_complete", "content": content}
        except Exception as fallback_error:
            print(f"Pre-conclusion fallback failed: {fallback_error}")
            error_content = "暫定的な結論の生成中にエラーが発生しました。議論を続行します。"
            state["preliminary_conclusion"] = error_content
            yield {"type": "pre_conclusion_complete", "content": error_content}

async def conclusion_node_streaming(state):
    """Streaming version of conclusion_node with improved error handling."""
    from .graph import llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
    
    if state["preliminary_conclusion"] and state["final_comments"]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""あなたは議論の結論をまとめる専門家です。以下の情報を統合して、最終的な結論を作成してください。

**重要な指示:**
1. 暫定的な結論案を基礎として使用してください
2. 参加者の最終意見を十分に考慮して、必要な修正や補足を行ってください
3. 最終的な結論は包括的で、全ての重要な論点を含むようにしてください
4. 意見が分かれた場合は、それも明確に記述してください

議論のテーマ: {state["topic"]}"""),
            HumanMessage(content=f"""
**暫定的な結論案:**
{state['preliminary_conclusion']}

**参加者の最終意見:**
{chr(10).join(state['final_comments'])}

**完全な議論記録:**
{chr(10).join(state['full_transcript'])}

上記の情報を統合して、最終的な結論を作成してください。
""")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="あなたは議論の結論をまとめる専門家です。以下の議論の完全な記録を読み、最終的な結論を客観的に要約してください。意見が分かれた場合は、それも明確に記述してください。議論のテーマ:" + state["topic"]),
            HumanMessage(content="\n".join(state["full_transcript"])) 
        ])
    
    chain = prompt | llm
    full_conclusion = ""
    
    try:
        async for chunk in chain.astream({}):
            if hasattr(chunk, 'content'):
                content_text = _extract_clean_content_from_chunk(chunk.content)
                if content_text:
                    full_conclusion += content_text
                    yield {"type": "conclusion_chunk", "chunk": content_text}
        
        state["conclusion"] = full_conclusion
        yield {"type": "conclusion_complete", "conclusion": full_conclusion}
        
    except Exception as e:
        print(f"Conclusion streaming failed: {e}")
        # Fallback to non-streaming
        try:
            result = await chain.ainvoke({})
            content = _extract_text_from_response(result)
            state["conclusion"] = content
            yield {"type": "conclusion_complete", "conclusion": content}
        except Exception as fallback_error:
            print(f"Conclusion fallback failed: {fallback_error}")
            error_content = "最終結論の生成中にエラーが発生しました。議論の記録は保存されています。"
            state["conclusion"] = error_content
            yield {"type": "conclusion_complete", "conclusion": error_content}

def _extract_clean_content_from_chunk(content):
    """Extract clean content from OpenAI Responses API chunk format"""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                elif 'text' in item and 'index' not in item:
                    text_parts.append(item['text'])
                # Skip metadata-only items
                elif len(item) == 1 and 'index' in item:
                    continue
        return ''.join(text_parts)
    elif isinstance(content, str):
        # Filter out common OpenAI Responses API artifacts
        artifacts = [
            "{'index': 0}", "{'index':0}", 
            '{"index":0}', '{"index": 0}',
            '[{"index": 0}]'
        ]
        if content.strip() not in artifacts:
            return content
    return ""

def _extract_text_from_response(response):
    """Extract text content from various response formats"""
    if hasattr(response, 'content'):
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            return _extract_clean_content_from_chunk(response.content)
    elif isinstance(response, str):
        return response
    elif isinstance(response, dict) and 'content' in response:
        return response['content']
    return str(response)
