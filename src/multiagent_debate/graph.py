"""
New LangGraph Workflow for Conversational Agents (Async Corrected)
"""
import os
import numpy as np
import pathlib
from typing import List
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from .state import ConversationState
from .agents import ConversationalAgent, llm
from pydantic import BaseModel, Field

# Load .env from project root
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize embeddings for convergence score calculation
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# --- Facilitator Decision Model ---
class FacilitatorDecision(BaseModel):
    action: str = Field(description="The action to take: 'continue', 'propose_conclusion', or 'call_vote'")
    reasoning: str = Field(description="Explanation of why this action was chosen")
    message: str = Field(description="Message to display to participants about the facilitation decision")

# --- Helper Functions for Content Extraction ---
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

# --- Graph Nodes (Async) ---

async def agent_node_streaming(state: ConversationState):
    """Executes the current speaker's turn with streaming output."""
    speaker_name = state["next_speaker"]
    agent_names = list(state["agent_states"].keys())

    if speaker_name not in agent_names and speaker_name != "Conclusion":
        print(f"[Warning] Invalid next_speaker: '{speaker_name}'. Using round-robin.")
        if state["full_transcript"]:
            last_speaker_name = state["full_transcript"][-1].split(":")[0].strip("[]").split(" ")[-1]
            try:
                last_speaker_index = agent_names.index(last_speaker_name)
                next_speaker_index = (last_speaker_index + 1) % len(agent_names)
                speaker_name = agent_names[next_speaker_index]
            except ValueError:
                speaker_name = agent_names[0] if agent_names else "Conclusion"
        else:
            speaker_name = agent_names[0] if agent_names else "Conclusion"
        state["next_speaker"] = speaker_name

    current_agent_state = state["agent_states"][speaker_name]
    agent = ConversationalAgent(current_agent_state, state["topic"], agent_names)

    # Stream the agent's response
    decision = None
    full_response_text = ""
    
    try:
        async for event in agent.astream_decision():
            if event["type"] == "chunk":
                content = event["content"]
                full_response_text += content
                yield {"type": "agent_message_chunk", "agent_name": speaker_name, "chunk": content}
            elif event["type"] == "complete":
                decision = event["decision"]
                break
    except Exception as e:
        print(f"Streaming failed for {speaker_name}, falling back to non-streaming: {e}")
        # Fallback to non-streaming
        try:
            decision = await agent.chain.ainvoke({
                "persona": agent.agent_state["persona"],
                "subjective_view": agent.agent_state["subjective_view"],
                "topic": agent.topic,
                "agent_names_str": ", ".join(agent.all_agent_names),
                "chat_history": agent.agent_state["chat_history"],
            })
        except Exception as fallback_error:
            print(f"Non-streaming fallback also failed for {speaker_name}: {fallback_error}")
            decision = agent._create_emergency_response(str(fallback_error))
    
    if decision is None:
        # Emergency fallback
        print(f"No decision received from {speaker_name}, creating emergency response")
        from pydantic import BaseModel
        class EmergencyDecision(BaseModel):
            thoughts: str = "Unable to generate proper response"
            response: str = full_response_text if full_response_text.strip() else "議論を続けましょう。"
            next_agent: str = agent_names[0] if agent_names else "Conclusion"
            ready_to_conclude: bool = False
        decision = EmergencyDecision()

    # Process the decision
    ai_message = AIMessage(content=decision.response, name=speaker_name)
    human_message = HumanMessage(content=f"（{speaker_name}の発言）: {decision.response}", name="human")

    for name, agent_state in state["agent_states"].items():
        if name == speaker_name:
            agent_state["chat_history"].append(ai_message)
        else:
            agent_state["chat_history"].append(human_message)

    state["next_speaker"] = decision.next_agent
    state["current_turn"] += 1
    state["ready_flags"].append(decision.ready_to_conclude)
    
    if any(marker in decision.response for marker in ["どう思う？", "どう考える？", "意見を聞かせて", "君はどう", "あなたはどう"]):
        question = f"{speaker_name}: {decision.response.split('？')[0]}？"
        if question not in state["pending_questions"]:
            state["pending_questions"].append(question)
        
        # Prevent self-nomination if a question was just asked
        if decision.next_agent == speaker_name:
            other_agents = [name for name in agent_names if name != speaker_name]
            if other_agents:
                import random
                decision.next_agent = random.choice(other_agents)
                print(f" -> [DEBUG] {speaker_name} tried to self-nominate after asking a question. Redirecting to {decision.next_agent}")
    
    turn_log = f"[Turn {state['current_turn']}] {speaker_name}: {decision.response}"
    state["full_transcript"].append(turn_log)
    state["logger"].info(turn_log)
    print(turn_log)
    print(f" -> Next Speaker: {decision.next_agent}")

    # Signal completion of agent's turn
    yield {"type": "agent_message_complete", "agent_name": speaker_name, "message": decision.response}

async def agent_node(state: ConversationState) -> ConversationState:
    """Executes the current speaker's turn asynchronously with streaming."""
    speaker_name = state["next_speaker"]
    agent_names = list(state["agent_states"].keys())

    if speaker_name not in agent_names and speaker_name != "Conclusion":
        print(f"[Warning] Invalid next_speaker: '{speaker_name}'. Using round-robin.")
        if state["full_transcript"]:
            last_speaker_name = state["full_transcript"][-1].split(":")[0].strip("[]").split(" ")[-1]
            try:
                last_speaker_index = agent_names.index(last_speaker_name)
                next_speaker_index = (last_speaker_index + 1) % len(agent_names)
                speaker_name = agent_names[next_speaker_index]
            except ValueError:
                speaker_name = agent_names[0] if agent_names else "Conclusion"
        else:
            speaker_name = agent_names[0] if agent_names else "Conclusion"
        state["next_speaker"] = speaker_name

    current_agent_state = state["agent_states"][speaker_name]
    agent = ConversationalAgent(current_agent_state, state["topic"], agent_names)

    # Use streaming decision method
    decision = None
    try:
        async for event in agent.astream_decision():
            if event["type"] == "chunk":
                # This will be caught by orchestrator streaming events
                pass
            elif event["type"] == "complete":
                decision = event["decision"]
                break
    except Exception as e:
        print(f"Agent streaming failed: {e}")
    
    if decision is None:
        # Fallback to non-streaming if streaming fails
        try:
            decision = await agent.chain.ainvoke({
                "persona": agent.agent_state["persona"],
                "subjective_view": agent.agent_state["subjective_view"],
                "topic": agent.topic,
                "agent_names_str": ", ".join(agent.all_agent_names),
                "chat_history": agent.agent_state["chat_history"],
            })
        except Exception as fallback_error:
            print(f"Agent fallback failed: {fallback_error}")
            decision = agent._create_emergency_response(str(fallback_error))

    ai_message = AIMessage(content=decision.response, name=speaker_name)
    human_message = HumanMessage(content=f"（{speaker_name}の発言）: {decision.response}", name="human")

    for name, agent_state in state["agent_states"].items():
        if name == speaker_name:
            agent_state["chat_history"].append(ai_message)
        else:
            agent_state["chat_history"].append(human_message)

    state["next_speaker"] = decision.next_agent
    state["current_turn"] += 1
    state["ready_flags"].append(decision.ready_to_conclude)
    
    if any(marker in decision.response for marker in ["どう思う？", "どう考える？", "意見を聞かせて", "君はどう", "あなたはどう"]):
        question = f"{speaker_name}: {decision.response.split('？')[0]}？"
        if question not in state["pending_questions"]:
            state["pending_questions"].append(question)
        
        # Prevent self-nomination if a question was just asked
        if decision.next_agent == speaker_name:
            other_agents = [name for name in agent_names if name != speaker_name]
            if other_agents:
                import random
                decision.next_agent = random.choice(other_agents)
                print(f" -> [DEBUG] {speaker_name} tried to self-nominate after asking a question. Redirecting to {decision.next_agent}")
    
    turn_log = f"[Turn {state['current_turn']}] {speaker_name}: {decision.response}"
    state["full_transcript"].append(turn_log)
    state["logger"].info(turn_log)
    print(turn_log)
    print(f" -> Next Speaker: {decision.next_agent}")

    return state

async def update_metrics_node(state: ConversationState) -> ConversationState:
    """Updates monitoring metrics asynchronously."""
    current_turn = state["current_turn"]
    if current_turn > 0:
        latest_statement = state["full_transcript"][-1]
        spoken_content = latest_statement.split(": ", 1)[1] if ": " in latest_statement else latest_statement
        try:
            latest_embedding = await embeddings.aembed_query(spoken_content)
            state["statement_embeddings"].append(latest_embedding)
            
            if len(state["statement_embeddings"]) >= 2:
                last_embedding = state["statement_embeddings"][-1]
                prev_embedding = state["statement_embeddings"][-2]
                dot_product = np.dot(last_embedding, prev_embedding)
                magnitude_1 = np.linalg.norm(last_embedding)
                magnitude_2 = np.linalg.norm(prev_embedding)
                state["convergence_score"] = max(0.0, dot_product / (magnitude_1 * magnitude_2)) if magnitude_1 > 0 and magnitude_2 > 0 else 0.0
            else:
                state["convergence_score"] = 0.0
        except Exception as e:
            print(f"[Warning] Failed to calculate embedding: {e}")
            state["convergence_score"] = 0.0
    
    ready_count = sum(state["ready_flags"])
    total_flags = len(state["ready_flags"])
    readiness_ratio = ready_count / total_flags if total_flags > 0 else 0.0
    
    current_speaker = state["next_speaker"]
    latest_statement = state["full_transcript"][-1] if state["full_transcript"] else ""
    answered_questions = [q for q in state["pending_questions"] if q.split(":")[0] != current_speaker and current_speaker in latest_statement]
    for answered in answered_questions:
        state["pending_questions"].remove(answered)
    
    total_questions_asked = state["current_turn"] - len(state["pending_questions"])
    state["discussion_depth"] = total_questions_asked / max(1, state["current_turn"])
    
    pending_count = len(state["pending_questions"])
    print(f" -> Metrics: Convergence: {state['convergence_score']:.3f}, Readiness: {ready_count}/{total_flags} ({readiness_ratio:.3f}), Depth: {state['discussion_depth']:.3f}, Pending Q: {pending_count}")
    
    return state

async def facilitator_node(state: ConversationState) -> ConversationState:
    """Facilitator evaluates the debate asynchronously."""
    print("\n--- Facilitator Evaluation ---")
    ready_count = sum(state["ready_flags"])
    total_flags = len(state["ready_flags"])
    readiness_ratio = ready_count / total_flags if total_flags > 0 else 0.0
    recent_turns = state["full_transcript"][-3:] if len(state["full_transcript"]) >= 3 else state["full_transcript"]
    
    facilitator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""あなたは議論のファシリテータです。議論全体を俯瞰し、以下の情報を元に最適な次のアクションを決定してください。

**利用可能なアクション:**
- "continue": 議論を継続させる（まだ議論が浅い、または新しい視点が期待できる場合）
- "propose_conclusion": 議論をまとめるフェーズへの移行を提案する（十分に議論が深まった場合）
- "call_vote": 意見が明確に分かれている場合に多数決を促す

**判断基準（人間らしい会話を最優先）:**
1. **NEVER interrupt if there are pending questions** - 質問が残っている場合は絶対に継続
2. **Continue if discussion_depth < 0.7** - 議論の深さが不十分な場合は継続
3. **Continue if current_turn < max_turns * 0.6** - 最大ターンの60%未満は基本的に継続
4. 議論が真に停滞している場合（同じ論点の3回以上の反復）のみ "propose_conclusion"
5. 準備完了率 > 80% かつ 残りターン < 3 の場合のみ "propose_conclusion"

議論のテーマ: {topic}"""),
        HumanMessage(content=f"""
**現在の議論状況:**
- 現在のターン: {state['current_turn']} / {state['max_turns']}
- 議論の進行率: {(state['current_turn'] / state['max_turns'] * 100):.1f}%
- 収束スコア: {state['convergence_score']:.3f}
- 準備完了率: {readiness_ratio:.3f} ({ready_count}/{total_flags})
- 議論の深さ: {state['discussion_depth']:.3f}
- 未回答の質問数: {len(state['pending_questions'])}

**未回答の質問:**
{chr(10).join(state['pending_questions']) if state['pending_questions'] else "なし"}

**直近の議論内容:**
{chr(10).join(recent_turns)}

**重要:** 未回答の質問がある場合は必ず "continue" を選択してください。議論の自然な流れを最優先してください。
""")
    ])
    
    facilitator_llm = llm.with_structured_output(FacilitatorDecision, strict=True)
    facilitator_chain = facilitator_prompt | facilitator_llm
    
    try:
        decision = await facilitator_chain.ainvoke({"topic": state["topic"]})
    except Exception as e:
        print(f"Facilitator decision failed: {e}, using default continue action")
        decision = FacilitatorDecision(
            action="continue",
            reasoning="Technical error occurred, continuing discussion",
            message="議論を継続します。"
        )
    
    state["facilitator_action"] = decision.action
    state["facilitator_message"] = decision.message
    print(f"Facilitator Decision: {decision.action}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Message: {decision.message}")
    state["logger"].info(f"Facilitator Decision: {decision.action} - {decision.reasoning}")
    return state

async def pre_conclusion_node(state: ConversationState) -> ConversationState:
    """Generates a preliminary conclusion asynchronously."""
    print("\n--- Pre-Conclusion: Preparing Draft Summary ---")
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
    
    try:
        result = await chain.ainvoke({"topic": state["topic"]})
        full_response = _extract_text_from_response(result)
    except Exception as e:
        print(f"Pre-conclusion generation failed: {e}")
        full_response = "暫定的な結論の生成中にエラーが発生しました。これまでの議論の主要なポイントを手動でまとめることをお勧めします。"
    
    state["preliminary_conclusion"] = full_response
    print(f"Preliminary Conclusion: {state['preliminary_conclusion']}")
    state["logger"].info(f"Preliminary conclusion generated: {state['preliminary_conclusion']}")
    return state

async def final_comment_node(state: ConversationState) -> ConversationState:
    """Allows agents to provide final comments asynchronously."""
    print("\n--- Final Comments: Last Chance for Input ---")
    agent_names = list(state["agent_states"].keys())
    
    async def get_comment(agent_name):
        print(f"\n--- {agent_name}の最終意見 ---")
        agent_state = state["agent_states"][agent_name]
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""あなたは{agent_name}です。ファシリテータが作成した暫定的な結論案を確認し、最終的な意見を述べてください。

**あなたの役割:** {agent_state['persona']}

**指示:**
1. 暫定的な結論案を読んで、内容が議論を適切に反映しているか確認してください
2. 重要な論点の見落としや誤解がないかチェックしてください  
3. 必要であれば補足や修正を提案してください
4. 簡潔に（2-3文程度で）最終意見を述べてください
5. もし結論案に満足している場合は、その旨を述べてください"""),
            HumanMessage(content=f"""
**暫定的な結論案:**
{state['preliminary_conclusion']}

上記の結論案について、あなたの最終的な意見を述べてください。重要な見落としや修正が必要な点があれば指摘し、なければ結論案への賛同を表明してください。
""")
        ])
        chain = prompt | llm
        
        try:
            result = await chain.ainvoke({})
            comment_text = _extract_text_from_response(result)
        except Exception as e:
            print(f"Final comment generation failed for {agent_name}: {e}")
            comment_text = f"技術的な問題により、{agent_name}の最終コメントを生成できませんでした。"
        
        print(f"{agent_name}: {comment_text}")
        state["logger"].info(f"Final comment from {agent_name}: {comment_text}")
        return f"[{agent_name}] {comment_text}"

    # Concurrently get final comments from all agents
    try:
        final_comments = await asyncio.gather(*[get_comment(name) for name in agent_names])
        state["final_comments"] = final_comments
    except Exception as e:
        print(f"Final comments generation failed: {e}")
        state["final_comments"] = [f"[Error] 最終コメントの生成中にエラーが発生しました: {str(e)}"]
    
    return state

async def conclusion_node(state: ConversationState) -> ConversationState:
    """Generates the final conclusion asynchronously."""
    print("\n--- Generating Final Conclusion ---")
    
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
    
    try:
        result = await chain.ainvoke({})
        full_conclusion = _extract_text_from_response(result)
    except Exception as e:
        print(f"Final conclusion generation failed: {e}")
        full_conclusion = f"""最終結論の生成中にエラーが発生しました。

**議論のテーマ:** {state['topic']}

**議論の記録:**
{chr(10).join(state['full_transcript'][-5:]) if state['full_transcript'] else '記録がありません'}

技術的な問題により、完全な結論を生成できませんでしたが、上記の議論記録を参考にしてください。"""
    
    state["conclusion"] = full_conclusion
    print(f"Final Conclusion: {state['conclusion']}")
    return state

# --- Conditional Routing (remains synchronous) ---
def route_after_metrics(state: ConversationState) -> str:
    if state["next_speaker"] == "Conclusion": return "conclusion_node"
    if state["current_turn"] >= state["max_turns"]: return "conclusion_node"
    if state["pending_questions"]:
        print(f" -> Continuing: {len(state['pending_questions'])} pending questions must be answered")
        return "agent_node"
    if state["current_turn"] > 0 and state["current_turn"] % state["facilitator_check_interval"] == 0:
        return "facilitator_node"
    ready_count = sum(state["ready_flags"])
    total_flags = len(state["ready_flags"])
    readiness_ratio = ready_count / total_flags if total_flags > 0 else 0.0
    remaining_turns = state["max_turns"] - state["current_turn"]
    if (state["convergence_score"] > 0.98 and readiness_ratio > 0.8 and remaining_turns <= 2 and state["discussion_depth"] > 0.7):
        print(f" -> Auto-termination: Very high confidence (Conv: {state['convergence_score']:.3f}, Ready: {readiness_ratio:.3f}, Depth: {state['discussion_depth']:.3f})")
        return "conclusion_node"
    return "agent_node"

def route_after_facilitator(state: ConversationState) -> str:
    if state["facilitator_action"] == "propose_conclusion":
        return "pre_conclusion_node"
    elif state["facilitator_action"] == "call_vote":
        print(" -> Facilitator called for vote, but continuing discussion for now")
        return "agent_node"
    else:
        return "agent_node"

# --- Graph Definition ---
def create_debate_graph() -> StateGraph:
    workflow = StateGraph(ConversationState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("update_metrics_node", update_metrics_node)
    workflow.add_node("facilitator_node", facilitator_node)
    workflow.add_node("pre_conclusion_node", pre_conclusion_node)
    workflow.add_node("final_comment_node", final_comment_node)
    workflow.add_node("conclusion_node", conclusion_node)
    workflow.set_entry_point("agent_node")
    workflow.add_edge("agent_node", "update_metrics_node")
    workflow.add_conditional_edges("update_metrics_node", route_after_metrics, {
        "agent_node": "agent_node",
        "facilitator_node": "facilitator_node",
        "conclusion_node": "conclusion_node",
    })
    workflow.add_conditional_edges("facilitator_node", route_after_facilitator, {
        "agent_node": "agent_node",
        "pre_conclusion_node": "pre_conclusion_node",
    })
    workflow.add_edge("pre_conclusion_node", "final_comment_node")
    workflow.add_edge("final_comment_node", "conclusion_node")
    workflow.add_edge("conclusion_node", END)
    return workflow.compile()
