"""
Conversational Agent with Dynamic, Strict Output Control
"""
import os
from typing import List, Literal, Type
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pathlib

from .state import AgentState

# Load .env from project root
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# --- LLM Configuration ---
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.8,
    streaming=True,
    use_responses_api=True,
    use_previous_response_id=True,
    api_key=os.getenv("OPENAI_API_KEY")
    )



# --- Prompt Template for Streaming ---
STREAMING_PROMPT_STR = """
**あなたの情報:**
{persona}

**他の参加者に対するあなたの主観的な視点:**
{subjective_view}

**現在の議論トピック:**
{topic}

**議論のルール:**
1.  **役割を演じる:** あなたのペルソナと、他の参加者への視点に基づいて、一貫した意見を述べてください。また、同年代に話すように、カジュアルにタメ口で話してください。発言はより簡単に、長くならないようにしてください。また、話し言葉で話すようにしてね。あとは人間らしい会話、反応をするようにしてください。
2.  **議論を深める:** 単に同意するだけでなく、あえて異なる視点を提示したり、疑問を投げかけたり、批判的な意見を述べることを推奨します。
3.  **次の発言者を指名:** 発言の最後に、必ず次の発言者を指名してください。

**重要:** 
- あなたの発言内容のみを出力してください。JSON形式やフィールド名などは一切含めないでください。
- 参加者への質問は自然な形で含めてください。
- 次に話すべき人の名前は発言の最後で自然に指名してください（例：「田中、どう思う？」）。
- 利用可能な参加者: {agent_names_str}
"""

# --- Prompt Template for Structured Output ---
PROMPT_TEMPLATE_STR = """
**あなたの情報:**
{persona}

**他の参加者に対するあなたの主観的な視点:**
{subjective_view}

**現在の議論トピック:**
{topic}

**議論のルール:**
1.  **役割を演じる:** あなたのペルソナと、他の参加者への視点に基づいて、一貫した意見を述べてください。また、同年代に話すように、カジュアルにタメ口で話してください。発言はより簡単に、長くならないようにしてください。また、話し言葉で話すようにしてね。あとは人間らしい会話、反応をするようにしてください。
2.  **議論を深める:** 単に同意するだけでなく、あえて異なる視点を提示したり、疑問を投げかけたり、批判的な意見を述べることを推奨します。
3.  **次の発言者を指名:** 発言の最後に、必ず次の発言者を指名してください。

**指名のルール:**
- あなたの応答は、必ず指定されたツール（AgentDecision）を呼び出す形で出力してください。
- あなたの応答は、必ずJSON形式で出力してください。
- `thoughts`フィールドには、あなたの思考プロセスを記述してください。
- `response`フィールドには、あなたの発言を記述してください。
- `next_agent`フィールドには、必ず以下のリストから名前を一つ選択してください: {agent_names_str}
- 議論を終結させたい場合のみ、'Conclusion' を指名してください。
- `ready_to_conclude`フィールドで、議論が十分に深まり結論を出せる状態かどうかを判断してください。
- 上記以外の名前を生成することは固く禁じられています。
"""

# --- Conversational Agent Class ---
class ConversationalAgent:
    def __init__(self, agent_state: AgentState, topic: str, all_agent_names: List[str]):
        self.agent_state = agent_state
        self.topic = topic
        self.all_agent_names = all_agent_names

        # Dynamically create the Pydantic model with Literal types for strict validation
        ValidNextAgents = Literal[tuple(all_agent_names + ["Conclusion"])]
        
        class AgentDecision(BaseModel):
            thoughts: str = Field(description="Your internal thoughts or reasoning before making a statement.")
            response: str = Field(description="Your statement in the discussion.")
            next_agent: ValidNextAgents = Field(description="The name of the agent who should speak next.")
            ready_to_conclude: bool = Field(description="Whether you think this discussion is ready to reach a conclusion.")

        # Create the LLM chain with the dynamic model and forced tool-calling
        self.structured_llm = llm.with_structured_output(
            AgentDecision, 
            method="json_mode"
        )
        
        # Structured output chain for final decision
        structured_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        self.chain = structured_prompt | self.structured_llm
        
        # Streaming chain for natural language output
        streaming_prompt = ChatPromptTemplate.from_messages([
            ("system", STREAMING_PROMPT_STR),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        self.streaming_chain = streaming_prompt | llm

    def invoke(self) -> BaseModel:
        """Invoke the agent to get its decision."""
        agent_names_str = ", ".join(self.all_agent_names)
        return self.chain.invoke({
            "persona": self.agent_state["persona"],
            "subjective_view": self.agent_state["subjective_view"],
            "topic": self.topic,
            "agent_names_str": agent_names_str,
            "chat_history": self.agent_state["chat_history"],
        })

    async def astream_decision(self):
        """Stream the agent's natural language response, then get structured decision."""
        agent_names_str = ", ".join(self.all_agent_names)
        input_data = {
            "persona": self.agent_state["persona"],
            "subjective_view": self.agent_state["subjective_view"],
            "topic": self.topic,
            "agent_names_str": agent_names_str,
            "chat_history": self.agent_state["chat_history"],
        }
        
        # Stream the natural language response
        full_response = ""
        async for chunk in self.streaming_chain.astream(input_data):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if isinstance(content, list):
                    # Handle OpenAI Responses API format: [{'type': 'text', 'text': '...', 'index': 0}]
                    content_str = ""
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            content_str += item['text']
                        elif isinstance(item, dict) and 'type' in item and item['type'] == 'text' and 'text' in item:
                            content_str += item['text']
                        # Skip dict items that only contain metadata like {'index': 0}
                        elif isinstance(item, dict) and len(item) == 1 and 'index' in item:
                            continue
                        elif not isinstance(item, dict):
                            content_str += str(item)
                else:
                    content_str = str(content)
                    # Filter out common OpenAI Responses API artifacts
                    if content_str.strip() in ["{'index': 0}", "{'index':0}", "{\"index\":0}", "{\"index\": 0}"]:
                        content_str = ""
                
                full_response += content_str
                if content_str:  # Only yield non-empty content
                    yield {"type": "chunk", "content": content_str}
        
        # Now get structured decision using the natural language response
        try:
            # Get structured decision
            structured_decision = await self.chain.ainvoke(input_data)
            
            # Create final decision using streamed response text and structured metadata
            class StreamedDecision(BaseModel):
                thoughts: str = Field(default="Generated from streaming response")
                response: str
                next_agent: str
                ready_to_conclude: bool
            
            final_decision = StreamedDecision(
                response=full_response.strip(),
                next_agent=structured_decision.next_agent,
                ready_to_conclude=structured_decision.ready_to_conclude
            )
            yield {"type": "complete", "decision": final_decision, "full_response": full_response}
            
        except Exception as e:
            print(f"Structured decision failed: {e}, falling back to parsing")
            # Fallback: parse next agent from natural language response
            next_agent = self._parse_next_agent_from_text(full_response)
            
            class FallbackDecision(BaseModel):
                thoughts: str = Field(default="Parsed from natural language")
                response: str
                next_agent: str
                ready_to_conclude: bool = Field(default=False)
            
            decision = FallbackDecision(
                response=full_response.strip(),
                next_agent=next_agent
            )
            yield {"type": "complete", "decision": decision, "full_response": full_response}
    
    def _parse_next_agent_from_text(self, text: str) -> str:
        """Extract the next agent name from natural language text."""
        # Look for common patterns like "田中、どう思う？" or "鈴木の意見は？"
        for agent_name in self.all_agent_names:
            if agent_name in text:
                # Check if it appears near the end (likely a nomination)
                agent_pos = text.rfind(agent_name)
                if agent_pos > len(text) * 0.5:  # In the latter half of the text
                    return agent_name
        
        # If no clear nomination found, round robin
        if self.all_agent_names:
            return self.all_agent_names[0]
        return "Conclusion"
