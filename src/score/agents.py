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



# --- Prompt Template ---
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
        
        # Unified prompt for both streaming and structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        self.chain = prompt | self.structured_llm

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
        """Get structured response using only JSON output (single API call)."""
        agent_names_str = ", ".join(self.all_agent_names)
        input_data = {
            "persona": self.agent_state["persona"],
            "subjective_view": self.agent_state["subjective_view"],
            "topic": self.topic,
            "agent_names_str": agent_names_str,
            "chat_history": self.agent_state["chat_history"],
        }
        
        # Use only structured output to avoid double API calls
        try:
            final_decision = await self.chain.ainvoke(input_data)
            
            # Simulate streaming by yielding the response field as a single chunk
            if hasattr(final_decision, 'response'):
                response_text = final_decision.response
                yield {"type": "chunk", "content": response_text}
            
            yield {"type": "complete", "decision": final_decision, "full_response": final_decision.response if hasattr(final_decision, 'response') else ""}
            
        except Exception as e:
            print(f"Structured decision failed: {e}")
            
            # Simple fallback without additional API calls
            class EmergencyDecision:
                def __init__(self):
                    self.thoughts = "Structured output failed"
                    self.response = "I need more time to analyze this topic."
                    self.next_agent = self.all_agent_names[0] if self.all_agent_names else "Conclusion"
                    self.ready_to_conclude = False
            
            emergency_decision = EmergencyDecision()
            yield {"type": "complete", "decision": emergency_decision, "full_response": emergency_decision.response}
