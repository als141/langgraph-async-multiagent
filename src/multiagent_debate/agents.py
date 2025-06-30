"""
Conversational Agent with Dynamic, Strict Output Control
"""
import os
import asyncio
import json
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

# --- LLM Configuration with Latest OpenAI Responses API ---
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Latest recommended model for cost-performance
    temperature=0.8,
    streaming=True,
    output_version="responses/v1",  # Use new output format
    use_responses_api=True,
    use_previous_response_id=True,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,
    timeout=30
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

        # Create the LLM chain with structured output and strict mode
        self.structured_llm = llm.with_structured_output(
            AgentDecision, 
            method="json_schema",
            strict=True
        )
        
        # Unified prompt for both streaming and structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE_STR),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        self.chain = prompt | self.structured_llm
        self.streaming_chain = prompt | llm

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

    def _extract_clean_content(self, content):
        """Extract clean content from OpenAI Responses API format"""
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

    async def _safe_llm_call(self, input_data, max_retries=3):
        """Robust LLM call with error handling"""
        for attempt in range(max_retries):
            try:
                result = await self.chain.ainvoke(input_data)
                return result
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Final fallback
                    return self._create_emergency_response(str(e))
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _create_emergency_response(self, error_msg: str):
        """Create emergency response when all LLM calls fail"""
        class EmergencyDecision(BaseModel):
            thoughts: str = Field(default=f"Error occurred: {error_msg}")
            response: str = Field(default="申し訳ありません、技術的な問題で適切な回答ができません。")
            next_agent: str = Field(default=self.all_agent_names[0] if self.all_agent_names else "Conclusion")
            ready_to_conclude: bool = Field(default=False)
        
        return EmergencyDecision()

    async def astream_decision(self):
        """Stream response with improved OpenAI Responses API handling"""
        agent_names_str = ", ".join(self.all_agent_names)
        input_data = {
            "persona": self.agent_state["persona"],
            "subjective_view": self.agent_state["subjective_view"],
            "topic": self.topic,
            "agent_names_str": agent_names_str,
            "chat_history": self.agent_state["chat_history"],
        }
        
        full_response = ""
        
        try:
            # Stream the raw response with improved content extraction
            async for chunk in self.streaming_chain.astream(input_data):
                if hasattr(chunk, 'content'):
                    content_text = self._extract_clean_content(chunk.content)
                    if content_text:
                        full_response += content_text
                        yield {"type": "chunk", "content": content_text}
            
            # Get structured decision with retry logic
            final_decision = await self._safe_llm_call(input_data)
            yield {"type": "complete", "decision": final_decision, "full_response": full_response}
            
        except Exception as e:
            print(f"Streaming failed: {e}, attempting JSON parsing fallback")
            
            # Advanced JSON parsing fallback
            try:
                parsed_decision = self._parse_json_response(full_response)
                yield {"type": "complete", "decision": parsed_decision, "full_response": full_response}
                
            except Exception as parse_error:
                print(f"JSON parsing failed: {parse_error}, using emergency response")
                
                # Final text-based fallback
                emergency_decision = self._create_text_based_decision(full_response)
                yield {"type": "complete", "decision": emergency_decision, "full_response": full_response}

    def _parse_json_response(self, text: str):
        """Advanced JSON parsing with multiple strategies"""
        # Strategy 1: Find JSON object boundaries
        try:
            start_brace = text.find('{')
            end_brace = text.rfind('}')
            if start_brace != -1 and end_brace != -1:
                json_str = text[start_brace:end_brace + 1]
                parsed_json = json.loads(json_str)
                return self._validate_and_create_decision(parsed_json)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Line-by-line extraction
        try:
            lines = text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if '{' in line:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if '}' in line and in_json:
                    break
            
            json_str = '\n'.join(json_lines)
            parsed_json = json.loads(json_str)
            return self._validate_and_create_decision(parsed_json)
        except (json.JSONDecodeError, IndexError):
            pass
        
        raise ValueError("Could not parse JSON from response")

    def _validate_and_create_decision(self, parsed_json: dict):
        """Validate and create AgentDecision from parsed JSON"""
        response_text = parsed_json.get("response", "")
        next_agent = parsed_json.get("next_agent", self.all_agent_names[0] if self.all_agent_names else "Conclusion")
        ready_to_conclude = parsed_json.get("ready_to_conclude", False)
        thoughts = parsed_json.get("thoughts", "Parsed from JSON")
        
        # Validate next_agent
        valid_agents = self.all_agent_names + ["Conclusion"]
        if next_agent not in valid_agents:
            next_agent = self._parse_next_agent_from_text(response_text)
        
        class ParsedDecision(BaseModel):
            thoughts: str
            response: str
            next_agent: str
            ready_to_conclude: bool
        
        return ParsedDecision(
            thoughts=thoughts,
            response=response_text,
            next_agent=next_agent,
            ready_to_conclude=ready_to_conclude
        )

    def _create_text_based_decision(self, text: str):
        """Create decision from raw text as final fallback"""
        response_text = self._extract_response_from_text(text)
        next_agent = self._parse_next_agent_from_text(response_text)
        
        class TextBasedDecision(BaseModel):
            thoughts: str = Field(default="Text-based parsing fallback")
            response: str
            next_agent: str
            ready_to_conclude: bool = Field(default=False)
        
        return TextBasedDecision(
            response=response_text.strip() if response_text.strip() else "議論を続けましょう。",
            next_agent=next_agent
        )
    
    def _extract_response_from_text(self, text: str) -> str:
        """Extract response content from JSON text."""
        try:
            # Try to find and parse JSON
            start_brace = text.find('{')
            end_brace = text.rfind('}')
            if start_brace != -1 and end_brace != -1:
                json_str = text[start_brace:end_brace + 1]
                parsed = json.loads(json_str)
                return parsed.get("response", text)
        except:
            pass
        
        # Fallback: return the text as is, cleaned up
        cleaned_text = text.strip()
        # Remove common JSON artifacts
        if cleaned_text.startswith('{"') and cleaned_text.endswith('"}'):
            try:
                parsed = json.loads(cleaned_text)
                return parsed.get("response", cleaned_text)
            except:
                pass
        
        return cleaned_text
    
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
