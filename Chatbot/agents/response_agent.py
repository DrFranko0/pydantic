from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from dataclasses import dataclass
from typing import List, Dict
from models.conversation import ResponseGenerationResult
from config import MODEL_NAME

@dataclass
class ResponseDependencies:
    conversation_history: List[ModelMessage]
    detected_intent: str
    detected_entities: Dict[str, str]
    knowledge_context: str = ""
    sentiment_score: float = 0.0

response_agent = Agent(
    f"gemini:{MODEL_NAME}",
    deps_type=ResponseDependencies,
    result_type=ResponseGenerationResult,
    system_prompt="""
    You are a helpful customer support assistant. Your goal is to provide clear, 
    accurate, and helpful responses to customer inquiries.
    
    When responding:
    1. Be polite and professional at all times
    2. Address the customer's intent directly
    3. Provide relevant information based on the entities detected
    4. Adjust your tone based on the customer's sentiment
    5. Be concise but thorough
    6. If you cannot confidently help the customer, indicate that human assistance is needed
    
    Remember to only use information you are certain about. Don't make up information.
    """
)

@response_agent.tool
async def generate_response(ctx: RunContext[ResponseDependencies]) -> ResponseGenerationResult:
    prompt = f"""
    Based on the conversation history and detected intent, generate a helpful response.
    
    Intent: {ctx.deps.detected_intent}
    Entities: {ctx.deps.detected_entities}
    Sentiment: {ctx.deps.sentiment_score} (-1 negative to 1 positive)
    
    Additional context: {ctx.deps.knowledge_context}
    
    Generate a helpful, accurate response addressing the customer's needs.
    """
    
    result = await response_agent.run(prompt, message_history=ctx.deps.conversation_history)
    
    return result.data
