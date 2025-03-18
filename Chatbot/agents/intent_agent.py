from pydantic_ai import Agent
from models.conversation import IntentAnalysisResult
from config import MODEL_NAME

intent_agent = Agent(
    f"gemini:{MODEL_NAME}",
    result_type=IntentAnalysisResult,
    system_prompt="""
    You are an intent analysis AI assistant for a customer support system.
    Your job is to analyze customer messages to:
    1. Identify the primary intent of the message
    2. Extract relevant entities (products, order numbers, dates, etc.)
    3. Analyze sentiment (-1 to 1, negative to positive)
    4. Provide a confidence score for your intent detection
    
    Be precise and thorough in your analysis.
    """
)
