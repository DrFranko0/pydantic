from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel, Field
from config import MODEL_NAME

class SentimentResult(BaseModel):
    score: float = Field(description="Sentiment score from -1 (very negative) to 1 (very positive)")
    key_indicators: List[str] = Field(description="Key phrases that indicate the sentiment")

@dataclass
class SentimentDependencies:
    message: str

sentiment_agent = Agent(
    f"gemini:{MODEL_NAME}",
    deps_type=SentimentDependencies,
    result_type=SentimentResult,
    system_prompt="""
    You are a sentiment analysis expert. Analyze the sentiment of customer messages
    on a scale from -1 (very negative) to 1 (very positive).
    
    Also identify key phrases that indicate sentiment. Be as objective as possible.
    """
)

@sentiment_agent.tool
async def analyze_sentiment(ctx: RunContext[SentimentDependencies]) -> SentimentResult:
    result = await sentiment_agent.run(
        f"Analyze the sentiment of this customer message: {ctx.deps.message}"
    )
    return result.data
