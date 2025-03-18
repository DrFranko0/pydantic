from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field

class SearchQueries(BaseModel):
    queries: List[str] = Field(
        description="List of optimized search queries to find information on the research question"
    )

def create_search_agent(model: GeminiModel) -> Agent:
    agent = Agent(
        model=model,
        result_type=SearchQueries,
        system_prompt=(
            "You are a search query optimization expert. "
            "Your goal is to generate effective search queries for finding "
            "high-quality information about a specific research question. "
            "Create queries that will yield authoritative, diverse, and relevant results. "
            "Consider different phrasings, technical terms, and variations that might "
            "help discover useful information."
        )
    )
    return agent
