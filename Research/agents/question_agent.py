from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field

class Questions(BaseModel):
    questions: List[str] = Field(
        description="List of focused research questions to explore the topic in depth"
    )

def create_question_agent(model: GeminiModel) -> Agent:
    agent = Agent(
        model=model,
        result_type=Questions,
        system_prompt=(
            "You are a research question formulation expert. "
            "Your goal is to generate focused, insightful questions that will guide "
            "a comprehensive research investigation on a given topic. "
            "Generate questions that will explore different facets of the topic, "
            "covering important aspects, recent developments, controversies, "
            "and future implications. "
            "Each question should be clear, specific, and answerable through research."
        )
    )
    return agent
