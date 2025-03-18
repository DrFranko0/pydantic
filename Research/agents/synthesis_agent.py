from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field
from models.report_schema import ReportSection

class ResearchSynthesis(BaseModel):
    sections: List[ReportSection] = Field(
        description="List of synthesized report sections based on the findings"
    )

def create_synthesis_agent(model: GeminiModel) -> Agent:
    agent = Agent(
        model=model,
        result_type=ResearchSynthesis,
        system_prompt=(
            "You are a research synthesis expert. "
            "Your goal is to analyze a collection of findings from various sources "
            "and organize them into coherent, well-structured sections. "
            "For each section, provide a clear title and comprehensive content "
            "that integrates information from multiple sources. "
            "Focus on identifying patterns, connections, and key insights across "
            "the findings. Ensure that the synthesis is balanced, thorough, and "
            "accurately represents the research findings."
        )
    )
    return agent
