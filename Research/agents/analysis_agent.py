from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field
from models.report_schema import Finding

class ContentAnalysis(BaseModel):
    findings: List[Finding] = Field(
        description="List of extracted findings from the content"
    )
    
def create_analysis_agent(model: GeminiModel) -> Agent:
    agent = Agent(
        model=model,
        result_type=ContentAnalysis,
        system_prompt=(
            "You are a content analysis expert. "
            "Your goal is to carefully analyze content from web pages and "
            "extract relevant, factual information related to a research question. "
            "For each piece of content, identify key findings, supporting evidence, "
            "and relevant details. Assign a relevance score between 0 and 1 to each finding. "
            "Ignore irrelevant content, advertisements, and unrelated information. "
            "Focus on extracting accurate, high-quality information that directly "
            "addresses the research question."
        )
    )
    return agent
