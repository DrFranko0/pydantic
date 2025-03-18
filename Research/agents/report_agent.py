from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field
from models.report_schema import ResearchReport, ReportSection, Reference

def create_report_agent(model: GeminiModel) -> Agent:
    agent = Agent(
        model=model,
        result_type=ResearchReport,
        system_prompt=(
            "You are a research report generation expert. "
            "Your goal is to create a comprehensive, well-structured research report "
            "based on the provided sections and references. "
            "Begin with an executive summary that highlights the key findings. "
            "Organize the sections in a logical flow, ensuring that the content "
            "is coherent and well-integrated. "
            "Include all references properly formatted at the end of the report. "
            "Ensure that the report is professional, balanced, and accurately "
            "represents the research findings."
        )
    )
    return agent
