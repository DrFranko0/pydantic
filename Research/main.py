import asyncio
import typer
import logfire
import logging
import os
from typing import Optional
from models.gemini import GeminiModelProvider
from models.research_state import ResearchState
from agents.question_agent import create_question_agent
from agents.search_agent import create_search_agent
from agents.analysis_agent import create_analysis_agent
from agents.synthesis_agent import create_synthesis_agent
from agents.report_agent import create_report_agent
from tools.web_search import WebSearch
from tools.content_extraction import ContentExtractor
from tools.citation_generator import CitationGenerator
from graph.research_workflow import (
    GenerateQuestions, 
    ResearchQuestions, 
    GenerateSearchQueries, 
    ExecuteSearch, 
    AnalyzeSearchResults, 
    SynthesizeFindings, 
    GenerateReport
)
from pydantic_graph import Graph
from config.settings import settings

app = typer.Typer(help="Autonomous Research Assistant")

logfire.configure(token='pylf_v1_us_1VjTXQWkHjwRRPH7CmsW70JqwyvkdJSY0xQRG8RdT36C')
logger = logging.getLogger("research_assistant")

async def run_research(topic: str, output_file: Optional[str] = None) -> None:
    logger.info(f"Starting research on topic: {topic}")
    
    model_provider = GeminiModelProvider(model_name=settings.GEMINI_MODEL)
    model = model_provider.get_model()
    
    web_search = WebSearch()
    content_extractor = ContentExtractor()
    citation_generator = CitationGenerator()
    
    question_agent = create_question_agent(model)
    search_agent = create_search_agent(model)
    analysis_agent = create_analysis_agent(model)
    synthesis_agent = create_synthesis_agent(model)
    report_agent = create_report_agent(model)
    
    research_state = ResearchState(topic=topic)
    
    research_graph = Graph(
        nodes=[
            GenerateQuestions,
            ResearchQuestions,
            GenerateSearchQueries,
            ExecuteSearch,
            AnalyzeSearchResults,
            SynthesizeFindings,
            GenerateReport
        ]
    )
    
    logger.info("Starting research workflow")
    deps = {
        "question_agent": question_agent,
        "search_agent": search_agent, 
        "analysis_agent": analysis_agent,
        "synthesis_agent": synthesis_agent,
        "report_agent": report_agent,
        "web_search": web_search,
        "content_extractor": content_extractor,
        "citation_generator": citation_generator
    }
    
    result = await research_graph.run(
        GenerateQuestions(),
        state=research_state,
        deps=deps
    )
    
    if isinstance(result.value, str):
        logger.info("Research completed")
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.value)
            logger.info(f"Research report saved to {output_file}")
        
        return result.value
    else:
        logger.error(f"Research failed: {result.value}")
        return None

@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic"),
    output: Optional[str] = typer.Option(None, help="Output file path for the research report")
):
    asyncio.run(run_research(topic, output))

if __name__ == "__main__":
    app()