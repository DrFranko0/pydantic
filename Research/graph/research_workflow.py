from dataclasses import dataclass, field
from typing import List, Optional
from pydantic_graph import BaseNode, GraphRunContext, Edge, End

from models.research_state import ResearchState, Finding, ReportSection, Reference

@dataclass
class GenerateQuestions(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "ResearchQuestions | End[str]":
        if ctx.state.questions:
            return ResearchQuestions(question_index=0)
            
        result = await ctx.deps["question_agent"].run(
            f"Generate 5 key research questions about: {ctx.state.topic}",
            message_history=ctx.state.agent_memory
        )
        
        ctx.state.agent_memory += result.all_messages()
        ctx.state.questions = result.data.questions
        
        if not ctx.state.questions:
            return End("Failed to generate research questions")
            
        return ResearchQuestions(question_index=0)

@dataclass
class ResearchQuestions(BaseNode[ResearchState]):
    question_index: int
    
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "GenerateSearchQueries | SynthesizeFindings":
        if self.question_index >= len(ctx.state.questions):
            return SynthesizeFindings()
            
        current_question = ctx.state.questions[self.question_index]
        

        if current_question not in ctx.state.findings:
            ctx.state.findings[current_question] = []
            
        return GenerateSearchQueries(question=current_question)

@dataclass
class GenerateSearchQueries(BaseNode[ResearchState]):
    question: str
    
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "ExecuteSearch | ResearchQuestions":
        result = await ctx.deps["search_agent"].run(
            f"Generate effective search queries for the research question: {self.question}",
            message_history=ctx.state.agent_memory
        )
        
        ctx.state.agent_memory += result.all_messages()
        search_queries = result.data.queries
        
        if not search_queries:
            # If no queries were generated, move to the next question
            question_index = next((i for i, q in enumerate(ctx.state.questions) if q == self.question), 0) + 1
            return ResearchQuestions(question_index=question_index)
        
        return ExecuteSearch(question=self.question, queries=search_queries, query_index=0)

@dataclass
class ExecuteSearch(BaseNode[ResearchState]):
    question: str
    queries: List[str]
    query_index: int
    
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "AnalyzeSearchResults | ResearchQuestions":
        if self.query_index >= len(self.queries):
            question_index = next((i for i, q in enumerate(ctx.state.questions) if q == self.question), 0) + 1
            return ResearchQuestions(question_index=question_index)
        
        current_query = self.queries[self.query_index]
        search_results = ctx.deps["web_search"].search(current_query)
        
        if not search_results:
            return ExecuteSearch(
                question=self.question, 
                queries=self.queries, 
                query_index=self.query_index + 1
            )
        
        return AnalyzeSearchResults(
            question=self.question,
            queries=self.queries,
            query_index=self.query_index,
            search_results=search_results,
            result_index=0
        )

@dataclass
class AnalyzeSearchResults(BaseNode[ResearchState]):
    question: str
    queries: List[str]
    query_index: int
    search_results: List
    result_index: int
    
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "AnalyzeSearchResults | ExecuteSearch":
        if self.result_index >= len(self.search_results):
            # Move to the next query
            return ExecuteSearch(
                question=self.question,
                queries=self.queries,
                query_index=self.query_index + 1
            )
        
        current_result = self.search_results[self.result_index]
        
        content = ctx.deps["content_extractor"].extract_content(current_result.url)
        
        if not content or len(content) < 100:
            return AnalyzeSearchResults(
                question=self.question,
                queries=self.queries,
                query_index=self.query_index,
                search_results=self.search_results,
                result_index=self.result_index + 1
            )
        
        analysis_prompt = (
            f"Research Question: {self.question}\n\n"
            f"Content from {current_result.url}:\n\n{content[:5000]}..."
        )
        
        analysis_result = await ctx.deps["analysis_agent"].run(
            analysis_prompt,
            message_history=ctx.state.agent_memory
        )
        
        ctx.state.agent_memory += analysis_result.all_messages()
        
        if analysis_result.data.findings:
            for finding in analysis_result.data.findings:
                finding.source_url = current_result.url
                ctx.state.findings[self.question].append(finding)
            
            citation = ctx.deps["citation_generator"].generate_citation(
                title=current_result.title,
                url=current_result.url
            )
            
            if not any(ref.url == citation.url for ref in ctx.state.references):
                ctx.state.references.append(citation)
        
        return AnalyzeSearchResults(
            question=self.question,
            queries=self.queries,
            query_index=self.query_index,
            search_results=self.search_results,
            result_index=self.result_index + 1
        )

@dataclass
class SynthesizeFindings(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> "GenerateReport | End[str]":
        all_findings = []
        for question, findings in ctx.state.findings.items():
            all_findings.extend(findings)
        
        if not all_findings:
            return End("No relevant findings were discovered during research")
        
        findings_by_question = {}
        for question, findings in ctx.state.findings.items():
            findings_by_question[question] = [
                {"content": finding.content, "source_url": finding.source_url, "relevance_score": finding.relevance_score}
                for finding in findings
            ]
        
        synthesis_result = await ctx.deps["synthesis_agent"].run(
            f"Synthesize the following research findings on '{ctx.state.topic}':\n{findings_by_question}",
            message_history=ctx.state.agent_memory
        )
        
        ctx.state.agent_memory += synthesis_result.all_messages()
        ctx.state.report_sections = synthesis_result.data.sections
        
        return GenerateReport()

@dataclass
class GenerateReport(BaseNode[ResearchState]):
    async def run(self, ctx: GraphRunContext[ResearchState]) -> End[str]:
        
        report_sections = [
            {
                "title": section.title,
                "content": section.content,
                "findings": [
                    {"content": finding.content, "source_url": finding.source_url, "relevance_score": finding.relevance_score}
                    for finding in section.findings
                ]
            }
            for section in ctx.state.report_sections
        ]
        
        references = [
            {
                "title": ref.title,
                "url": ref.url,
                "author": ref.author,
                "date": ref.date,
                "accessed_date": ref.accessed_date
            }
            for ref in ctx.state.references
        ]
        
        # Generate the report
        report_result = await ctx.deps["report_agent"].run(
            f"Generate a research report on '{ctx.state.topic}' using the following sections and references:\n" +
            f"Sections: {report_sections}\n" +
            f"References: {references}",
            message_history=ctx.state.agent_memory
        )
        
        ctx.state.agent_memory += report_result.all_messages()
        
        report_data = report_result.data
        
        markdown_report = f"# Research Report: {ctx.state.topic}\n\n"
        markdown_report += f"## Executive Summary\n\n{report_data.executive_summary}\n\n"
        
        for section in report_data.sections:
            markdown_report += f"## {section.title}\n\n{section.content}\n\n"
        
        markdown_report += "## References\n\n"
        for i, reference in enumerate(report_data.references, 1):
            markdown_report += f"{i}. {ctx.deps['citation_generator'].format_apa_citation(reference)}\n"
        
        return End(markdown_report)
