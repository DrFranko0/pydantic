from typing import List, Optional
from pydantic import BaseModel, Field

class Finding(BaseModel):
    content: str = Field(description="The extracted content from the source")
    source_url: str = Field(description="URL of the source")
    relevance_score: float = Field(description="Relevance score between 0 and 1", ge=0, le=1)

class Reference(BaseModel):
    title: str = Field(description="Title of the reference")
    url: str = Field(description="URL of the reference")
    author: Optional[str] = Field(None, description="Author of the reference")
    date: Optional[str] = Field(None, description="Publication date")
    accessed_date: str = Field("", description="Date accessed")

class ReportSection(BaseModel):
    title: str = Field(description="Title of the section")
    content: str = Field(description="Content of the section")
    findings: List[Finding] = Field(default_factory=list, description="Supporting findings")

class ResearchReport(BaseModel):
    topic: str = Field(description="Research topic")
    executive_summary: str = Field(description="Executive summary of the research")
    sections: List[ReportSection] = Field(description="Sections of the report")
    references: List[Reference] = Field(description="References used in the report")
