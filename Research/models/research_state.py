from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pydantic_ai.messages import ModelMessage

@dataclass
class Finding:
    content: str
    source_url: str
    relevance_score: float = 0.0

@dataclass
class Reference:
    title: str
    url: str
    author: Optional[str] = None
    date: Optional[str] = None
    accessed_date: str = ""

@dataclass
class ReportSection:
    title: str
    content: str
    findings: List[Finding] = field(default_factory=list)

@dataclass
class ResearchState:
    topic: str
    questions: List[str] = field(default_factory=list)
    findings: Dict[str, List[Finding]] = field(default_factory=dict)
    agent_memory: List[ModelMessage] = field(default_factory=list)
    report_sections: List[ReportSection] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
