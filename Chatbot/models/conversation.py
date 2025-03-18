from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage

class Entity(BaseModel):
    name: str = Field(description="Name of the entity identified in the conversation")
    type: str = Field(description="Type of entity (e.g., product, order_number, date)")
    value: str = Field(description="Value of the entity")

class IntentAnalysisResult(BaseModel):
    intent: str = Field(description="Primary user intent")
    entities: List[Entity] = Field(default_factory=list, description="Entities extracted from the message")
    sentiment: float = Field(description="Sentiment score (-1 to 1, negative to positive)")
    confidence: float = Field(description="Confidence score for intent detection (0-1)")

class ResponseGenerationResult(BaseModel):
    response: str = Field(description="Response to the user's query")
    needs_human: bool = Field(default=False, description="Whether the query needs human attention")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested actions for the user or agent")

@dataclass
class ConversationState:
    customer_id: str
    issue_category: str = ""
    sentiment: float = 0.0
    conversation_history: List[ModelMessage] = field(default_factory=list)
    resolved: bool = False
    entities: Dict[str, str] = field(default_factory=dict)
    needs_human: bool = False
    confidence: float = 1.0
