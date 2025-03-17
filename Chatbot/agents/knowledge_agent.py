from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import List, Dict
from pydantic import BaseModel, Field
from config import KNOWLEDGE_MODEL_NAME

class KnowledgeResult(BaseModel):
    relevant_info: str = Field(description="Relevant information from the knowledge base")
    confidence: float = Field(description="Confidence in the retrieved information (0-1)")
    sources: List[str] = Field(default_factory=list, description="Sources of the information")

@dataclass
class KnowledgeDependencies:
    query: str
    entities: Dict[str, str]

knowledge_agent = Agent(
    f"gemini:{KNOWLEDGE_MODEL_NAME}",
    deps_type=KnowledgeDependencies,
    result_type=KnowledgeResult,
    system_prompt="""
    You are a knowledge retrieval expert for a customer support system.
    Your job is to provide relevant information from our knowledge base
    that will help answer customer queries.
    
    Retrieve only the most relevant information and indicate your confidence
    in the information provided.
    """
)

@knowledge_agent.tool
async def retrieve_knowledge(ctx: RunContext[KnowledgeDependencies]) -> KnowledgeResult:
    
    # For this example, we'll simulate knowledge retrieval with the LLM
    
    prompt = f"""
    Retrieve relevant knowledge to help answer this customer query:
    
    Query: {ctx.deps.query}
    Entities: {ctx.deps.entities}
    
    Provide the most helpful information, your confidence in this information,
    and what sources you would cite if this were a real knowledge base.
    """
    
    result = await knowledge_agent.run(prompt)
    return result.data
