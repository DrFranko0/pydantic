from pydantic_graph import Graph
from .nodes import AnalyzeIntent, RespondToUser, EscalateToHuman
from models.conversation import ConversationState
from typing import Dict, Any, List
from pydantic_ai.messages import ModelMessage, UserMessage

class ConversationGraph:
    def __init__(self, customer_id: str):
        self.state = ConversationState(customer_id=customer_id)
        self.graph = Graph[ConversationState, Dict[str, Any]](
            state=self.state,
            first_node=None 
        )
    
    async def start_conversation(self, initial_message: str) -> Dict[str, Any]:
        self.graph.first_node = AnalyzeIntent(user_message=initial_message)
        result = await self.graph.run()
        return result
    
    async def continue_conversation(self, user_message: str) -> Dict[str, Any]:
        if self.state.needs_human or self.state.resolved:
            return {
                "error": "Conversation already ended or handed off to human",
                "conversation": self.state.conversation_history,
                "needs_human": self.state.needs_human,
                "resolved": self.state.resolved
            }
        
        self.graph.first_node = AnalyzeIntent(user_message=user_message)
        result = await self.graph.run()
        return result
    
    def get_conversation_history(self) -> List[ModelMessage]:
        return self.state.conversation_history
    
    def add_human_message(self, message: str, is_agent: bool = False) -> None:
        role = "assistant" if is_agent else "user"
        self.state.conversation_history.append(UserMessage(content=message, role=role))