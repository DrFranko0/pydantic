from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
from models.conversation import ConversationState
from pydantic_ai.messages import UserMessage
from agents import intent_agent, response_agent, knowledge_agent
from typing import Dict, Any

@dataclass
class AnalyzeIntent(BaseNode[ConversationState]):
    user_message: str
    
    async def run(self, ctx: GraphRunContext[ConversationState]) -> "RespondToUser | EscalateToHuman":
        ctx.state.conversation_history.append(UserMessage(content=self.user_message))
        
        analysis_result = await intent_agent.run(
            f"Analyze this customer message: {self.user_message}"
        )
    
        ctx.state.sentiment = analysis_result.data.sentiment
        ctx.state.confidence = analysis_result.data.confidence
        ctx.state.issue_category = analysis_result.data.intent
        
        for entity in analysis_result.data.entities:
            ctx.state.entities[entity.name] = entity.value
        
        if ctx.state.sentiment < -0.7 or analysis_result.data.confidence < 0.4:
            return EscalateToHuman(reason="High negative sentiment or low confidence")
        elif len(ctx.state.conversation_history) > 10:
            return EscalateToHuman(reason="Conversation too long without resolution")
        else:
            return RespondToUser()

@dataclass
class RespondToUser(BaseNode[ConversationState]):
    async def run(self, ctx: GraphRunContext[ConversationState]) -> "AnalyzeIntent | EscalateToHuman | End[Dict[str, Any]]":
        knowledge_context = ""
        if ctx.state.entities:
            knowledge_result = await knowledge_agent.run(
                f"Retrieve information relevant to: {ctx.state.issue_category}",
                deps={
                    "query": ctx.state.issue_category,
                    "entities": ctx.state.entities
                }
            )
            knowledge_context = knowledge_result.data.relevant_info
        
        response_result = await response_agent.run(
            f"Generate a response for a {ctx.state.issue_category} query",
            deps={
                "conversation_history": ctx.state.conversation_history,
                "detected_intent": ctx.state.issue_category,
                "detected_entities": ctx.state.entities,
                "knowledge_context": knowledge_context,
                "sentiment_score": ctx.state.sentiment
            }
        )
        
        ctx.state.conversation_history.append(UserMessage(content=response_result.data.response, role="assistant"))
        
        if response_result.data.needs_human:
            return EscalateToHuman(reason="Response agent requested human assistance")
        
        if "resolved" in response_result.data.suggested_actions or len(response_result.data.suggested_actions) == 0:
            ctx.state.resolved = True
            return End({
                "conversation": ctx.state.conversation_history,
                "resolved": True,
                "sentiment": ctx.state.sentiment,
                "entities": ctx.state.entities
            })
        
        return AnalyzeIntent(user_message="[WAITING FOR USER INPUT]")

@dataclass
class EscalateToHuman(BaseNode[ConversationState]):
    reason: str = "Unspecified reason"
    
    async def run(self, ctx: GraphRunContext[ConversationState]) -> End[Dict[str, Any]]:
        ctx.state.needs_human = True
        
        handoff_message = f"I'll connect you with a human agent who can better assist you with this. A customer support representative will be with you shortly."
        ctx.state.conversation_history.append(UserMessage(content=handoff_message, role="assistant"))
        
        return End({
            "conversation": ctx.state.conversation_history,
            "resolved": False,
            "needs_human": True,
            "escalation_reason": self.reason,
            "sentiment": ctx.state.sentiment,
            "entities": ctx.state.entities
        })
