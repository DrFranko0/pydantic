from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext, End
from models.conversation import ConversationState
from pydantic_ai.messages import UserMessage
from agents.intent_agent import intent_agent
from agents.knowledge_agent import knowledge_agent
from agents.response_agent import response_agent
from agents.sentiment_agent import sentiment_agent
from tools.product_lookup import product_lookup_tool
from tools.order_status import order_status_tool
from tools.customer_records import customer_records_tool
from typing import Dict, Any, Optional

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
        product_info = None
        order_info = None
        customer_info = None
        
        if "product_id" in ctx.state.entities or "product_name" in ctx.state.entities:
            try:
                product_lookup_result = await product_lookup_tool.run(
                    deps={
                        "product_id": ctx.state.entities.get("product_id"),
                        "product_name": ctx.state.entities.get("product_name")
                    }
                )
                product_info = product_lookup_result.data
            except Exception as e:
                print(f"Error looking up product: {e}")
        
        if "order_id" in ctx.state.entities:
            try:
                order_status_result = await order_status_tool.run(
                    deps={
                        "order_id": ctx.state.entities.get("order_id"),
                        "customer_id": ctx.state.customer_id
                    }
                )
                order_info = order_status_result.data
            except Exception as e:
                print(f"Error looking up order: {e}")
        
        if "customer_id" in ctx.state.entities or ctx.state.issue_category == "account_inquiry":
            try:
                customer_record_result = await customer_records_tool.run(
                    deps={"customer_id": ctx.state.customer_id}
                )
                customer_info = customer_record_result.data
            except Exception as e:
                print(f"Error looking up customer: {e}")
        
        knowledge_context = ""
        if ctx.state.entities:
            try:
                knowledge_result = await knowledge_agent.retrieve_knowledge(
                    deps={
                        "query": ctx.state.issue_category,
                        "entities": ctx.state.entities
                    }
                )
                knowledge_context = knowledge_result.relevant_info
            except Exception as e:
                print(f"Error retrieving knowledge: {e}")
        
        tool_context = ""
        if product_info:
            tool_context += f"Product info: {product_info.dict()}\n"
        if order_info:
            tool_context += f"Order info: {order_info.dict()}\n"
        if customer_info:
            tool_context += f"Customer info: {customer_info.dict()}\n"
        
        response_result = await response_agent.generate_response(
            deps={
                "conversation_history": ctx.state.conversation_history,
                "detected_intent": ctx.state.issue_category,
                "detected_entities": ctx.state.entities,
                "knowledge_context": f"{knowledge_context}\n{tool_context}".strip(),
                "sentiment_score": ctx.state.sentiment
            }
        )
        
        ctx.state.conversation_history.append(UserMessage(content=response_result.response, role="assistant"))
        
        if response_result.needs_human:
            return EscalateToHuman(reason="Response agent requested human assistance")
        
        if "resolved" in response_result.suggested_actions:
            ctx.state.resolved = True
            return End({
                "conversation": ctx.state.conversation_history,
                "resolved": True,
                "sentiment": ctx.state.sentiment,
                "entities": ctx.state.entities
            })
        
        return End({
            "waiting_for_input": True,
            "conversation": ctx.state.conversation_history,
            "resolved": False,
            "sentiment": ctx.state.sentiment,
            "entities": ctx.state.entities
        })

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
