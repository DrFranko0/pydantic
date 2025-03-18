from typing import Dict, List, Any
from pydantic_ai.messages import ModelMessage, UserMessage

class HumanHandoffManager:
    def __init__(self):
        self.pending_handoffs: Dict[str, Dict[str, Any]] = {}
    
    def queue_handoff(self, customer_id: str, conversation_data: Dict[str, Any]) -> str:
        handoff_id = f"handoff_{customer_id}_{len(self.pending_handoffs)}"
        self.pending_handoffs[handoff_id] = {
            "customer_id": customer_id,
            "conversation": conversation_data.get("conversation", []),
            "escalation_reason": conversation_data.get("escalation_reason", "No reason provided"),
            "entities": conversation_data.get("entities", {}),
            "sentiment": conversation_data.get("sentiment", 0.0),
            "status": "pending"
        }
        return handoff_id
    
    def get_pending_handoffs(self) -> List[Dict[str, Any]]:
        return [
            {
                "handoff_id": handoff_id,
                "customer_id": data["customer_id"],
                "escalation_reason": data["escalation_reason"],
                "sentiment": data["sentiment"],
                "status": data["status"]
            }
            for handoff_id, data in self.pending_handoffs.items()
            if data["status"] == "pending"
        ]
    
    def get_handoff_details(self, handoff_id: str) -> Dict[str, Any]:
        return self.pending_handoffs.get(handoff_id, {})
    
    def claim_handoff(self, handoff_id: str, agent_id: str) -> bool:
        if handoff_id in self.pending_handoffs and self.pending_handoffs[handoff_id]["status"] == "pending":
            self.pending_handoffs[handoff_id]["status"] = "claimed"
            self.pending_handoffs[handoff_id]["agent_id"] = agent_id
            return True
        return False
