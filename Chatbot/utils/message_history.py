from typing import List, Dict, Any
from pydantic_ai.messages import ModelMessage, UserMessage
import json
import os

class MessageHistoryManager:
    def __init__(self, storage_dir: str = "conversation_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_conversation(self, customer_id: str, conversation: List[ModelMessage]) -> str:
        import time
        timestamp = int(time.time())
        filename = f"{customer_id}_{timestamp}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        serialized_conversation = []
        for message in conversation:
            if isinstance(message, UserMessage):
                serialized_conversation.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                serialized_conversation.append({
                    "role": "system" if not hasattr(message, "role") else message.role,
                    "content": str(message)
                })
        
        with open(filepath, "w") as f:
            json.dump(serialized_conversation, f, indent=2)
        
        return filepath
    
    def load_conversation(self, filepath: str) -> List[ModelMessage]:
        with open(filepath, "r") as f:
            serialized_conversation = json.load(f)
        
        conversation = []
        for message in serialized_conversation:
            conversation.append(UserMessage(
                content=message["content"],
                role=message["role"]
            ))
        
        return conversation
