import os
from typing import Optional, List, Dict, Any
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import ModelMessage

class GeminiModelProvider:
    def __init__(self, model_name: str = "models/gemini-pro"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.model = GeminiModel(
            model=model_name,
            api_key=api_key
        )
    
    def get_model(self) -> GeminiModel:
        return self.model
