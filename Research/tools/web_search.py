import os
import json
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class WebSearch:
    def __init__(self):
        self.api_key = os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable not set")
        
        self.url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        payload = json.dumps({
            "q": query,
            "num": num_results
        })
        
        response = requests.post(self.url, headers=self.headers, data=payload)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "organic" in data:
            for item in data["organic"][:num_results]:
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", "")
                    )
                )
        
        return results
