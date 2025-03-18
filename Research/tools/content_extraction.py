import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
import trafilatura

class ContentExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_content(self, url: str) -> str:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded, include_comments=False, 
                                             include_tables=True, no_fallback=False)
                if content and len(content.strip()) > 100:
                    return content

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            text = soup.get_text(separator='\n')
            
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = '\n'.join(lines)
            
            return content
        
        except Exception as e:
            return f"Error extracting content: {str(e)}"
