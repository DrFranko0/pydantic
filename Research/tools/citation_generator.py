import datetime
from typing import Optional
from models.research_state import Reference

class CitationGenerator:
    def __init__(self):
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    def generate_citation(self, 
                          title: str, 
                          url: str, 
                          author: Optional[str] = None,
                          date: Optional[str] = None) -> Reference:
        return Reference(
            title=title,
            url=url,
            author=author if author else "Unknown",
            date=date if date else "n.d.",
            accessed_date=self.today
        )
    
    def format_apa_citation(self, reference: Reference) -> str:
        if reference.author and reference.author != "Unknown":
            author_part = f"{reference.author}. "
        else:
            author_part = ""
        
        if reference.date and reference.date != "n.d.":
            date_part = f"({reference.date}). "
        else:
            date_part = "(n.d.). "
        
        return f"{author_part}{date_part}{reference.title}. Retrieved on {reference.accessed_date} from {reference.url}"
