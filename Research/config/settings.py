import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    GOOGLE_API_KEY: str
    SERPER_API_KEY: str
    
    GEMINI_MODEL: str = "models/gemini-pro"
    
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()