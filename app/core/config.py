"""
Application Configuration
Loads settings from environment variables.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    APP_NAME: str = "IDLock API"
    DEBUG: bool = False
    API_BASE_URL: str = "http://localhost:8000"  # Base URL for file serving
    
    # Database - SQLite for local dev, PostgreSQL for production
    DATABASE_URL: str = "sqlite:///./idlock.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Vector DB (Pinecone)
    PINECONE_API_KEY: str = ""
    PINECONE_ENV: str = "us-east-1"
    PINECONE_INDEX: str = "idlock-characters"
    
    # Image Generation (Gemini 2.5 Flash - Nano Banana)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-image"  # Nano Banana image generation model
    # Model for character consistency (supports up to 5 reference images for identity preservation)
    # gemini-3-pro-image-preview is specifically designed for maintaining character consistency
    GEMINI_MODEL_CHARACTER: str = "gemini-3-pro-image-preview"
    
    # Video Generation (Veo 3.1 - Native audio and dialogue support)
    VEO_MODEL: str = "veo-3.1-generate-preview"  # Best quality with native audio
    VEO_POLL_INTERVAL: int = 10  # Seconds between status checks
    VEO_MAX_WAIT_TIME: int = 360  # Max wait time (6 minutes)
    
    # LLM for prompts/vectors/summarization (Groq - Free)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    
    # Storage - S3 settings (optional)
    S3_BUCKET: str = ""
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_REGION: str = "us-east-1"
    
    # Local storage fallback
    LOCAL_STORAGE_PATH: str = "./uploads"
    USE_LOCAL_STORAGE: bool = True
    
    # Google Cloud Storage (for Cloud Run deployment)
    USE_GCS: bool = False
    GCS_BUCKET_UPLOADS: str = "cineai-uploads"
    GCS_BUCKET_OUTPUTS: str = "cineai-outputs"
    GCP_PROJECT_ID: str = ""
    
    # Cloud SQL connection (for Cloud Run)
    CLOUD_SQL_CONNECTION_NAME: str = ""  # Format: PROJECT_ID:REGION:INSTANCE
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Worker settings
    JOB_TIMEOUT_EXTRACTION: int = 60
    JOB_TIMEOUT_GENERATION: int = 120
    JOB_TIMEOUT_REFINE: int = 90
    
    # Identity settings
    IDR_THRESHOLD: float = 0.7
    SEMANTIC_WEIGHT: float = 0.6
    EPISODIC_WEIGHT: float = 0.4
    EPISODIC_DECAY: float = 0.6
    EPISODIC_TOP_K: int = 10
    
    @field_validator('GEMINI_API_KEY', 'GROQ_API_KEY', 'PINECONE_API_KEY', mode='before')
    @classmethod
    def strip_api_keys(cls, v):
        """Strip whitespace and newlines from API keys loaded from secrets."""
        if isinstance(v, str):
            return v.strip()
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
