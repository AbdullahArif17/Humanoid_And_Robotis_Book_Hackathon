"""
Configuration management for the AI-Native Book RAG Chatbot application.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: str = Field(default="postgresql://user:password@localhost/dbname", alias="DATABASE_URL")

    # Qdrant settings
    qdrant_url: Optional[str] = Field(default="https://your-qdrant-cluster.qdrant.tech", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default="your-qdrant-api-key", alias="QDRANT_API_KEY")

    # OpenAI settings (kept for backward compatibility)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")  # Optional field
    openai_model: str = Field(default="gpt-4-turbo", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_EMBEDDING_MODEL")  # Model for embeddings
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")  # Optional, for custom endpoints
    openai_timeout: int = Field(default=30, alias="OPENAI_TIMEOUT")  # Request timeout in seconds
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")  # Number of retries for failed requests

    # Google AI settings
    google_api_key: str = Field(alias="GOOGLE_API_KEY")  # Required field
    google_model: str = Field(default="models/gemini-2.5-flash", alias="GOOGLE_MODEL")
    google_temperature: float = Field(default=0.7, alias="GOOGLE_TEMPERATURE")  # 0.0-1.0
    google_max_output_tokens: int = Field(default=1000, alias="GOOGLE_MAX_OUTPUT_TOKENS")
    google_top_p: float = Field(default=0.9, alias="GOOGLE_TOP_P")  # 0.0-1.0
    google_top_k: int = Field(default=40, alias="GOOGLE_TOP_K")  # 1-40

    # Application settings
    debug: bool = Field(default=False, alias="DEBUG")  # Default to False in production
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    secret_key: str = Field(default="your-secret-key-here", alias="SECRET_KEY")

    # API settings
    allowed_origins: str = Field(default="https://your-username.github.io", alias="ALLOWED_ORIGINS")

    # Performance settings
    response_max_tokens: int = Field(default=1000, alias="RESPONSE_MAX_TOKENS")
    embedding_chunk_size: int = Field(default=1000, alias="EMBEDDING_CHUNK_SIZE")  # tokens

    class Config:
        env_file = ".env"
        populate_by_name = True  # Allow both alias and field name


settings = Settings()


def validate_settings():
    """
    Validate settings at startup.
    
    Raises:
        ValueError: If required settings are missing or invalid
    """
    errors = []
    
    # Validate required settings
    if not settings.google_api_key or settings.google_api_key == "":
        errors.append("GOOGLE_API_KEY is required")
    
    if not settings.database_url or settings.database_url.startswith("postgresql://user:password"):
        errors.append("DATABASE_URL must be configured with a valid PostgreSQL connection string")
    
    if not settings.qdrant_url or settings.qdrant_url.startswith("https://your-qdrant-cluster"):
        errors.append("QDRANT_URL must be configured with a valid Qdrant URL")
    
    # Validate numeric ranges
    if settings.google_temperature < 0.0 or settings.google_temperature > 1.0:
        errors.append("GOOGLE_TEMPERATURE must be between 0.0 and 1.0")
    
    if settings.google_top_p < 0.0 or settings.google_top_p > 1.0:
        errors.append("GOOGLE_TOP_P must be between 0.0 and 1.0")
    
    if settings.google_top_k < 1 or settings.google_top_k > 40:
        errors.append("GOOGLE_TOP_K must be between 1 and 40")
    
    if settings.openai_timeout < 1:
        errors.append("OPENAI_TIMEOUT must be at least 1 second")
    
    if settings.openai_max_retries < 0:
        errors.append("OPENAI_MAX_RETRIES must be non-negative")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))


def get_settings():
    """Get the application settings instance."""
    return settings