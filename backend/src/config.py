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

    # OpenAI settings
    openai_api_key: str = Field(alias="OPENAI_API_KEY")  # Required field
    openai_model: str = Field(default="gpt-4-turbo", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_EMBEDDING_MODEL")  # Model for embeddings
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")  # Optional, for custom endpoints
    openai_timeout: int = Field(default=30, alias="OPENAI_TIMEOUT")  # Request timeout in seconds
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")  # Number of retries for failed requests

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


def get_settings():
    """Get the application settings instance."""
    return settings