"""
OpenAI client wrapper for the AI-Native Book RAG Chatbot application.
Handles AI completions, embeddings, and other OpenAI API interactions.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """
    OpenAI client wrapper for handling API interactions.

    This class provides methods for:
    - Generating chat completions
    - Creating embeddings
    - Managing API configuration
    """

    def __init__(self):
        """Initialize the OpenAI client with configuration."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,  # Optional, for custom endpoints
            timeout=self.settings.openai_timeout,
            max_retries=self.settings.openai_max_retries,
        )
        self.model = self.settings.openai_model

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a completion using the OpenAI API.

        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API

        Returns:
            Generated completion text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )

            if stream:
                # For streaming, we'd need to handle differently
                # For now, return the first choice
                logger.warning("Streaming not fully implemented yet")
                return ""
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (lists of floats)
        """
        try:
            # OpenAI API limits batch size, so we may need to chunk large requests
            # The limit is typically 2048 texts per request
            batch_size = min(len(texts), 2048)
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = await self.client.embeddings.create(
                    model=self.settings.openai_embedding_model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def generate_completion_with_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a completion using the query and retrieved context chunks.

        Args:
            query: User's original query
            context_chunks: List of context chunks with source information
            system_prompt: Optional custom system prompt

        Returns:
            Generated completion with proper citations
        """
        if not system_prompt:
            system_prompt = (
                "You are an AI assistant for a technical book on Physical AI & Humanoid Robotics. "
                "Answer the user's question based on the provided context. "
                "Always cite your sources using the format [source_title, section_path]. "
                "If the answer cannot be found in the provided context, say so explicitly. "
                "Be concise but thorough in your responses, and maintain a professional tone."
            )

        # Format context chunks into a readable format
        context_text = "\n\n".join([
            f"Source: {chunk.get('title', 'Unknown')} ({chunk.get('section_path', 'Unknown')})\n"
            f"Content: {chunk.get('content_body', '')}"
            for chunk in context_chunks
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]

        return await self.generate_completion(messages)

    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is working by making a simple request.

        Returns:
            True if the API key is valid, False otherwise
        """
        try:
            # Make a simple request to validate the API key
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False


# Global instance for convenience
_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """
    Get the global OpenAI client instance.

    Returns:
        OpenAI client instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client