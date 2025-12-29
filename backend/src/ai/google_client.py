"""
Google AI client wrapper for the AI-Native Book RAG Chatbot application.
Handles Google AI completions, embeddings, and other Google AI API interactions.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel

from src.config import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    logger.warning("Google AI SDK not available. Install with: pip install google-generativeai")
    GOOGLE_AI_AVAILABLE = False


class GoogleAIClient:
    """
    Google AI client wrapper for handling API interactions.

    This class provides methods for:
    - Generating chat completions using Gemini
    - Managing API configuration
    """

    def __init__(self):
        """Initialize the Google AI client with configuration."""
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google GenAI SDK is not installed. Please install with: pip install google-genai")

        self.settings = get_settings()
        genai.configure(api_key=self.settings.google_api_key)

        self.model = self.settings.google_model
        self.generation_config = GenerationConfig(
            temperature=self.settings.google_temperature,
            max_output_tokens=self.settings.google_max_output_tokens,
            top_p=self.settings.google_top_p,
            top_k=self.settings.google_top_k,
        )

        # Initialize the generative model
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config,
            system_instruction=self._get_system_instruction()
        )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        NOTE: Google AI doesn't provide native embedding API, so we'll use a fallback
        method or recommend using a different embedding service.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (lists of floats)
        """
        logger.warning("Google AI does not provide native embeddings. Using fallback method.")

        # For now, we'll return dummy embeddings with the expected dimensions
        # In a real implementation, you might want to use Sentence Transformers,
        # OpenAI embeddings, or another embedding service

        # Assuming 768 dimensions for BERT-like embeddings as a fallback
        dummy_embedding = [0.0] * 768

        return [dummy_embedding for _ in texts]

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the model."""
        return (
            "You are an AI assistant for a technical book on Physical AI & Humanoid Robotics. "
            "Answer the user's question based on the provided context. "
            "Always cite your sources using the format [source_title, section_path]. "
            "If the answer cannot be found in the provided context, say so explicitly. "
            "Be concise but thorough in your responses, and maintain a professional tone."
        )

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a completion using the Google AI API.

        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            Generated completion text
        """
        try:
            # For Google AI, we'll use the non-chat model for completion since
            # chat models expect a conversation flow
            # Extract the last user message as the prompt
            last_user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break

            if not last_user_message:
                last_user_message = "Hello"

            # Create generation config with overrides if provided
            generation_config = self.generation_config
            if temperature is not None or max_tokens is not None:
                generation_config = GenerationConfig(
                    temperature=temperature or self.settings.google_temperature,
                    max_output_tokens=max_tokens or self.settings.google_max_output_tokens,
                    top_p=self.settings.google_top_p,
                    top_k=self.settings.google_top_k,
                )

            # Google AI's generate_content is synchronous, so we run it in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    last_user_message,
                    generation_config=generation_config
                )
            )

            return response.text

        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise

    async def generate_completion_with_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ):
        """
        Generate a completion using the query and retrieved context chunks.

        Args:
            query: User's original query
            context_chunks: List of context chunks with source information
            system_prompt: Optional custom system prompt

        Returns:
            Generated completion object with response text and sources
        """
        # Format context chunks into a readable format
        context_text = "\n\n".join([
            f"Source: {chunk.get('title', 'Unknown')} ({chunk.get('section_path', 'Unknown')})\n"
            f"Content: {chunk.get('content_body', '')}"
            for chunk in context_chunks
        ])

        # Create the full prompt with context
        full_prompt = f"""Context:\n{context_text}\n\nQuestion: {query}"""

        try:
            # Use the model to generate a response
            # Since the Google client is synchronous, we run it in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.generate_content, full_prompt)

            # For now, return a simple response object with text and empty sources
            # In a full implementation, you'd want to extract sources from the response
            class ResponseObj:
                def __init__(self, text, sources):
                    self.response_text = text
                    self.sources = sources

            # Extract sources from context_chunks to return as source information
            sources = [
                {
                    'title': chunk.get('title', 'Unknown'),
                    'section_path': chunk.get('section_path', 'Unknown'),
                    'confidence': chunk.get('score', 0.0)
                }
                for chunk in context_chunks
            ]

            return ResponseObj(response_text=response.text, sources=sources)
        except Exception as e:
            logger.error(f"Error generating completion with context: {str(e)}")
            raise

    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is working by making a simple request.

        Returns:
            True if the API key is valid, False otherwise
        """
        try:
            # Make a simple request to validate the API key
            # Since the Google client is synchronous, we run it in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.generate_content, "Say hello")
            return len(response.text) > 0
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False


# Global instance for convenience
_google_client: Optional[GoogleAIClient] = None


def get_google_client() -> GoogleAIClient:
    """
    Get the global Google AI client instance.

    Returns:
        Google AI client instance
    """
    global _google_client
    if _google_client is None:
        _google_client = GoogleAIClient()
    return _google_client