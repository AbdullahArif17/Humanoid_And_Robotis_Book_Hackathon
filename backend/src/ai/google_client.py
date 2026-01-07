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
            raise ImportError("Google Generative AI SDK is not installed. Please install with: pip install google-generativeai")

        self.settings = get_settings()

        # Check if Google API key is configured
        if not self.settings.google_api_key or self.settings.google_api_key == "your-google-api-key-here":
            logger.warning("Google API key not configured. Google AI features will not work until API key is set.")
            self.client = None
            return

        try:
            genai.configure(api_key=self.settings.google_api_key)

            # Try to use the configured model, fall back to alternatives if not available
            # Google AI API expects model names with "models/" prefix
            available_models = [
                self.settings.google_model,           # User-configured model
                "models/gemini-2.5-flash",           # Current recommended model
                "models/gemini-3-flash",             # Alternative model
                "models/gemini-2.5-flash-tts",       # Alternative model
                "models/gemini-robotics-er-1.5-preview", # Specialized model
            ]

            self.client = None
            successful_model = None

            for model_name in available_models:
                try:
                    self.model = model_name
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

                    # If we get here, the model initialization was successful
                    successful_model = model_name
                    logger.info(f"Successfully initialized Google AI client with model: {self.model}")
                    break
                except Exception as model_error:
                    logger.warning(f"Failed to initialize model {model_name}: {str(model_error)}")
                    self.client = None
                    continue

            if self.client is None:
                logger.error("Failed to initialize any Google AI model")
                raise Exception("No available Google AI model could be initialized")
            else:
                self.model = successful_model  # Ensure the model is set to the successful one

        except Exception as e:
            logger.error(f"Failed to initialize Google AI client: {str(e)}")
            logger.warning("Google AI client not available. Some features may not work.")
            self.client = None

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
            "You are an enthusiastic and helpful AI assistant for a technical book on Physical AI & Humanoid Robotics. "
            "Answer the user's question based on the provided context in a friendly, approachable manner. "
            "Always cite your sources using the format [source_title, section_path]. "
            "If the answer cannot be found in the provided context, say so politely and offer to help with related topics. "
            "Use a conversational tone that's welcoming to both beginners and experts. "
            "Include encouraging language and offer to provide more details if needed. "
            "Keep responses informative but approachable, and always maintain a positive, supportive attitude."
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
        if self.client is None:
            logger.error("Google AI client not initialized. Cannot generate completion.")
            return "AI service is not available. Please check that the API key is properly configured."

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
    ) -> str:
        """
        Generate a completion using the query and retrieved context chunks.

        Args:
            query: User's original query
            context_chunks: List of context chunks with source information
            system_prompt: Optional custom system prompt

        Returns:
            Generated completion text
        """
        if self.client is None:
            logger.error("Google AI client not initialized. Cannot generate completion with context.")
            # Return a default response indicating the service is unavailable
            return "AI service is not available. Please check that the API key is properly configured."

        # Format context chunks into a readable format with better structure
        context_sections = []
        for chunk in context_chunks:
            title = chunk.get('title', 'Unknown')
            section_path = chunk.get('section_path', 'Unknown')
            content_body = chunk.get('content_body', '')

            # Only include chunks with substantial content
            if content_body.strip():
                context_section = f"📚 Source: {title}\n📍 Section: {section_path}\n📖 Content: {content_body}"
                context_sections.append(context_section)

        # Combine context sections
        context_text = "\n\n" + "="*50 + "\nCONTEXT FROM BOOK:\n" + "="*50 + "\n\n"
        context_text += "\n\n" + "-"*30 + "\n".join(context_sections)
        context_text += "\n\n" + "="*50 + "\nANSWER THE FOLLOWING QUESTION USING THE CONTEXT ABOVE:\n" + "="*50

        # Create the full prompt with context
        full_prompt = f"""{context_text}\n\n🎯 Question: {query}

Please provide a helpful, friendly response that:
1. Directly answers the question using the context
2. Cites sources using [Title, Section] format
3. Offers additional helpful information if available
4. Maintains a positive, supportive tone"""

        try:
            # Use the model to generate a response
            # Since the Google client is synchronous, we run it in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.generate_content, full_prompt)

            return response.text
        except Exception as e:
            logger.error(f"Error generating completion with context: {str(e)}")
            raise

    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is working by making a simple request.

        Returns:
            True if the API key is valid, False otherwise
        """
        if self.client is None:
            logger.warning("Google AI client not initialized. Cannot validate API key.")
            return False

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
        try:
            _google_client = GoogleAIClient()
        except ImportError as e:
            logger.error(f"Failed to initialize Google AI client: {str(e)}")
            # Create a client instance even if initialization fails to avoid breaking the service
            _google_client = GoogleAIClient.__new__(GoogleAIClient)  # Create empty instance
            _google_client.settings = None
            _google_client.client = None
            _google_client.model = "fallback-model"
            _google_client.generation_config = None
    return _google_client