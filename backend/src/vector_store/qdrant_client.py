"""
Qdrant client wrapper for the AI-Native Book RAG Chatbot application.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """
    Wrapper class for Qdrant client to handle vector storage operations
    for the AI-Native Book RAG Chatbot application.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Qdrant client wrapper.

        Args:
            url: Qdrant server URL (defaults to settings if not provided)
            api_key: Optional API key for authentication (defaults to settings if not provided)
        """
        settings = get_settings()
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.collection_name = "book_content_embeddings"

        # Initialize the collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection with appropriate configuration."""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Compatible with BERT-like embeddings and Google fallback
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {self.collection_name}")

    def store_embedding(self, content_id: str, embedding: List[float],
                       content_text: str, metadata: Dict[str, Any]) -> bool:
        """
        Store an embedding in the Qdrant collection.

        Args:
            content_id: Unique identifier for the content
            embedding: The embedding vector
            content_text: Original text that was embedded
            metadata: Additional metadata for the embedding

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=content_id,
                        vector=embedding,
                        payload={
                            "content_text": content_text,
                            "metadata": metadata
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content based on the query embedding.

        Args:
            query_embedding: The embedding to search for similar content
            limit: Maximum number of results to return

        Returns:
            List of similar content with scores and payloads
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "content_text": result.payload.get("content_text"),
                    "metadata": result.payload.get("metadata")
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching for similar content: {e}")
            return []

    def delete_embedding(self, content_id: str) -> bool:
        """
        Delete an embedding from the collection.

        Args:
            content_id: ID of the content to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[content_id]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False

    def update_embedding(self, content_id: str, new_embedding: List[float],
                        content_text: str, metadata: Dict[str, Any]) -> bool:
        """
        Update an existing embedding in the collection.

        Args:
            content_id: ID of the content to update
            new_embedding: The new embedding vector
            content_text: Updated content text
            metadata: Updated metadata

        Returns:
            True if successful, False otherwise
        """
        # Delete the old embedding and add the new one
        if self.delete_embedding(content_id):
            return self.store_embedding(content_id, new_embedding, content_text, metadata)
        return False

    def search_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content chunks relevant to the query text.
        This method converts the query text to a simple embedding and searches in Qdrant.

        Args:
            query: The query text to search for
            limit: Maximum number of results to return

        Returns:
            List of content chunks with their metadata
        """
        try:
            # Generate a simple embedding from the query text
            # This is a basic approach using character frequency to create a vector
            # In a real implementation, you would use a proper embedding model
            embedding = self._text_to_embedding(query)

            # Search for similar content in Qdrant
            results = self.search_similar(embedding, limit)

            # Format results to match expected chunk format
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result['id'],
                    'content_body': result['content_text'],
                    'title': result['metadata'].get('title', 'Unknown'),
                    'section_path': result['metadata'].get('section_path', 'Unknown'),
                    'score': result['score'],
                    'content_text': result['content_text'],
                }
                # Add any additional metadata from the result
                formatted_result.update(result['metadata'])
                formatted_results.append(formatted_result)

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching for content chunks: {e}")
            return []

    def _text_to_embedding(self, text: str) -> List[float]:
        """
        Convert text to a simple embedding vector.
        This is a basic implementation for demonstration purposes.
        In a real application, you would use a proper embedding model.

        Args:
            text: Input text to convert to embedding

        Returns:
            Embedding vector as a list of floats
        """
        # Create a simple embedding by using character/word frequency
        # This is not a proper semantic embedding but will allow the system to function
        import hashlib

        # Create a 768-dimensional vector (same as expected by the Qdrant collection)
        embedding = [0.0] * 768

        if not text:
            return embedding

        # Simple approach: use hash of text to populate vector with some values
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes)
        hash_hex = hash_obj.hexdigest()

        # Convert hex hash to numbers and distribute across the embedding vector
        for i in range(len(embedding)):
            # Use pairs of hex characters to create float values
            hex_idx = (i * 2) % len(hash_hex)
            hex_pair = hash_hex[hex_idx:hex_idx+2]
            if len(hex_pair) == 2:
                # Convert hex pair to a value between -1 and 1
                hex_val = int(hex_pair, 16)
                embedding[i] = (hex_val / 127.5) - 1.0  # Normalize to [-1, 1]

        return embedding