"""
Qdrant client wrapper for the AI-Native Book RAG Chatbot application.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """
    Wrapper class for Qdrant client to handle vector storage operations
    for the AI-Native Book RAG Chatbot application.
    """

    def __init__(self, url: str, api_key: Optional[str] = None):
        """
        Initialize the Qdrant client wrapper.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
        """
        self.client = QdrantClient(url=url, api_key=api_key)
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
                    size=1536,  # Standard OpenAI embedding size
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