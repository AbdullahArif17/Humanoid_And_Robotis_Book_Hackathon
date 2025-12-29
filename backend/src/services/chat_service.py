"""
ChatService for the AI-Native Book RAG Chatbot application.
Handles chat operations, query processing, and response generation.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.models.conversation import Conversation
from src.models.user_query import UserQuery
from src.models.chatbot_response import ChatbotResponse
from src.models.book_content import BookContent
from src.vector_store.qdrant_client import QdrantClientWrapper
from src.ai.google_client import GoogleAIClient
from src.utils.exceptions import ValidationError, DatabaseError, RAGError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ChatService:
    """
    Service class for handling chat operations.

    This service provides methods for:
    - Creating and managing conversations
    - Processing user queries
    - Generating AI responses using RAG
    - Storing and retrieving chat history
    """

    def __init__(self, db_session: Session, qdrant_client: QdrantClientWrapper, google_client: GoogleAIClient):
        """
        Initialize the ChatService with required dependencies.

        Args:
            db_session: SQLAlchemy database session
            qdrant_client: Qdrant client wrapper for vector search
            google_client: Google AI client for AI completions
        """
        self.db_session = db_session
        self.qdrant_client = qdrant_client
        self.google_client = google_client

    def create_conversation(self, user_id: Optional[str] = None, title: Optional[str] = None) -> Conversation:
        """
        Create a new conversation.

        Args:
            user_id: Optional user identifier
            title: Optional conversation title

        Returns:
            Created Conversation instance
        """
        conversation = Conversation(
            id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),  # Add required session_id
            user_id=user_id,
            title=title or "New Conversation",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        try:
            self.db_session.add(conversation)
            self.db_session.commit()
            self.db_session.refresh(conversation)

            logger.info(f"Created new conversation: {conversation.id}")
            return conversation
        except IntegrityError as e:
            self.db_session.rollback()
            logger.error(f"Database integrity error creating conversation: {str(e)}")
            raise DatabaseError(f"Failed to create conversation: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error creating conversation: {str(e)}")
            raise DatabaseError(f"Unexpected error creating conversation: {str(e)}")

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by its ID.

        Args:
            conversation_id: Unique identifier of the conversation

        Returns:
            Conversation instance or None if not found
        """
        try:
            conversation = self.db_session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            return conversation
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
            raise DatabaseError(f"Error retrieving conversation: {str(e)}")

    def get_conversations_by_user(self, user_id: str, skip: int = 0, limit: int = 20) -> List[Conversation]:
        """
        Retrieve conversations for a specific user with pagination.

        Args:
            user_id: User identifier
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of Conversation instances
        """
        try:
            conversations = self.db_session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
            return conversations
        except Exception as e:
            logger.error(f"Error retrieving conversations for user {user_id}: {str(e)}")
            raise DatabaseError(f"Error retrieving conversations: {str(e)}")

    def process_query(
        self,
        conversation_id: str,
        query_text: str,
        selected_text: Optional[str] = None,
        context_window: int = 5
    ) -> ChatbotResponse:
        """
        Process a user query and generate an AI response using RAG.

        Args:
            conversation_id: Unique identifier of the conversation
            query_text: User's query text
            selected_text: Optional selected text for context
            context_window: Number of surrounding chunks to include

        Returns:
            Generated ChatbotResponse instance
        """
        # Validate inputs
        if not query_text or not query_text.strip():
            raise ValidationError("Query text cannot be empty")

        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValidationError(f"Conversation with ID '{conversation_id}' does not exist")

        try:
            # Create user query record
            user_query = UserQuery(
                id=str(uuid.uuid4()),
                session_id=conversation.session_id,  # Use the conversation's session_id
                conversation_id=conversation_id,
                query_text=query_text,
                selected_text=selected_text,
                query_type="full_book" if not selected_text else "selected_text",  # Set appropriate query type
                timestamp=datetime.utcnow()
            )
            self.db_session.add(user_query)
            self.db_session.flush()  # Flush to get the ID without committing

            # Perform semantic search in Qdrant
            search_text = query_text
            if selected_text:
                # Combine query and selected text for more targeted search
                search_text = f"{query_text} {selected_text}"

            relevant_chunks = self.qdrant_client.search_chunks(
                query=search_text,
                limit=context_window * 2  # Get more than needed for better selection
            )

            # Filter chunks to ensure they're relevant
            filtered_chunks = []
            for chunk in relevant_chunks:
                # Simple relevance check - could be enhanced with more sophisticated filtering
                if len(chunk.get('content_body', '')) > 10:  # At least 10 characters
                    filtered_chunks.append(chunk)

            # Limit to the exact context window
            context_chunks = filtered_chunks[:context_window]

            # Generate AI response using the context
            ai_response_text = self.google_client.generate_completion_with_context(
                query=query_text,
                context_chunks=context_chunks
            )

            # Create chatbot response record
            chatbot_response = ChatbotResponse(
                id=str(uuid.uuid4()),
                query_id=user_query.id,
                response_text=ai_response_text,
                sources=[{
                    'title': chunk.get('title'),
                    'section_path': chunk.get('section_path'),
                    'confidence': chunk.get('score', 0.0)
                } for chunk in context_chunks],
                timestamp=datetime.utcnow()
            )
            self.db_session.add(chatbot_response)
            self.db_session.commit()

            # Update conversation timestamp
            conversation.updated_at = datetime.utcnow()
            self.db_session.commit()

            logger.info(f"Processed query for conversation {conversation_id}, response: {chatbot_response.id}")
            return chatbot_response

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error processing query: {str(e)}")
            raise RAGError(f"Error processing query: {str(e)}")

    def get_chat_history(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a conversation.

        Args:
            conversation_id: Unique identifier of the conversation
            limit: Maximum number of messages to return

        Returns:
            List of chat messages with user queries and AI responses
        """
        try:
            # Join UserQuery and ChatbotResponse tables to get the full chat history
            history = self.db_session.query(UserQuery, ChatbotResponse).outerjoin(
                ChatbotResponse, UserQuery.id == ChatbotResponse.query_id
            ).filter(
                UserQuery.conversation_id == conversation_id
            ).order_by(UserQuery.timestamp.asc()).limit(limit).all()

            chat_history = []
            for user_query, chatbot_response in history:
                message_pair = {
                    'user_query': {
                        'id': user_query.id,
                        'text': user_query.query_text,
                        'selected_text': user_query.selected_text,
                        'timestamp': user_query.timestamp.isoformat() if user_query.timestamp else None
                    },
                    'ai_response': None
                }

                if chatbot_response:
                    message_pair['ai_response'] = {
                        'id': chatbot_response.id,
                        'text': chatbot_response.response_text,
                        'sources': chatbot_response.sources,
                        'timestamp': chatbot_response.timestamp.isoformat() if chatbot_response.timestamp else None
                    }

                chat_history.append(message_pair)

            return chat_history
        except Exception as e:
            logger.error(f"Error retrieving chat history for conversation {conversation_id}: {str(e)}")
            raise DatabaseError(f"Error retrieving chat history: {str(e)}")

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update the title of a conversation.

        Args:
            conversation_id: Unique identifier of the conversation
            title: New title for the conversation

        Returns:
            True if update was successful, False if conversation not found
        """
        if not title or not title.strip():
            raise ValidationError("Title cannot be empty")

        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        try:
            conversation.title = title
            conversation.updated_at = datetime.utcnow()
            self.db_session.commit()

            logger.info(f"Updated conversation title: {conversation_id}")
            return True
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error updating conversation title {conversation_id}: {str(e)}")
            raise DatabaseError(f"Error updating conversation title: {str(e)}")

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all associated queries and responses.

        Args:
            conversation_id: Unique identifier of the conversation

        Returns:
            True if deletion was successful, False if conversation not found
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False

        try:
            # Delete associated user queries and chatbot responses
            # Due to foreign key constraints, we need to delete child records first
            self.db_session.query(ChatbotResponse).filter(
                ChatbotResponse.query_id.in_(
                    self.db_session.query(UserQuery.id).filter(
                        UserQuery.conversation_id == conversation_id
                    )
                )
            ).delete(synchronize_session=False)

            # Delete user queries
            self.db_session.query(UserQuery).filter(
                UserQuery.conversation_id == conversation_id
            ).delete(synchronize_session=False)

            # Delete the conversation
            self.db_session.delete(conversation)
            self.db_session.commit()

            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            raise DatabaseError(f"Error deleting conversation: {str(e)}")