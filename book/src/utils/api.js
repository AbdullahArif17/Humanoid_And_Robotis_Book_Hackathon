/**
 * API client module for the AI-Native Book RAG Chatbot
 * Handles communication with the backend API
 */

import { handleApiError, validateConversationId, validateContentId, validateQueryParams, validateInput, logError } from './errorHandler';

const DEFAULT_API_URL = 'https://abdullah017-humanoid-and-robotis-book.hf.space';

class ApiClient {
  constructor(apiUrl = DEFAULT_API_URL) {
    this.apiUrl = apiUrl;
  }

  /**
   * Create a new conversation
   * @param {Object} options - Options for creating conversation
   * @param {string} [options.userId] - Optional user identifier
   * @param {string} [options.title] - Optional conversation title
   * @returns {Promise<Object>} The created conversation
   */
  async createConversation({ userId = null, title = null } = {}) {
    try {
      // Validate inputs if needed
      if (title && title.length > 200) {
        throw new Error('Title is too long (maximum 200 characters)');
      }

      const response = await fetch(`${this.apiUrl}/api/v1/chat/conversations/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          title: title,
        }),
      });

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'createConversation', error);
      throw error;
    }
  }

  /**
   * Get a conversation by ID
   * @param {string} conversationId - The conversation ID
   * @returns {Promise<Object>} The conversation
   */
  async getConversation(conversationId) {
    try {
      // Validate conversation ID
      const validationErrors = validateConversationId(conversationId);
      if (validationErrors.length > 0) {
        throw new Error(`Invalid conversation ID: ${validationErrors.join(', ')}`);
      }

      const response = await fetch(`${this.apiUrl}/api/v1/chat/conversations/${conversationId}`);

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'getConversation', error);
      throw error;
    }
  }

  /**
   * Get conversations for a user
   * @param {string} userId - The user ID
   * @param {number} [skip=0] - Number of records to skip
   * @param {number} [limit=20] - Maximum number of records to return
   * @returns {Promise<Array>} List of conversations
   */
  async getConversationsByUser(userId, skip = 0, limit = 20) {
    try {
      if (!userId || typeof userId !== 'string') {
        throw new Error('User ID is required and must be a string');
      }

      if (typeof skip !== 'number' || skip < 0) {
        throw new Error('Skip must be a non-negative number');
      }

      if (typeof limit !== 'number' || limit < 1 || limit > 100) {
        throw new Error('Limit must be a number between 1 and 100');
      }

      const response = await fetch(
        `${this.apiUrl}/api/v1/chat/conversations/?user_id=${encodeURIComponent(userId)}&skip=${skip}&limit=${limit}`
      );

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'getConversationsByUser', error);
      throw error;
    }
  }

  /**
   * Process a query in a conversation
   * @param {string} conversationId - The conversation ID
   * @param {string} queryText - The user's query text
   * @param {string} [selectedText=null] - Optional selected text for context
   * @param {number} [contextWindow=5] - Number of surrounding chunks to include
   * @returns {Promise<Object>} The AI response
   */
  async processQuery(conversationId, queryText, selectedText = null, contextWindow = 5) {
    try {
      // Validate inputs
      const validationErrors = [
        ...validateConversationId(conversationId),
        ...validateInput(queryText)
      ];

      if (validationErrors.length > 0) {
        throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
      }

      if (typeof contextWindow !== 'number' || contextWindow < 1 || contextWindow > 20) {
        throw new Error('Context window must be a number between 1 and 20');
      }

      if (selectedText && selectedText.length > 5000) {
        throw new Error('Selected text is too long (maximum 5,000 characters)');
      }

      const response = await fetch(`${this.apiUrl}/api/v1/chat/conversations/${conversationId}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_text: queryText,
          selected_text: selectedText,
          context_window: contextWindow,
        }),
      });

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'processQuery', error);
      throw error;
    }
  }

  /**
   * Get chat history for a conversation
   * @param {string} conversationId - The conversation ID
   * @param {number} [limit=50] - Maximum number of messages to return
   * @returns {Promise<Array>} List of chat messages
   */
  async getChatHistory(conversationId, limit = 50) {
    try {
      // Validate conversation ID
      const validationErrors = validateConversationId(conversationId);
      if (validationErrors.length > 0) {
        throw new Error(`Invalid conversation ID: ${validationErrors.join(', ')}`);
      }

      if (typeof limit !== 'number' || limit < 1 || limit > 200) {
        throw new Error('Limit must be a number between 1 and 200');
      }

      const response = await fetch(
        `${this.apiUrl}/api/v1/chat/conversations/${conversationId}/history?limit=${limit}`
      );

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'getChatHistory', error);
      throw error;
    }
  }

  /**
   * Update conversation title
   * @param {string} conversationId - The conversation ID
   * @param {string} title - The new title
   * @returns {Promise<Object>} Success response
   */
  async updateConversationTitle(conversationId, title) {
    try {
      // Validate inputs
      const validationErrors = [
        ...validateConversationId(conversationId),
        ...validateInput(title)
      ];

      if (validationErrors.length > 0) {
        throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
      }

      if (title.length > 200) {
        throw new Error('Title is too long (maximum 200 characters)');
      }

      const response = await fetch(`${this.apiUrl}/api/v1/chat/conversations/${conversationId}/title`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title }),
      });

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'updateConversationTitle', error);
      throw error;
    }
  }

  /**
   * Delete a conversation
   * @param {string} conversationId - The conversation ID
   * @returns {Promise<Object>} Success response
   */
  async deleteConversation(conversationId) {
    try {
      // Validate conversation ID
      const validationErrors = validateConversationId(conversationId);
      if (validationErrors.length > 0) {
        throw new Error(`Invalid conversation ID: ${validationErrors.join(', ')}`);
      }

      const response = await fetch(`${this.apiUrl}/api/v1/chat/conversations/${conversationId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'deleteConversation', error);
      throw error;
    }
  }

  /**
   * Get content by ID
   * @param {string} contentId - The content ID
   * @returns {Promise<Object>} The content
   */
  async getContentById(contentId) {
    try {
      // Validate content ID
      const validationErrors = validateContentId(contentId);
      if (validationErrors.length > 0) {
        throw new Error(`Invalid content ID: ${validationErrors.join(', ')}`);
      }

      const response = await fetch(`${this.apiUrl}/api/v1/content/${contentId}`);

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'getContentById', error);
      throw error;
    }
  }

  /**
   * Get content by section path
   * @param {string} sectionPath - The section path
   * @returns {Promise<Object>} The content
   */
  async getContentBySectionPath(sectionPath) {
    try {
      if (!sectionPath || typeof sectionPath !== 'string') {
        throw new Error('Section path is required and must be a string');
      }

      if (sectionPath.length > 200) {
        throw new Error('Section path is too long (maximum 200 characters)');
      }

      const response = await fetch(`${this.apiUrl}/api/v1/content/path/${encodeURIComponent(sectionPath)}`);

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'getContentBySectionPath', error);
      throw error;
    }
  }

  /**
   * Search for content
   * @param {string} query - The search query
   * @param {string} [moduleId=null] - Optional module ID to limit search
   * @returns {Promise<Array>} List of matching content
   */
  async searchContent(query, moduleId = null) {
    try {
      // Validate inputs
      const validationErrors = validateQueryParams(query);
      if (validationErrors.length > 0) {
        throw new Error(`Validation failed: ${validationErrors.join(', ')}`);
      }

      if (moduleId && typeof moduleId !== 'string') {
        throw new Error('Module ID must be a string if provided');
      }

      let url = `${this.apiUrl}/api/v1/content/search/?query=${encodeURIComponent(query)}`;
      if (moduleId) {
        url += `&module_id=${encodeURIComponent(moduleId)}`;
      }

      const response = await fetch(url);

      if (!response.ok) {
        return await handleApiError(response);
      }

      return await response.json();
    } catch (error) {
      logError('ApiClient', 'searchContent', error);
      throw error;
    }
  }
}

// Create a default instance
const apiClient = new ApiClient();

export default apiClient;
export { ApiClient };