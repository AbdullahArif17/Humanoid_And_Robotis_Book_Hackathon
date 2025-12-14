/**
 * Error handling utilities for the AI-Native Book RAG Chatbot
 */

/**
 * Standard error class for API errors
 */
export class ApiError extends Error {
  constructor(message, status, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Handle API errors consistently
 * @param {Response} response - The fetch response
 * @returns {Promise<Object>} Error details
 */
export const handleApiError = async (response) => {
  let errorData = { detail: `HTTP error! status: ${response.status}` };

  try {
    errorData = await response.json();
  } catch (e) {
    // If response is not JSON, use default error message
    errorData.detail = `HTTP error! status: ${response.status}`;
  }

  throw new ApiError(errorData.detail, response.status, errorData);
};

/**
 * Validate user input
 * @param {string} input - The input to validate
 * @returns {Array} Array of validation errors
 */
export const validateInput = (input) => {
  const errors = [];

  if (!input || input.trim().length === 0) {
    errors.push('Input cannot be empty');
  }

  if (input.length > 10000) {
    errors.push('Input is too long (maximum 10,000 characters)');
  }

  return errors;
};

/**
 * Validate conversation ID
 * @param {string} conversationId - The conversation ID to validate
 * @returns {Array} Array of validation errors
 */
export const validateConversationId = (conversationId) => {
  const errors = [];

  if (!conversationId || typeof conversationId !== 'string') {
    errors.push('Conversation ID is required and must be a string');
  }

  // Basic UUID validation
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(conversationId)) {
    errors.push('Invalid conversation ID format');
  }

  return errors;
};

/**
 * Validate content ID
 * @param {string} contentId - The content ID to validate
 * @returns {Array} Array of validation errors
 */
export const validateContentId = (contentId) => {
  const errors = [];

  if (!contentId || typeof contentId !== 'string') {
    errors.push('Content ID is required and must be a string');
  }

  // Basic UUID validation
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(contentId)) {
    errors.push('Invalid content ID format');
  }

  return errors;
};

/**
 * Validate query parameters
 * @param {string} query - The search query
 * @param {number} [limit] - The limit parameter
 * @returns {Array} Array of validation errors
 */
export const validateQueryParams = (query, limit = null) => {
  const errors = [];

  if (!query || query.trim().length === 0) {
    errors.push('Query cannot be empty');
  }

  if (query.length > 1000) {
    errors.push('Query is too long (maximum 1,000 characters)');
  }

  if (limit !== null && (typeof limit !== 'number' || limit < 1 || limit > 1000)) {
    errors.push('Limit must be a number between 1 and 1000');
  }

  return errors;
};

/**
 * Log error with consistent format
 * @param {string} component - Component name
 * @param {string} operation - Operation name
 * @param {Error} error - Error object
 */
export const logError = (component, operation, error) => {
  console.error(`[${component}] ${operation}:`, {
    message: error.message,
    name: error.name,
    stack: error.stack,
    ...(error.status && { status: error.status }),
    ...(error.detail && { detail: error.detail })
  });
};

/**
 * Format error message for display
 * @param {Error} error - Error object
 * @returns {string} Formatted error message
 */
export const formatErrorMessage = (error) => {
  if (error instanceof ApiError) {
    return `API Error: ${error.message}`;
  }

  if (error instanceof TypeError && error.message.includes('fetch')) {
    return 'Network error: Unable to connect to the server. Please check your connection.';
  }

  return `Error: ${error.message || 'An unexpected error occurred'}`;
};

/**
 * Error boundary component for React
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    logError('ErrorBoundary', 'Error caught', error);
    console.error('Error info:', errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong.</h2>
          <p>{formatErrorMessage(this.state.error)}</p>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}