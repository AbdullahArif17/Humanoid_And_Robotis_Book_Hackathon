import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../../utils/api';
import './PremiumChatInterface.css';

/**
 * Premium ChatInterface component for the AI-Native Book RAG Chatbot
 * Features a modern, professional design with enhanced UX
 */
const PremiumChatInterface = ({
  apiUrl = 'https://abdullah017-humanoid-and-robotis-book.hf.space',
  className = ''
}) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState(null);
  const [isOpen, setIsOpen] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to create a new conversation
  const createConversation = async () => {
    try {
      const data = await apiClient.createConversation();
      setConversationId(data.id);
      return data.id;
    } catch (err) {
      console.error('Error creating conversation:', err);
      setError('Failed to create conversation. Please try again.');
      return null;
    }
  };

  // Function to send a message to the backend
  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    // Use existing conversation ID or create a new one
    let currentConversationId = conversationId;
    if (!currentConversationId) {
      currentConversationId = await createConversation();
      if (!currentConversationId) return;
    }

    // Add user message to UI immediately
    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    try {
      // Call backend API to process query (without selected text)
      const data = await apiClient.processQuery(
        currentConversationId,
        inputText
      );

      // Add AI response to messages
      const aiMessage = {
        id: Date.now(),
        text: data.response,
        sender: 'ai',
        timestamp: new Date().toISOString(),
        sources: data.sources || []
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError(`Failed to get response: ${err.message}`);

      // Add error message to UI
      const errorMessage = {
        id: Date.now(),
        text: `Sorry, I encountered an error: ${err.message}`,
        sender: 'ai',
        timestamp: new Date().toISOString(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  // Handle key down in textarea
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Clear chat history
  const clearChat = async () => {
    if (conversationId) {
      try {
        await apiClient.deleteConversation(conversationId);
      } catch (err) {
        console.error('Error deleting conversation:', err);
        // Still clear the local state even if the API call fails
      }
    }
    setMessages([]);
    setConversationId(null);
    setError(null);
  };

  // Toggle chat visibility
  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  if (!isOpen) {
    return (
      <button
        className="chat-trigger-button"
        onClick={toggleChat}
        aria-label="Open AI Assistant"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6v-2h12v2zm0-3H6V6h12v2z"/>
        </svg>
      </button>
    );
  }

  return (
    <div className={`premium-chat-container ${className}`}>
      <div className="chat-header">
        <div className="chat-title-section">
          <h3>AI Assistant</h3>
          <p className="chat-subtitle">Physical AI & Humanoid Robotics</p>
        </div>
        <div className="chat-actions">
          <button
            className="clear-chat-btn"
            onClick={clearChat}
            disabled={messages.length === 0}
            title="Clear chat history"
          >
            Clear
          </button>
          <button
            className="close-chat-btn"
            onClick={toggleChat}
            title="Close chat"
          >
            ×
          </button>
        </div>
      </div>

      {error && (
        <div className="chat-error">
          {error}
        </div>
      )}

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h4>Hello! I'm your AI assistant for Humanoid Robotics.</h4>
            <p>Ask me questions about ROS 2, Gazebo/Unity simulation, NVIDIA Isaac, or Vision-Language-Action systems.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`chat-message ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
            >
              <div className="message-header">
                <span className="sender-name">
                  {message.sender === 'user' ? 'You' : 'AI Assistant'}
                </span>
                <span className="timestamp">
                  {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className="message-content">
                {message.text}

                {message.sender === 'ai' && message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <strong>Sources:</strong>
                    <ul>
                      {message.sources.map((source, index) => (
                        <li key={index}>
                          <a
                            href={`/docs/${source.section_path}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {source.title}
                          </a>
                          {source.confidence && (
                            <span className="confidence"> (Confidence: {(source.confidence * 100).toFixed(1)}%)</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {message.isError && (
                  <div className="error-indicator">
                    ⚠️ An error occurred processing this response
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="chat-message ai-message">
            <div className="message-header">
              <span className="sender-name">AI Assistant</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about humanoid robotics..."
          rows="3"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default PremiumChatInterface;