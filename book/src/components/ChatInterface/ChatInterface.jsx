import React, { useState, useEffect, useRef } from 'react';
import { useSelection } from '../../contexts/SelectionContext';
import apiClient from '../../utils/api'; // Correct import path
import './ChatInterface.css';

/**
 * ChatInterface component for the AI-Native Book RAG Chatbot
 * Provides an interactive chat interface that connects to the backend API
 */
const ChatInterface = ({ apiUrl = 'https://abdullah017-humanoid-and-robotis-book.hf.space' }) => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState(null);
  const { selectedText, clearSelection } = useSelection();

  // Update the API client if the URL changes
  useEffect(() => {
    // Update API client with new URL if it changes
    if (apiClient && apiClient.setBaseUrl) {
      apiClient.setBaseUrl(apiUrl);
    }
  }, [apiUrl]);

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
      timestamp: new Date().toISOString(),
      selectedText: selectedText || null
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    try {
      // Call backend API to process query
      const data = await apiClient.processQuery(
        currentConversationId,
        inputText,
        selectedText || null
      );

      // Add AI response to messages
      const aiMessage = {
        id: data.id,
        text: data.response_text,
        sender: 'ai',
        timestamp: data.timestamp,
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
      clearSelection(); // Clear selected text after sending
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

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>AI Assistant for Humanoid Robotics</h3>
        <button
          className="clear-chat-btn"
          onClick={clearChat}
          disabled={messages.length === 0}
          title="Clear chat history"
        >
          Clear Chat
        </button>
      </div>

      {error && (
        <div className="chat-error">
          {error}
        </div>
      )}

      {selectedText && (
        <div className="selected-text-preview">
          <strong>Selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
        </div>
      )}

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <p>Hello! I'm your AI assistant for the Humanoid Robotics book.</p>
            <p>Ask me questions about ROS 2, Gazebo/Unity simulation, NVIDIA Isaac, or Vision-Language-Action systems.</p>
            <p>You can also select text on the page and ask questions about it!</p>
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

export default ChatInterface;