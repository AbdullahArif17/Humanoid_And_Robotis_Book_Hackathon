import React, { useState, useEffect, useRef } from 'react';
import apiClient from '../../utils/api';
import './EmbeddedChatInterface.css';

/**
 * Embedded ChatInterface component for the AI-Native Book RAG Chatbot
 * Designed to be embedded within a container on the homepage
 */
const EmbeddedChatInterface = ({
	apiUrl = 'https://abdullah017-humanoid-and-robotis-book.hf.space',
	className = ''
}) => {
	const [messages, setMessages] = useState([]);
	const [inputText, setInputText] = useState('');
	const [isLoading, setIsLoading] = useState(false);
	const [conversationId, setConversationId] = useState(null);
	const [error, setError] = useState(null);

	const messagesEndRef = useRef(null);
	const textareaRef = useRef(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

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

	const sendMessage = async () => {
		if (!inputText.trim() || isLoading) return;

		let currentConversationId = conversationId;
		if (!currentConversationId) {
			currentConversationId = await createConversation();
			if (!currentConversationId) return;
		}

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
			const data = await apiClient.processQuery(
				currentConversationId,
				inputText
			);

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

	const handleSubmit = (e) => {
		e.preventDefault();
		sendMessage();
	};

	const handleKeyDown = (e) => {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			sendMessage();
		}
	};

	const clearChat = async () => {
		if (conversationId) {
			try {
				await apiClient.deleteConversation(conversationId);
			} catch (err) {
				console.error('Error deleting conversation:', err);
			}
		}
		setMessages([]);
		setConversationId(null);
		setError(null);
	};

	return (
		<div className={`embedded-chat-container ${className}`}>
			<div className="embedded-chat-header">
				<div className="embedded-chat-title">
					<span className="chat-icon">🤖</span>
					<div>
						<h3>AI Assistant</h3>
						<p>Ask me anything about the book!</p>
					</div>
				</div>
				<button
					className="embedded-clear-btn"
					onClick={clearChat}
					disabled={messages.length === 0}
					title="Clear chat"
				>
					🗑️ Clear
				</button>
			</div>

			{error && (
				<div className="embedded-chat-error">
					⚠️ {error}
				</div>
			)}

			<div className="embedded-chat-messages">
				{messages.length === 0 ? (
					<div className="embedded-chat-welcome">
						<div className="welcome-icon">👋</div>
						<h4>Welcome! I'm your AI assistant</h4>
						<p>Ask me about ROS 2, simulation, NVIDIA Isaac, or Vision-Language-Action systems. I'm powered by RAG to give you accurate answers from the book!</p>
						<div className="welcome-suggestions">
							<span>Try asking:</span>
							<button onClick={() => setInputText("What is ROS 2?")}>What is ROS 2?</button>
							<button onClick={() => setInputText("How does Gazebo simulation work?")}>How does Gazebo work?</button>
						</div>
					</div>
				) : (
					messages.map((message) => (
						<div
							key={message.id}
							className={`embedded-message ${message.sender === 'user' ? 'user' : 'assistant'}`}
						>
							<div className="message-avatar">
								{message.sender === 'user' ? '👤' : '🤖'}
							</div>
							<div className="message-bubble">
								<div className="message-text">{message.text}</div>
								{message.sender === 'ai' && message.sources && message.sources.length > 0 && (
									<div className="message-sources">
										<span>📚 Sources:</span>
										{message.sources.map((source, index) => (
											<a key={index} href={`/docs/${source.section_path}`}>
												{source.title}
											</a>
										))}
									</div>
								)}
							</div>
						</div>
					))
				)}
				{isLoading && (
					<div className="embedded-message assistant">
						<div className="message-avatar">🤖</div>
						<div className="message-bubble">
							<div className="typing-dots">
								<span></span>
								<span></span>
								<span></span>
							</div>
						</div>
					</div>
				)}
				<div ref={messagesEndRef} />
			</div>

			<form className="embedded-chat-input" onSubmit={handleSubmit}>
				<textarea
					ref={textareaRef}
					value={inputText}
					onChange={(e) => setInputText(e.target.value)}
					onKeyDown={handleKeyDown}
					placeholder="Type your question here..."
					rows="2"
					disabled={isLoading}
				/>
				<button
					type="submit"
					disabled={!inputText.trim() || isLoading}
					className="embedded-send-btn"
				>
					{isLoading ? '...' : '➤'}
				</button>
			</form>
		</div>
	);
};

export default EmbeddedChatInterface;
