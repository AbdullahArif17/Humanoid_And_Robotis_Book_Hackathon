import React, { useState, useEffect, useCallback } from 'react';

/**
 * SelectionHandler component that manages text selection across the book
 * Provides context for selected text to be used by the chat interface
 */
const SelectionHandler = ({ children }) => {
  const [selectedText, setSelectedText] = useState('');

  // Function to get selected text from the page
  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection.toString().trim();

    // Only update if there's actual selected text
    if (text && text.length > 0) {
      // Limit to 500 characters to avoid sending too much text to the API
      const limitedText = text.length > 500 ? text.substring(0, 500) + '...' : text;
      setSelectedText(limitedText);
    } else {
      setSelectedText('');
    }
  }, []);

  useEffect(() => {
    // Add event listeners for text selection
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', (e) => {
      // Handle text selection with keyboard (Ctrl+A, etc.)
      if (e.key === 'Escape') {
        setSelectedText('');
      }
    });

    // Cleanup event listeners on unmount
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
          setSelectedText('');
        }
      });
    };
  }, [handleSelection]);

  // Function to clear the selected text
  const clearSelection = () => {
    setSelectedText('');
    window.getSelection().removeAllRanges();
  };

  return (
    <div className="selection-handler">
      {children}
      {selectedText && (
        <div className="selection-indicator">
          <span className="selected-text-preview">
            Selected: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
          </span>
          <button
            className="clear-selection-btn"
            onClick={clearSelection}
            title="Clear selection"
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
};

export default SelectionHandler;