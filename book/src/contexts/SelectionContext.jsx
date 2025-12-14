import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

const SelectionContext = createContext();

export const useSelection = () => {
  const context = useContext(SelectionContext);
  if (!context) {
    throw new Error('useSelection must be used within a SelectionProvider');
  }
  return context;
};

export const SelectionProvider = ({ children }) => {
  const [selectedText, setSelectedText] = useState('');

  // Function to get selected text from the page
  const updateSelection = useCallback(() => {
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

  // Clear the selection
  const clearSelection = useCallback(() => {
    setSelectedText('');
    window.getSelection().removeAllRanges();
  }, []);

  useEffect(() => {
    // Add event listeners for text selection
    document.addEventListener('mouseup', updateSelection);
    document.addEventListener('keyup', (e) => {
      // Handle text selection with keyboard (Ctrl+A, etc.)
      if (e.key === 'Escape') {
        clearSelection();
      }
    });

    // Cleanup event listeners on unmount
    return () => {
      document.removeEventListener('mouseup', updateSelection);
      document.removeEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
          clearSelection();
        }
      });
    };
  }, [updateSelection, clearSelection]);

  const value = {
    selectedText,
    clearSelection
  };

  return (
    <SelectionContext.Provider value={value}>
      {children}
    </SelectionContext.Provider>
  );
};