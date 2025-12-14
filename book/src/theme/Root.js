import React from 'react';
import { SelectionProvider } from '../contexts/SelectionContext';

export default function Root({ children }) {
  return <SelectionProvider>{children}</SelectionProvider>;
}