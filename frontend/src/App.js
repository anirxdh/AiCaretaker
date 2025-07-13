import React, { useState } from 'react';
import './App.css';
import ChatBox from './components/ChatBox';

function App() {
  return (
    <div className="App">
      <h1>Care Assistant</h1>
      <ChatBox />
    </div>
  );
}

export default App;