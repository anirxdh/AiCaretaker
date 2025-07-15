// App.js
import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function UserSelection({ onSelect }) {
  return (
    <div className="user-selection">
      <h2>Who are you?</h2>
      <button className="user-selection-btn" onClick={() => onSelect("user_john")}>John</button>
      <button className="user-selection-btn" onClick={() => onSelect("user_mary")}>Mary</button>
    </div>
  );
}

function TypingAnimation() {
  return (
    <div className="care-chat-typing" aria-live="polite">
      <span>Agent is typing</span>
      <span className="care-chat-typing-dot">•</span>
      <span className="care-chat-typing-dot">•</span>
      <span className="care-chat-typing-dot">•</span>
    </div>
  );
}

function CareAssistantChat({ userId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (userId) {
      setLoading(true);
      fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "", user_id: userId }),
      })
        .then(res => res.json())
        .then(data => {
          setMessages([{ sender: "agent", text: data.response }]);
          setLoading(false);
        });
    }
  }, [userId]);

  useEffect(() => {
    if (!userId) return;
    const checkFollowups = async () => {
      try {
        const res = await fetch("/check-followups", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId }),
        });
        const data = await res.json();
        if (data.followups && data.followups.length > 0) {
          data.followups.forEach(followup => {
            setMessages(msgs => [...msgs, { sender: "agent", text: followup }]);
          });
        }
      } catch (error) {
        console.error("Error checking follow-ups:", error);
      }
    };
    checkFollowups();
    const interval = setInterval(checkFollowups, 10000);
    return () => clearInterval(interval);
  }, [userId]);

  useEffect(() => {
    // Scroll to bottom on new message
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setMessages(msgs => [...msgs, { sender: "user", text: input }]);
    setLoading(true);
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input, user_id: userId }),
    });
    const data = await res.json();
    setMessages(msgs => [...msgs, { sender: "agent", text: data.response }]);
    setInput("");
    setLoading(false);
  };

  return (
    <div className="app-root">
      <div className="care-chat-container">
        <h1 className="care-chat-title">ElderlyCare Assistant</h1>
        <div className="care-chat-area">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`care-chat-message-row ${msg.sender}`}
              style={{ justifyContent: msg.sender === "user" ? "flex-end" : "flex-start" }}
            >
              <div className={`care-chat-message ${msg.sender}`}>
                <div className="care-chat-message-label">
                  {msg.sender === "user" ? "You" : "Agent"}
                </div>
                <div>{msg.text}</div>
              </div>
            </div>
          ))}
          {loading && <TypingAnimation />}
          <div ref={chatEndRef} />
        </div>
        <form className="care-chat-form" onSubmit={handleSend} autoComplete="off">
          <input
            className="care-chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading}
            autoFocus
          />
          <button
            className="care-chat-send-btn"
            type="submit"
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default function App() {
  const [userId, setUserId] = useState(null);
  return (
    <div className="app-root">
      {!userId ? <UserSelection onSelect={setUserId} /> : <CareAssistantChat userId={userId} />}
    </div>
  );
}