// App.js
import React, { useState, useEffect } from "react";

// User selection component
function UserSelection({ onSelect }) {
  return (
    <div style={{ textAlign: "center", marginTop: 100 }}>
      <h2>Who are you?</h2>
      <button onClick={() => onSelect("user_john")} style={{ margin: 10, padding: 10, fontSize: 18 }}>John</button>
      <button onClick={() => onSelect("user_mary")} style={{ margin: 10, padding: 10, fontSize: 18 }}>Mary</button>
    </div>
  );
}

// Chat UI component
function CareAssistantChat({ userId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // On mount (or when userId changes), get agent greeting
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

  // Check for follow-up messages every 10 seconds
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
          // Add each follow-up message to the chat
          data.followups.forEach(followup => {
            setMessages(msgs => [...msgs, { sender: "agent", text: followup }]);
          });
        }
      } catch (error) {
        console.error("Error checking follow-ups:", error);
      }
    };

    // Check immediately and then every 10 seconds
    checkFollowups();
    const interval = setInterval(checkFollowups, 10000);
    
    return () => clearInterval(interval);
  }, [userId]);

  // Handle sending a message
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
    <div style={{ maxWidth: 800, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Care Assistant</h1>
      <div style={{
        border: "1px solid #ccc",
        borderRadius: 8,
        padding: 24,
        minHeight: 200,
        background: "#fafbfc"
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            margin: "12px 0",
            textAlign: msg.sender === "user" ? "right" : "left"
          }}>
            <b>{msg.sender === "user" ? "You" : "Agent"}:</b> {msg.text}
          </div>
        ))}
        {loading && <div style={{ color: "#888" }}>Agent is typing...</div>}
      </div>
      <form onSubmit={handleSend} style={{ marginTop: 16, display: "flex" }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          style={{ flex: 1, fontSize: 16, padding: 8 }}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button type="submit" style={{ marginLeft: 8, fontSize: 16 }} disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
}

// Main App
export default function App() {
  const [userId, setUserId] = useState(null);

  return (
    <div>
      {!userId
        ? <UserSelection onSelect={setUserId} />
        : <CareAssistantChat userId={userId} />
      }
    </div>
  );
}