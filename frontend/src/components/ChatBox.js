import React, { useState } from 'react';

function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    const res = await fetch("http://localhost:5050/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input })
    });
    const data = await res.json();
    setMessages([...messages, { type: "user", text: input }, { type: "agent", text: data.response }]);
    setInput("");
  };

  return (
    <div>
      <div style={{ minHeight: 300, border: "1px solid #ccc", padding: 10 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ textAlign: msg.type === "user" ? "right" : "left" }}>
            <b>{msg.type === "user" ? "You" : "Agent"}:</b> {msg.text}
          </div>
        ))}
      </div>
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

export default ChatBox;