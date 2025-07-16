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
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
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

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = event => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        setTranscribing(true);
        try {
          const res = await fetch("/transcribe", {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          if (data.transcription) {
            setInput(data.transcription);
          } else {
            console.error("Transcription failed:", data.error);
          }
        } catch (err) {
          console.error("Transcription error:", err);
          alert("Failed to transcribe audio. Please try again.");
        } finally {
          setTranscribing(false);
        }
        
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      alert("Could not access microphone. Please check your permissions.");
      console.error("Microphone access error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div className="app-root">
      <div className="care-chat-container">
        <h1 className="care-chat-title">Care Assistant</h1>
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
          {recording && (
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              Recording...
            </div>
          )}
          {transcribing && (
            <div className="transcribing-indicator">
              Transcribing your message...
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        <form className="care-chat-form" onSubmit={handleSend} autoComplete="off">
          <input
            className="care-chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={loading || recording || transcribing}
            autoFocus
          />
          {!recording ? (
            <button
              className="care-chat-mic-btn"
              type="button"
              onClick={startRecording}
              disabled={loading || transcribing}
              title="Record voice message"
            >
              🎤
            </button>
          ) : (
            <button
              className="care-chat-stop-btn"
              type="button"
              onClick={stopRecording}
              title="Stop recording"
            >
              ⏹️ Stop
            </button>
          )}
          <button
            className="care-chat-send-btn"
            type="submit"
            disabled={loading || !input.trim() || recording || transcribing}
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