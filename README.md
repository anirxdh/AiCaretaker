# Elderly Care Agent

AI-powered health assistant for elderly users with appointment booking, health monitoring, and emergency response capabilities.

## Project Structure

```
elderly-agent/
├── backend/          # Flask API server
└── frontend/         # React web application
```

## Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the `backend/` directory with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   GOOGLE_APPLICATION_CREDENTIALS=path_to_your_service_account_key.json
   ```

4. **Run the Flask server:**
   ```bash
   python app.py
   ```
   The backend will start on `http://localhost:5050`

## Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   The frontend will start on `http://localhost:3000`

## Running the Full Application

1. **Terminal 1 - Start Backend:**
   ```bash
   cd backend
   python app.py
   ```

2. **Terminal 2 - Start Frontend:**
   ```bash
   cd frontend
   npm start
   ```

3. **Open your browser:**
   Navigate to `http://localhost:3000`

## Features

- **Health Monitoring**: Track vitals, food intake, and medical records via RAG
- **Appointment Booking**: Book doctor appointments with calendar integration
- **Voice Input**: Record and transcribe voice messages using Whisper API
- **Emergency Response**: Automated 911 calling and emergency contact notification
- **Follow-up Reminders**: Automated check-ins after health concerns

## API Endpoints

- `POST /chat` - Send chat messages to the AI agent
- `POST /check-followups` - Check for pending follow-up messages
- `POST /transcribe` - Transcribe audio files to text

## Technologies

**Backend:**
- Flask (Python web framework)
- LangChain (AI agent framework)
- OpenAI GPT-4 (LLM)
- Pinecone (Vector database)
- Google Calendar API (Appointment scheduling)

**Frontend:**
- React 18
- Modern CSS with dark theme
- MediaRecorder API (Voice recording)
- Fetch API (HTTP requests)