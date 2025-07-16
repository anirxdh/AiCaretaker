from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
from agent import agent_response, get_pending_followups
import os
import openai
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)

# OpenAI client will be initialized when needed

user_memories = {}
pending_followups = {}
emergency_states = {}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message")
        user_id = data.get("user_id")
        response = agent_response(user_input, user_id=user_id)
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full error in your terminal
        return jsonify({"error": str(e)}), 500

@app.route('/check-followups', methods=['POST'])
def check_followups():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        followups = get_pending_followups(user_id)
        return jsonify({"followups": followups})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_filepath = tmp_file.name
        
        try:
            # Transcribe using Whisper with new OpenAI API format
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(tmp_filepath, 'rb') as audio:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    language="en"  # Optional: specify language
                )
            
            transcription = response.text.strip()
            
            # Clean up the temporary file
            os.unlink(tmp_filepath)
            
            return jsonify({"transcription": transcription})
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_filepath):
                os.unlink(tmp_filepath)
            print(f"Whisper API error: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Transcribe endpoint error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5050)
