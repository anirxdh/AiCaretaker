from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
from agent import agent_response, get_pending_followups

app = Flask(__name__)
CORS(app)

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

if __name__ == '__main__':
    app.run(port=5050)
