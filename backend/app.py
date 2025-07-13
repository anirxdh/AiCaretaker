from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.4)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=(
        "You are a friendly AI assistant for elderly care. "
        "You remember the user's name and previous messages in this conversation. "
        "Be helpful and conversational.\n"
        "Conversation so far:\n{chat_history}\n"
        "User: {input}\n"
        "Assistant:"
    )
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def get_agent_response(message):
    return chain.run(input=message)

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message")
        response = get_agent_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full error in your terminal
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5050)
