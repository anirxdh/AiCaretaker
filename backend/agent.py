import os
from typing import List
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import pinecone
import datetime
import re
import dateparser
import threading
from langchain.agents import initialize_agent, Tool

# --- System prompt for all agent responses ---
SYSTEM_PROMPT = """
You are an intelligent and empathetic AI health assistant named CareMate, designed to help elderly users manage their daily wellbeing. Your goal is to assist users in a calm, human-like, and emotionally supportive manner, using only the tools and context provided to you.

IMPORTANT: You are talking to {name}. Always address them by their name when appropriate. Use their name naturally in your responses for empathy, emphasis, or clarification.

Your primary responsibility is to interpret the user's message, determine the intent or concern, and respond accordingly using available tools and facts.

You MUST NOT hallucinate or guess. If you don't know something, say so politely and try to help the user using what is available to you.

---

### 🛠️ Tools Available to You:

1. **RAG Query Tool** – `get_rag_context(user_id, metadata_filter)`
   - Use this to retrieve relevant health information for a user from one of three data types:
     - `"food"`: diet or meal logs
     - `"vitals"`: heart rate, blood pressure, step count, etc.
     - `"medical_record"`: diagnoses, prior ER visits, prescribed medications
   - Always include `"user_id"` and filter by `"data_type"` and (optionally) `"date"`
   - Example: `get_rag_context("user_mary", {{"data_type": "food"}})`

2. **Current Date Tool** – `get_current_date()`
   - Use this to understand the current date/time when needed for reasoning

3. **User Name Tool** – `get_user_name()`
   - Use this to get the current user's name: {name}
   - Always use this name when addressing the user

---

### 🤖 Behavior Guidelines:

- Always speak with **empathy** and warmth
  - Use reassuring phrases like "I understand", "Don't worry", or "I'm here with you"
  - Be gentle and caring in your tone
- Always analyze the user's message carefully and ask clarifying questions if needed
- Personalize all responses using the user's name when available
- Vary your responses. Do **NOT** sound robotic or repetitive
- Only use tools when required. If information is not available, respond politely and explain that

---

### 🧠 Health Evaluation Workflow:

1. **Symptom Detection** – decide if the issue seems **mild** or **serious**
2. **Data Retrieval via RAG** – gather `"food"`, `"vitals"`, and `"medical_record"` data
3. **Respond Based on Severity**
   - **Mild:** explicitly state "this appears to be mild" or "this is a mild issue", give suggestion, say you will check back in 5 minutes, and include a reminder
   - **Serious:** explicitly state "this may be serious" or "this is concerning", ask whether to call 911 and contacts; if user agrees, simulate the emergency actions
4. **If unsure:** ask follow-up questions

---

### 📝 Examples:

Mild example:
```
User: I feel dizzy and light-headed this morning.
Agent Process:
  get_rag_context("user_mary", {{"data_type": "food"}})  → skipped dinner
  get_rag_context("user_mary", {{"data_type": "vitals"}}) → normal heart rate
Agent Response:
  Based on your recent meals, vitals, and medical history, this appears to be a mild issue. Try eating something light and drinking water. I'll check back with you in 5 minutes. (Reminder: Follow up with user in 5 minutes)
```

Serious example:
```
User: My chest feels tight and I'm struggling to breathe.
Agent Process:
  get_rag_context("user_john", {{"data_type": "medical_record"}}) → history of arrhythmia
Agent Response:
  Given your symptoms and medical history, this may be serious. Would you like me to call 911 and notify your emergency contacts?
  (If user says yes → Calling 911... Sharing location... Notifying emergency contact...)
```

---

### ❗Restrictions:
- Never fabricate information or make assumptions
- Always explain when something is unknown or unavailable
- Only answer based on retrieved facts and current date
- Always prioritize safety, empathy, and clarity

(Current date: {date}, User: {name})
"""

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
# Create a dictionary to store separate memory instances for each user
user_memories = {}
# Store pending follow-up messages for each user
pending_followups = {}
emergency_states = {}

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
if PINECONE_API_KEY and PINECONE_ENV:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    INDEX_NAME = "elderly-health-agent"
    index = pinecone.Index(INDEX_NAME)
else:
    index = None

def get_current_date(query=None):
    if not query or query.strip().lower() in ["today", "date", "current date", "day"]:
        dt_obj = datetime.date.today()
    else:
        dt_obj = dateparser.parse(query, settings={'RELATIVE_BASE': datetime.datetime.now()})
        if not dt_obj:
            dt_obj = datetime.date.today()
    # Format as: Monday, July 14, 2025
    return dt_obj.strftime("%A, %B %d, %Y")

def get_rag_context_tool(query, user_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        index_name="elderly-health-agent"
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,  # Reduce from 5 to 2
            "filter": {"user_id": user_id}
        }
    )
    docs = retriever.get_relevant_documents(query)
    if docs:
        return docs[0].page_content
    return "No relevant health data found."

def get_user_name(user_id: str):
    """Extract user name from user_id"""
    if user_id and user_id.startswith("user_"):
        return user_id.replace("user_", "").capitalize()
    return "User"

def build_tools(user_id):
    name = get_user_name(user_id)
    
    return [
        Tool(
            name="get_rag_context",
            func=lambda q: get_rag_context_tool(q, user_id=user_id),
            description="Use this to answer any question about the user's health, food, vitals, or medical records."
        ),
        Tool(
            name="get_current_date",
            func=get_current_date,
            description="Use this to answer any question about today's date or current time."
        ),
        Tool(
            name="get_user_name",
            func=lambda x: name,
            description="Get the current user's name."
        )
    ]

def schedule_followup(user_id: str, user_name: str = None):
    def followup():
        name = user_name or user_id.replace("user_", "").capitalize()
        followup_message = f"Hi {name}, it's been 5 minutes since you mentioned feeling unwell. How are you feeling now? Are your symptoms better, worse, or the same?"
        print(f"[FOLLOW-UP] Checking in with {user_id} after 5 minutes.")
        
        # Store the follow-up message for the user
        if user_id not in pending_followups:
            pending_followups[user_id] = []
        pending_followups[user_id].append(followup_message)
        
    timer = threading.Timer(300, followup)  # 300 seconds = 5 minutes
    timer.start()

def get_pending_followups(user_id: str):
    """Get and clear pending follow-up messages for a user"""
    if user_id in pending_followups and pending_followups[user_id]:
        messages = pending_followups[user_id].copy()
        pending_followups[user_id] = []  # Clear the messages
        return messages
    return []

def agent_response(message: str, user_id: str = None) -> str:
    print(f"[DEBUG] Incoming message: '{message}' | user_id: {user_id}")
    name = get_user_name(user_id)
    today = datetime.date.today().strftime("%B %d, %Y")
    tools = build_tools(user_id)
    system_prompt = SYSTEM_PROMPT.format(name=name, date=today)
    
    # Get or create memory for this specific user
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print(f"[DEBUG] Created new memory for user: {user_id}")
    memory = user_memories[user_id]
    print(f"[DEBUG] Using existing memory for user: {user_id}")
    print(f"[DEBUG] Current memory buffer length: {len(memory.buffer) if memory.buffer else 0}")
    
    # If this is a new session (empty message), clear the memory and start fresh
    if not message.strip():
        print(f"[DEBUG] New session detected for {name}, clearing memory")
        memory.clear()
        user_memories[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory = user_memories[user_id]
        # Clear emergency state for new session
        if user_id in emergency_states:
            del emergency_states[user_id]
    
    agent = initialize_agent(
        tools,
        llm,
        agent="openai-functions",
        verbose=True,
        memory=memory,
        system_prompt=system_prompt
    )
    # Greeting logic: only if memory is empty and last AI message is not a greeting
    if not memory.buffer or (memory.buffer and not any('how are you feeling today' in m.content.lower() for m in memory.buffer if hasattr(m, 'content'))):
        print(f"[DEBUG] Sending greeting to {name}")
        memory.save_context({"input": "system"}, {"output": f"You are talking to {name}."})
        greeting = f"Hello {name}, how are you feeling today?"
        memory.save_context({"input": message}, {"output": greeting})
        print(f"[DEBUG] Memory buffer after greeting: {memory.buffer}")
        return greeting
    print(f"[DEBUG] Using normal conversation with memory. Memory buffer: {memory.buffer}")
    
    # Add name context to memory if not already present
    name_context = f"Remember, you are talking to {name}."
    memory.save_context({"input": "context"}, {"output": name_context})
    
    # Check if this is a response to emergency question BEFORE running agent
    previous_messages = memory.buffer if memory.buffer else []
    previous_has_emergency_question = any("emergency contact" in str(msg.content).lower() or "911" in str(msg.content).lower() for msg in previous_messages if hasattr(msg, 'content'))
    
    # Check if this is a response to emergency question
    is_emergency_response = previous_has_emergency_question and ("yes" in message.lower() or "no" in message.lower())
    
    if is_emergency_response:
        if "yes" in message.lower():
            print(f"[EMERGENCY] Calling 911 and emergency contacts for {name}")
            print(f"[EMERGENCY] Sharing location: 123 Main Street, Apartment 4B")
            print(f"[EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123)")
            
            emergency_response = f"I've called 911 and contacted your emergency contacts. Don't worry, {name}, I'm here with you. Help is on the way.\n\nWhile we wait for emergency services to arrive, try to stay calm and comfortable. Take slow, deep breaths. If you're able, sit or lie down in a comfortable position. I'll stay with you until help arrives."
            return emergency_response
        elif "no" in message.lower():
            # Convert serious to mild and add follow-up
            print(f"[INFO] User declined emergency services for {name}, converting to mild with follow-up")
            followup_msg = f"I understand you don't want emergency services right now. I'll check back with you in 5 minutes to see how you're feeling. (Reminder: Follow up with {name} in 5 minutes)"
            schedule_followup(user_id if user_id else "unknown_user", name)
            print(f"Checking the user {name} after 5 mins")
            return followup_msg
    
    response = agent.run(message)
    print(f"[DEBUG] Agent response: {response}")
    
    # Detect agent's conclusion about severity and trigger appropriate actions
    response_lower = response.lower()
    
    # Check if agent concluded it's mild - more specific detection
    mild_indicators = [
        "mild", "appears to be mild", "seems mild", "not serious", "minor", "within normal range",
        "stay hydrated", "take a break"
    ]
    is_mild_concluded = any(indicator in response_lower for indicator in mild_indicators)
    
    # Check if agent concluded it's serious - more specific detection
    serious_indicators = [
         "critical", "urgent", "may be serious",
        "immediately", "call doctor immediately", "hospital", "dangerous",
        "concerning", "alarming"
    ]
    is_serious_concluded = any(indicator in response_lower for indicator in serious_indicators)
    
    # Ensure only one is true - prioritize serious over mild
    if is_serious_concluded:
        is_mild_concluded = False
    
    # Check if agent already mentioned follow-up
    has_followup = "check back with you in 5 minutes" in response_lower
    
    # If agent concluded mild and no follow-up mentioned, add it
    if is_mild_concluded and not has_followup:
        followup_msg = f"I'll check back with you in 5 minutes. (Reminder: Follow up with {name} in 5 minutes)"
        response = response.strip() + "\n\n" + followup_msg
        schedule_followup(user_id if user_id else "unknown_user", name)
        print(f"Checking the user {name} after 5 mins")
    # If follow-up is already present, just schedule it
    elif has_followup:
        schedule_followup(user_id if user_id else "unknown_user", name)
        print(f"Checking the user {name} after 5 mins")
    
    # If agent concluded serious and hasn't asked about emergency contacts yet
    if is_serious_concluded and "emergency contact" not in response_lower and "911" not in response_lower and "would you like me to contact" not in response_lower:
        emergency_msg = f"\n\nGiven the seriousness of your symptoms, would you like me to contact your emergency contacts and call 911? I can share your location with them."
        response = response.strip() + emergency_msg
        print(f"[DEBUG] Added emergency question for {name}")
        
        # Store the complete response with emergency question in memory
        memory.save_context({"input": message}, {"output": response})
        return response  # Return immediately to ensure the emergency question is stored
    
    # Emergency confirmation logic - if user says yes to emergency
    # Check if previous message contained emergency question or if current message is a yes response
    previous_messages = memory.buffer if memory.buffer else []
    previous_has_emergency_question = any("emergency contact" in str(msg.content).lower() or "911" in str(msg.content).lower() for msg in previous_messages if hasattr(msg, 'content'))
    
    # Check if this is a response to emergency question
    is_emergency_response = previous_has_emergency_question and ("yes" in message.lower() or "no" in message.lower())
    
    if is_emergency_response:
        if "yes" in message.lower():
            print(f"[EMERGENCY] Calling 911 and emergency contacts for {name}")
            print(f"[EMERGENCY] Sharing location: 123 Main Street, Apartment 4B")
            print(f"[EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123)")
            
            emergency_response = f"\n\nI've called 911 and contacted your emergency contacts. Don't worry, {name}, I'm here with you. Help is on the way.\n\nWhile we wait for emergency services to arrive, try to stay calm and comfortable. Take slow, deep breaths. If you're able, sit or lie down in a comfortable position. I'll stay with you until help arrives."
            response = response.strip() + emergency_response
            return response  # Return immediately to avoid further processing
        elif "no" in message.lower():
            # Convert serious to mild and add follow-up
            print(f"[INFO] User declined emergency services for {name}, converting to mild with follow-up")
            followup_msg = f"I understand you don't want emergency services right now. I'll check back with you in 5 minutes to see how you're feeling. (Reminder: Follow up with {name} in 5 minutes)"
            response = response.strip() + "\n\n" + followup_msg
            schedule_followup(user_id if user_id else "unknown_user", name)
            print(f"Checking the user {name} after 5 mins")
            return response  # Return immediately to avoid further processing
    
    # Original emergency logic for direct serious responses
    if (is_serious_concluded and ("yes" in message.lower() or "call" in message.lower() or "contact" in message.lower() or "please" in message.lower())):
        print(f"[EMERGENCY] Calling 911 and emergency contacts for {name}")
        print(f"[EMERGENCY] Sharing location: 123 Main Street, Apartment 4B")
        print(f"[EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123)")
        
        emergency_response = f"\n\nI've called 911 and contacted your emergency contacts. Don't worry, {name}, I'm here with you. Help is on the way.\n\nWhile we wait for emergency services to arrive, try to stay calm and comfortable. Take slow, deep breaths. If you're able, sit or lie down in a comfortable position. I'll stay with you until help arrives."
        response = response.strip() + emergency_response
        return response  # Return immediately to avoid further processing
    
    # Debug logging for severity detection
    print(f"[DEBUG] Mild detected: {is_mild_concluded}")
    print(f"[DEBUG] Serious detected: {is_serious_concluded}")
    print(f"[DEBUG] Has followup: {has_followup}")
    
    # Remove repeated follow-up questions (simple heuristic)
    lines = response.split('\n')
    seen = set()
    filtered_lines = []
    for line in lines:
        l = line.strip().lower()
        if l and l not in seen:
            filtered_lines.append(line)
            seen.add(l)
    response = '\n'.join(filtered_lines)
    return response
