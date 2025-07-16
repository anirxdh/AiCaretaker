import os
from typing import List, Optional
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
from dateparser.search import search_dates

# --- System prompt for all agent responses ---
SYSTEM_PROMPT = """
You are an intelligent and empathetic AI health assistant named CareMate, designed to help elderly users manage their daily wellbeing. Your goal is to assist users in a calm, human-like, and emotionally supportive manner, using only the tools and context provided to you.

IMPORTANT: You are talking to {name}. Always address them by their name when appropriate. Use their name naturally in your responses for empathy, emphasis, or clarification.

Your primary responsibility is to interpret the user's message, determine the intent or concern, and respond accordingly using available tools and facts.

You MUST NOT hallucinate or guess. If you don't know something, say so politely and try to help the user using what is available to you.

---

CONVERSATION MEMORY (CRITICAL):
- You have access to the full conversation history (memory) between you and the user.
- ALWAYS use this memory to resolve ambiguous, follow-up, or context-dependent questions.
- If the user refers to something mentioned earlier (e.g., "that day", "the food I had before", "my last result"), use previous messages to infer what they mean and answer accordingly.
- If a question is unclear, ask a clarifying question, but first try to answer using memory.
- Never ignore previous conversation context when responding.

---

EMERGENCY STATE HANDLING:
- If the user is in an emergency state (e.g., chest pain, 911 called), always keep responses short, factual, and focused on the emergency.
- Do NOT give generic health or posture advice during an emergency.
- Only return to normal conversation if the user says they are okay or help has arrived.
- Example: If the user says "I cannot sit properly" after a heart pain emergency, remind them to stay calm, help is on the way, and not to move unless necessary.

---

DATE-AWARE HEALTH DATA RETRIEVAL (RAG):
- Whenever the user reports a symptom or asks any question related to their health, body, or condition, retrieve data using `get_rag_context`.
- If the user mentions a date (e.g., “yesterday”, “July 14”), convert it to `YYYY-MM-DD`.
- If no date is mentioned, call `get_current_date()` to use today’s date.
- Always add this date in the metadata filter like:
  {{"user_id": "user_john", "data_type": "vitals", "timestamp": "2025-07-15"}}
- Only return data that matches that date exactly.

---

TOOLS AVAILABLE TO YOU:

1. RAG Query Tool — `get_rag_context(user_id, metadata_filter)`
   - Used to retrieve data from:
     - "food" — diet or meals
     - "vitals" — heart rate, BP, steps, etc.
     - "medical_record" — diagnoses, medications, ER visits

2. Current Date Tool — `get_current_date()`
   - Get today’s date

3. User Name Tool — `get_user_name()`
   - Get the current user's name ({name})

---

BEHAVIOR GUIDELINES:
- Speak with warmth and empathy
- Reassure the user (e.g., “I’m here with you”)
- Use the user's name naturally
- Vary your phrasing; avoid repetition
- Only use tools when necessary
- Never guess if something is unknown

---

HEALTH EVALUATION WORKFLOW:

1. SYMPTOM DETECTION:
   - Detect health-related complaints (e.g., “my chest hurts”, “I feel dizzy”)
   - Trigger RAG for vitals, food, medical_record

2. DETERMINE SEVERITY:
   - Mild = low-risk, fatigue, hydration-related
   - Serious = heart, breathing, fainting, severe pain

3. RESPOND ACCORDINGLY:
   - Mild case:
     - Say “This appears to be a mild issue...”
     - Give light suggestions (rest, hydration)
     - Say: “I’ll check back with you in 5 minutes.” Include a reminder.
   - Serious case:
     - Say: “This may be serious based on your symptoms and history.”
     - Ask: “Should I call 911 and notify your emergency contacts?”
     - If confirmed: print → `Calling 911... Sharing location... Notifying emergency contact...`

4. IF UNCERTAIN:
   - Ask a follow-up question
   - Never assume or guess

---

EXAMPLES:

Mild Case:
User: I feel dizzy and light-headed this morning.

Agent Process:
get_rag_context("user_mary", {{"data_type": "food", "timestamp": "2025-07-15"}}) → skipped dinner  
get_rag_context("user_mary", {{"data_type": "vitals", "timestamp": "2025-07-15"}}) → normal heart rate

Agent Response:
Based on your meals, vitals, and health records, this appears to be a mild issue. Try drinking some water and having a small meal. I’ll check back with you in 5 minutes. (Reminder: Follow up with Mary in 5 minutes)

---

Serious Case:
User: My chest feels tight and I'm struggling to breathe.

Agent Process:
get_rag_context("user_john", {{"data_type": "medical_record", "timestamp": "2025-07-15"}}) → history of arrhythmia

Agent Response:
Given your medical history and symptoms, this may be serious. Would you like me to call 911 and notify your emergency contacts?  
(If user says yes → Calling 911... Sharing location... Notifying emergency contact...)

---

RESTRICTIONS:
- Never fabricate or assume answers
- Do not mix dates unless explicitly asked
- Be clear, safe, and helpful always

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

def _infer_data_type_from_query(q: str) -> Optional[str]:
    q_lower = q.lower()
    if any(word in q_lower for word in ["food", "meal", "diet", "breakfast", "lunch", "dinner"]):
        return "food"
    if any(word in q_lower for word in ["vitals", "heart", "blood pressure", "oxygen", "pulse", "steps", "step count"]):
        return "vitals"
    if any(word in q_lower for word in ["medical", "diagnosis", "record", "history", "prescribed", "medication"]):
        return "medical_record"
    return None

def _extract_date_from_query(q: str) -> Optional[str]:
    """Return a YYYY-MM-DD string if a date-like expression is found in the query."""
    # First try search_dates to find any date expression in the sentence
    results = search_dates(q, settings={'RELATIVE_BASE': datetime.datetime.now()})
    if results:
        for txt, dt in results:
            txt_l = txt.lower().strip()
            # Accept if the match contains any digit OR clearly date-related keywords
            if any(ch.isdigit() for ch in txt) or any(k in txt_l for k in [
                "yesterday", "today", "tomorrow", "last", "ago", "week", "month", "year"]):
                return dt.strftime('%Y-%m-%d')
    # Fallback to direct parse of the whole query only if query itself hints at a date
    date_keywords = [
        "yesterday", "today", "tomorrow", "last", "ago", "week", "month", "year",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"
    ]
    if any(ch.isdigit() for ch in q) or any(k in q.lower() for k in date_keywords):
        dt = dateparser.parse(q, settings={'RELATIVE_BASE': datetime.datetime.now()})
        if dt:
            return dt.strftime('%Y-%m-%d')
    return None


def get_rag_context_tool(query, user_id):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
        index_name="elderly-health-agent"
    )
    date_str = _extract_date_from_query(query)
    if not date_str:
        date_str = datetime.date.today().strftime('%Y-%m-%d')

    data_type = _infer_data_type_from_query(query)
    metadata_filter = {
        "user_id": {"$eq": user_id},
        "date": {"$eq": date_str}
    }
    if data_type:
        metadata_filter["data_type"] = {"$eq": data_type}

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": metadata_filter
        }
    )
    docs = retriever.get_relevant_documents(query)
    if docs:
        return docs[0].page_content
    return f"No health data found for {date_str}."

def get_user_name(user_id: str):
    """Extract user name from user_id"""
    if user_id and user_id.startswith("user_"):
        return user_id.replace("user_", "").capitalize()
    return "User"

def build_tools(user_id, current_message: str):
    """Return the list of tools, ensuring get_rag_context_tool receives the full user message for correct date parsing."""
    name = get_user_name(user_id)

    return [
        Tool(
            name="get_rag_context",
            # We ignore the LLM-provided query (`q`) and instead use the full user message so
            # that date and data-type inference is always accurate.
            func=lambda q=None: get_rag_context_tool(current_message, user_id=user_id),
            description="Retrieve health data (food, vitals, medical_record) for the user. Automatically infers date and data type from the current question. Use this to answer any question about the user's health, diet, vitals, or medical history. If user has any problem with thier health use this data to understand and give your prediction on if its mild or serious as well.Alwasys use todays date for the data retrieval unless user specifies yesterday or any other day for prediction of symptoms"
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

    # Detect symptom keywords
    symptom_keywords = [
        "pain", "hurt", "ache", "dizzy", "dizziness", "light-headed", "lightheaded",
        "tight", "pressure", "nausea", "breath", "breathing", "faint", "bleeding",
        "vomit", "palpitation", "arrhythmia", "cramp", "chest", "heart"
    ]

    extra_context = ""
    symptom_facts = None
    if any(word in message.lower() for word in symptom_keywords):
        # Determine the date to use (parse from message or default today)
        date_str_symptom = _extract_date_from_query(message) or datetime.date.today().strftime('%Y-%m-%d')
        # Fetch data for all three types
        food_ctx = get_rag_context_tool(f"food {date_str_symptom}", user_id)
        vitals_ctx = get_rag_context_tool(f"vitals {date_str_symptom}", user_id)
        med_ctx = get_rag_context_tool(f"medical record {date_str_symptom}", user_id)

        extra_context = (
            f"\n\n---\nFood on {date_str_symptom}: {food_ctx}\n"
            f"Vitals on {date_str_symptom}: {vitals_ctx}\n"
            f"Medical record on {date_str_symptom}: {med_ctx}\n---\n"
        )
        
        # Store a concise factual summary to prepend later (only for symptom messages)
        symptom_facts = (
            f"Based on your records for {date_str_symptom}:\n"
            f"• Vitals: {vitals_ctx}\n"
            f"• Food intake: {food_ctx}\n"
            f"• Medical record: {med_ctx}\n\n"
        )

    tools = build_tools(user_id, message)
    system_prompt = extra_context + SYSTEM_PROMPT.format(name=name, date=today)
    
    # Emergency state: track as dict with 'active' and 'reason'
    if user_id not in emergency_states:
        emergency_states[user_id] = {"active": False, "reason": None}

    # If emergency is active, keep responses contextual and varied until cleared
    if emergency_states[user_id]["active"]:
        # Clear emergency when user confirms help has arrived or they feel okay
        if any(p in message.lower() for p in ["help arrived", "paramedics", "ambulance", "i'm fine", "i am fine", "feel better", "i'm okay", "im okay", "i feel okay"]):
            emergency_states[user_id] = {"active": False, "reason": None}
            return (
                f"I'm relieved help has arrived, {name}. I'm here if you need anything else or have questions while you recover."
            )

        state = emergency_states[user_id]
        turn = state.get("turn", 0) + 1
        state["turn"] = turn  # persist updated turn count
        reason = state.get("reason", "your symptoms")

        if turn == 1:
            return (
                f"I'm still here with you, {name}. Help is on the way.\n"
                "On a scale of 0–10, how intense is the pain right now?\n"
                "Keep breathing slowly. Let me know immediately if anything changes."
            )
        elif turn == 2:
            return (
                f"Emergency services should be close, {name}. If you aren't allergic and have aspirin nearby, keep one handy but wait for paramedics to advise before taking it.\n"
                "Are you feeling shortness of breath or light-headedness?"
            )
        elif turn == 3:
            return (
                f"Stay seated or lie in the most comfortable position, {name}. If possible, loosen any tight clothing around your chest.\n"
                "Can you tell me if the pain is spreading to your arm, jaw, or back?"
            )
        else:
            # Generic supportive message for subsequent turns
            return (
                f"I'm right here with you, {name}. Help should arrive any moment. Continue slow, steady breaths.\n"
                "If anything gets worse, speak out so I can update emergency responders." 
            )

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
            print(f"🚨 [EMERGENCY ALERT] User {name} (ID: {user_id}) has requested emergency assistance!")
            print(f"📞 [EMERGENCY] Calling 911 for {name}...")
            print(f"📍 [EMERGENCY] Sharing location with emergency services: 123 Main Street, Apartment 4B, Minneapolis, MN 55455")
            print(f"👤 [EMERGENCY] User details: {name}, Age: 72, Medical history: Cardiac arrhythmia")
            print(f"📱 [EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123) - Daughter")
            print(f"📱 [EMERGENCY] Notifying emergency contact: Dr. Michael Chen (555-0456) - Primary Care Physician")
            print(f"⏰ [EMERGENCY] Emergency initiated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🚑 [EMERGENCY] Estimated arrival time: 8-12 minutes")
            
            # Set emergency state active and store reason from last agent message if possible
            last_agent_message = None
            for msg in reversed(memory.buffer):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    last_agent_message = msg.content
                    break
            reason = None
            if last_agent_message:
                # Try to extract a reason (e.g., "chest pain", "heart pain") from last agent message
                import re
                match = re.search(r'serious issue.*?([\w\s]+)[\.,]', last_agent_message, re.IGNORECASE)
                if match:
                    reason = match.group(1).strip()
                elif "chest" in last_agent_message.lower():
                    reason = "chest pain"
                elif "heart" in last_agent_message.lower():
                    reason = "heart pain"
                else:
                    reason = "a medical emergency"
            else:
                reason = "a medical emergency"
            emergency_states[user_id] = {"active": True, "reason": reason}

            emergency_response = f"\n\nI've called 911 and contacted your emergency contacts. Don't worry, {name}, I'm here with you. Help is on the way.\n\nWhile we wait for emergency services to arrive, try to stay calm and comfortable. Take slow, deep breaths. If you're able, sit or lie down in a comfortable position. I'll stay with you until help arrives."
            return emergency_response
        elif "no" in message.lower():
            print(f"[INFO] User declined emergency services for {name}, converting to mild with follow-up")
            followup_msg = f"I understand you don't want emergency services right now. I'll check back with you in 5 minutes to see how you're feeling. (Reminder: Follow up with {name} in 5 minutes)"
            emergency_states[user_id] = {"active": False, "reason": None}
            return followup_msg
    
    response = agent.run(message)
    
    # If symptom facts were gathered, prepend them so the user sees concrete data
    if symptom_facts:
        response = symptom_facts + response
        
    print(f"[DEBUG] Agent response: {response}")
    
    # Detect agent's conclusion about severity and trigger appropriate actions
    response_lower = response.lower()
    
    # Check if agent concluded it's mild - more specific detection
    mild_indicators = [
        "mild", "appears to be mild", "seems mild", "not serious", "minor", "within normal range",
        "stay hydrated", "take a break" , "normal", "stable"
    ]
    is_mild_concluded = any(indicator in response_lower for indicator in mild_indicators)
    
    # Check if agent concluded it's serious - more specific detection
    serious_indicators = [
         "critical", "urgent", "may be serious",
        "immediate", "call doctor immediately", "hospital", "dangerous", "alarming"
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
            print(f"🚨 [EMERGENCY ALERT] User {name} (ID: {user_id}) has requested emergency assistance!")
            print(f"📞 [EMERGENCY] Calling 911 for {name}...")
            print(f"📍 [EMERGENCY] Sharing location with emergency services: 123 Main Street, Apartment 4B, Minneapolis, MN 55455")
            print(f"👤 [EMERGENCY] User details: {name}, Age: 72, Medical history: Cardiac arrhythmia")
            print(f"📱 [EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123) - Daughter")
            print(f"📱 [EMERGENCY] Notifying emergency contact: Dr. Michael Chen (555-0456) - Primary Care Physician")
            print(f"⏰ [EMERGENCY] Emergency initiated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🚑 [EMERGENCY] Estimated arrival time: 8-12 minutes")
            
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
        print(f"🚨 [EMERGENCY ALERT] User {name} (ID: {user_id}) has requested emergency assistance!")
        print(f"📞 [EMERGENCY] Calling 911 for {name}...")
        print(f"📍 [EMERGENCY] Sharing location with emergency services: 123 Main Street, Apartment 4B, Minneapolis, MN 55455")
        print(f"👤 [EMERGENCY] User details: {name}, Age: 72, Medical history: Cardiac arrhythmia")
        print(f"📱 [EMERGENCY] Notifying emergency contact: Sarah Johnson (555-0123) - Daughter")
        print(f"📱 [EMERGENCY] Notifying emergency contact: Dr. Michael Chen (555-0456) - Primary Care Physician")
        print(f"⏰ [EMERGENCY] Emergency initiated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚑 [EMERGENCY] Estimated arrival time: 8-12 minutes")
        
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
