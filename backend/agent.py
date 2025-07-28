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
from appointments import get_available_slots, get_slots_by_specialty, get_specialty_recommendation, format_slots_for_display, book_appointment, get_booking_confirmation_message
import string

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

4. get_appointments(specialty, week_range) – Retrieves available doctors and appointment slots for the week, based on the medical specialty needed.

5. book_appointment(slot_number, reason) – Books the selected slot, schedules it on the user's calendar, and sends a confirmation email.

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

APPOINTMENT BOOKING WORKFLOW:

1. APPOINTMENT REQUEST DETECTION:
   - Detect appointment-related requests (e.g., "I need an appointment", "book a doctor", "schedule appointment")
   - PRIORITY: When appointment booking is detected, DO NOT analyze symptoms or call RAG
   - Use get_appointments() to show ALL available slots

2. SPECIALTY RECOMMENDATION:
   - Only recommend specialty if user explicitly asks for a specific type of doctor
   - Otherwise, show all available slots across all specialties
   - Let user choose based on their own preference

3. BOOKING PROCESS:
   - Show all available slots with get_appointments() for user to review
   - Wait for user to choose a specific slot number from the displayed list
   - When user selects a slot (e.g., "I want slot 3" or "book slot 5"), use book_appointment(slot_number, reason)
   - Provide confirmation with booking details and send email

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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
# Create a dictionary to store separate memory instances for each user
user_memories = {}
# Store pending follow-up messages for each user
pending_followups = {}
emergency_states = {}

pending_appointment = {}  # user_id -> {'slot_number': int, 'slot_details': dict, 'reason': str, 'summary': str}
pending_slots = {}  # user_id -> list of slots last shown

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

def get_appointments_tool(specialty: str, week_range: str, current_message: str, user_id: str = None):
    """Get available appointments - show all slots for user to choose from"""
    all_slots = get_available_slots()
    # Determine if we should filter by specialty
    filtered_specialties = ["cardiology", "neurology", "geriatrics", "internal medicine", "general medicine"]
    if specialty and specialty.lower() in filtered_specialties:
        slots = get_slots_by_specialty(specialty)
    else:
        slots = all_slots
    # Always store the shown slots for this user
    if user_id:
        pending_slots[user_id] = slots
        print(f"[DEBUG] Set pending_slots for user {user_id}: {[f'{s['doctor']} {s['date']} {s['time']}' for s in slots]}")
    return format_slots_for_display(slots)

def book_appointment_tool(slot_number: str, reason: str, user_id: str, current_message: str):
    """Book an appointment by slot number"""
    try:
        slot_num = int(slot_number) if slot_number else 1
        patient_name = get_user_name(user_id)
        
        # If no reason provided, try to extract from message
        if not reason:
            # Extract symptoms from current message
            symptom_keywords = ["pain", "hurt", "ache", "dizzy", "tight", "pressure", "nausea", "breath", "faint", "bleeding", "vomit", "palpitation", "arrhythmia", "cramp", "chest", "heart", "headache", "memory", "nerve", "stroke", "seizure", "mobility", "balance", "fall", "chronic", "diabetes"]
            found_symptoms = [word for word in symptom_keywords if word in current_message.lower()]
            reason = f"Patient reported: {', '.join(found_symptoms)}" if found_symptoms else "General consultation"
        
        # Book the appointment
        result = book_appointment(slot_num, patient_name, reason, user_id)
        
        if result["success"]:
            return get_booking_confirmation_message(result["booking"])
        else:
            return result["message"]
            
    except (ValueError, TypeError):
        return "Please provide a valid slot number to book an appointment."

def set_pending_for_direct_slot(user_id, slot_details, summary):
    pending_slots[user_id] = [slot_details]
    pending_appointment[user_id] = {
        'slot_number': 1,
        'slot_details': slot_details,
        'reason': summary,
        'summary': summary
    }
    print(f"[DEBUG] Set pending_appointment and pending_slots for user {user_id} (direct slot): {slot_details['doctor']} {slot_details['date']} {slot_details['time']}")

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
        ),
        Tool(
            name="get_appointments",
            func=lambda specialty=None, week_range=None: get_appointments_tool(specialty, week_range, current_message, user_id),
            description="Show all available appointment slots for the user to choose from. If user mentions a specific specialty (e.g., 'cardiology', 'neurology'), filter by that specialty. Otherwise, show all available slots across all specialties."
        ),
        Tool(
            name="book_appointment",
            func=lambda slot_number=None, reason=None: book_appointment_tool(slot_number, reason, user_id, current_message),
            description="Book an appointment by slot number. Use this after showing available slots with get_appointments. The slot_number should be the number from the displayed list."
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

def normalize_confirmation(msg):
    msg = msg.strip().lower().strip(string.punctuation)
    return msg

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
    
    # Detect appointment booking keywords
    appointment_keywords = [
        "appointment", "book", "schedule", "see doctor", "see a doctor", "make appointment",
        "need to see", "want to see", "doctor visit", "medical appointment", "consultation"
    ]
    
    # Detect slot selection keywords
    slot_selection_keywords = [
        "slot", "choose", "select", "want slot", "pick slot", "book slot", "number", "option"
    ]

    extra_context = ""
    symptom_facts = None
    
    # Check for appointment booking intent FIRST (prioritize over symptom analysis)
    appointment_intent = any(word in message.lower() for word in appointment_keywords)
    slot_selection_intent = any(word in message.lower() for word in slot_selection_keywords) and any(char.isdigit() for char in message)
    
    # Only do symptom analysis if user is NOT requesting an appointment
    if not appointment_intent and any(word in message.lower() for word in symptom_keywords):
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
    
    if appointment_intent:
        # Add appointment context to help the agent understand the intent
        extra_context += "\n\n---\nAPPOINTMENT BOOKING REQUEST DETECTED\n"
        extra_context += "The user is requesting to book a doctor appointment.\n"
        extra_context += "Use get_appointments() to show ALL available slots for user to choose from.\n"
        extra_context += "Wait for user to select a specific slot number, then use book_appointment() to confirm.\n"
        extra_context += "DO NOT call RAG or analyze symptoms - focus only on appointment booking.\n---\n"
    
    # Check for slot confirmation intent (yes/no)
    norm_msg = normalize_confirmation(message)
    confirmation_yes = norm_msg in ["yes", "y", "confirm", "ok", "okay", "s", "sure","yes sure", "Yes", "Yes.","yes."]
    confirmation_no = norm_msg in ["no", "n", "not ok", "not okay"]


    # Debug logging for confirmation
    if user_id in pending_appointment:
        print(f"[DEBUG] Pending appointment found for user: {user_id}")
        print(f"[DEBUG] User message for confirmation: '{message}' (normalized: '{norm_msg}')")

    # If user is confirming a pending appointment
    if user_id in pending_appointment and confirmation_yes:
        print(f"[DEBUG] Booking appointment for user: {user_id}")
        slot_info = pending_appointment[user_id]
        slot_number = slot_info['slot_number']
        reason = slot_info['reason']
        summary = slot_info['summary']
        # Map the selected slot from pending_slots to the global list
        slots_list = pending_slots.get(user_id)
        if not slots_list or slot_number < 1 or slot_number > len(slots_list):
            del pending_appointment[user_id]
            if user_id in pending_slots:
                del pending_slots[user_id]
            return "Sorry, that slot is no longer available. Please choose another."
        slot_details = slots_list[slot_number - 1]
        all_slots = get_available_slots()
        global_slot_index = None
        for idx, slot in enumerate(all_slots):
            if (
                slot['date'] == slot_details['date'] and
                slot['time'] == slot_details['time'] and
                slot['doctor'] == slot_details['doctor']
            ):
                global_slot_index = idx + 1  # 1-based index for booking
                break
        if global_slot_index is None:
            del pending_appointment[user_id]
            if user_id in pending_slots:
                del pending_slots[user_id]
            return "Sorry, that slot is no longer available. Please choose another."
        result = book_appointment(global_slot_index, get_user_name(user_id), summary, user_id)
        del pending_appointment[user_id]
        if user_id in pending_slots:
            del pending_slots[user_id]
        if result["success"]:
            return get_booking_confirmation_message(result["booking"])
        else:
            return result["message"]
    # If user declines the slot
    if user_id in pending_appointment and confirmation_no:
        print(f"[DEBUG] User declined appointment for user: {user_id}")
        del pending_appointment[user_id]
        if user_id in pending_slots:
            del pending_slots[user_id]
        return "No problem. Please choose another slot number from the available appointments."

    # If user selects a slot, prompt for confirmation instead of booking
    if slot_selection_intent:
        import re
        match = re.search(r'slot\s*(\d+)', message.lower())
        slot_number = None
        if match:
            slot_number = int(match.group(1))
        else:
            numbers = re.findall(r'\d+', message)
            if numbers:
                slot_number = int(numbers[0])
        if slot_number is None:
            return "Please specify a valid slot number."
        # Use the last shown slots for this user
        slots_list = pending_slots.get(user_id)
        print(f"[DEBUG] Slot selection for user {user_id}: slot_number={slot_number}, pending_slots={[f'{s['doctor']} {s['date']} {s['time']}' for s in slots_list] if slots_list else None}")
        if not slots_list or slot_number < 1 or slot_number > len(slots_list):
            return f"Invalid slot number. Please choose between 1 and {len(slots_list) if slots_list else 0}."
        slot_details = slots_list[slot_number - 1]
        # Try to get recent symptom/vitals summary from memory
        summary = "General consultation"
        if user_id in user_memories:
            # Look for last symptom message and vitals
            mem = user_memories[user_id].buffer
            last_symptom = None
            for m in reversed(mem):
                if hasattr(m, 'content') and any(word in m.content.lower() for word in symptom_keywords):
                    last_symptom = m.content
                    break
            if last_symptom:
                # Try to get vitals from last RAG call
                vitals = get_rag_context_tool("vitals", user_id)
                summary = f"Patient reported: {last_symptom}\nRecent vitals: {vitals}"
        # Store pending slot selection
        pending_appointment[user_id] = {
            'slot_number': slot_number,
            'slot_details': slot_details,
            'reason': summary,
            'summary': summary
        }
        slot_str = f"{slot_details['day']}, {slot_details['date']} at {slot_details['time']} with {slot_details['doctor']} ({slot_details['specialty']})"
        return f"You chose: {slot_str}. Are you okay with this timing? (yes/no)"

    # Detect direct slot proposal in agent output (e.g., 'Would you like to book this appointment slot?')
    # This should be triggered when the agent proposes a specific slot by doctor/date
    # For this, after showing a single slot, call set_pending_for_direct_slot(user_id, slot_details, summary)
    # Example: If message contains 'Would you like to book this appointment slot?' and a slot is in context, set pending
    # (This may require you to add this logic wherever you generate such a proposal in your agent code)

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
    
    # Check if agent concluded it's serious - EXPANDED detection for safety
    serious_indicators = [
        "critical", "urgent", "emergency", "Emergency",
        "call doctor immediately", "dangerous", "alarming",
        "seek medical attention", "emergency room",
        "heart-related symptoms","require immediate medical attention"
        "high blood pressure", "low oxygen", "symptoms can be serious"
        "tachycardia", "hypertension", "call 911",
        "see a doctor immediately", "abnormal readings", "require immediate attention"
    ]
    is_serious_concluded = any(indicator in response_lower for indicator in serious_indicators)
    
    # Check if agent concluded it's mild - ONLY if not serious
    mild_indicators = [
        "rest", "mild", "appears to be mild", "seems mild", "not serious", "minor", 
        "within normal range", "stay hydrated", "take a break", "dizziness","stable condition",
        "no immediate concern", "routine", "minor issue"
    ]
    # Only check for mild if it's NOT already flagged as serious
    is_mild_concluded = not is_serious_concluded and any(indicator in response_lower for indicator in mild_indicators)
    
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
