import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from google_calendar_integration import google_integration

# Predefined appointment slots from July 18-29, 2025
# 3 slots for each specialty to allow choice based on health problem
APPOINTMENT_SLOTS = [
    # General Medicine - 3 slots
    {
        "date": "2025-07-29",
        "day": "Tuesday",
        "time": "9:45 AM",
        "available": True,
        "doctor": "Dr. Emily Rodriguez",
        "specialty": "General Medicine",
        "description": "Annual checkups, general health concerns, routine care"
    },
    {
        "date": "2025-08-05",
        "day": "Tuesday",
        "time": "10:30 AM",
        "available": True,
        "doctor": "Dr. David Thompson",
        "specialty": "General Medicine",
        "description": "Annual checkups, general health concerns, routine care"
    },
    {
        "date": "2025-08-13",
        "day": "Wednesday",
        "time": "2:00 PM",
        "available": True,
        "doctor": "Dr. Sarah Johnson",
        "specialty": "General Medicine",
        "description": "Annual checkups, general health concerns, routine care"
    },

    # Cardiology - 3 slots
    {
        "date": "2025-07-30",
        "day": "Wednesday",
        "time": "11:15 AM",
        "available": True,
        "doctor": "Dr. Michael Chen",
        "specialty": "Cardiology",
        "description": "Heart conditions, chest pain, blood pressure, arrhythmia"
    },
    {
        "date": "2025-08-08",
        "day": "Friday",
        "time": "3:30 PM",
        "available": True,
        "doctor": "Dr. Lisa Park",
        "specialty": "Cardiology",
        "description": "Heart conditions, chest pain, blood pressure, arrhythmia"
    },
    {
        "date": "2025-08-15",
        "day": "Friday",
        "time": "1:00 PM",
        "available": True,
        "doctor": "Dr. James Anderson",
        "specialty": "Cardiology",
        "description": "Heart conditions, chest pain, blood pressure, arrhythmia"
    },

    # Internal Medicine - 3 slots
    {
        "date": "2025-07-31",
        "day": "Thursday",
        "time": "10:00 AM",
        "available": True,
        "doctor": "Dr. Maria Garcia",
        "specialty": "Internal Medicine",
        "description": "Complex medical conditions, chronic diseases, comprehensive care"
    },
    {
        "date": "2025-08-06",
        "day": "Wednesday",
        "time": "4:45 PM",
        "available": True,
        "doctor": "Dr. Robert Wilson",
        "specialty": "Internal Medicine",
        "description": "Complex medical conditions, chronic diseases, comprehensive care"
    },
    {
        "date": "2025-08-12",
        "day": "Tuesday",
        "time": "3:15 PM",
        "available": True,
        "doctor": "Dr. Thomas Brown",
        "specialty": "Internal Medicine",
        "description": "Complex medical conditions, chronic diseases, comprehensive care"
    },

    # Geriatrics - 3 slots
    {
        "date": "2025-08-01",
        "day": "Friday",
        "time": "9:00 AM",
        "available": True,
        "doctor": "Dr. Jennifer Lee",
        "specialty": "Geriatrics",
        "description": "Elderly care, age-related conditions, mobility issues, memory concerns"
    },
    {
        "date": "2025-08-07",
        "day": "Thursday",
        "time": "2:30 PM",
        "available": True,
        "doctor": "Dr. Patricia Martinez",
        "specialty": "Geriatrics",
        "description": "Elderly care, age-related conditions, mobility issues, memory concerns"
    },
    {
        "date": "2025-08-14",
        "day": "Thursday",
        "time": "11:00 AM",
        "available": True,
        "doctor": "Dr. William Davis",
        "specialty": "Geriatrics",
        "description": "Elderly care, age-related conditions, mobility issues, memory concerns"
    },

    # Neurology - 3 slots
    {
        "date": "2025-07-29",
        "day": "Monday",
        "time": "1:45 PM",
        "available": True,
        "doctor": "Dr. Amanda White",
        "specialty": "Neurology",
        "description": "Headaches, dizziness, memory problems, nerve issues, stroke follow-up"
    },
    {
        "date": "2025-08-04",
        "day": "Monday",
        "time": "3:30 PM",
        "available": True,
        "doctor": "Dr. Christopher Taylor",
        "specialty": "Neurology",
        "description": "Headaches, dizziness, memory problems, nerve issues, stroke follow-up"
    },
    {
        "date": "2025-08-11",
        "day": "Monday",
        "time": "10:00 AM",
        "available": True,
        "doctor": "Dr. Kevin Miller",
        "specialty": "Neurology",
        "description": "Headaches, dizziness, memory problems, nerve issues, stroke follow-up"
    }
]



def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")

def get_available_slots() -> List[Dict]:
    """Get all available appointment slots"""
    current_date = get_current_date()
    # Filter out past dates
    available_slots = [
        slot for slot in APPOINTMENT_SLOTS 
        if slot["date"] >= current_date and slot["available"]
    ]
    return available_slots

def get_slots_for_week(week_offset: int = 0) -> List[Dict]:
    """Get available slots for a specific week (0 = current week, 1 = next week, etc.)"""
    current_date = datetime.now()
    target_week_start = current_date + timedelta(weeks=week_offset)
    target_week_end = target_week_start + timedelta(days=6)
    
    available_slots = get_available_slots()
    week_slots = [
        slot for slot in available_slots
        if target_week_start.date() <= datetime.strptime(slot["date"], "%Y-%m-%d").date() <= target_week_end.date()
    ]
    return week_slots

def get_slots_by_specialty(specialty: str) -> List[Dict]:
    """Get available slots for a specific specialty"""
    available_slots = get_available_slots()
    specialty_slots = [
        slot for slot in available_slots
        if specialty.lower() in slot["specialty"].lower()
    ]
    return specialty_slots

def get_specialty_recommendation(symptoms: str) -> str:
    """Recommend a specialty based on symptoms"""
    symptoms_lower = symptoms.lower()
    
    if any(word in symptoms_lower for word in ["chest", "heart", "blood pressure", "arrhythmia", "palpitation"]):
        return "Cardiology"
    elif any(word in symptoms_lower for word in ["headache", "dizzy", "memory", "nerve", "stroke", "seizure"]):
        return "Neurology"
    elif any(word in symptoms_lower for word in ["elderly", "age", "mobility", "balance", "fall"]):
        return "Geriatrics"
    elif any(word in symptoms_lower for word in ["chronic", "diabetes", "complex", "multiple"]):
        return "Internal Medicine"
    else:
        return "General Medicine"

def format_slots_for_display(slots: List[Dict]) -> str:
    """Format appointment slots for display"""
    if not slots:
        return "No available appointments found for the requested time period."
    
    # Group slots by specialty
    specialty_groups = {}
    for slot in slots:
        specialty = slot['specialty']
        if specialty not in specialty_groups:
            specialty_groups[specialty] = []
        specialty_groups[specialty].append(slot)
    
    formatted = "🏥 Available Appointment Slots:\n\n"
    slot_number = 1
    
    for specialty, specialty_slots in specialty_groups.items():
        formatted += f"📋 {specialty}\n"
        formatted += f"   {specialty_slots[0]['description']}\n\n"
        
        for slot in specialty_slots:
            formatted += f"   {slot_number}. {slot['day']}, {slot['date']} at {slot['time']}\n"
            formatted += f"      👨‍⚕️ {slot['doctor']}\n\n"
            slot_number += 1
        
        formatted += "\n"
    
    formatted += "💡 Choose a slot number based on your health concern. For example:\n"
    formatted += "   • General Medicine: Annual checkups, routine care\n"
    formatted += "   • Cardiology: Heart conditions, chest pain, blood pressure\n"
    formatted += "   • Internal Medicine: Complex conditions, chronic diseases\n"
    formatted += "   • Geriatrics: Elderly care, age-related issues\n"
    formatted += "   • Neurology: Headaches, dizziness, memory problems\n"
    
    return formatted

def book_appointment(slot_index: int, patient_name: str, reason: str, user_id: str) -> Dict:
    """Book an appointment and return booking details"""
    available_slots = get_available_slots()
    
    if slot_index < 1 or slot_index > len(available_slots):
        return {
            "success": False,
            "message": f"Invalid slot number. Please choose between 1 and {len(available_slots)}"
        }
    
    selected_slot = available_slots[slot_index - 1]
    
    # Mark slot as unavailable
    for slot in APPOINTMENT_SLOTS:
        if (slot["date"] == selected_slot["date"] and 
            slot["time"] == selected_slot["time"] and 
            slot["doctor"] == selected_slot["doctor"]):
            slot["available"] = False
            break
    
    # Create booking details
    booking = {
        "patient_name": patient_name,
        "user_id": user_id,
        "appointment_date": selected_slot["date"],
        "appointment_time": selected_slot["time"],
        "doctor": selected_slot["doctor"],
        "specialty": selected_slot["specialty"],
        "reason": reason,
        "booking_id": f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "confirmed"
    }
    
    # Use real Google Calendar integration
    calendar_event = google_integration.create_calendar_event(booking)
    
    # Send confirmation email
    email_sent = google_integration.send_confirmation_email(booking, calendar_event)
    
    return {
        "success": True,
        "message": f"Appointment booked successfully!",
        "booking": booking,
        "calendar_event": calendar_event
    }



def get_booking_confirmation_message(booking: Dict) -> str:
    """Generate a confirmation message for the booking"""
    return f"""✅ Appointment Confirmed!

📅 Date: {booking['appointment_date']} ({datetime.strptime(booking['appointment_date'], '%Y-%m-%d').strftime('%A')})
⏰ Time: {booking['appointment_time']}
👨‍⚕️ Doctor: {booking['doctor']} ({booking['specialty']})
🏥 Location: Main Medical Center, 123 Healthcare Ave
📋 Reason: {booking['reason']}
🆔 Booking ID: {booking['booking_id']}

Your appointment has been added to your Google Calendar, and I have sent you an email with the appointment details.
You'll receive a reminder 24 hours before your appointment.

Please arrive 15 minutes early to complete any necessary paperwork. If you need to reschedule or cancel, please call us at least 24 hours in advance.""" 