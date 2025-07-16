#!/usr/bin/env python3
"""
Test script to demonstrate the updated appointment booking flow
"""

from agent import agent_response

def test_appointment_flow():
    """Test the complete appointment booking flow"""
    
    # Test user ID
    user_id = "user_john"
    
    print("=== APPOINTMENT BOOKING FLOW TEST ===\n")
    
    # Step 1: User requests an appointment
    print("1. User: 'I need to book a doctor appointment'")
    response1 = agent_response("I need to book a doctor appointment", user_id)
    print(f"Agent: {response1}\n")
    
    # Step 2: User chooses a specific slot
    print("2. User: 'I want slot 3'")
    response2 = agent_response("I want slot 3", user_id)
    print(f"Agent: {response2}\n")
    
    print("=== FLOW COMPLETE ===")

if __name__ == "__main__":
    test_appointment_flow() 