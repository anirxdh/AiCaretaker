#!/usr/bin/env python3

import requests
import json
import time

# Debug test to see what's happening with pending appointments
def debug_test():
    base_url = "http://127.0.0.1:5050"
    
    print("=== DEBUG TEST ===")
    
    # Step 1: Request specific doctor
    print("\n1. Requesting specific doctor...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Dr. Christopher Taylor.",
        "user_id": "debug_user"
    })
    result = response.json()
    print(f"Response: {result}")
    
    # Check if the response contains booking confirmation question
    if "Would you like to book this appointment?" in result.get("response", ""):
        print("✓ Auto-detection trigger phrase found in response")
    else:
        print("✗ Auto-detection trigger phrase NOT found in response")
        
    time.sleep(1)
    
    # Step 2: Try to confirm
    print("\n2. Confirming with 'Yes'...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "Yes",
        "user_id": "debug_user"
    })
    result = response.json()
    print(f"Response: {result}")
    
    # Check if the response indicates successful booking
    if "appointment" in result.get("response", "").lower() and "booked" in result.get("response", "").lower():
        print("✓ Appointment appears to be booked")
    elif "hello" in result.get("response", "").lower():
        print("✗ Got greeting instead of booking confirmation")
    else:
        print("? Unexpected response")

if __name__ == "__main__":
    debug_test() 