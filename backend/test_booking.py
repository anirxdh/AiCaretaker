#!/usr/bin/env python3

import requests
import json
import time

# Test the appointment booking flow
def test_appointment_booking():
    base_url = "http://127.0.0.1:5050"
    
    # Step 1: Initial greeting
    print("Step 1: Initial greeting")
    response = requests.post(f"{base_url}/chat", json={
        "message": "",
        "user_id": "user_test"
    })
    print(f"Response: {response.json()}")
    
    time.sleep(1)
    
    # Step 2: Request appointment for dizziness
    print("\nStep 2: Request appointment for dizziness")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book an appointment for my dizziness.",
        "user_id": "user_test"
    })
    print(f"Response: {response.json()}")
    
    time.sleep(1)
    
    # Step 3: Request specific doctor
    print("\nStep 3: Request specific doctor")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Christopher Taylor.",
        "user_id": "user_test"
    })
    print(f"Response: {response.json()}")
    
    time.sleep(1)
    
    # Step 4: Confirm with yes
    print("\nStep 4: Confirm with yes")
    response = requests.post(f"{base_url}/chat", json={
        "message": "Yes, please book it.",
        "user_id": "user_test"
    })
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_appointment_booking() 