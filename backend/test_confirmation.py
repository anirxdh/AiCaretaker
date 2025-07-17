#!/usr/bin/env python3

import requests
import json
import time

# Test just the confirmation part
def test_confirmation():
    base_url = "http://127.0.0.1:5050"
    
    # Step 1: Request specific doctor (this should set pending_appointment)
    print("Step 1: Request specific doctor")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Dr. Christopher Taylor.",
        "user_id": "user_test2"
    })
    print(f"Response: {response.json()}")
    
    time.sleep(2)
    
    # Step 2: Confirm with yes (this should book the appointment)
    print("\nStep 2: Confirm with yes")
    response = requests.post(f"{base_url}/chat", json={
        "message": "Yes",
        "user_id": "user_test2"
    })
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_confirmation() 