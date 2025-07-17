#!/usr/bin/env python3

import requests
import json
import time

# Test the "sure book it" confirmation
def test_sure_book():
    base_url = "http://127.0.0.1:5050"
    
    print("=== TESTING 'SURE BOOK IT' CONFIRMATION ===")
    
    # Step 1: Request specific doctor
    print("\n1. Requesting specific doctor...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Dr. Christopher Taylor.",
        "user_id": "sure_test_user"
    })
    result = response.json()
    print(f"Response: {result}")
    
    time.sleep(1)
    
    # Step 2: Confirm with "sure book it"
    print("\n2. Confirming with 'sure book it'...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "sure book it",
        "user_id": "sure_test_user"
    })
    result = response.json()
    print(f"Response: {result}")
    
    # Check if booking was successful
    if "booked" in result.get("response", "").lower() or "confirmed" in result.get("response", "").lower():
        print("✓ Appointment appears to be booked successfully!")
    else:
        print("✗ Booking may not have worked")

if __name__ == "__main__":
    test_sure_book() 