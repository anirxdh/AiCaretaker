#!/usr/bin/env python3

import requests
import json
import time

# Simple test to verify confirmation logic
def simple_test():
    base_url = "http://127.0.0.1:5050"
    
    print("=== SIMPLE CONFIRMATION TEST ===")
    
    # Step 1: Request doctor (should trigger auto-detection)
    print("\n1. Requesting doctor...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Dr. Christopher Taylor.",
        "user_id": "simple_test"
    })
    result = response.json()
    print(f"Step 1 Response: {result.get('response', 'ERROR')[:100]}...")
    
    # Step 2: Test "yes" confirmation
    print("\n2. Testing 'yes' confirmation...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "yes",
        "user_id": "simple_test"
    })
    result = response.json()
    print(f"Step 2 Response: {result.get('response', 'ERROR')[:100]}...")
    
    # Step 3: Test "sure book it" confirmation (new user)
    print("\n3. Setting up new user for 'sure book it' test...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "I need to book with Dr. Christopher Taylor.",
        "user_id": "simple_test2"
    })
    result = response.json()
    print(f"Step 3 Setup Response: {result.get('response', 'ERROR')[:100]}...")
    
    time.sleep(1)
    
    print("\n4. Testing 'sure book it' confirmation...")
    response = requests.post(f"{base_url}/chat", json={
        "message": "sure book it",
        "user_id": "simple_test2"
    })
    result = response.json()
    print(f"Step 4 Response: {result.get('response', 'ERROR')[:100]}...")

if __name__ == "__main__":
    simple_test() 