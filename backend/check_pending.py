#!/usr/bin/env python3

# Import the agent module to check pending appointments
import sys
sys.path.append('.')
from agent import pending_appointment, pending_slots

def check_pending_state():
    print("=== CHECKING PENDING STATE ===")
    print(f"pending_appointment: {pending_appointment}")
    print(f"pending_slots: {pending_slots}")

if __name__ == "__main__":
    check_pending_state() 