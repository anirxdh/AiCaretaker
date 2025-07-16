import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import json

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    print("Google APIs not available. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# If modifying these scopes, delete the file token.pickle.
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.send'
]

class GoogleCalendarIntegration:
    def __init__(self):
        self.creds = None
        self.calendar_service = None
        self.gmail_service = None
        self.setup_credentials()
    
    def setup_credentials(self):
        """Set up Google API credentials"""
        if not GOOGLE_APIS_AVAILABLE:
            print("[WARNING] Google APIs not available. Using simulation mode.")
            return
        
        # The file token.pickle stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                # Check if credentials file exists
                if not os.path.exists('credentials.json'):
                    print("[ERROR] credentials.json not found!")
                    print("Please download your Google API credentials from:")
                    print("https://console.cloud.google.com/apis/credentials")
                    print("Save as 'credentials.json' in the backend directory")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=8080)
            
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)
        
        # Build the services
        try:
            self.calendar_service = build('calendar', 'v3', credentials=self.creds)
            self.gmail_service = build('gmail', 'v1', credentials=self.creds)
            print("[SUCCESS] Google APIs connected successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to build Google services: {e}")
    
    def create_calendar_event(self, booking: Dict) -> Dict:
        """Create a real Google Calendar event"""
        if not self.calendar_service:
            return self._simulate_calendar_event(booking)
        
        try:
            # Parse the appointment time
            appointment_datetime = datetime.strptime(
                f"{booking['appointment_date']} {booking['appointment_time']}", 
                "%Y-%m-%d %I:%M %p"
            )
            
            # Set event duration to 30 minutes
            end_datetime = appointment_datetime + timedelta(minutes=30)
            
            event = {
                'summary': f"Doctor Appointment - {booking['patient_name']}",
                'description': f"""Appointment Details:
• Patient: {booking['patient_name']}
• Doctor: {booking['doctor']} ({booking['specialty']})
• Reason: {booking['reason']}
• Booking ID: {booking['booking_id']}

Please arrive 15 minutes early to complete paperwork.""",
                'start': {
                    'dateTime': appointment_datetime.isoformat(),
                    'timeZone': 'America/New_York',
                },
                'end': {
                    'dateTime': end_datetime.isoformat(),
                    'timeZone': 'America/New_York',
                },
                'attendees': [
                    {'email': 'anirudhvasudevan11@gmail.com'},  # Would be actual doctor's email
                    {'email': 'anirudhcodesbetter@gmail.com'}     # Would be patient's email
                ],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                        {'method': 'popup', 'minutes': 60}        # 1 hour before
                    ],
                },
                'location': 'Main Medical Center, 123 Healthcare Ave, New York, NY',
                'colorId': '1'  # Blue color for medical appointments
            }
            
            event = self.calendar_service.events().insert(
                calendarId='primary', 
                body=event,
                sendUpdates='all'  # Send email notifications to attendees
            ).execute()
            
            print(f"[CALENDAR] Event created: {event.get('htmlLink')}")
            
            return {
                'event_id': event['id'],
                'event_url': event['htmlLink'],
                'status': 'confirmed',
                'real_calendar': True
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to create calendar event: {e}")
            return self._simulate_calendar_event(booking)
    
    def send_confirmation_email(self, booking: Dict, calendar_event: Dict) -> bool:
        """Send confirmation email via Gmail API"""
        if not self.gmail_service:
            return self._simulate_email_send(booking)
        
        try:
            # Create email content
            subject = f"Appointment Confirmed - {booking['appointment_date']}"
            
            body = f"""Dear {booking['patient_name']},

Your appointment has been successfully confirmed!

📅 Appointment Details:
• Date: {booking['appointment_date']} ({datetime.strptime(booking['appointment_date'], '%Y-%m-%d').strftime('%A')})
• Time: {booking['appointment_time']}
• Doctor: {booking['doctor']} ({booking['specialty']})
• Location: Main Medical Center, 123 Healthcare Ave
• Reason: {booking['reason']}
• Booking ID: {booking['booking_id']}

📋 Important Information:
• Please arrive 15 minutes early to complete necessary paperwork
• Bring your ID and insurance card
• If you need to reschedule or cancel, please call us at least 24 hours in advance

📅 Calendar Event:
Your appointment has been added to your calendar: {calendar_event.get('event_url', 'N/A')}

If you have any questions, please don't hesitate to contact us.

Best regards,
Your Healthcare Team"""

            # Create the email message
            message = {
                'raw': self._create_message(
                    sender='anirudhvasudevan11@gmail.com',
                    to='anirudhcodesbetter@gmail.com',  # Would be actual patient email
                    subject=subject,
                    message_text=body
                )
            }
            
            # Send the email
            sent_message = self.gmail_service.users().messages().send(
                userId='me', body=message
            ).execute()
            
            print(f"[EMAIL] Confirmation email sent: {sent_message['id']}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return self._simulate_email_send(booking)
    
    def _create_message(self, sender: str, to: str, subject: str, message_text: str) -> str:
        """Create a message for Gmail API"""
        import base64
        from email.mime.text import MIMEText
        
        message = MIMEText(message_text)
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject
        
        return base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    
    def _simulate_calendar_event(self, booking: Dict) -> Dict:
        """Simulate calendar event creation"""
        print(f"[SIMULATION] Creating calendar event for {booking['patient_name']}")
        print(f"[SIMULATION] Event: {booking['appointment_date']} at {booking['appointment_time']}")
        
        return {
            'event_id': f"CAL-{booking['booking_id']}",
            'event_url': f"https://calendar.google.com/event?eid=CAL-{booking['booking_id']}",
            'status': 'confirmed',
            'real_calendar': False
        }
    
    def _simulate_email_send(self, booking: Dict) -> bool:
        """Simulate email sending"""
        print(f"[SIMULATION] Sending confirmation email to {booking['patient_name']}")
        print(f"[SIMULATION] Email content: Appointment confirmed for {booking['appointment_date']}")
        return True

# Global instance
google_integration = GoogleCalendarIntegration() 