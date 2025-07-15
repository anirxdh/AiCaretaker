# Google API Setup Guide

This guide will help you set up Google Calendar and Gmail APIs for the appointment booking system.

## Prerequisites

1. A Google account
2. Python 3.7+ installed
3. Access to Google Cloud Console

## Step 1: Enable Google APIs

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Calendar API
   - Gmail API

## Step 2: Create Credentials

1. In the Google Cloud Console, go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Desktop application" as the application type
4. Give it a name (e.g., "AI Caretaker Appointment System")
5. Click "Create"
6. Download the JSON file and rename it to `credentials.json`
7. Place `credentials.json` in the `elderly-agent/backend/` directory

## Step 3: Install Dependencies

```bash
cd elderly-agent/backend
pip install -r requirements.txt
```

## Step 4: Configure Email Settings

Edit the `google_calendar_integration.py` file to update the email addresses:

```python
# In the create_calendar_event method, update these lines:
'attendees': [
    {'email': 'your-doctor-email@example.com'},  # Replace with actual doctor email
    {'email': 'patient-email@example.com'}       # Replace with actual patient email
]

# In the send_confirmation_email method, update these lines:
sender='your-healthcare-email@example.com',  # Replace with your healthcare email
to='patient-email@example.com',              # Replace with actual patient email
```

## Step 5: First Run Authentication

1. Run the backend application:
   ```bash
   python app.py
   ```

2. On first run, a browser window will open asking you to authorize the application
3. Sign in with your Google account and grant the requested permissions
4. The application will create a `token.pickle` file to store your credentials

## Step 6: Test the Integration

1. Start a conversation with the AI agent
2. Request an appointment booking
3. Check your Google Calendar to see if the event was created
4. Check your email for the confirmation message

## Troubleshooting

### Common Issues:

1. **"credentials.json not found"**
   - Make sure you downloaded the credentials file and placed it in the backend directory
   - Ensure the file is named exactly `credentials.json`

2. **"Google APIs not available"**
   - Install the required packages: `pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client`

3. **Authentication errors**
   - Delete the `token.pickle` file and restart the application
   - Make sure you're using the correct Google account

4. **Calendar permission errors**
   - Ensure the Google Calendar API is enabled in your Google Cloud Console
   - Check that your Google account has calendar access

### Security Notes:

- Keep your `credentials.json` and `token.pickle` files secure
- Don't commit these files to version control
- Add them to your `.gitignore` file

### Production Considerations:

For production use, consider:
- Using service accounts instead of OAuth 2.0
- Implementing proper email templates
- Adding error handling and retry logic
- Setting up webhook notifications for calendar changes

## API Scopes Used

The application requests the following permissions:
- `https://www.googleapis.com/auth/calendar` - Full access to Google Calendar
- `https://www.googleapis.com/auth/gmail.send` - Send emails via Gmail

These scopes allow the application to:
- Create, read, update, and delete calendar events
- Send emails on your behalf
- Manage calendar reminders and notifications 