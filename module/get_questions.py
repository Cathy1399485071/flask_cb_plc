import gspread
from google.oauth2.service_account import Credentials


# Define the scopes
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Load credentials
credentials = Credentials.from_service_account_file(
    "../responsecollection-dc52f198ecb4.json",
    scopes=scopes,
)

# Authorize and open the sheet
client = gspread.authorize(credentials)

# Open the Google Sheet by name
sheet = client.open("Questions Collection (Responses)").sheet1

# Get all responses
responses = sheet.get_all_records()

# Print the responses
for response in responses:
    print(response)