import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the project root
load_dotenv()

# Read Gemini API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=google_api_key,
    temperature=0
)

# Send a simple test request
response = llm.invoke("Reply with exactly: Gemini connection successful")

print("Gemini response:")
print(response.content)

# Check Langfuse credentials without printing secrets
langfuse_keys = [
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_BASE_URL"
]

for key in langfuse_keys:
    if os.getenv(key):
        print(f"{key}: configured")
    else:
        print(f"{key}: missing")