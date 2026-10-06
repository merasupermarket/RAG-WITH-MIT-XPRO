#calculates the usage of the OpenRouter API key and prints it to the console.

import os
import requests
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
"https://openrouter.ai/api/v1/key",
headers={
"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
}
)

data = response.json().get("data", {})

usage = data.get("usage", 0)
limit = data.get("limit", 0)
limit_remaining = data.get("limit_remaining", 0)
usage_percentage = (usage / limit) * 100 if limit else 0

print(f"Usage: {usage_percentage:.7f}%")
print(f"Limit: ${limit:.2f}")
print(f"Used: ${usage}")
print(f"Remaining: ${limit_remaining}")