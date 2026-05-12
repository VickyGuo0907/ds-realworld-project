import os, requests
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("DISCORD_TOKEN", "MTUwMTYwNzk4MzYzMTU2NDgwMA.GZEIkR.tjhXpO2_g_X9MKYjs1qz2N6YUlAhe3Hrij9Ocw").strip()

# Sanity check on the token itself
print(f"Token length: {len(token)}")
print(f"Token starts with: {token[:10]}...")
print(f"Has whitespace? {token != token.strip()}")
print(f"Has quotes? {token.startswith(chr(34)) or token.startswith(chr(39))}")
print()

# Actual REST API test
r = requests.get(
    "https://discord.com/api/v10/users/@me",
    headers={"Authorization": f"Bot {token}"},
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:300]}")