import requests


BASE_URL = "http://127.0.0.1:5000"

SESSION_ID = "79dbbb87-695d-4d31-8524-d0237f31a850"

AUDIO_PATH = "test_audio.ogg"


url = (
    f"{BASE_URL}"
    f"/api/lsrw/sessions/"
    f"{SESSION_ID}/response"
)


print("Sending speaking response...")
print("Session ID:", SESSION_ID)
print("Audio:", AUDIO_PATH)


with open(AUDIO_PATH, "rb") as audio:

    files = {
        "audio": (
            "test_audio.ogg",
            audio,
            "audio/ogg"
        )
    }

    response = requests.post(
        url,
        files=files
    )


print()
print("Status Code:", response.status_code)

print("Response:")

try:
    print(response.json())
except Exception:
    print(response.text)