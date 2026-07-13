import requests

# Use the model name you downloaded in Ollama
model_id = "granite3-moe:3b"   

prompt = "You are a travel expert. Create a 3-day Tokyo food itinerary with neighborhoods."

print(f"Sending prompt to local Ollama model '{model_id}'...")

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_id,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        print("\nResult:\n", response.json()["response"])
    else:
        print(f"\nError: Received status code {response.status_code} from Ollama.")
except requests.exceptions.ConnectionError:
    print("\nError: Could not connect to local Ollama server. Please ensure Ollama is running.")