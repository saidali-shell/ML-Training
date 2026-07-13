import requests

task = "Summarize this text: AI is transforming every industry..."
prompt = f"Example:\n  Task: Translate 'hello' to French\n  Output: bonjour\n\n---\n\nTask: {task}\nOutput:"
r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1:8b", "prompt": prompt, "stream": False})
print(r.json()["response"])
