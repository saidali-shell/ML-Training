import requests

task = "What is the capital of France?"
r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1:8b", "prompt": task, "stream": False})
print(r.json()["response"])
