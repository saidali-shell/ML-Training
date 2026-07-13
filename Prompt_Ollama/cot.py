import requests

task = "If a shirt costs $40 and is 25% off, what is the final price?"
prompt = f"""{task}
Let's think step by step:"""

r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1:8b", "prompt": prompt, "stream": False})
print(r.json()["response"])
