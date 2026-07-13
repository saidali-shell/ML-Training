import requests

task = "What is 7 * 8?"
prompt = """Examples:
Q: What is 2 + 3?
A: 5
Q: What is 12 * 2?
A: 24
Q: What is 9 / 3?
A: 3

Q: What is 7 * 8?
A:"""

r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1:8b", "prompt": prompt, "stream": False})
print(r.json()["response"])
