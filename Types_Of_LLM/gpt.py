from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

chat = pipeline("text-generation", model=mdl, tokenizer=tok)
print(chat("You are a concise assistant.\nUser: Explain vector databases in 2 lines.\nAssistant:", max_new_tokens=80)[0]["generated_text"])