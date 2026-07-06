from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "squeezebert/squeezebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
  
text = f"The capital of France is {tokenizer.mask_token}."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
logits = outputs.logits[0, mask_idx, :]
top5 = torch.topk(logits, 5, dim=-1).indices[0].tolist()

print(f"Input: {text}")
print(f"Top-5 predictions: {[tokenizer.decode([t]) for t in top5]}")