from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

text = (
    "Scientists at a research lab announced on Tuesday that they had "
    "developed a new battery technology capable of charging electric "
    "vehicles in under five minutes. The breakthrough, published in a "
    "peer-reviewed journal, could significantly reduce range anxiety "
    "for EV owners and accelerate adoption of electric vehicles "
    "worldwide, researchers said."
)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=4,
        no_repeat_ngram_size=2,
    )

summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: {text}")
print(f"Summary output: {summary!r}")