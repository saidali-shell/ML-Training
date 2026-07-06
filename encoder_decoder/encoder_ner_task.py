# file: ner_inference.py
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="MatteoFasulo/ModernBERT-base-NER",
    aggregation_strategy="simple"
)

text = "Apple was founded  Steve Jobs  Cupertino, California."
entities = ner(text)

for e in entities:
    print(e["entity_group"], e["word"], e["score"])
