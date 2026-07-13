# -------------------------------------------------
#  slm.py – tiny 4‑bit model for quick inference
# -------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Qwen 1.8B Chat, already quantised to 4‑bit
model_id = "Qwen/Qwen1.8B-Chat-q4_0"

# Load tokenizer (same for all variants)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the **quantised** model – bitsandbytes will handle the 4‑bit tensors
mdl = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",          # lets 🤗 decide CPU/CPU‑offload
    torch_dtype=torch.float16,   # 4‑bit models use float16 internally
    trust_remote_code=True,
)

# Build a simple generation pipeline
pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

# ----------------------------------------------------------------------
# Example query – replace with whatever you need
print(
    pipe(
        "Give 3 K8s troubleshooting tips:",
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
    )[0]["generated_text"]
)
