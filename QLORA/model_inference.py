import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

models_to_test = {
    "Base": "Qwen/Qwen2.5-1.5B-Instruct",              # downloads from HF if not cached
    "Fine-tuned": r"C:\Users\MuhammedSaidali\Desktop\Qlora-vscode\qwen_qlora_merged_model"  # your local merged model
}

test_prompts = [
    "Give three tips for staying healthy.",
    "What are the three primary colors?",
]

def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end = time.time()

    generated_tokens = output[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    duration = end - start
    tokens_per_sec = len(generated_tokens) / duration if duration > 0 else 0
    return response, duration, len(generated_tokens), tokens_per_sec

for label, path in models_to_test.items():
    print(f"\n{'='*80}\nLoading {label} model from: {path}\n{'='*80}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map="cpu")

    for prompt in test_prompts:
        response, duration, num_tokens, tps = generate_response(model, tokenizer, prompt)
        print(f"\nPROMPT: {prompt}")
        print(f"[{label}] {response}")
        print(f"[Time: {duration:.2f}s | Tokens: {num_tokens} | Speed: {tps:.2f} tok/s]")

    del model, tokenizer  # free memory before loading the next model