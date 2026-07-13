from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ----------------------------------------------------------------------
# Tiny, CPU‑only image captioning using BLIP (≈ 400 MB)
# ----------------------------------------------------------------------
model_id = "Salesforce/blip-image-captioning-base"

# Load processor and model directly (no pipeline lookup needed)
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Load the target image (make sure the file exists)
img = Image.open(r"C:\Users\MuhammedSaidali\Desktop\LLM-Types\image.webp")   # <-- change this path

# ----------------------------------------------------------------------
# Generate a caption and print it
# ----------------------------------------------------------------------
inputs = processor(images=img, return_tensors="pt")
# Ensure inputs are on the same device as the model (CPU)
inputs = {k: v.to("cpu") for k, v in inputs.items()}
output_ids = model.generate(**inputs)
caption = processor.decode(output_ids[0], skip_special_tokens=True)
print("Result:\n", caption)