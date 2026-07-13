"""
Variational Autoencoder — Inference Demo
Two parts:
  1. Posterior sampling + prior sampling using SD1.4's VAE
  2. Denoising test using stabilityai/sd-vae-ft-mse

Fix applied: SD VAEs are trained on images normalized to [-1, 1], not [0, 1].
All encode/decode steps below use the correct range.
"""

import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

IMG_PATH = r"C:\Users\MuhammedSaidali\Desktop\Autoencoders\demo.jpg"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "variational")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cpu"
DTYPE = torch.bfloat16

to_tensor_pm1 = T.Compose([
    T.ToTensor(),                              # [0,1], shape (C,H,W)
    lambda x: (x * 2 - 1).unsqueeze(0),        # -> [-1,1], shape (1,C,H,W)
])

def to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a [-1,1] decoded tensor back to a savable PIL image."""
    x = ((x + 1) / 2).clamp(0, 1)
    return T.ToPILImage()(x.squeeze(0).float().cpu())


# ---------------------------------------------------------------------------
# Part 1: VAE — posterior reconstruction + naive prior sampling (SD1.4 VAE)
# ---------------------------------------------------------------------------
print("[1] Loading SD1.4 VAE...")
vae1 = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    torch_dtype=DTYPE,
).to(DEVICE)
vae1.eval()

img = Image.open(IMG_PATH).convert("RGB").resize((512, 512))
x = to_tensor_pm1(img).to(DTYPE)

with torch.no_grad():
    dist = vae1.encode(x).latent_dist

    for i in range(4):
        z = dist.sample()
        recon = vae1.decode(z).sample
        to_pil(recon).save(os.path.join(OUT_DIR, f"vae_sample_{i}.png"))

    # NOTE: SD's true latent space is scaled (~0.18215) and is not simple
    # unit-variance Gaussian noise. Decoding raw N(0,1) noise here is a
    # naive prior sample and will look like a blurry/colorful blob rather
    # than a coherent image — this is expected, not a bug. Real generation
    # requires the diffusion denoising process on top of this latent space.
    z_prior = torch.randn(1, 4, 64, 64).to(DEVICE, DTYPE)
    gen = vae1.decode(z_prior).sample
    to_pil(gen).save(os.path.join(OUT_DIR, "generated_from_prior.png"))

print("[Variational Autoencoder — Generation]")
print(f"  Saved to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Part 2: Denoising test (stabilityai/sd-vae-ft-mse)
# ---------------------------------------------------------------------------
print("\n[2] Loading sd-vae-ft-mse...")
vae2 = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=DTYPE,
).to(DEVICE)
vae2.eval()

img2 = Image.open(IMG_PATH).convert("RGB").resize((256, 256))
x2 = to_tensor_pm1(img2).to(DTYPE)

# add noise in [-1,1] space, then re-clamp to a valid range
noise = torch.randn_like(x2) * 0.4          # 0.4 in [-1,1] range ~ 0.2 in [0,1] range
x2_noisy = (x2 + noise).clamp(-1, 1)

with torch.no_grad():
    latent = vae2.encode(x2_noisy).latent_dist.sample()
    denoised = vae2.decode(latent).sample.clamp(-1, 1)

to_pil(x2).save(os.path.join(OUT_DIR, "original.png"))
to_pil(x2_noisy).save(os.path.join(OUT_DIR, "noisy.png"))
to_pil(denoised).save(os.path.join(OUT_DIR, "denoised.png"))

mse_after = (denoised.float() - x2.float()).pow(2).mean().item()
mse_noise = (x2_noisy.float() - x2.float()).pow(2).mean().item()

print("[Denoising Autoencoder]")
print(f"  Noise added MSE : {mse_noise:.6f}")
print(f"  After denoising : {mse_after:.6f}")
print(f"  Saved to {OUT_DIR}")