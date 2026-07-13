# Autoencoders

A hands-on collection of autoencoder implementations covering the major variants — from plain deterministic compression to variational, adversarial, adaptive, and sparse autoencoders. Built for learning and experimentation on CPU.

---

## Table of Contents

- [What's Inside](#whats-inside)
- [File Overview](#file-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Outputs](#outputs)
- [Key Concepts](#key-concepts)
- [Dependencies](#dependencies)

---

## What's Inside

| # | File | Type | Training | Dataset | Model Source |
|---|------|------|----------|---------|-------------|
| 1 | `autoencoders.py` | **Plain AE** | From scratch | MNIST | Self-built |
| 2 | `variational_encoders.py` | **VAE** | Pretrained only | Your image | HuggingFace (`CompVis/stable-diffusion-v1-4`, `stabilityai/sd-vae-ft-mse`) |
| 3 | `adversarial.py` | **AAE** | From scratch | MNIST | Self-built |
| 4 | `adaptive.py` | **Adaptive AE** | From scratch | MNIST | Self-built |
| 5 | `sparse_encoders.py` | **SAE** | Pretrained only | GPT-2 text | SAELens / TransformerLens (`gpt2-small-res-jb`) |

---

## File Overview

### 1. `autoencoders.py` — Plain Autoencoder

**What it does:**
Trains a simple convolutional autoencoder on MNIST handwritten digits from scratch. The encoder compresses a 28×28 grayscale image into a 32-dimensional latent vector, and the decoder reconstructs it back to 28×28.

**Key characteristics:**
- **Deterministic** — same input always produces the same latent vector (no sampling, no randomness)
- **Loss** — reconstruction error only (MSE), no KL divergence term
- **Output** — reconstruction quality improves epoch by epoch; final inference saves originals vs reconstructions

**Analogies:**
- Lossy ZIP compressor for images
- Finds a compact "essence" of each digit

**Run time:** ~1-2 minutes on CPU.

---

### 2. `variational_encoders.py` — Variational Autoencoder

**What it does:**
Loads two pretrained Stable Diffusion VAEs from HuggingFace and demonstrates:

1. **Posterior sampling** — encodes your input image into a distribution (not a fixed vector), then samples 4 different latents from it. Each decodes to a slightly different reconstruction, showing the probabilistic nature of VAEs.
2. **Prior sampling** — generates an image from pure random noise `N(0,1)` by decoding through the VAE.
3. **Denoising** — adds Gaussian noise to your image, encodes the noisy version, and reconstructs a cleaner version.

**Key characteristics:**
- **Probabilistic** — encoder outputs a distribution (mean + std), sampling introduces randomness
- **Loss (during training)** — reconstruction + KL divergence (not used here, inference only)
- **Important fix** — SD VAEs expect inputs in `[-1, 1]` range, not `[0, 1]`. The code handles this correctly.

---

### 3. `adversarial.py` — Adversarial Autoencoder (AAE)

**What it does:**
Trains an AAE on MNIST from scratch. Instead of using KL divergence to regularize the latent space (like a VAE), it uses a **discriminator network** (a GAN-style approach).

**How it works:**
Three networks are trained simultaneously:
1. **Encoder** — compresses input to latent vector `z`
2. **Decoder** — reconstructs image from `z`
3. **Discriminator** — tries to tell apart "real" `N(0,1)` noise from the encoder's `z` outputs

Two loss signals drive the training:
- **Reconstruction loss** (encoder + decoder) — same as plain AE
- **Adversarial loss** (encoder vs discriminator) — encoder tries to fool the discriminator into thinking its outputs are real `N(0,1)` samples

**Result:** The latent space is shaped to match a Gaussian distribution, enabling meaningful prior sampling — generating new digits from pure noise actually works.

**Improvements in this version:**
- Label smoothing (0.9 instead of 1.0) prevents discriminator overconfidence
- Adjusted learning rates for stable GAN training

---

### 4. `adaptive.py` — Adaptive Autoencoder

**What it does:**
Trains an adaptive autoencoder on MNIST from scratch. The key innovation is a **gate network** that decides, per input, how many of the 32 latent dimensions to keep active.

**How it works:**
1. Encoder produces 32 raw latent values + a budget score in `[0, 1]`
2. `budget × 32` = number of dimensions to keep (rounded to integer `k`)
3. Only the top-`k` largest-magnitude latent values are kept; the rest are zeroed
4. Decoder reconstructs from the masked latent

**Why this matters:**
A plain AE always uses all 32 dimensions for every image. But a simple digit (e.g., "1") needs fewer dimensions than a complex one (e.g., "8"). The gate network learns this adaptively.

**Loss:** reconstruction MSE + small penalty on average budget used (encourages efficiency).

**Run time:** ~1-2 minutes on CPU.

---

### 5. `sparse_encoders.py` — Sparse Autoencoder (SAE)

**What it does:**
Loads a pretrained sparse autoencoder (via SAELens) that was trained on GPT-2 Small's internal residual stream activations. This is the technique behind **Anthropic's monosemanticity research** — finding interpretable features inside language models.

**How it works:**
1. Loads GPT-2 Small via TransformerLens
2. Runs a prompt through the model, caching activations at layer 8
3. Encodes those activations through the SAE → sparse, overcomplete feature vector
4. Decodes back to the original activation space
5. Prints per-token top active features

**Key characteristics:**
- **Sparse** — only a small fraction of the ~30,000 features activate per token
- **Overcomplete** — feature dimension is much larger than input dimension (30K vs 768)
- **Interpretable** — each feature often corresponds to a specific concept (e.g., "the" in a grammatical context)

**Output:** Average active features per token (L0 norm), reconstruction MSE, and top-5 feature IDs per token (lookup up on Neuronpedia).

---

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** For `sparse_encoders.py`, you may also need:
```bash
pip install sae-lens transformer-lens
```

---

## Usage

**From-scratch training scripts (AE, AAE, Adaptive):**
```bash
python autoencoders.py        # Plain AE — trains in ~1-2 min
python adversarial.py         # AAE — trains in ~2-3 min
python adaptive.py            # Adaptive AE — trains in ~1-2 min
```

**Pretrained model inference scripts (VAE, SAE):**
```bash
python variational_encoders.py   # Requires demo.jpg in root folder
python sparse_encoders.py        # Downloads GPT-2 + SAE on first run
```

**For the VAE script:** Place a `demo.jpg` image in the project root folder.

---

## Outputs

All generated images are saved to structured folders under `outputs/`:

```
outputs/
│
├── autoencoder/
│   ├── ae_originals.png           # 8 MNIST test digits (originals)
│   └── ae_reconstructions.png     # Reconstructed versions (same digits)
│
├── variational/
│   ├── vae_sample_0.png           # Posterior sample 1
│   ├── vae_sample_1.png           # Posterior sample 2
│   ├── vae_sample_2.png           # Posterior sample 3
│   ├── vae_sample_3.png           # Posterior sample 4
│   ├── generated_from_prior.png   # Generated from pure noise
│   ├── original.png               # Clean reference (denoising)
│   ├── noisy.png                  # Noisy input (denoising)
│   └── denoised.png               # Reconstructed (cleaner)
│
├── adversarial/
│   ├── aae_originals.png               # 8 MNIST test digits
│   ├── aae_reconstructions.png         # Reconstructed versions
│   └── aae_generated_from_prior.png    # Generated from prior (N(0,1))
│
└── adaptive/
    ├── adaptive_originals.png          # 8 MNIST test digits
    └── adaptive_reconstructions.png    # Reconstructed versions
```

---

## Key Concepts

| Type | Latent Space | Randomness | Regularization | Can Generate? |
|------|-------------|------------|----------------|---------------|
| **Plain AE** | Fixed vector | None | None | No |
| **VAE** | Distribution (mean + std) | Sampling from posterior | KL divergence | Yes (via prior) |
| **AAE** | Fixed vector (adversarially shaped) | None (but distribution is shaped) | Discriminator (GAN) | Yes (via prior) |
| **Adaptive AE** | Fixed vector (variable sparsity) | None | Budget penalty | No |
| **SAE** | Sparse, overcomplete vector | None | Sparsity (L1 or top-k) | No (interpretability) |

### Autoencoder Variants Explained

**Plain Autoencoder**
Encoder → latent vector `z` → Decoder. All dimensions always used. Loss = reconstruction error only.

**Variational Autoencoder (VAE)**
Encoder outputs parameters of a distribution (mean `μ`, std `σ`). Sampling produces `z`. Loss = reconstruction + KL divergence (pushes distribution toward `N(0,1)`). The smooth latent space enables generation.

**Adversarial Autoencoder (AAE)**
Same architecture as VAE but replaces KL divergence with a discriminator. The discriminator learns to distinguish real `N(0,1)` samples from encoder outputs. The encoder learns to fool it, shaping the latent space implicitly.

**Adaptive Autoencoder**
Adds a gate network that predicts a budget per input. Only the top-k dimensions of the latent vector are kept, where `k` depends on the input's complexity.

**Sparse Autoencoder (SAE)**
Overcomplete bottleneck (larger dimension than input) with sparsity constraint. Used for interpretability — each sparse feature often represents a human-interpretable concept.

---

## Dependencies

- `torch>=2.0.0` — Core deep learning framework
- `torchvision>=0.15.0` — Datasets, transforms, image utilities (MNIST, save_image)
- `diffusers>=0.30.0` — Pretrained VAE models (`AutoencoderKL`)
- `transformers>=4.40.0` — HuggingFace hub integration
- `Pillow>=10.0.0` — Image loading and saving

**For SAE only:**
- `sae-lens` — Pretrained sparse autoencoders
- `transformer-lens` — Language model internals

---

## Notes

- All scripts are designed to run on **CPU** — no GPU required.
- The `data/` folder is auto-created when scripts download MNIST.
- The `venv/` folder, `data/`, `outputs/`, and model weight files (`.safetensors`, `.bin`) are excluded from version control via `.gitignore`.
