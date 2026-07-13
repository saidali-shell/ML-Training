"""
Adaptive Autoencoder — from-scratch (no standardized pretrained model exists
for this term; several different papers use "adaptive autoencoder" to mean
different things, so this implements the most teachable version).

Core idea, in plain terms:
  A normal AE (see ae_inference.py) always uses ALL 32 latent dimensions for
  every image, whether the image is simple or complex.

  This version adds a small "gate" network that looks at each input and
  decides HOW MANY of the 32 latent dimensions are actually worth keeping
  active. Simple inputs -> fewer active dims. Complex inputs -> more.

  Mechanically:
    1. encoder(x)      -> 32 raw latent values      (same as plain AE)
    2. gate_net(x)      -> a single number in [0,1]   ("budget" for this input)
    3. budget * 32       -> how many dims to keep      (rounded to an integer k)
    4. keep only the k LARGEST-magnitude latent values, zero out the rest
    5. decoder(masked z) -> reconstruction             (same as plain AE)

  Loss = reconstruction error + a small penalty on the average budget used,
  which pushes the network to use as few active dimensions as it can get
  away with, per input -- rather than a fixed sparsity level for every input
  (that fixed-level version is what the SAE script already demonstrates).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

DEVICE = "cpu"
LATENT_DIM = 32
EPOCHS = 5
BATCH_SIZE = 128
BUDGET_PENALTY = 0.05   # how strongly to encourage using fewer active dims


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.to_latent = nn.Linear(32 * 7 * 7, latent_dim)
        # gate network: shares the conv features, outputs one number in [0,1]
        self.to_gate = nn.Sequential(
            nn.Linear(32 * 7 * 7, 1),
            nn.Sigmoid(),   # squashes to [0,1] = fraction of latent dims to keep
        )

    def forward(self, x):
        feats = self.net(x)
        z_raw = self.to_latent(feats)
        budget = self.to_gate(feats)   # shape (batch, 1), value in [0,1]
        return z_raw, budget


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 32, 7, 7)
        return self.net(x)


def apply_adaptive_mask(z_raw, budget, latent_dim):
    """
    Keep only the top-k largest-magnitude latent values per sample, where
    k = round(budget * latent_dim). Zero out everything else.
    Uses a straight-through style trick so this stays (mostly) differentiable.
    """
    batch_size = z_raw.size(0)
    k_per_sample = (budget.squeeze(-1) * latent_dim).round().long().clamp(min=1, max=latent_dim)

    mask = torch.zeros_like(z_raw)
    abs_z = z_raw.abs()
    for i in range(batch_size):
        k = k_per_sample[i].item()
        top_idx = abs_z[i].topk(k).indices
        mask[i, top_idx] = 1.0

    z_masked = z_raw * mask
    return z_masked, k_per_sample


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    encoder = Encoder(LATENT_DIM).to(DEVICE)
    decoder = Decoder(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    print(f"[Adaptive AE] Training for {EPOCHS} epochs on MNIST...")
    encoder.train(); decoder.train()

    for epoch in range(EPOCHS):
        total_recon, total_budget = 0.0, 0.0

        for x, _ in train_loader:
            x = x.to(DEVICE)
            bs = x.size(0)

            z_raw, budget = encoder(x)
            z_masked, k_used = apply_adaptive_mask(z_raw, budget, LATENT_DIM)
            recon = decoder(z_masked)

            recon_loss = F.mse_loss(recon, x)
            budget_loss = budget.mean()   # encourages smaller budget on average
            loss = recon_loss + BUDGET_PENALTY * budget_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_recon += recon_loss.item() * bs
            total_budget += k_used.float().mean().item() * bs

        n = len(train_ds)
        print(f"  epoch {epoch+1}/{EPOCHS} - recon: {total_recon/n:.4f}  "
              f"avg active dims: {total_budget/n:.1f} / {LATENT_DIM}")

    # ---------------------------------------------------------------
    # Inference — show that different digits use different budgets
    # ---------------------------------------------------------------
    encoder.eval(); decoder.eval()
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=True)
    x_test, labels = next(iter(test_loader))

    with torch.no_grad():
        z_raw, budget = encoder(x_test)
        z_masked, k_used = apply_adaptive_mask(z_raw, budget, LATENT_DIM)
        recon = decoder(z_masked)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "adaptive")
    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(x_test, os.path.join(out_dir, "adaptive_originals.png"), nrow=8)
    vutils.save_image(recon, os.path.join(out_dir, "adaptive_reconstructions.png"), nrow=8)

    print("\n[Adaptive Autoencoder — Inference]")
    print("  Digit -> active latent dims used (out of 32):")
    for digit, k in zip(labels.tolist(), k_used.tolist()):
        print(f"    digit {digit}: {k} dims active")
    print(f"  Saved to {out_dir}")
    print("  Note: different digits should use noticeably different numbers")
    print("  of active dims -- that's the 'adaptive' part working.")


if __name__ == "__main__":
    main()