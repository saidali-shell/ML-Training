"""
Plain Autoencoder (AE) — deterministic, no distribution, no KL term.
Trains a small conv AE on MNIST (auto-downloaded) then runs inference
on a few test images. Runs fine on CPU in ~1-2 minutes.

Key difference vs VAE:
  - encoder(x) -> single fixed latent vector z (no sampling, no randomness)
  - loss = reconstruction error only (no KL divergence term)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

DEVICE = "cpu"
LATENT_DIM = 32
EPOCHS = 3
BATCH_SIZE = 128


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 28->14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14->7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),                     # deterministic z
        )

    def forward(self, x):
        return self.net(x)   # <- single fixed vector, no sampling


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


class AE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = AE(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"[Autoencoder] Training for {EPOCHS} epochs on MNIST...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)   # <- reconstruction loss only, no KL term

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_ds)
        print(f"  epoch {epoch+1}/{EPOCHS} - recon MSE: {avg_loss:.6f}")

    # ---- Inference ----
    model.eval()
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=True)
    x_test, _ = next(iter(test_loader))

    with torch.no_grad():
        recon, z = model(x_test)

        # confirm determinism: same input -> identical latent every time
        z_again = model.encoder(x_test)
        max_diff = (z - z_again).abs().max().item()

    import os
    out = os.path.join(os.path.dirname(__file__), "outputs", "autoencoder")
    os.makedirs(out, exist_ok=True)
    vutils.save_image(x_test, os.path.join(out, "ae_originals.png"), nrow=8)
    vutils.save_image(recon, os.path.join(out, "ae_reconstructions.png"), nrow=8)

    print("\n[Autoencoder — Inference]")
    print(f"  Latent shape           : {tuple(z.shape)}")
    print(f"  Determinism check (should be 0.0): {max_diff}")
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()