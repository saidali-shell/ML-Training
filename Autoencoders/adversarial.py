"""
Adversarial Autoencoder (AAE) — from-scratch, no pretrained checkpoint exists
for this architecture, same situation as the plain AE.

Key difference vs VAE:
  - VAE regularizes the latent space using a KL-divergence loss (an explicit
    formula comparing the latent distribution to N(0,1)).
  - AAE regularizes the latent space using a DISCRIMINATOR network instead —
    a second neural net is trained to distinguish "real" N(0,1) samples from
    the encoder's actual outputs. The encoder is trained to fool it.
    This is the GAN idea applied to an autoencoder's latent space.

Three networks, two loss signals:
  1. Encoder + Decoder  -> reconstruction loss (like a plain AE)
  2. Encoder (as generator) + Discriminator -> adversarial loss (like a GAN)
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
EPOCHS = 15
BATCH_SIZE = 128


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 28->14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14->7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


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


class Discriminator(nn.Module):
    """Tries to tell apart: real N(0,1) noise  vs  encoder's latent output z."""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),   # 1 = "real" (from N(0,1)), 0 = "fake" (from encoder)
        )

    def forward(self, z):
        return self.net(z)


def main():
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    encoder = Encoder(LATENT_DIM).to(DEVICE)
    decoder = Decoder(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator(LATENT_DIM).to(DEVICE)

    opt_recon = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=5e-5)   # slower disc
    opt_gen = torch.optim.Adam(encoder.parameters(), lr=2e-4)          # faster generator

    print(f"[AAE] Training for {EPOCHS} epochs on MNIST...")
    encoder.train(); decoder.train(); discriminator.train()

    for epoch in range(EPOCHS):
        total_recon, total_disc, total_gen = 0.0, 0.0, 0.0

        for x, _ in train_loader:
            x = x.to(DEVICE)
            bs = x.size(0)

            # ---- Phase 1: reconstruction (same as plain AE) ----
            z = encoder(x)
            recon = decoder(z)
            recon_loss = F.mse_loss(recon, x)

            opt_recon.zero_grad()
            recon_loss.backward()
            opt_recon.step()

            # ---- Phase 2: train discriminator to separate real N(0,1) vs encoder z ----
            with torch.no_grad():
                z_fake = encoder(x)
            z_real = torch.randn(bs, LATENT_DIM, device=DEVICE)

            pred_real = discriminator(z_real)
            pred_fake = discriminator(z_fake)
            # label smoothing (0.9 instead of 1.0): stops discriminator from
            # getting overconfident, which is what starved the encoder's
            # gradient signal in the run above (disc loss -> 0.02, gen loss -> 8.2)
            real_labels = torch.full((bs, 1), 0.9)
            fake_labels = torch.zeros(bs, 1)
            disc_loss = F.binary_cross_entropy(pred_real, real_labels) + \
                        F.binary_cross_entropy(pred_fake, fake_labels)

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # ---- Phase 3: train encoder to fool the discriminator ----
            z_gen = encoder(x)
            pred_gen = discriminator(z_gen)
            gen_loss = F.binary_cross_entropy(pred_gen, torch.ones(bs, 1))  # wants disc to say "real"

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            total_recon += recon_loss.item() * bs
            total_disc += disc_loss.item() * bs
            total_gen += gen_loss.item() * bs

        n = len(train_ds)
        print(f"  epoch {epoch+1}/{EPOCHS} - recon: {total_recon/n:.4f}  "
              f"disc: {total_disc/n:.4f}  gen: {total_gen/n:.4f}")

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    encoder.eval(); decoder.eval()
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=True)
    x_test, _ = next(iter(test_loader))

    with torch.no_grad():
        # 1. Reconstruction
        z = encoder(x_test)
        recon = decoder(z)

        # 2. Generation from prior — unlike the plain AE, this SHOULD work
        #    reasonably well, because the discriminator explicitly trained
        #    the encoder's output distribution to match N(0,1).
        z_prior = torch.randn(8, LATENT_DIM)
        generated = decoder(z_prior)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "adversarial")
    os.makedirs(out_dir, exist_ok=True)
    vutils.save_image(x_test, os.path.join(out_dir, "aae_originals.png"), nrow=8)
    vutils.save_image(recon, os.path.join(out_dir, "aae_reconstructions.png"), nrow=8)
    vutils.save_image(generated, os.path.join(out_dir, "aae_generated_from_prior.png"), nrow=8)

    print("\n[Adversarial Autoencoder — Inference]")
    print(f"  Latent shape: {tuple(z.shape)}")
    print(f"  Saved to {out_dir}")
    print("  Note: unlike the plain AE, prior-sampled generation should look")
    print("  like plausible digits here — that's the adversarial regularization working.")


if __name__ == "__main__":
    main()